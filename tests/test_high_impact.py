"""Tests for high-impact enhancements:
- Casing and sequel protection in matching
- Token/prefix inverted-index candidate duplicate detection
- Non-destructive resolver auto-merging (zero data loss)
- Multi-format exports (Parquet & SQLite with indexing)
- Review queue tracking, auto-resolution, and match overrides
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.merge_pipeline import (
    CANONICAL_SCHEMA,
    SourceConfig,
    build_name_match_key,
    get_name_confidence_score,
    identify_potential_duplicates,
    normalize_source,
    _run_fuzzy_dedup,
)
from utils.resolvers import resolver as default_resolver
from utils.csv_export import write_to_parquet, write_to_sqlite, export_dataset
from utils.review_queue import (
    load_review_queue,
    save_review_queue,
    get_review_status,
    auto_resolve_queue,
    record_override,
)


class TestCasingAndSequels:
    """Test title case normalization and sequel collision prevention."""

    def test_build_name_match_key_case_insensitivity(self):
        k_lower = build_name_match_key("super mario bros.")
        k_upper = build_name_match_key("SUPER MARIO BROS.")
        k_title = build_name_match_key("Super Mario Bros.")
        assert k_lower == k_title
        assert k_upper == k_title

    def test_build_name_match_key_roman_casing(self):
        k_lower = build_name_match_key("final fantasy vii")
        k_upper = build_name_match_key("FINAL FANTASY VII")
        assert k_lower == k_upper
        assert "VII" in k_lower

    def test_sequel_mismatch_penalty(self):
        # Different sequel numbers must not match with high confidence
        scores = get_name_confidence_score("Super Mario Bros.", "Super Mario Bros. 2")
        assert scores["confidence"] < 0.50

        scores_roman = get_name_confidence_score("Final Fantasy VII", "Final Fantasy VIII")
        assert scores_roman["confidence"] < 0.50

    def test_matching_sequels_allowed(self):
        # Roman vs Arabic digit for the SAME sequel number must match
        scores = get_name_confidence_score("Final Fantasy VII", "Final Fantasy 7")
        assert scores["confidence"] >= 0.85


class TestFuzzyCandidateDetection:
    """Test inverted-index blocking and candidate generation."""

    def test_different_length_candidates_found(self):
        df = pd.DataFrame([
            {"name": "Chrono Trigger", "platform": "SNES"},
            {"name": "Chrono Trigger (USA)", "platform": "SNES"},
        ])
        dups = identify_potential_duplicates(df, threshold=0.8)
        assert len(dups) == 1
        idx1, idx2, conf = dups[0]
        assert conf >= 0.85

    def test_platform_isolation_prevents_cross_platform_merge(self):
        df = pd.DataFrame([
            {"name": "Tetris", "platform": "Game Boy"},
            {"name": "Tetris", "platform": "NES"},
        ])
        dups = identify_potential_duplicates(df, threshold=0.8)
        assert len(dups) == 0


class TestNonDestructiveAutoMerge:
    """Test that auto-merge merges fields with resolvers instead of discarding data."""

    def test_auto_merge_preserves_missing_fields(self):
        df = pd.DataFrame([
            {
                "name": "Super Mario Bros.",
                "platform": "NES",
                "summary": "Full detailed description of saving Princess Peach in the Mushroom Kingdom.",
                "rating": np.nan,
                "genres": "Platformer",
            },
            {
                "name": "Super Mario Bros",
                "platform": "NES",
                "summary": "Short intro.",
                "rating": 9.2,
                "genres": "Action",
            },
        ])

        deduped, review_q = _run_fuzzy_dedup(
            df,
            auto_merge_high_threshold=0.90,
            resolver_map=default_resolver,
        )

        assert len(deduped) == 1
        kept = deduped.iloc[0]
        # Longer summary kept
        assert "Mushroom Kingdom" in kept["summary"]
        # Rating preserved from dropped row
        assert kept["rating"] == 9.2
        # Genres collected uniquely
        assert "Action" in kept["genres"] and "Platformer" in kept["genres"]


class TestMultiFormatExports:
    """Test Parquet and SQLite database exports."""

    def test_write_to_parquet(self, tmp_path):
        df = pd.DataFrame([
            {"name": "Super Mario 64", "platform": "Nintendo 64", "rating": 9.5, "release_year": 1996},
        ])
        pq_path = tmp_path / "games.parquet"
        write_to_parquet(df, pq_path, CANONICAL_SCHEMA)
        assert pq_path.exists()
        loaded = pd.read_parquet(pq_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["name"] == "Super Mario 64"

    def test_write_to_sqlite_with_indices(self, tmp_path):
        df = pd.DataFrame([
            {"name": "Chrono Trigger", "platform": "SNES", "release_year": 1995},
        ])
        db_path = tmp_path / "games.db"
        write_to_sqlite(df, db_path, table_name="test_games", create_indices=True)
        assert db_path.exists()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, platform, release_year FROM test_games")
            rows = cursor.fetchall()
            assert rows == [("Chrono Trigger", "SNES", 1995)]

            # Verify indices were created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = [r[0] for r in cursor.fetchall()]
            assert "idx_test_games_name" in indices
            assert "idx_test_games_platform" in indices

    def test_export_dataset_all_formats(self, tmp_path):
        df = pd.DataFrame([
            {"name": "Zelda", "platform": "NES"},
        ])
        base = tmp_path / "catalog"
        exported = export_dataset(df, base, formats=("csv", "parquet", "sqlite"))
        assert "csv" in exported and exported["csv"].exists()
        assert "parquet" in exported and exported["parquet"].exists()
        assert "sqlite" in exported and exported["sqlite"].exists()


class TestReviewQueueAndOverrides:
    """Test review queue helpers and override application."""

    def test_review_status_and_auto_resolve(self, tmp_path):
        q_path = tmp_path / "review_queue.csv"
        ov_path = tmp_path / "match_overrides.json"

        q_df = pd.DataFrame([
            {
                "name1": "Zelda: Ocarina of Time",
                "name2": "Zelda: Ocarina of Time (Collector's)",
                "confidence": 0.92,
                "platform1": "Nintendo 64",
                "platform2": "Nintendo 64",
                "merged": False,
            },
            {
                "name1": "Mario Kart",
                "name2": "Mario Party",
                "confidence": 0.81,
                "platform1": "Nintendo 64",
                "platform2": "Nintendo 64",
                "merged": False,
            },
        ])
        save_review_queue(q_df, q_path)

        status = get_review_status(q_path)
        assert status["total_pairs"] == 2
        assert status["pending_pairs"] == 2

        resolved = auto_resolve_queue(q_path, ov_path, min_confidence=0.90)
        assert resolved == 1

        with open(ov_path) as f:
            overrides = json.load(f)
        assert "Nintendo 64" in overrides

    def test_normalize_source_applies_overrides(self, tmp_path):
        ov_path = tmp_path / "match_overrides.json"
        with open(ov_path, "w") as f:
            json.dump({"SNES": {"Chrono Hack": "Chrono Trigger"}}, f)

        # Monkeypatch override path loader
        import utils.merge_pipeline as mp_module
        orig_loader = mp_module.load_match_overrides
        mp_module.load_match_overrides = lambda: json.loads(open(ov_path).read())

        try:
            cfg = SourceConfig(name="test_src")
            df = pd.DataFrame([
                {"name": "Chrono Hack", "platform": "SNES"},
            ])
            normalized = normalize_source(df, cfg, target_columns=["name", "platform"], key_columns=["name", "platform"])
            assert normalized.iloc[0]["name"] == "Chrono Trigger"
        finally:
            mp_module.load_match_overrides = orig_loader

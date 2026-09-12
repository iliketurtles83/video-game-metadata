"""Tests for high-impact enhancements:
- Casing and sequel protection in matching
- Token/prefix inverted-index candidate duplicate detection
- Non-destructive resolver auto-merging (zero data loss)
- Multi-format exports (Parquet & SQLite with indexing)
- Review queue tracking, auto-resolution, and match overrides
- Provenance tracking (_source column across merge pipeline)
- Rating scale normalization (0-100 / 0-1 → 0-10)
- Genre translation config loading
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


class TestProvenanceTracking:
    """Tests for _source column provenance tracking across the merge pipeline."""

    def test_normalize_source_adds_source_column(self):
        """normalize_source should add _source column with the config name."""
        df = pd.DataFrame([
            {"name": "Super Mario Bros.", "platform": "NES"},
            {"name": "Zelda", "platform": "NES"},
        ])
        cfg = SourceConfig(name="launchbox")
        result = normalize_source(
            df, cfg,
            target_columns=["name", "platform"],
            key_columns=["name", "platform"],
        )
        assert "_source" in result.columns
        assert (result["_source"] == "launchbox").all()

    def test_source_column_survives_fuzzy_dedup(self):
        """_source should be preserved/merged after fuzzy dedup."""
        df = pd.DataFrame([
            {"name": "Chrono Trigger", "platform": "SNES", "_source": "launchbox", "summary": "A classic RPG"},
            {"name": "Chrono Trigger", "platform": "SNES", "_source": "gametdb", "summary": None},
        ])
        result, _ = _run_fuzzy_dedup(df, resolver_map=default_resolver)
        assert "_source" in result.columns
        # After dedup, _source should contain both source names
        source_val = str(result.iloc[0]["_source"])
        assert "launchbox" in source_val

    def test_source_column_collects_unique_across_merge(self):
        """When rows from different sources merge, _source should collect all contributing sources."""
        from utils.merge_pipeline import merge_into_main
        from utils.resolvers import resolver as r
        main_df = pd.DataFrame([
            {"name": "Pac-Man", "platform": "Arcade", "_source": "launchbox", "_name_match_key": "Pac-Man"},
        ])
        source_df = pd.DataFrame([
            {"name": "Pac-Man", "platform": "Arcade", "_source": "gametdb", "_name_match_key": "Pac-Man"},
        ])
        result = merge_into_main(
            main_df, source_df,
            key_columns=["_name_match_key", "platform"],
            resolver_map=r,
        )
        source_val = str(result.iloc[0]["_source"])
        assert "launchbox" in source_val
        assert "gametdb" in source_val


class TestRatingNormalization:
    """Tests for normalize_rating_scales() in data_cleaning.py."""

    def test_100_scale_normalized_to_10(self):
        """Ratings on a 0-100 scale should be divided by 10."""
        from utils.data_cleaning import normalize_rating_scales
        df = pd.DataFrame({"rating": [85, 90, 50, 100], "user_rating": [7.5, 8.0, 5.0, 10.0]})
        result = normalize_rating_scales(df)
        # 85 -> 8.5, 90 -> 9.0, 50 -> 5.0, 100 -> 10.0
        assert result["rating"].iloc[0] == pytest.approx(8.5, abs=0.01)
        assert result["rating"].iloc[1] == pytest.approx(9.0, abs=0.01)
        assert result["rating"].iloc[2] == pytest.approx(5.0, abs=0.01)
        # user_rating already in 0-10 range, should be unchanged
        assert result["user_rating"].iloc[0] == pytest.approx(7.5, abs=0.01)

    def test_01_scale_normalized_to_10(self):
        """Ratings on a 0-1 scale should be multiplied by 10."""
        from utils.data_cleaning import normalize_rating_scales
        df = pd.DataFrame({"rating": [0.85, 0.5], "user_rating": [0.75, 0.9]})
        result = normalize_rating_scales(df)
        assert result["rating"].iloc[0] == pytest.approx(8.5, abs=0.01)
        assert result["user_rating"].iloc[0] == pytest.approx(7.5, abs=0.01)

    def test_already_normalized_unchanged(self):
        """Ratings already in 0-10 range should not be modified."""
        from utils.data_cleaning import normalize_rating_scales
        df = pd.DataFrame({"rating": [7.0, 8.5, 10.0, 1.0], "user_rating": [5.0, 9.0, 3.0, 2.0]})
        result = normalize_rating_scales(df)
        assert result["rating"].iloc[0] == pytest.approx(7.0, abs=0.01)
        assert result["rating"].iloc[1] == pytest.approx(8.5, abs=0.01)
        assert result["user_rating"].iloc[2] == pytest.approx(3.0, abs=0.01)

    def test_clipping_extreme_values(self):
        """Values beyond 100 should be clipped to target_scale."""
        from utils.data_cleaning import normalize_rating_scales
        df = pd.DataFrame({"rating": [150, -5], "user_rating": [10.0, 0.0]})
        result = normalize_rating_scales(df)
        assert result["rating"].iloc[0] <= 10.0
        assert result["rating"].iloc[1] == 0.0

    def test_nan_values_preserved(self):
        """NaN ratings should remain NaN after normalization."""
        from utils.data_cleaning import normalize_rating_scales
        df = pd.DataFrame({"rating": [np.nan, 7.0], "user_rating": [5.0, np.nan]})
        result = normalize_rating_scales(df)
        assert pd.isna(result["rating"].iloc[0])
        assert pd.isna(result["user_rating"].iloc[1])

    def test_cleaning_pipeline_includes_normalization(self):
        """run_cleaning_pipeline should normalize ratings when normalize_ratings=True."""
        from utils.data_cleaning import run_cleaning_pipeline
        df = pd.DataFrame({
            "name": ["Game A", "Game B"],
            "rating": [85, 7.0],
            "user_rating": [0.75, 8.0],
            "genres": ["Action", "RPG"],
            "release_date": [None, None],
            "players": [None, None],
            "cooperative": [None, None],
            "release_year": [None, None],
        })
        result = run_cleaning_pipeline(df, normalize_ratings=True)
        # 85 should be normalized to 8.5
        assert result["rating"].iloc[0] == pytest.approx(8.5, abs=0.1)
        # 0.75 should be normalized to 7.5
        assert result["user_rating"].iloc[0] == pytest.approx(7.5, abs=0.1)


class TestGenreTranslationConfig:
    """Tests for genre translation config file loading."""

    def test_genre_translation_file_exists_and_valid(self):
        """config/genre_translations.json should exist and be valid JSON."""
        config_path = Path(__file__).parent.parent / "config" / "genre_translations.json"
        assert config_path.exists(), "genre_translations.json should exist"
        with open(config_path, "r", encoding="utf-8") as f:
            translations = json.load(f)
        assert isinstance(translations, dict)
        assert len(translations) >= 40, f"Expected 40+ translations, got {len(translations)}"
        # Verify key Portuguese terms are mapped
        assert translations.get("Ação") == "Action"
        assert translations.get("Plataforma") == "Platform"
        assert translations.get("Estratégia") == "Strategy"

    def test_translate_genres_with_config_map(self):
        """translate_genres should correctly translate Portuguese genres."""
        from utils.data_cleaning import translate_genres
        config_path = Path(__file__).parent.parent / "config" / "genre_translations.json"
        with open(config_path, "r", encoding="utf-8") as f:
            translation_map = json.load(f)
        df = pd.DataFrame({"genres": ["Ação", "Plataforma, Aventura", "Estratégia"]})
        result = translate_genres(df, translation_map=translation_map)
        assert result["genres"].iloc[0] == "Action"
        assert "Platform" in result["genres"].iloc[1]
        assert "Adventure" in result["genres"].iloc[1]
        assert result["genres"].iloc[2] == "Strategy"

    def test_clean_config_references_genre_translations(self):
        """clean_config.json should reference genre_translations.json."""
        from utils.pipeline import load_merge_config
        config_path = Path(__file__).parent.parent / "config" / "clean_config.json"
        config = load_merge_config(str(config_path))
        translate_step = config.get("steps", {}).get("translate_genres", {})
        assert translate_step.get("enabled") is True
        assert translate_step.get("map_path") == "config/genre_translations.json"


class TestDateCleaningAndReconciliation:
    """Tests for placeholder date removal, truncated year cleanup, and date/year alignment."""

    def test_clean_placeholder_dates_on_modern_platforms(self):
        """1980-01-01 and 1970-01-01 on post-1980 platforms should be cleared to NaT."""
        from utils.data_cleaning import clean_placeholder_dates
        df = pd.DataFrame([
            {"name": "Zelda: Breath of the Wild", "platform": "Nintendo Switch", "release_date": "1980-01-01"},
            {"name": "12 Family Games", "platform": "Nintendo DS", "release_date": "1980-01-01"},
            {"name": "Retro Space", "platform": "Arcade", "release_date": "1980-01-01"},
            {"name": "Mario 64", "platform": "Nintendo 64", "release_date": "1996-06-23"},
        ])
        result = clean_placeholder_dates(df)
        assert pd.isna(result.iloc[0]["release_date"])
        assert pd.isna(result.iloc[1]["release_date"])
        # Arcade existed in 1980, so it should not be cleared
        assert pd.notna(result.iloc[2]["release_date"])
        # Valid date preserved
        assert result.iloc[3]["release_date"] == pd.Timestamp("1996-06-23")

    def test_clean_release_years_invalid_truncated(self):
        """Truncated release years like 19, 199, 200 should be set to NA."""
        from utils.data_cleaning import clean_release_years
        df = pd.DataFrame({
            "name": ["Game 1", "Game 2", "Game 3", "Game 4", "Pinball 1933"],
            "release_year": [200, 199, 19, 1995, 1933],
        })
        result = clean_release_years(df, min_year=1900, max_year=2030)
        assert pd.isna(result["release_year"].iloc[0])
        assert pd.isna(result["release_year"].iloc[1])
        assert pd.isna(result["release_year"].iloc[2])
        assert result["release_year"].iloc[3] == 1995
        assert result["release_year"].iloc[4] == 1933

    def test_derive_release_year_reconciles_conflicts(self):
        """When release_date contradicts release_year, reconcile_conflicts=True should align to release_date."""
        from utils.data_cleaning import derive_release_year
        df = pd.DataFrame([
            {"name": "Game A", "release_date": "2016-12-31", "release_year": 2015},
            {"name": "Game B", "release_date": "1992-12-31", "release_year": 1993},
            {"name": "Game C", "release_date": "1997-02-21", "release_year": pd.NA},
            {"name": "Game D", "release_date": None, "release_year": 2008},
        ])
        result = derive_release_year(df, reconcile_conflicts=True)
        # 2016-12-31 -> year 2016
        assert result["release_year"].iloc[0] == 2016
        # 1992-12-31 -> year 1992
        assert result["release_year"].iloc[1] == 1992
        # Missing year filled
        assert result["release_year"].iloc[2] == 1997
        # Undated game keeps its original release_year
        assert result["release_year"].iloc[3] == 2008

    def test_full_cleaning_pipeline_resolves_all_date_conflicts(self):
        """End-to-end cleaning pipeline handles dummy dates, truncated years, and aligns date/year."""
        from utils.data_cleaning import run_cleaning_pipeline
        df = pd.DataFrame([
            {
                "name": "12 Family Games",
                "platform": "Nintendo DS",
                "release_date": "1980-01-01",
                "release_year": 2008,
                "genres": "Compilation",
                "rating": 85,
                "user_rating": 6.7,
                "players": "1",
                "cooperative": None,
            },
            {
                "name": "0-to-X",
                "platform": "Nintendo Entertainment System",
                "release_date": "2016-12-31",
                "release_year": 2015,
                "genres": "Puzzle",
                "rating": None,
                "user_rating": 0.59,
                "players": "1",
                "cooperative": None,
            },
            {
                "name": "Fruit Machine Arcade",
                "platform": "Arcade",
                "release_date": None,
                "release_year": 200,
                "genres": "Casino",
                "rating": 50,
                "user_rating": 7.0,
                "players": "1",
                "cooperative": None,
            },
        ])
        cleaned = run_cleaning_pipeline(df)
        # 12 Family Games: dummy date cleared, original 2008 year retained!
        assert pd.isna(cleaned.iloc[0]["release_date"])
        assert cleaned.iloc[0]["release_year"] == 2008
        # 0-to-X: 2016-12-31 and release_year aligned to 2016 (zero conflict!)
        assert cleaned.iloc[1]["release_year"] == 2016
        assert cleaned.iloc[1]["release_date"].year == 2016
        # Fruit Machine: truncated year 200 cleared to NA
        assert pd.isna(cleaned.iloc[2]["release_year"])
        # Rating scale normalized
        assert cleaned.iloc[0]["rating"] == pytest.approx(8.5, abs=0.1)
        assert cleaned.iloc[1]["user_rating"] == pytest.approx(5.9, abs=0.1)


class TestDatasetValidation:
    """Tests for utils/validate_dataset.py."""

    def test_validate_dataset_valid_data(self, tmp_path):
        """Clean dataset adhering to rules should pass validation."""
        from utils.validate_dataset import validate_dataset
        df = pd.DataFrame({
            "name": ["Game 1", "Game 2"],
            "platform": ["NES", "SNES"],
            "release_date": ["1990-01-01", "1992-05-15"],
            "release_year": [1990, 1992],
            "genres": ["Action", "RPG"],
            "developer": ["Dev 1", "Dev 2"],
            "publisher": ["Pub 1", "Pub 2"],
            "players": [1, 2],
            "cooperative": [False, True],
            "rating": [8.5, 9.0],
            "user_rating": [7.8, 8.4],
            "summary": ["A game", "Another game"],
            "filename": ["g1", "g2"],
            "version": ["1.0", "1.0"],
        })
        csv_file = tmp_path / "valid.csv"
        df.to_csv(csv_file, index=False)
        passed, report = validate_dataset(csv_file)
        assert passed is True
        assert len(report["errors"]) == 0

    def test_validate_dataset_detects_rating_out_of_bounds(self, tmp_path):
        """Ratings > 10.0 should trigger validation errors."""
        from utils.validate_dataset import validate_dataset
        df = pd.DataFrame({
            "name": ["Game 1"],
            "platform": ["NES"],
            "release_date": ["1990-01-01"],
            "release_year": [1990],
            "genres": ["Action"],
            "developer": ["Dev 1"],
            "publisher": ["Pub 1"],
            "players": [1],
            "cooperative": [False],
            "rating": [85],  # 85 > 10.0!
            "user_rating": [7.8],
            "summary": ["A game"],
            "filename": ["g1"],
            "version": ["1.0"],
        })
        csv_file = tmp_path / "bad_rating.csv"
        df.to_csv(csv_file, index=False)
        passed, report = validate_dataset(csv_file)
        assert passed is False
        assert any("rating > 10.0" in e for e in report["errors"])

    def test_validate_dataset_detects_date_year_mismatch(self, tmp_path):
        """Contradictions between release_date and release_year should be flagged."""
        from utils.validate_dataset import validate_dataset
        df = pd.DataFrame({
            "name": ["Game 1"],
            "platform": ["NES"],
            "release_date": ["2016-12-31"],
            "release_year": [2015],  # Mismatch!
            "genres": ["Action"],
            "developer": ["Dev 1"],
            "publisher": ["Pub 1"],
            "players": [1],
            "cooperative": [False],
            "rating": [8.5],
            "user_rating": [7.8],
            "summary": ["A game"],
            "filename": ["g1"],
            "version": ["1.0"],
        })
        csv_file = tmp_path / "mismatch.csv"
        df.to_csv(csv_file, index=False)
        passed, report = validate_dataset(csv_file)
        assert passed is False
        assert any("release_date.year != release_year" in e for e in report["errors"])

    def test_validate_dataset_detects_untranslated_genres(self, tmp_path):
        """Portuguese genre terms should be flagged if present."""
        from utils.validate_dataset import validate_dataset
        df = pd.DataFrame({
            "name": ["Game 1"],
            "platform": ["NES"],
            "release_date": ["1990-01-01"],
            "release_year": [1990],
            "genres": ["Ação, Plataforma"],  # Untranslated!
            "developer": ["Dev 1"],
            "publisher": ["Pub 1"],
            "players": [1],
            "cooperative": [False],
            "rating": [8.5],
            "user_rating": [7.8],
            "summary": ["A game"],
            "filename": ["g1"],
            "version": ["1.0"],
        })
        csv_file = tmp_path / "untranslated.csv"
        df.to_csv(csv_file, index=False)
        passed, report = validate_dataset(csv_file)
        assert passed is False
        assert any("untranslated Portuguese genre" in e for e in report["errors"])



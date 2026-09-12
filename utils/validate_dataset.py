"""Dataset validation utility.

Validates cleaned and merged game metadata datasets against domain rules,
schema expectations, and data quality constraints defined in the Definition of Done.

Usage:
    python utils/validate_dataset.py [--input output/cleaned_games.csv] [--parquet output/cleaned_games.parquet] [--sqlite output/cleaned_games.db]
    python -m utils validate [--input output/cleaned_games.csv]
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.merge_pipeline import CANONICAL_SCHEMA


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Check that canonical columns are present."""
    errors = []
    required_core = ["name", "platform"]
    for col in required_core:
        if col not in df.columns:
            errors.append(f"Missing mandatory core column: '{col}'")

    for col in CANONICAL_SCHEMA:
        if col not in df.columns:
            errors.append(f"Missing canonical schema column: '{col}'")

    return errors


def validate_ratings(df: pd.DataFrame) -> list[str]:
    """Check that ratings conform to the 0.0 - 10.0 scale."""
    errors = []
    for col in ["rating", "user_rating"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        too_high = (vals > 10.0).sum()
        too_low = (vals < 0.0).sum()
        if too_high > 0:
            errors.append(f"Column '{col}' has {too_high} records with rating > 10.0 (max: {vals.max()})")
        if too_low > 0:
            errors.append(f"Column '{col}' has {too_low} records with rating < 0.0 (min: {vals.min()})")

    return errors


def validate_date_year_consistency(df: pd.DataFrame) -> list[str]:
    """Verify that release_date.year and release_year never contradict."""
    errors = []
    if "release_date" not in df.columns or "release_year" not in df.columns:
        return errors

    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")
    parsed_years = pd.to_numeric(df["release_year"], errors="coerce")

    mask_both = parsed_dates.notna() & parsed_years.notna()
    if mask_both.any():
        date_years = parsed_dates.loc[mask_both].dt.year
        year_values = parsed_years.loc[mask_both]
        mismatches = (date_years != year_values).sum()
        if mismatches > 0:
            errors.append(f"Found {mismatches} records where release_date.year != release_year")

    # Check for truncated/impossible release years
    if parsed_years.notna().any():
        valid_years = parsed_years.dropna()
        invalid_years = ((valid_years < 1900) | (valid_years > 2030)).sum()
        if invalid_years > 0:
            sample_invalid = valid_years[(valid_years < 1900) | (valid_years > 2030)].head(5).tolist()
            errors.append(f"Found {invalid_years} records with release_year outside 1900-2030 (samples: {sample_invalid})")

    return errors


def validate_genres(df: pd.DataFrame, translation_map_path: Optional[str] = "config/genre_translations.json") -> list[str]:
    """Check for untranslated foreign genre terms."""
    errors = []
    if "genres" not in df.columns:
        return errors

    map_path = Path(translation_map_path) if translation_map_path else None
    if map_path and map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                translations = json.load(f)
            # Only check terms that actually differ from their English translation
            pt_terms = [k for k, v in translations.items() if k.lower() != v.lower()]
            genre_series = df["genres"].dropna().astype(str)
            untranslated_terms = []
            untranslated_count = 0
            for term in pt_terms:
                pattern = r'(?<![a-zA-Z0-9À-ÿ])' + re.escape(term) + r'(?![a-zA-Z0-9À-ÿ])'
                matches = int(genre_series.str.contains(pattern, regex=True).sum())
                if matches > 0:
                    untranslated_terms.append(f"{term} ({matches})")
                    untranslated_count += matches
            if untranslated_count > 0:
                sample_str = ", ".join(untranslated_terms[:5])
                errors.append(f"Found {untranslated_count} occurrences of untranslated Portuguese genre terms (samples: {sample_str})")
        except Exception as e:
            errors.append(f"Failed to check genre translations: {e}")

    return errors


def validate_multi_format_parity(
    csv_path: Optional[Path],
    parquet_path: Optional[Path],
    sqlite_path: Optional[Path],
    table_name: str = "cleaned_games",
) -> list[str]:
    """Verify that CSV, Parquet, and SQLite exports have matching row counts."""
    errors = []
    counts: dict[str, int] = {}

    if csv_path and csv_path.exists():
        try:
            counts["csv"] = len(pd.read_csv(csv_path, usecols=[0], low_memory=False))
        except Exception as e:
            errors.append(f"Could not read CSV export: {e}")

    if parquet_path and parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            counts["parquet"] = pq.read_metadata(str(parquet_path)).num_rows
        except Exception as e:
            errors.append(f"Could not read Parquet export: {e}")

    if sqlite_path and sqlite_path.exists():
        try:
            with sqlite3.connect(sqlite_path) as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                counts["sqlite"] = cur.fetchone()[0]
        except Exception as e:
            errors.append(f"Could not read SQLite export: {e}")

    if len(counts) > 1:
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            errors.append(f"Export row count mismatch between formats: {counts}")

    return errors


def generate_completeness_report(df: pd.DataFrame) -> dict[str, Any]:
    """Generate completeness summary per column and platform."""
    total_rows = len(df)
    col_stats = {}
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        pct = round(non_null / total_rows * 100, 2) if total_rows > 0 else 0.0
        col_stats[col] = {
            "non_null_count": non_null,
            "completeness_pct": pct,
        }

    return {
        "total_rows": total_rows,
        "unique_platforms": int(df["platform"].nunique()) if "platform" in df.columns else 0,
        "unique_names": int(df["name"].nunique()) if "name" in df.columns else 0,
        "columns": col_stats,
    }


def validate_dataset(
    input_path: str | Path = "output/cleaned_games.csv",
    parquet_path: Optional[str | Path] = None,
    sqlite_path: Optional[str | Path] = None,
    translation_map: Optional[str | Path] = "config/genre_translations.json",
) -> tuple[bool, dict[str, Any]]:
    """Run full validation against a dataset.

    Returns:
        Tuple of (passed: bool, report: dict).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return False, {"error": f"File not found: {input_path}"}

    print(f"Loading dataset from {input_path}...")
    if input_path.suffix == ".pkl":
        df = pd.read_pickle(input_path)
    elif input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    all_errors: list[str] = []

    # 1. Schema check
    schema_errors = validate_schema(df)
    all_errors.extend(schema_errors)

    # 2. Ratings scale check
    rating_errors = validate_ratings(df)
    all_errors.extend(rating_errors)

    # 3. Date / Year consistency
    date_errors = validate_date_year_consistency(df)
    all_errors.extend(date_errors)

    # 4. Genre translations
    genre_errors = validate_genres(df, translation_map_path=str(translation_map) if translation_map else None)
    all_errors.extend(genre_errors)

    # 5. Multi-format parity
    if parquet_path is None and input_path.with_suffix(".parquet").exists():
        parquet_path = input_path.with_suffix(".parquet")
    if sqlite_path is None and input_path.with_suffix(".db").exists():
        sqlite_path = input_path.with_suffix(".db")

    parity_errors = validate_multi_format_parity(
        csv_path=input_path if input_path.suffix == ".csv" else None,
        parquet_path=Path(parquet_path) if parquet_path else None,
        sqlite_path=Path(sqlite_path) if sqlite_path else None,
    )
    all_errors.extend(parity_errors)

    completeness = generate_completeness_report(df)

    passed = len(all_errors) == 0

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Rows:         {completeness['total_rows']}")
    print(f"Unique Platforms:   {completeness['unique_platforms']}")
    print(f"Unique Titles:      {completeness['unique_names']}")
    print("-" * 60)
    print("Column Completeness:")
    for col, stat in completeness["columns"].items():
        bar = "█" * int(stat["completeness_pct"] // 5)
        print(f"  {col:<16} {stat['completeness_pct']:>6.2f}%  {bar}")
    print("-" * 60)

    if passed:
        print("RESULT: PASSED (All validation checks satisfied!)")
    else:
        print(f"RESULT: FAILED ({len(all_errors)} issues found)")
        for err in all_errors:
            print(f"  - {err}")
    print("=" * 60)

    report = {
        "passed": passed,
        "errors": all_errors,
        "completeness": completeness,
    }
    return passed, report


def main():
    parser = argparse.ArgumentParser(description="Validate game metadata dataset")
    parser.add_argument("--input", "-i", default="output/cleaned_games.csv",
                        help="Path to input dataset file (CSV, Parquet, or pickle)")
    parser.add_argument("--parquet", "-p", default=None,
                        help="Path to Parquet file for parity check")
    parser.add_argument("--sqlite", "-s", default=None,
                        help="Path to SQLite DB file for parity check")
    parser.add_argument("--translations", "-t", default="config/genre_translations.json",
                        help="Path to genre translations JSON")
    args = parser.parse_args()

    passed, _ = validate_dataset(
        input_path=args.input,
        parquet_path=args.parquet,
        sqlite_path=args.sqlite,
        translation_map=args.translations,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

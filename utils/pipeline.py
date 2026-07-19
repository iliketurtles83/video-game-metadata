"""
CLI pipeline runner for the video game metadata merge pipeline.

Loads source configurations from JSON, builds SourceConfig objects,
runs the merge pipeline, applies cleaning, and exports CSVs.

Usage:
    python -m utils pipeline run --config config/merge_config.json
    python -m utils pipeline clean --config config/clean_config.json --input output/merged_df.pkl
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.merge_pipeline import SourceConfig, run_merge_pipeline
from utils.gamelist_parser import load_all_gamelists
from utils.data_cleaning import run_cleaning_pipeline
from utils.csv_export import write_to_csv
from utils.merge_pipeline import CANONICAL_SCHEMA


# ---------------------------------------------------------------------------
# Config -> SourceConfig helpers
# ---------------------------------------------------------------------------

def _build_transforms(transforms_config: list[dict]) -> dict[str, Any]:
    """Convert serializable transform specs to pandas Series -> Series functions."""
    transforms: dict[str, Any] = {}
    for spec in transforms_config:
        column = spec["column"]
        operation = spec["operation"]

        if operation == "scale":
            factor = spec["factor"]

            def make_scale(f):
                return lambda s: pd.to_numeric(s, errors="coerce") * f

            transforms[column] = make_scale(factor)
        else:
            print(f"  [warn] unknown transform operation '{operation}', skipping")
    return transforms


def _build_loader(path: str, loader_filter: Optional[dict] = None):
    """Build a DataFrameLoader callable from config."""
    if loader_filter is None:
        return None

    col = loader_filter["column"]
    val = loader_filter["value"]

    def loader():
        df = pd.read_csv(path, low_memory=False)
        return df[df[col] == val]

    return loader


def config_to_source_config(name: str, cfg: dict, base_path: str = ".") -> SourceConfig:
    """Convert a source config dict (from JSON) into a SourceConfig dataclass instance."""
    source_path = str(Path(base_path) / cfg["path"])
    rename_map = cfg.get("rename_map", {})
    platform_map = cfg.get("platform_map", {})
    constants = cfg.get("constants", {})
    read_csv_kwargs = cfg.get("read_csv_kwargs", {})
    transforms = _build_transforms(cfg.get("transforms", []))
    loader = _build_loader(source_path, cfg.get("loader_filter"))

    return SourceConfig(
        name=name,
        path=source_path,
        rename_map=rename_map,
        platform_map=platform_map,
        constants=constants,
        transforms=transforms,
        read_csv_kwargs=read_csv_kwargs,
        loader=loader,
    )


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments from JSON text.

    Strips comments while preserving strings (e.g. URLs containing //).
    """
    import re
    result = []
    i = 0
    in_string = False
    escape = False

    while i < len(text):
        ch = text[i]

        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue

        if ch == '\\' and in_string:
            result.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string:
            result.append(ch)
            i += 1
            continue

        # Not in a string — look for comments
        if i + 1 < len(text) and text[i + 1] == '/':
            # Line comment: skip to end of line
            while i < len(text) and text[i] != '\n':
                i += 1
            continue

        if i + 1 < len(text) and text[i + 1] == '*':
            # Block comment: skip to */
            i += 2
            while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                i += 1
            i += 2  # skip */
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def load_merge_config(config_path: str) -> dict[str, Any]:
    """Load and validate a merge config JSON file (supports // and /* */ comments)."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = _strip_json_comments(raw)
    config = json.loads(cleaned)

    return config


def build_sources(config: dict, base_path: str = "."):
    """Build main SourceConfig and list of source configs from merge config."""
    main_cfg = config["main"]
    main = config_to_source_config(main_cfg["name"], main_cfg, base_path)

    sources = []
    for src in config.get("sources", []):
        sources.append(config_to_source_config(src["name"], src, base_path))

    # Add gamelist source if enabled
    gamelists = config.get("gamelists", {})
    if gamelists.get("enabled", False):
        gamelist_dir = gamelists.get("dir", "lists")
        sources.append(
            SourceConfig(
                name="gamelist",
                loader=lambda: load_all_gamelists(lists_dir=gamelist_dir),
            )
        )

    return main, sources


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_merge(config_path: str, output_dir: Optional[str] = None):
    """Run the full merge pipeline from a config file."""
    config = load_merge_config(config_path)
    base_path = str(Path(config_path).parent.parent)

    print(f"Loading config from: {config_path}")
    main, sources = build_sources(config, base_path)
    print(f"Main source: {main.name}")
    print(f"Additional sources: {len(sources)}")

    pipeline_cfg = config.get("pipeline", {})
    output_cfg = config.get("output", {})

    if output_dir:
        pipeline_cfg.setdefault("output_dir", output_dir)

    merged_df = run_merge_pipeline(
        main_config=main,
        source_configs=sources,
        key_columns=pipeline_cfg.get("key_columns", ["name", "platform"]),
        use_name_match_key=pipeline_cfg.get("use_name_match_key", True),
        duplicate_detection_threshold=pipeline_cfg.get("duplicate_detection_threshold", 0.8),
        collapse_platforms=pipeline_cfg.get("collapse_platforms", False),
        output_dir=pipeline_cfg.get("output_dir"),
    )

    print(f"\nMerged: {len(merged_df)} rows, {merged_df['platform'].nunique()} unique platforms")

    # Save merged pickle
    merged_pkl = output_cfg.get("merged_pkl", "output/merged_df.pkl")
    Path(merged_pkl).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_pickle(merged_pkl)
    print(f"Merged DataFrame saved to: {merged_pkl}")

    # Save merged CSV
    merged_csv = output_cfg.get("merged_csv")
    if merged_csv:
        Path(merged_csv).parent.mkdir(parents=True, exist_ok=True)
        write_to_csv(merged_df, merged_csv, CANONICAL_SCHEMA)
        print(f"Merged CSV saved to: {merged_csv}")

    return merged_df


def run_clean(config_path: Optional[str] = None, input_pkl: Optional[str] = None, merged_df: Optional[pd.DataFrame] = None):
    """Run the cleaning pipeline."""
    if merged_df is None:
        if input_pkl is None:
            input_pkl = "output/merged_df.pkl"
        if not Path(input_pkl).exists():
            print(f"Error: input file not found: {input_pkl}")
            print("Run 'pipeline run' first to generate the merged dataset.")
            sys.exit(1)
        print(f"Loading merged DataFrame from: {input_pkl}")
        merged_df = pd.read_pickle(input_pkl)

    if config_path:
        config = load_merge_config(config_path)
    else:
        config = _default_clean_config()

    steps = config.get("steps", {})
    columns = config.get("columns", {})

    # Build genre translation map if configured
    genre_translation_map = None
    translate_step = steps.get("translate_genres", {})
    if translate_step.get("enabled", True):
        map_path = translate_step.get("map_path")
        if map_path and Path(map_path).exists():
            with open(map_path, "r", encoding="utf-8") as f:
                genre_translation_map = json.load(f)
            print(f"Loaded genre translation map: {len(genre_translation_map)} mappings")

    cleaning_kwargs = {
        "normalize_genres_col": columns.get("genres", "genres"),
        "normalize_date_col": columns.get("release_date", "release_date"),
        "parse_players_col": columns.get("players", "players"),
        "cooperative_col": columns.get("cooperative", "cooperative"),
        "derive_year_col": columns.get("release_year", "release_year"),
        "genre_translation_map": genre_translation_map,
    }

    round_step = steps.get("round_ratings", {})
    if round_step.get("enabled", True):
        cleaning_kwargs["round_columns"] = round_step.get("columns", ["rating", "user_rating"])
        cleaning_kwargs["round_decimals"] = round_step.get("decimals", 1)

    print("Running cleaning pipeline...")
    cleaned_df = run_cleaning_pipeline(merged_df, **cleaning_kwargs)

    output_cfg = config.get("output", {})
    cleaned_pkl = output_cfg.get("cleaned_pkl", "output/cleaned_df.pkl")
    cleaned_csv = output_cfg.get("cleaned_csv", "output/cleaned_games.csv")

    Path(cleaned_pkl).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_pickle(cleaned_pkl)
    print(f"Cleaned DataFrame saved to: {cleaned_pkl}")

    Path(cleaned_csv).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df["version"] = pd.Timestamp.utcnow().isoformat()
    write_to_csv(cleaned_df, cleaned_csv, CANONICAL_SCHEMA)
    print(f"Cleaned CSV saved to: {cleaned_csv} ({len(cleaned_df)} rows)")

    return cleaned_df


def _default_clean_config() -> dict:
    """Return a minimal default cleaning config when no config file is provided."""
    return {
        "columns": {
            "genres": "genres",
            "release_date": "release_date",
            "players": "players",
            "cooperative": "cooperative",
            "release_year": "release_year",
        },
        "steps": {
            "translate_genres": {"enabled": False, "map_path": None},
            "normalize_genres": {"enabled": True},
            "normalize_dates": {"enabled": True},
            "parse_players": {"enabled": True},
            "infer_cooperative": {"enabled": True},
            "derive_year": {"enabled": True},
            "round_ratings": {"enabled": True, "columns": ["rating", "user_rating"], "decimals": 1},
        },
        "output": {
            "cleaned_pkl": "output/cleaned_df.pkl",
            "cleaned_csv": "output/cleaned_games.csv",
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for `python -m utils pipeline run|clean`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video game metadata pipeline runner",
        prog="python -m utils",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run merge pipeline from config")
    run_parser.add_argument("--config", "-c", default="config/merge_config.json",
                            help="Path to merge config JSON (default: config/merge_config.json)")
    run_parser.add_argument("--output-dir", "-o", default=None,
                            help="Override output directory for audit files")

    # --- clean ---
    clean_parser = subparsers.add_parser("clean", help="Run cleaning pipeline")
    clean_parser.add_argument("--config", "-c", default=None,
                              help="Path to clean config JSON (optional, uses defaults if omitted)")
    clean_parser.add_argument("--input", "-i", default=None,
                              help="Path to merged DataFrame pickle (default: output/merged_df.pkl)")

    # --- full (merge + clean in one shot) ---
    full_parser = subparsers.add_parser("full", help="Run merge + clean in one shot")
    full_parser.add_argument("--merge-config", "-m", default="config/merge_config.json",
                             help="Path to merge config JSON")
    full_parser.add_argument("--clean-config", "-n", default=None,
                             help="Path to clean config JSON (optional)")
    full_parser.add_argument("--output-dir", "-o", default=None,
                             help="Override output directory for audit files")

    args = parser.parse_args()

    if args.command == "run":
        run_merge(args.config, args.output_dir)
    elif args.command == "clean":
        run_clean(args.config, args.input)
    elif args.command == "full":
        merged_df = run_merge(args.merge_config, args.output_dir)
        run_clean(args.clean_config, merged_df=merged_df)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

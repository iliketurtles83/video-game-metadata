# Video Game Metadata Pipeline

Merge, clean, and analyze video game metadata from multiple sources.

This project is built for a practical workflow:
- ingest metadata from CSV and `gamelist.xml`
- normalize schema and platform names
- deduplicate and clean records
- analyze quality and coverage

## Project Goals

- Build a canonical game metadata table from heterogeneous sources.
- Keep platform naming consistent across CSV sources and filesystem folders.
- Produce outputs usable for frontend browsing, curation, and downstream analysis.

## Repository Layout

- `csv/`: raw and processed tabular datasets.
- `config/`: JSON config files for the CLI pipeline runner.
- `utils/`: reusable pipeline modules.
- `output/`: generated artifacts for testing and exports.
- `scripts/`: helper shell scripts for export flows.

## End-to-End Workflow

### CLI (recommended for reproducible runs)

Run the full pipeline (merge → clean → export) from the repo root:

```bash
# Full pipeline: merge + clean in one shot
python -m utils full

# Full pipeline with Parquet and SQLite DB exports
python -m utils full --export-all

# Merge only (produces merged_df.pkl + merged CSV)
python -m utils run

# Clean only (reads merged_df.pkl, applies cleaning, exports CSV)
python -m utils clean

# Inspect and resolve ambiguous fuzzy matches (review queue)
python -m utils review --status                     # Check queue status
python -m utils review --auto-resolve 0.88          # Auto-approve high confidence pairs
python -m utils review                              # Interactive terminal review

# Validate dataset schema, value ranges, and date consistency
python -m utils validate                            # Run validation checks
```

Both `run` and `full` accept `--config` to point to a custom merge config:

```bash
python -m utils full --config config/merge_config.json --export-all
```

Config files live in `config/`:
- `config/merge_config.json` — source paths, column mappings, pipeline settings
- `config/clean_config.json` — cleaning steps, column names, genre translation maps
- `config/match_overrides.json` — human-curated title match overrides saved from review queue

See the config files for fully commented examples of every option.

### Notebooks (for exploration and experimentation)

Run notebooks for exploration and inspection:

1. `notebooks/01-game_data_exploration.ipynb`
	Inspect source quality, field distribution, and platform naming issues on raw CSVs.
2. `notebooks/02-data_analysis.ipynb`
	Unified output data viewer, interactive title/genre/year search, SQL querying, and deep EDA on the cleaned dataset.

## Canonical Schema (Core Columns)

The merged dataset is normalized around fields like:
- `name`
- `platform`
- `filename`
- `summary`
- `release_date`
- `release_year`
- `genres`
- `developer`
- `publisher`
- `players`
- `cooperative`
- `rating`
- `user_rating`

Additional columns may exist depending on source coverage and enrichment rules.

## Mapping Files and Normalization Rules

Two mapping files serve different responsibilities:

### `utils/platform_registry.json`
- Used by both `utils/merge_pipeline.py` and `utils/gamelist_parser.py`
- Normalizes platform labels from metadata sources
- Maps folder keys (for example `psx`, `nes`, `sfc`) to canonical platform names
- Ensures parsed `lists/<platform>/` data aligns with the merged dataset
- Example: aliases like `PSP` and `PlayStation Portable` resolve to one canonical platform

## Key Modules

### `utils/merge_pipeline.py`
- source loading and schema alignment
- platform normalization and datatype harmonization
- deduplication and multi-source merge logic

### `utils/pipeline.py`
- CLI runner (`python -m utils`) for reproducible pipeline execution
- reads JSON configs, builds SourceConfigs, orchestrates merge + clean + export

### `utils/data_cleaning.py`
- cleanup helpers for text, null handling, and normalized field formatting
- post-merge consistency operations used by cleaning notebooks/scripts

### `utils/gamelist_parser.py`
- parses `gamelist.xml` into DataFrames
- tolerates missing tags and normalizes values
- applies folder-to-platform mapping for merge compatibility

## Inputs and Outputs

### Primary Inputs
- source CSV files in `csv/` (for example `launchbox.csv`, `mobygames.csv`, DAT exports)
- console gamelists in `lists/<platform>/gamelist.xml`

### Primary Outputs
- merged datasets such as `csv/combined.csv`, `csv/all_games.csv`
- cleaned datasets such as `csv/game_dataset_cleaned.csv`

## Running From Scripts (Optional)

If you prefer shell workflows over notebooks, the CLI is the primary interface:

```bash
python -m utils run        # merge only
python -m utils clean      # clean only
python -m utils full       # merge + clean
```

For legacy shell helpers, see `scripts/`:
- `scripts/export_tables.sh` — mdb-export for ARRM databases

## Practical Notes

- Keep mapping JSON files up to date before major merges.
- Re-run analysis after cleaning to catch regressions early.
- Track schema changes explicitly when introducing new metadata sources.

## License

See `LICENSE`.

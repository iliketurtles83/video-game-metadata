# Video Game Metadata Pipeline

Merge, clean, analyze, and apply video game metadata across multiple sources and console gamelists.

This project is built for a practical workflow:
- ingest metadata from CSV and `gamelist.xml`
- normalize schema and platform names
- deduplicate and clean records
- analyze quality and coverage
- write enriched metadata back into per-platform `gamelist.xml`

## Project Goals

- Build a canonical game metadata table from heterogeneous sources.
- Keep platform naming consistent across CSV sources and filesystem folders.
- Preserve existing gamelist structure while updating metadata fields safely.
- Produce outputs usable for frontend browsing, curation, and downstream analysis.

## Repository Layout

- `csv/`: raw and processed tabular datasets.
- `lists/<platform>/gamelist.xml`: source gamelists per platform.
- `lists/<platform>/gamelist_updated.xml`: generated updated gamelists.
- `utils/`: reusable pipeline modules.
- `output/`: generated artifacts for testing and exports.
- `scripts/`: helper shell scripts for export/update flows.

## End-to-End Workflow

Run notebooks in this order:

1. `01-game_data_exploration.ipynb`
	Inspect source quality, field distribution, and platform naming issues.
2. `02-merge_game_data.ipynb`
	Merge all configured sources into a unified dataset.
3. `03-data_cleaning.ipynb`
	Apply post-merge cleanup and normalization rules.
4. `04-data_analysis.ipynb`
	Validate coverage, missingness, duplicates, and output quality.
5. `05-update-gamelists.ipynb`
	Apply merged metadata into platform-specific `gamelist.xml` files.

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

### `utils/platform_mappings.json`
- Used by `utils/merge_pipeline.py`
- Normalizes platform labels from metadata sources
- Example: aliases like `PSP` and `PlayStation Portable` resolve to one canonical platform

### `utils/gamelist_folder_mappings.json`
- Used by `utils/gamelist_parser.py`
- Maps folder keys (for example `psx`, `nes`, `sfc`) to canonical platform names
- Ensures parsed `lists/<platform>/` data aligns with the merged dataset

## Key Modules

### `utils/merge_pipeline.py`
- source loading and schema alignment
- platform normalization and datatype harmonization
- deduplication and multi-source merge logic

### `utils/data_cleaning.py`
- cleanup helpers for text, null handling, and normalized field formatting
- post-merge consistency operations used by cleaning notebooks/scripts

### `utils/gamelist_parser.py`
- parses `gamelist.xml` into DataFrames
- tolerates missing tags and normalizes values
- applies folder-to-platform mapping for merge compatibility

### `utils/update_gamelist.py`
- writes merged metadata back into gamelist XML entries
- updates desired metadata fields while preserving sensitive existing tags (for example image/media tags)

### `utils/gamelist_builder.py`
- helper logic for constructing/updating gamelist structures from DataFrame outputs

## Inputs and Outputs

### Primary Inputs
- source CSV files in `csv/` (for example `launchbox.csv`, `mobygames.csv`, DAT exports)
- console gamelists in `lists/<platform>/gamelist.xml`

### Primary Outputs
- merged datasets such as `csv/combined.csv`, `csv/all_games.csv`
- cleaned datasets such as `csv/game_dataset_cleaned.csv`
- updated gamelists in each `lists/<platform>/gamelist_updated.xml`

## Running From Scripts (Optional)

If you prefer shell workflows over notebooks, helper scripts are available in `scripts/`:
- `scripts/cpgamelist.sh`
- `scripts/export_tables.sh`

Use these after reviewing notebook logic so script execution matches your current mapping and cleaning assumptions.

## Practical Notes

- Keep mapping JSON files up to date before major merges.
- Re-run analysis after cleaning to catch regressions early.
- Test gamelist updates on a small platform subset first (for example `output/test_games/`) before full rollout.
- Track schema changes explicitly when introducing new metadata sources.

## License

See `LICENSE`.

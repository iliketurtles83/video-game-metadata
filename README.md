# Video Game Metadata Merge Pipeline

Tools for merging, cleaning, and enriching video game metadata from multiple sources.

The project produces a unified dataset that can be used for:
- Web display and browsing
- Updating `gamelist.xml` files
- Data analysis and machine learning workflows

## What This Project Does

This repository combines metadata from different formats and sources into one canonical table.

Typical sources include:
- ARRM exports (CSV)
- `gamelist.xml` files from RetroPie/EmulationStation-style setups
- Public datasets (for example, from Google Dataset Search)

The pipeline normalizes field names, platform names, datatypes, and duplicate records.

## Canonical Dataset Shape

The merge pipeline targets a consistent schema, including columns such as:
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

Notes:
- Multi-value fields (for example `developer`, `publisher`, `genres`) are normalized during processing.
- Platform names are standardized through mapping files so equivalent variants merge cleanly.

## Platform Mapping Rules

Two mapping files are used for different purposes:

- `utils/platform_mappings.json`
- Used by `utils/merge_pipeline.py`
- Maps platform name variants (for example `PSP`, `PlayStation Portable`) to canonical names during merge

- `utils/gamelist_folder_mappings.json`
- Used by `utils/gamelist_parser.py`
- Maps folder names (for example `psx`, `nes`) to canonical platform names


## Main Workflow (Notebooks)

- `01-game_data_exploration.ipynb`: inspect raw sources
- `02-merge_game_data.ipynb`: configure sources and run the merge pipeline
- `03-data_cleaning.ipynb`: additional cleanup and normalization
- `04-data_analysis.ipynb`: analysis and validation
- `05-update_gamelists.ipynb`: write merged metadata back to `gamelist.xml`

## Key Python Modules

- `utils/merge_pipeline.py`
- Source normalization and schema alignment
- Deduplication and source-to-source merging
- Platform variant normalization via `platform_mappings.json`

- `utils/gamelist_parser.py`
- Loads and parses `gamelist.xml` into DataFrames
- Applies folder-to-platform mapping via `gamelist_folder_mappings.json`
- Handles missing XML fields and basic value normalization

- `utils/update_gamelist.py`
- Updates `gamelist.xml` with merged metadata
- Preserves fields that should not be overwritten (for example image-related tags)

## Inputs and Outputs

### Inputs
- Source CSV files in `csv/`
- Console-specific `gamelist.xml` files in `lists/<platform>/`

### Outputs
- Merged/cleaned CSV datasets in `csv/` and `output/`
- Updated `gamelist_updated.xml` files in each `lists/<platform>/` folder

## Quick Start

1. Open and run `01-game_data_exploration.ipynb` to inspect source data.
2. Run `02-merge_game_data.ipynb` to generate the combined dataset.
3. Run `03-data_cleaning.ipynb` and `04-data_analysis.ipynb` as needed.
4. Run `05-update_gamelists.ipynb` to apply metadata updates to `gamelist.xml` files.

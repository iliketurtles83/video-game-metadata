# AGENTS.md

This file provides essential context for agents working in this repository to avoid common mistakes and ramp up quickly.

## Project Overview
This is a video game metadata pipeline that merges, cleans, analyzes, and applies metadata across multiple sources and console gamelists.

## Key Commands and Workflows

### Running the pipeline notebooks (recommended):
1. `01-game_data_exploration.ipynb` - Inspect source quality and platform naming issues
2. `02-merge_game_data.ipynb` - Merge all configured sources into a unified dataset
3. `03-data_cleaning.ipynb` - Apply post-merge cleanup and normalization rules
4. `04-data_analysis.ipynb` - Validate coverage, missingness, duplicates, and output quality
5. `05-update-gamelists.ipynb` - Apply merged metadata into platform-specific `gamelist.xml` files

### Alternative shell workflow:
- Run `scripts/cpgamelist.sh` to extract gamelist.xml files from platform folders
- Use `scripts/export_tables.sh` to export tables from ARRM databases

## Architecture and Structure

### Directory Layout:
- `csv/`: raw and processed tabular datasets
- `lists/<platform>/gamelist.xml`: source gamelists per platform
- `lists/<platform>/gamelist_updated.xml`: generated updated gamelists
- `utils/`: reusable pipeline modules
- `output/`: generated artifacts for testing and exports
- `scripts/`: helper shell scripts

### Core Modules:
- `utils/merge_pipeline.py`: source loading, schema alignment, platform normalization, deduplication
- `utils/data_cleaning.py`: cleanup helpers for text, null handling, and normalized field formatting
- `utils/gamelist_parser.py`: parses `gamelist.xml` into DataFrames
- `utils/update_gamelist.py`: writes merged metadata back into gamelist XML entries
- `utils/gamelist_builder.py`: helper logic for constructing/updating gamelist structures

## Key Configuration Files
- `utils/platform_mappings.json`: Maps platform aliases to canonical platform names (used by merge_pipeline.py)
- `utils/gamelist_folder_mappings.json`: Maps folder keys to canonical platform names (used by gamelist_parser.py)

## Important Notes
- Keep mapping JSON files up to date before major merges
- Re-run analysis after cleaning to catch regressions early
- Test gamelist updates on a small platform subset first before full rollout
- Track schema changes explicitly when introducing new metadata sources
- The pipeline uses a canonical schema with specific expected column types
- Platform name normalization occurs in two places: merge_pipeline.py (source platforms) and gamelist_parser.py (folder platforms)
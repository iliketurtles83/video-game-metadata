# AGENTS.md

Video game metadata merge pipeline: ingest CSV/XML sources → normalize schema/platforms → deduplicate → clean → analyze.

## How to run

**Notebooks (recommended, run in order):**
1. `01-game_data_exploration.ipynb` — inspect source quality, field distribution, platform naming
2. `02-merge_game_data.ipynb` — merge all configured sources into unified dataset
3. `03-data_cleaning.ipynb` — post-merge cleanup and normalization
4. `04-data_analysis.ipynb` — validate coverage, missingness, duplicates

**Tests:** `python -m pytest tests/ -v` (8 tests, all unit tests on pipeline functions)
**Standalone test:** `python utils/test_pipeline_comprehensive.py` (pytest-style classes, runs outside pytest)

## Architecture

```
csv/          ← raw & processed CSVs (gitignored *.csv)
  combined.csv, all_games.csv, game_dataset_cleaned.csv
  launchbox.csv, mobygames.csv, dat_database_*.csv, gamelist_parsed.csv, games_on_gametdb.csv, recalbox_gamelist.csv
utils/        ← pipeline modules
  merge_pipeline.py   ← SourceConfig, run_merge_pipeline(), canonical schema, dedup, platform normalization
  resolvers.py        ← field merge strategies (pick_first, pick_longer, collect_unique, any_truthy, weighted_avg, prefer_specific)
  data_cleaning.py    ← genre normalization, player parsing, date normalization, release year derivation
  gamelist_parser.py  ← parse lists/<platform>/gamelist.xml into DataFrames
  csv_export.py       ← write DataFrames to CSV with type coercion
  platform_mappings.json       ← CSV source platform aliases → canonical names
  gamelist_folder_mappings.json ← folder names → canonical platform names
output/         ← generated artifacts (merged_df.pkl, gitignored)
scripts/        ← export_tables.sh (mdb-export for ARRM databases)
tests/          ← pytest test suite
```

## Canonical schema (merge_pipeline.py)

| Column | Type |
|---|---|
| name | string |
| filename | string |
| summary | string |
| platform | string |
| release_date | datetime64[ns] |
| release_year | Int64 |
| genres | string |
| developer | string |
| publisher | string |
| players | string |
| cooperative | boolean |
| rating | Int64 |
| user_rating | float64 |
| version | string |

## Key gotchas

- **Platform normalization** happens in two places: `merge_pipeline.py` (CSV source platforms via `platform_mappings.json`) and `gamelist_parser.py` (folder names via `gamelist_folder_mappings.json`). Keep both JSON files in sync before merging new sources.
- **Multi-value columns** (`platform`, `developer`, `publisher`, `genres`) are split, deduplicated, and re-joined as comma-separated strings. `normalize_source()` explodes `platform` into rows; other multi-value columns are flattened.
- **Deduplication** uses `name`+`platform` as key columns, with a `_name_match_key` for fuzzy name matching (rapidfuzz `token_set_ratio`, threshold 85 for platforms, 0.8 confidence for names).
- **`lists/` directory** (gitignored) contains `gamelist.xml` files per platform subdirectory — these feed `gamelist_parser.py`.
- **Mapping consistency**: `merge_pipeline.py` has a built-in consistency checker (`validate_platform_mapping_consistency()`) that prints warnings if keys exist in only one mapping file or resolve to different canonical names.
- **Name cleaning** strips region tags `(USA)`, `(PAL)`, `(Japan)`, ROM hack tags `[!]`, `[a]`, `[T+En]`, etc. Japanese/Chinese/Korean titles are preserved.
- **Tests** import via `sys.path.insert(0, parent)` — run from repo root. The `utils/test_pipeline_comprehensive.py` test file uses class-based pytest but is also runnable as a standalone script.
- No lint/typecheck/formatter config — this is a data exploration pipeline, not an application.

## Adding a new source

1. Place CSV in `csv/` (it's gitignored).
2. Add a `SourceConfig` in the merge notebook with: `name`, `path`, optional `rename_map` (source column → canonical column), optional `platform_map` overrides, optional `constants` (fixed column values), optional `transforms`.
3. If source uses platform names not in `platform_mappings.json`, add them.
4. Run the merge notebook and validate with `04-data_analysis.ipynb`.

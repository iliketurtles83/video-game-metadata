# AGENTS.md

Video game metadata merge pipeline: ingest CSV/XML sources → normalize schema/platforms → deduplicate → clean → analyze.

## How to run

**Notebooks (recommended, run in order):**
1. `notebooks/01-game_data_exploration.ipynb` — inspect source quality, field distribution, platform naming
2. `notebooks/02-merge_game_data.ipynb` — merge all configured sources into unified dataset
3. `notebooks/03-data_analysis.ipynb` — anomaly detection on merged data (missingness, platform naming, name quality, genre issues, date inconsistencies, duplicates) — EDA for downstream uses (gamelist updater, ML recommender, online database)
4. `notebooks/04-data_cleaning.ipynb` — post-merge cleanup and normalization informed by analysis findings

**Tests:** `python -m pytest tests/ -v` (99 tests across normalization, resolvers, fuzzy dedup, exports, provenance, date reconciliation, and validation)
**Standalone test:** `python utils/test_pipeline_comprehensive.py` (runs outside pytest)

## Architecture

```
csv/          ← raw & processed CSVs (gitignored *.csv)
  combined.csv, all_games.csv, game_dataset_cleaned.csv
  launchbox.csv, mobygames.csv, dat_database_*.csv, gamelist_parsed.csv, games_on_gametdb.csv, recalbox_gamelist.csv
config/       ← pipeline configuration JSON files
  merge_config.json   ← source paths, transforms, merge settings
  clean_config.json   ← post-merge cleaning pipeline settings
  match_overrides.json ← human-curated title match overrides from review queue
  genre_translations.json ← Portuguese to English genre mapping
docs/         ← documentation
  DATA_DICTIONARY.md  ← canonical dataset schema and quality invariants
utils/        ← pipeline modules
  merge_pipeline.py   ← SourceConfig, run_merge_pipeline(), canonical schema, dedup, platform normalization
  pipeline.py         ← CLI runner (run, clean, full, review, validate)
  resolvers.py        ← field merge strategies (pick_first, pick_longer, collect_unique, any_truthy, mean_rating, prefer_specific)
  data_cleaning.py    ← genre normalization, player parsing, date normalization, release year derivation
  validate_dataset.py ← dataset validation against quality constraints and schema
  gamelist_parser.py  ← parse lists/<platform>/gamelist.xml into DataFrames
  csv_export.py       ← write DataFrames to CSV, Parquet, and SQLite with indices
  review_queue.py     ← interactive & automated resolution of ambiguous fuzzy duplicates
  platform_registry.json ← unified canonical platform names & aliases
output/         ← generated artifacts (merged_df.pkl, review_queue.csv, gitignored)
scripts/        ← export_tables.sh (mdb-export for ARRM databases)
tests/          ← pytest test suite (test_matching.py, test_high_impact.py, etc.)
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
| rating | float64 |
| user_rating | float64 |
| version | string |

## Key gotchas

- **Platform normalization** happens in two places: `merge_pipeline.py` (CSV source platforms) and `gamelist_parser.py` (folder names). Both use `platform_registry.json`. Keep this file updated before merging new sources.
- **Multi-value columns** (`platform`, `developer`, `publisher`, `genres`) are split, deduplicated, and re-joined as comma-separated strings. `normalize_source()` explodes `platform` into rows; other multi-value columns are flattened.
- **Deduplication** uses `name`+`platform` as key columns, with a `_name_match_key` for fuzzy name matching (rapidfuzz `token_set_ratio`, threshold 85 for platforms, 0.8 confidence for names).
- **`lists/` directory** (gitignored) contains `gamelist.xml` files per platform subdirectory — these feed `gamelist_parser.py`.
- **Name cleaning** strips region tags `(USA)`, `(PAL)`, `(Japan)`, ROM hack tags `[!]`, `[a]`, `[T+En]`, etc. Japanese/Chinese/Korean titles are preserved.
- **Tests** import via `sys.path.insert(0, parent)` — run from repo root. The `utils/test_pipeline_comprehensive.py` test file uses class-based pytest but is also runnable as a standalone script.
- No lint/typecheck/formatter config — this is a data exploration pipeline, not an application.

## Adding a new source

1. Place CSV in `csv/` (it's gitignored).
2. Add a `SourceConfig` in the merge notebook with: `name`, `path`, optional `rename_map` (source column → canonical column), optional `platform_map` overrides, optional `constants` (fixed column values), optional `transforms`.
3. If source uses platform names not in `platform_registry.json`, add them.
4. Run the merge notebook and validate with `notebooks/04-data_analysis.ipynb`.

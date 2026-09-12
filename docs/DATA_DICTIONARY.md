# Video Game Metadata — Canonical Data Dictionary

This document defines the schema, value domains, constraints, and consumption guidelines for datasets produced by the `video-game-metadata` pipeline.

The dataset is available in three export formats: **CSV**, **Parquet** (Snappy compressed), and **SQLite** (with B-Tree indices).

---

## 1. Canonical Schema Specification

| Column | CSV Type | Parquet Type | SQLite Type | Nullable | Description | Example |
|---|---|---|---|---|---|---|
| `platform` | `string` | `string` | `TEXT` | No | Canonical platform name as standardized in `platform_registry.json`. | `"Super Nintendo Entertainment System"` |
| `name` | `string` | `string` | `TEXT` | No | Normalized game title with regional and ROM dump tags stripped. Title-cased with preserved Roman numerals. | `"Chrono Trigger"` |
| `filename` | `string` | `string` | `TEXT` | Yes | Normalized ROM or disc image stem (without extension or path). | `"Chrono Trigger (USA)"` |
| `summary` | `string` | `string` | `TEXT` | Yes | Longest and most complete overview / synopsis available across merged sources. | `"An epic role-playing game developed by Square..."` |
| `release_date` | `string` (`YYYY-MM-DD`) | `timestamp[ns]` | `TEXT` (`YYYY-MM-DD`) | Yes | ISO 8601 release date (date component only, UTC normalized). Null if only release year is known. | `"1995-03-11"` |
| `release_year` | `Int64` | `int64` | `INTEGER` | Yes | 4-digit release year (`1900 <= year <= 2030`). Aligned with `release_date.year` whenever `release_date` is non-null. | `1995` |
| `genres` | `string` | `string` | `TEXT` | Yes | Alphabetically sorted, comma-separated list of normalized title-case English genres. | `"Action, Adventure, Role Playing Game"` |
| `developer` | `string` | `string` | `TEXT` | Yes | Most specific primary developer(s). | `"Square"` |
| `publisher` | `string` | `string` | `TEXT` | Yes | Most specific primary publisher(s). | `"Square"` |
| `players` | `Int64` | `int64` | `INTEGER` | Yes | Maximum supported player count as a positive integer. | `2` |
| `cooperative` | `boolean` | `boolean` | `INTEGER` (0/1) | Yes | `True` if cooperative play is supported. Inferred `False` when `players == 1`. | `False` |
| `rating` | `Float64` | `float64` | `REAL` | Yes | Critic score normalized to a **0.0 – 10.0** continuous scale (rounded to 1 decimal place). | `9.2` |
| `user_rating` | `Float64` | `float64` | `REAL` | Yes | Community/audience score normalized to a **0.0 – 10.0** continuous scale (rounded to 1 decimal place). | `8.9` |
| `version` | `string` | `string` | `TEXT` | Yes | Pipeline export timestamp tag (ISO 8601 UTC). | `"2026-09-12T18:00:00Z"` |
| `_source` | `string` | `string` | `TEXT` | Yes | Comma-separated list of data sources that contributed to this record (provenance tracking). | `"launchbox, mobygames"` |

---

## 2. Value Domains & Quality Invariants

Every dataset passing the `python -m utils validate` gate adheres to the following rules:

1. **Ratings Scale Invariant**:
   - Both `rating` (critic) and `user_rating` (community) reside strictly in the range **[0.0, 10.0]** or are `NULL`.
   - Legacy 0–100 scales (Metacritic critic scores) and 0–1 scales (scraped gamelists) are automatically rescaled during post-merge cleaning.

2. **Date & Year Reconciliation Invariant**:
   - When both `release_date` and `release_year` are non-null, **`release_date.year == release_year` is guaranteed**.
   - Sentinel placeholder dates (e.g. `1980-01-01` from GameTDB on modern platforms like Switch, 3DS, DS, Wii, PS3) are stripped to `NaT`.
   - Malformed truncated decade years (`< 1900` or `> 2030`, e.g. `200`, `199`) are stripped to `NULL`.

3. **Genre Translation Invariant**:
   - Non-English genre terminology (e.g. Portuguese terms like *Ação*, *Plataforma*, *Esporte*) is translated into standard English genres using `config/genre_translations.json` before sorting and deduplication.

4. **Platform Normalization Invariant**:
   - Platform names are resolved against `platform_registry.json`. No unmapped raw alias variants (such as `psx`, `PS2`, `snes`) exist in output data.

5. **Title Cleaning & Deduplication Invariant**:
   - Titles are stripped of region identifiers `(USA)`, `(Japan)`, `(PAL)`, and ROM dump flags `[!]`, `[a]`.
   - Roman numerals are standardized (e.g., `Final Fantasy VII` and `Final Fantasy 7` match).
   - Sequel collisions are protected so that distinct numbered installments are never merged.

---

## 3. SQLite Database Indices

When exporting to SQLite (`output/cleaned_games.db`), the following B-Tree indices are pre-constructed:

```sql
CREATE INDEX idx_cleaned_games_name ON cleaned_games(name);
CREATE INDEX idx_cleaned_games_platform ON cleaned_games(platform);
CREATE INDEX idx_cleaned_games_name_platform ON cleaned_games(name, platform);
CREATE INDEX idx_cleaned_games_year ON cleaned_games(release_year);
```

---

## 4. Downstream Consumption Guidelines

### A. Gamelist Updater (`/home/jack/Projects/gamelist-updater`)
- Query pattern: Lookup by `(platform, name)` or `(platform, filename)`:
  ```python
  import sqlite3
  conn = sqlite3.connect("output/cleaned_games.db")
  cursor = conn.cursor()
  cursor.execute(
      "SELECT summary, release_year, genres, developer, publisher, rating "
      "FROM cleaned_games WHERE platform = ? AND name = ?",
      (platform_name, game_title)
  )
  ```

### B. Machine Learning Recommender
- Load via Parquet with zero-copy PyArrow:
  ```python
  import pandas as pd
  df = pd.read_parquet("output/cleaned_games.parquet")
  # Features: genres (one-hot or TF-IDF), summary (embeddings), developer, publisher, release_year, rating
  ```

### C. Games Database & API UI
- SQLite or PostgreSQL import:
  - Table name: `cleaned_games`
  - Canonical primary key: `(name, platform)`

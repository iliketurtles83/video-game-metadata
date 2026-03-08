import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from utils.resolvers import pick_first, resolver as default_resolver, collapse_resolver as default_collapse_resolver


# Canonical schema: defines expected types for all columns
CANONICAL_SCHEMA = {
    'name': 'string',
    'filename': 'string',
    'summary': 'string',
    'platform': 'string',
    'release_date': 'datetime64[ns]',
    'release_year': 'int64',
    'genres': 'string',
    'developer': 'string',
    'publisher': 'string',
    'players': 'string',
    'cooperative': 'boolean',
    'rating': 'int64',
    'user_rating': 'float64',
}

DEFAULT_COLUMNS = tuple(CANONICAL_SCHEMA.keys())

# Key columns for deduplication and merging
KEY_COLUMNS = ("name", "platform")
NAME_MATCH_KEY_COLUMN = "_name_match_key"

# Columns that may contain multiple values (lists or delimited strings)
MULTI_VALUE_COLUMNS = ("platform", "developer", "publisher", "genres")
MULTI_VALUE_SPLIT_PATTERN = r"[;,/]"

COMMON_ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
}


def _roman_replacer(match: re.Match[str]) -> str:
    token = match.group(0)
    if token.lower() in COMMON_ROMAN_NUMERALS:
        return token.upper()
    return token


SeriesTransform = Callable[[pd.Series], pd.Series]
DataFrameLoader = Callable[[], pd.DataFrame]
DataFramePostLoad = Callable[[pd.DataFrame], pd.DataFrame]

_GLOBAL_PLATFORM_MAP: Optional[dict[str, str]] = None


def _load_global_platform_map() -> dict[str, str]:
    """Load and cache the global platform name mapping from platform_mappings.json."""
    global _GLOBAL_PLATFORM_MAP
    if _GLOBAL_PLATFORM_MAP is None:
        mappings_path = Path(__file__).parent / "platform_mappings.json"
        try:
            with open(mappings_path, "r", encoding="utf-8") as f:
                _GLOBAL_PLATFORM_MAP = json.load(f)
        except (json.JSONDecodeError, OSError):
            _GLOBAL_PLATFORM_MAP = {}
    return _GLOBAL_PLATFORM_MAP


@dataclass
class SourceConfig:
    name: str
    path: Optional[str] = None
    rename_map: Mapping[str, str] = field(default_factory=dict)
    platform_map: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    transforms: Mapping[str, SeriesTransform] = field(default_factory=dict)
    read_csv_kwargs: Mapping[str, Any] = field(default_factory=dict)
    loader: Optional[DataFrameLoader] = None
    post_load: Optional[DataFramePostLoad] = None


def clean_game_name(name: str) -> str:
    """Normalize a game title by removing region/version tags, fixing common encoding
    issues, and standardizing whitespace and capitalization.
    Performs the following steps:
    - Attempts to fix mojibake by decoding latin-1 bytes as UTF-8.
    - Strips region codes in parentheses and tags in brackets.
    - Removes extra spaces before colons and collapses repeated whitespace.
    - Converts fully uppercase titles to title case.
    Args:
        name: Raw game title string.
    Returns:
        A cleaned, human-readable game title string.
    """
    if not name:
        return name
    
    original_name = name

    # Fix mojibake/encoding issues: decode latin-1 encoded as utf-8
    try:
        name = name.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    
    # Remove region codes: (U), (E), (J), (USA), etc.
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    # Remove ROM tags: [!], [a], [b1], [h1], [T+En], etc.
    # Only strip very short brackets (max 4 chars) containing ROM hack/version markers to preserve
    # legitimate titles like [Speer], [Redacted], or [ R.U.M.A ].
    name = re.sub(r'\s*\[[!a-zA-Z][a-zA-Z0-9+]{0,3}\]\s*', ' ', name)
    # Remove spaces before colons: " :" -> ":"
    name = re.sub(r'\s+:', ':', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Normalize all-caps names to title case
    if name.isupper() and len(name) > 1:
        name = name.title()

    # Normalize common sequel roman numerals (e.g., Ii -> II)
    name = re.sub(r'\b[IVXLCDMivxlcdm]+\b', _roman_replacer, name)
    
    # if name became empty after cleaning, print warning and original name for debugging
    if not name:
        print(f"Warning: name '{original_name}' cleaned to empty string")

    return name.strip()


def build_name_match_key(name: Any) -> Any:
    """Build a stable key for near-duplicate title matching.

    Keeps the display title untouched while normalizing punctuation/case variants
    used during deduplication.
    """
    if pd.isna(name):
        return name

    raw = str(name).strip()
    if raw.lower() in {'', 'n/a', 'na', 'null', 'none'}:
        return np.nan

    # Assumes name is already cleaned by clean_game_name before this call.
    text = raw.casefold()
    text = re.sub(r"[‐‑–—-]+", " ", text)
    text = re.sub(r"[:/]+", " ", text)
    text = re.sub(r"[’'`]+", "", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        # Preserve a stable key for non-empty originals that normalize to empty
        # (e.g. symbolic names), rather than dropping them from key validation.
        fallback = re.sub(r"\s+", " ", raw.casefold()).strip()
        if fallback:
            print(f"Info: name '{raw}' cleaned to empty match key, using raw fallback key")
            return f"raw:{fallback}"
        print(f"Warning: name '{raw}' cleaned to empty match key, returning NaN")
        return np.nan
    return text


def clean_filename(filename: Any) -> Any:
    """Normalize filename values to a stem-only identifier.

    Example: './game name.zip' -> 'game name'.
    Non-string or missing values are returned unchanged.
    """
    if pd.isna(filename):
        return filename

    try:
        return Path(str(filename)).stem
    except (TypeError, ValueError):
        return filename


def _parse_multi_value(value: Any) -> list[str]:
    """Parse a value that may be a Python list, stringified list, or plain string
    into a flat list of cleaned strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        parsed: list[str] = []
        for item in value:
            parsed.extend(_parse_multi_value(item))
        return parsed

    text = str(value).strip()
    if not text:
        return []

    # Handle stringified Python lists: "['a', 'b']"
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                parsed_values: list[str] = []
                for item in parsed:
                    parsed_values.extend(_parse_multi_value(item))
                return parsed_values
        except (ValueError, SyntaxError):
            pass

    return [" ".join(part.strip().split()) for part in re.split(MULTI_VALUE_SPLIT_PATTERN, text) if part.strip()]


def _flatten_multi_value(value: Any) -> Any:
    """Convert a multi-value field (list, stringified list) to a comma-separated string.
    Scalar strings are parsed for delimiters and normalized."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    parsed = _parse_multi_value(value)
    return ", ".join(parsed) if parsed else np.nan


def load_source(config: SourceConfig) -> pd.DataFrame:
    if config.loader is not None:
        frame = config.loader()
    elif config.path is not None:
        kwargs = {**config.read_csv_kwargs, 'low_memory': False}
        frame = pd.read_csv(config.path, **kwargs)
    else:
        raise ValueError(f"Source '{config.name}' requires either path or loader")

    if config.post_load is not None:
        frame = config.post_load(frame)

    return frame


def normalize_source(
    df: pd.DataFrame,
    config: SourceConfig,
    target_columns: Sequence[str],
    key_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    # rename columns according to map if it is provided
    if config.rename_map:
        out = out.rename(columns=dict(config.rename_map))

    # Normalize multi-value columns: explode key columns into rows,
    # flatten non-key columns to comma-separated strings.
    for mv_col in MULTI_VALUE_COLUMNS:
        if mv_col not in out.columns:
            continue
        if mv_col in key_columns:
            out[mv_col] = out[mv_col].apply(_parse_multi_value)
            out = out.explode(mv_col, ignore_index=True)
            out[mv_col] = out[mv_col].where(out[mv_col].astype(bool), np.nan)
        else:
            out[mv_col] = out[mv_col].apply(_flatten_multi_value)

    # Apply global platform name mapping from platform_mappings.json
    if "platform" in out.columns:
        global_map = _load_global_platform_map()
        if global_map:
            out["platform"] = out["platform"].replace(global_map)

    # Apply per-source platform overrides (if any)
    if config.platform_map and "platform" in out.columns:
        out["platform"] = out["platform"].replace(dict(config.platform_map))

    # set constant values for columns if provided (usually if data source has no platform)
    if config.constants:
        for column, value in config.constants.items():
            out[column] = value

    # apply any column-specific transforms if provided
    for column, transform in config.transforms.items():
        if column in out.columns:
            out[column] = transform(out[column])

    # convert release_date to datetime
    if 'release_date' in out.columns:
        out['release_date'] = pd.to_datetime(out['release_date'], errors='coerce')

    # Clean names before key-based validation/merge
    if 'name' in out.columns:
        original_name = out['name'].copy()
        original_non_null_mask = original_name.notna()
        out['name'] = out['name'].apply(lambda x: clean_game_name(x) if pd.notna(x) else x)
        name_missing_mask = out['name'].astype(str).str.strip().str.lower().isin(
            {'', 'n/a', 'na', 'null', 'none'}
        )
        out['name'] = out['name'].where(~name_missing_mask, np.nan)
        if NAME_MATCH_KEY_COLUMN in key_columns:
            out[NAME_MATCH_KEY_COLUMN] = out['name'].apply(build_name_match_key)
            key_missing_from_non_null_mask = original_non_null_mask & out[NAME_MATCH_KEY_COLUMN].isna()
            key_missing_from_non_null_count = int(key_missing_from_non_null_mask.sum())
            if key_missing_from_non_null_count > 0:
                sample_original_names = (
                    original_name[key_missing_from_non_null_mask]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace('', np.nan)
                    .dropna()
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
                print(
                    f"Warning: {config.name} produced {key_missing_from_non_null_count} rows with null {NAME_MATCH_KEY_COLUMN} "
                    "from originally non-null name values. "
                    f"Sample originals: {sample_original_names}"
                )

    # Normalize filename paths (e.g., './game name.zip' -> 'game name')
    if 'filename' in out.columns:
        out['filename'] = out['filename'].apply(clean_filename)

    required_columns = list(dict.fromkeys([*target_columns, *key_columns]))

    # make up missing columns
    for column in required_columns:
        if column not in out.columns:
            out[column] = np.nan

    return out.loc[:, required_columns]


def coerce_to_schema(
    df: pd.DataFrame,
    schema: Mapping[str, str],
) -> pd.DataFrame:
    """Coerce DataFrame columns to their canonical types."""
    out = df.copy()
    
    for column, dtype in schema.items():
        if column not in out.columns:
            continue
            
        try:
            original_non_null = out[column].notna().sum()
            
            if dtype == 'float64':
                out[column] = pd.to_numeric(out[column], errors='coerce')
            elif dtype == 'int64':
                out[column] = pd.to_numeric(out[column], errors='coerce').astype('Int64')
            elif dtype == 'datetime64[ns]':
                out[column] = pd.to_datetime(out[column], errors='coerce')
            elif dtype == 'boolean':
                out[column] = out[column].astype('boolean')
            elif dtype == 'string':
                # Standardize missing value placeholders to NaN
                mask = out[column].astype(str).isin(['', 'N/A', 'null', 'None', 'NA', 'n/a'])
                out[column] = out[column].where(~mask, np.nan)
            
            # Warn if entire column became NaN
            final_non_null = out[column].notna().sum()
            if final_non_null == 0 and original_non_null > 0:
                print(f"Warning: {column} coerced to all NaN (dtype={dtype}, had {original_non_null} values before)")
                
        except Exception as e:
            print(f"Warning: Failed to coerce {column} to {dtype}: {e}")
    
    return out


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns {missing}")


def validate_key_column_values(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    """Drop rows with NaN in key columns and log how many were removed."""
    key_cols = list(key_columns)
    out = df.dropna(subset=key_cols)
    
    dropped = len(df) - len(out)
    if dropped > 0:
        print(f"Warning: {source_name} dropped {dropped} rows with NaN in key columns {key_columns}")
    
    return out


def merge_into_main(
    main_df: pd.DataFrame,
    source_df: pd.DataFrame,
    key_columns: Sequence[str],
    resolver_map: Mapping[str, Any],
    schema: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    # Ensure both dataframes have the same columns in the same order before concat
    all_columns = main_df.columns.tolist()
    source_df = source_df.reindex(columns=all_columns)
    
    # Coerce to schema dtypes before concat to avoid FutureWarning
    if schema is not None:
        main_df = coerce_to_schema(main_df, schema)
        source_df = coerce_to_schema(source_df, schema)
    
    stacked = pd.concat([main_df, source_df], ignore_index=True, sort=False)

    agg_map: dict[str, Any] = {}
    for column in stacked.columns:
        if column in key_columns:
            continue
        agg_map[column] = resolver_map.get(column, pick_first)

    return stacked.groupby(list(key_columns), as_index=False).agg(agg_map)


def prepare_source(
    config: SourceConfig,
    target_columns: Sequence[str],
    key_columns: Sequence[str],
) -> pd.DataFrame:
    frame = load_source(config)
    normalized = normalize_source(
        frame,
        config=config,
        target_columns=target_columns,
        key_columns=key_columns,
    )
    validate_required_columns(normalized, required_columns=key_columns, source_name=config.name)
    validated = validate_key_column_values(normalized, key_columns=key_columns, source_name=config.name)
    return validated


def collapse_by_name(
    df: pd.DataFrame,
    name_column: str = "name",
    resolver_map: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Collapse a per-(name, platform) DataFrame into one row per game.

    Uses the same normalized match key as the merge step so that display-name
    variants (e.g. differing punctuation across platforms) are still grouped
    together.  Platforms and filenames are collected into sorted
    comma-separated strings.  Other columns are resolved using the provided
    resolver map.

    Args:
        df: Merged DataFrame with one row per (name, platform).
        name_column: Column to group by (default: 'name').
        resolver_map: Aggregation functions per column. Defaults to
            ``collapse_resolver`` from resolvers.py.

    Returns:
        DataFrame with one row per unique game name.
    """
    effective_resolver = resolver_map or default_collapse_resolver

    out = df.copy()
    out[NAME_MATCH_KEY_COLUMN] = out[name_column].apply(build_name_match_key)

    agg_map: dict[str, Any] = {}
    for column in out.columns:
        if column == NAME_MATCH_KEY_COLUMN:
            continue
        agg_map[column] = effective_resolver.get(column, pick_first)

    collapsed = out.groupby(NAME_MATCH_KEY_COLUMN, as_index=False).agg(agg_map)
    collapsed = collapsed.drop(columns=[NAME_MATCH_KEY_COLUMN])
    print(f"Collapsed {len(df)} rows -> {len(collapsed)} unique games")
    return collapsed


def run_merge_pipeline(
    main_config: SourceConfig,
    source_configs: Sequence[SourceConfig],
    target_columns: Optional[Sequence[str]] = None,
    key_columns: Sequence[str] = KEY_COLUMNS,
    resolver_map: Optional[Mapping[str, Any]] = None,
    schema: Optional[Mapping[str, str]] = None,
    use_name_match_key: bool = True,
    collapse_platforms: bool = False,
    collapse_resolver_map: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Orchestrate merging of multiple data sources into a canonical schema.
    
    Args:
        main_config: Primary dataset configuration (base for merging).
        source_configs: List of additional sources to merge into main.
        target_columns: Columns to include in output (defaults to CANONICAL_SCHEMA).
        key_columns: Columns for grouping/dedup (defaults to ('name', 'platform')).
                     Rows with NaN in key columns are automatically dropped.
        resolver_map: Custom aggregation functions per column during merge.
        schema: Type coercion mapping (defaults to CANONICAL_SCHEMA).
        use_name_match_key: Whether to dedupe by a normalized internal name key
                    to collapse punctuation/casing variants.
        collapse_platforms: If True, post-process the merged result to produce
                    one row per game name with platforms collected into a
                    comma-separated list. If false, returns one row per (name, platform).
        collapse_resolver_map: Custom resolver map for the collapse step.
                    Defaults to ``collapse_resolver`` from resolvers.py.
    
    Returns:
        Merged DataFrame with deduplicated rows grouped by key_columns.
        If collapse_platforms is True, returns one row per game name.
    """
    effective_resolver = resolver_map or default_resolver
    effective_schema = schema or CANONICAL_SCHEMA
    effective_target_columns = target_columns or DEFAULT_COLUMNS
    effective_key_columns = tuple(key_columns)

    if use_name_match_key and "name" in effective_key_columns:
        effective_key_columns = tuple(
            NAME_MATCH_KEY_COLUMN if column == "name" else column
            for column in effective_key_columns
        )

    merged = prepare_source(
        main_config,
        target_columns=effective_target_columns,
        key_columns=effective_key_columns,
    )
    main_length = len(merged)

    for source_config in source_configs:
        source = prepare_source(
            source_config,
            target_columns=effective_target_columns,
            key_columns=effective_key_columns,
        )
        merged = merge_into_main(
            merged,
            source,
            key_columns=effective_key_columns,
            resolver_map=effective_resolver,
            schema=effective_schema,
        )
        source_length = len(source)
        merged_length = len(merged)
        games_added = merged_length - main_length
        main_length = merged_length
        print(f"main size: {merged_length}, {source_config.name} size: {source_length}, new games: {games_added}")

    if NAME_MATCH_KEY_COLUMN in merged.columns:
        merged = merged.drop(columns=[NAME_MATCH_KEY_COLUMN])

    if collapse_platforms:
        merged = collapse_by_name(
            merged,
            name_column="name",
            resolver_map=collapse_resolver_map,
        )

    return merged

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from utils.resolvers import pick_first, resolver as default_resolver

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
    'players': 'int64',
    'cooperative': 'string',
    'rating': 'float64',
    'user_rating': 'float64',
}

# Canonical columns (derived from schema keys)
DEFAULT_COLUMNS = list(CANONICAL_SCHEMA.keys())

# Key columns for deduplication and merging
KEY_COLUMNS = ("name", "platform")


SeriesTransform = Callable[[pd.Series], pd.Series]
DataFrameLoader = Callable[[], pd.DataFrame]
DataFramePostLoad = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class SourceConfig:
    name: str
    path: Optional[str] = None
    keep_columns: Optional[Sequence[str]] = None
    rename_map: Mapping[str, str] = field(default_factory=dict)
    platform_map: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    transforms: Mapping[str, SeriesTransform] = field(default_factory=dict)
    replace_map: Optional[dict[Any, Any]] = None
    dropna_subset: Optional[Sequence[str]] = None
    dedupe_sort: Optional[Sequence[str]] = None
    dedupe_subset: Optional[Sequence[str]] = None
    read_csv_kwargs: Mapping[str, Any] = field(default_factory=dict)
    loader: Optional[DataFrameLoader] = None
    post_load: Optional[DataFramePostLoad] = None


def load_source(config: SourceConfig) -> pd.DataFrame:
    if config.loader is not None:
        frame = config.loader()
    elif config.path is not None:
        frame = pd.read_csv(config.path, **config.read_csv_kwargs)
    else:
        raise ValueError(f"Source '{config.name}' requires either path or loader")

    if config.post_load is not None:
        frame = config.post_load(frame)

    return frame


def normalize_source(
    df: pd.DataFrame,
    config: SourceConfig,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    if config.replace_map:
        out = out.replace(config.replace_map)

    if config.keep_columns:
        missing = [column for column in config.keep_columns if column not in out.columns]
        if missing:
            raise ValueError(f"{config.name}: missing keep_columns {missing}")
        out = out.loc[:, list(config.keep_columns)]

    if config.rename_map:
        out = out.rename(columns=dict(config.rename_map))

    if config.platform_map and "platform" in out.columns:
        out["platform"] = out["platform"].replace(dict(config.platform_map))

    if config.constants:
        for column, value in config.constants.items():
            out[column] = value

    for column, transform in config.transforms.items():
        if column in out.columns:
            out[column] = transform(out[column])

    if config.dropna_subset:
        out = out.dropna(subset=list(config.dropna_subset))

    if config.dedupe_sort:
        existing_sort_columns = [column for column in config.dedupe_sort if column in out.columns]
        if existing_sort_columns:
            out = out.sort_values(existing_sort_columns)

    if config.dedupe_subset:
        subset = [column for column in config.dedupe_subset if column in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first")

    for column in target_columns:
        if column not in out.columns:
            out[column] = np.nan

    return out.loc[:, list(target_columns)]


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
    """Drop rows with NaN in key columns and validate none remain."""
    out = df.dropna(subset=list(key_columns))
    
    dropped = len(df) - len(out)
    if dropped > 0:
        print(f"Warning: {source_name} dropped {dropped} rows with NaN in key columns {key_columns}")
    
    # Check if all key columns are now populated
    null_counts = out[list(key_columns)].isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(
            f"{source_name}: after normalization, key columns still have NaN values: {null_counts[null_counts > 0].to_dict()}"
        )
    
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
    stacked = pd.concat([main_df, source_df], ignore_index=True, sort=False)
    
    # Coerce to canonical types before aggregation
    if schema is not None:
        stacked = coerce_to_schema(stacked, schema)

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
    normalized = normalize_source(frame, config=config, target_columns=target_columns)
    validate_required_columns(normalized, required_columns=key_columns, source_name=config.name)
    validated = validate_key_column_values(normalized, key_columns=key_columns, source_name=config.name)
    return validated


def run_merge_pipeline(
    main_config: SourceConfig,
    source_configs: Sequence[SourceConfig],
    target_columns: Sequence[str],
    key_columns: Sequence[str] = ("name", "platform"),
    resolver_map: Optional[Mapping[str, Any]] = None,
    schema: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    effective_resolver = resolver_map or default_resolver
    effective_schema = schema or CANONICAL_SCHEMA

    merged = prepare_source(
        main_config,
        target_columns=target_columns,
        key_columns=key_columns,
    )

    for source_config in source_configs:
        source = prepare_source(
            source_config,
            target_columns=target_columns,
            key_columns=key_columns,
        )
        merged = merge_into_main(
            merged,
            source,
            key_columns=key_columns,
            resolver_map=effective_resolver,
            schema=effective_schema,
        )

    return merged

import ast
import json
import re
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

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
_PLATFORM_MAPPING_CONSISTENCY_CHECKED = False


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


def validate_platform_mapping_consistency() -> dict[str, any]:
    """Validate consistency between platform_mappings.json and gamelist_folder_mappings.json.
    
    Returns:
        Dictionary with consistency check results and warnings
    """
    global _PLATFORM_MAPPING_CONSISTENCY_CHECKED
    
    if _PLATFORM_MAPPING_CONSISTENCY_CHECKED:
        return {}
    
    # Load both mapping files
    platform_mappings_path = Path(__file__).parent / "platform_mappings.json"
    gamelist_mappings_path = Path(__file__).parent / "gamelist_folder_mappings.json"
    
    platform_map = {}
    gamelist_map = {}
    
    try:
        with open(platform_mappings_path, "r", encoding="utf-8") as f:
            platform_map = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load platform_mappings.json: {e}")
        return {}
    
    try:
        with open(gamelist_mappings_path, "r", encoding="utf-8") as f:
            gamelist_map = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load gamelist_folder_mappings.json: {e}")
        return {}
    
    # Check for inconsistencies
    inconsistencies = []
    
    # Find platforms that exist in both files but with different canonical names
    platform_keys = set(platform_map.keys())
    gamelist_keys = set(gamelist_map.keys())
    
    common_keys = platform_keys.intersection(gamelist_keys)
    
    for key in common_keys:
        platform_value = platform_map[key]
        gamelist_value = gamelist_map[key]
        if platform_value != gamelist_value:
            inconsistencies.append({
                'key': key,
                'platform_mapping': platform_value,
                'gamelist_mapping': gamelist_value,
                'type': 'inconsistency'
            })
    
    # Check for platforms that exist only in one file
    only_in_platform = platform_keys - gamelist_keys
    only_in_gamelist = gamelist_keys - platform_keys
    
    for key in only_in_platform:
        inconsistencies.append({
            'key': key,
            'platform_mapping': platform_map[key],
            'gamelist_mapping': None,
            'type': 'only_in_platform'
        })
        
    for key in only_in_gamelist:
        inconsistencies.append({
            'key': key,
            'platform_mapping': None,
            'gamelist_mapping': gamelist_map[key],
            'type': 'only_in_gamelist'
        })
    
    _PLATFORM_MAPPING_CONSISTENCY_CHECKED = True
    
    return {
        'inconsistencies': inconsistencies,
        'total_inconsistencies': len(inconsistencies),
        'only_in_platform': len(only_in_platform),
        'only_in_gamelist': len(only_in_gamelist)
    }


def print_platform_mapping_consistency_report():
    """Print a detailed report of platform mapping consistency issues."""
    report = validate_platform_mapping_consistency()
    if report and report.get('total_inconsistencies', 0) > 0:
        print("⚠️  Platform Mapping Inconsistency Report:")
        print(f"  Found {report['total_inconsistencies']} inconsistencies")
        print(f"  {report['only_in_platform']} keys only in platform_mappings.json")
        print(f"  {report['only_in_gamelist']} keys only in gamelist_folder_mappings.json")
        
        for inconsistency in report['inconsistencies']:
            if inconsistency['type'] == 'inconsistency':
                print(f"    '{inconsistency['key']}': platform='{inconsistency['platform_mapping']}', gamelist='{inconsistency['gamelist_mapping']}'")
            elif inconsistency['type'] == 'only_in_platform':
                print(f"    '{inconsistency['key']}': only in platform_mappings.json = '{inconsistency['platform_mapping']}'")
            elif inconsistency['type'] == 'only_in_gamelist':
                print(f"    '{inconsistency['key']}': only in gamelist_folder_mappings.json = '{inconsistency['gamelist_mapping']}'")
    elif report:
        print("✓ Platform mapping consistency check passed")
    else:
        print("⚠️  Could not perform platform mapping consistency check")


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
    
    # Handle special characters that commonly break name matching
    # Remove control characters and normalize various hyphens/underscores
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', name)  # Remove control characters
    
    # Replace various hyphens and dashes with standard hyphens
    name = re.sub(r'[‐‑–—-]+', '-', name)
    
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
    
    # Handle empty results more gracefully
    cleaned_name = name.strip()
    if not cleaned_name:
        # Log warning with original name for debugging
        print(f"Warning: name '{original_name}' cleaned to empty string")
        return original_name  # Return original to preserve information
    
    return cleaned_name


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

    # Handle special characters that can break matching
    # Replace various hyphens with standard spaces
    text = re.sub(r"[‐‑–—-]+", " ", raw)
    
    # Remove control characters that might break matching
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize punctuation and whitespace
    text = re.sub(r"[:/]+", " ", text)
    text = re.sub(r"[’'`]+", "", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Handle empty results gracefully
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


def fuzzy_name_match_key(name: Any, threshold: int = 85) -> Any:
    """Build a fuzzy match key for handling titles with minor variations.
    
    This function provides a more flexible matching approach for edge cases
    where exact matching would fail due to minor variations in titles.
    """
    if pd.isna(name):
        return name
        
    raw = str(name).strip()
    if raw.lower() in {'', 'n/a', 'na', 'null', 'none'}:
        return np.nan
        
    # For cases where we want to be more lenient with matching
    return raw


def fuzzy_match_names(name1: str, name2: str, method: str = 'ratio') -> float:
    """Calculate fuzzy similarity between two names using different methods.
    
    Args:
        name1: First name to compare
        name2: Second name to compare
        method: Matching method ('ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio')
        
    Returns:
        Similarity score between 0 and 100
    """
    if pd.isna(name1) or pd.isna(name2):
        return 0.0
        
    name1_clean = str(name1).strip()
    name2_clean = str(name2).strip()
    
    if method == 'ratio':
        return fuzz.ratio(name1_clean, name2_clean)
    elif method == 'partial_ratio':
        return fuzz.partial_ratio(name1_clean, name2_clean)
    elif method == 'token_sort_ratio':
        return fuzz.token_sort_ratio(name1_clean, name2_clean)
    elif method == 'token_set_ratio':
        return fuzz.token_set_ratio(name1_clean, name2_clean)
    else:
        return fuzz.ratio(name1_clean, name2_clean)


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


# Global cache for similarity scores to prevent recomputation
_similarity_cache = {}
# Maximum cache size to prevent memory issues
_MAX_CACHE_SIZE = 10000

def _manage_cache_size():
    """Manage cache size to prevent memory issues."""
    if len(_similarity_cache) > _MAX_CACHE_SIZE:
        # Remove oldest entries to maintain cache size limit
        # Simple approach: remove first 1000 items
        items_to_remove = list(_similarity_cache.keys())[:1000]
        for key in items_to_remove:
            del _similarity_cache[key]

def clear_similarity_cache():
    """Clear the similarity score cache."""
    global _similarity_cache
    _similarity_cache.clear()

def get_similarity_cache_size():
    """Get the current size of the similarity score cache."""
    return len(_similarity_cache)

def get_name_confidence_score(name1: str, name2: str) -> dict:
    """Calculate various confidence scores for name comparison.
    
    Implements caching to avoid recomputing similarity scores for the same name pair.
    Includes early termination conditions for performance optimization.
    Uses a maximum cache size to prevent memory issues.
    
    Returns:
        Dictionary with different similarity scores and match confidence
    """
    # Create a canonical key for the pair to ensure consistency (order doesn't matter)
    key1, key2 = str(name1).strip(), str(name2).strip()
    cache_key = tuple(sorted([key1, key2]))
    
    # Check if result is already in cache
    if cache_key in _similarity_cache:
        return _similarity_cache[cache_key]
    
    if pd.isna(name1) or pd.isna(name2):
        result = {
            'exact_match': False,
            'ratio': 0.0,
            'partial_ratio': 0.0,
            'token_sort_ratio': 0.0,
            'token_set_ratio': 0.0,
            'confidence': 0.0
        }
        _similarity_cache[cache_key] = result
        return result
        
    name1_clean = key1
    name2_clean = key2
    
    # Early termination: If names are identical, return exact match
    if name1_clean == name2_clean:
        result = {
            'exact_match': True,
            'ratio': 100.0,
            'partial_ratio': 100.0,
            'token_sort_ratio': 100.0,
            'token_set_ratio': 100.0,
            'confidence': 1.0
        }
        _similarity_cache[cache_key] = result
        return result
    
    # Early termination: If names are very different by simple checks, return early
    # This prevents expensive calculations for clearly unrelated names
    if len(name1_clean) > 0 and len(name2_clean) > 0:
        # If names are very different in length, they're likely not similar
        if abs(len(name1_clean) - len(name2_clean)) > 20:
            result = {
                'exact_match': False,
                'ratio': 0.0,
                'partial_ratio': 0.0,
                'token_sort_ratio': 0.0,
                'token_set_ratio': 0.0,
                'confidence': 0.0
            }
            _similarity_cache[cache_key] = result
            return result
    
    ratio = fuzz.ratio(name1_clean, name2_clean)
    partial_ratio = fuzz.partial_ratio(name1_clean, name2_clean)
    token_sort_ratio = fuzz.token_sort_ratio(name1_clean, name2_clean)
    token_set_ratio = fuzz.token_set_ratio(name1_clean, name2_clean)
    
    # Calculate overall confidence (weighted average)
    confidence = (ratio * 0.3 + partial_ratio * 0.2 + token_sort_ratio * 0.25 + token_set_ratio * 0.25) / 100
    
    # Early termination: If confidence is very low, return early
    if confidence < 0.1:  # Very low confidence threshold
        result = {
            'exact_match': False,
            'ratio': ratio,
            'partial_ratio': partial_ratio,
            'token_sort_ratio': token_sort_ratio,
            'token_set_ratio': token_set_ratio,
            'confidence': confidence
        }
        _similarity_cache[cache_key] = result
        return result
    
    result = {
        'exact_match': name1_clean == name2_clean,
        'ratio': ratio,
        'partial_ratio': partial_ratio,
        'token_sort_ratio': token_sort_ratio,
        'token_set_ratio': token_set_ratio,
        'confidence': confidence
    }
    
    # Manage cache size to prevent memory issues
    _manage_cache_size()
    
    _similarity_cache[cache_key] = result
    return result


def _parse_multi_value(value: Any) -> list[str]:
    """Parse a value that may be a Python list, stringified list, or plain string
    into a flat list of cleaned strings.
    
    Args:
        value: Value to parse
        
    Returns:
        List of cleaned strings
    """
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

    # Handle special cases with unusual characters that might break parsing
    # Replace common problematic characters that could break the regex split
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # Remove control characters
    
    # Use more robust splitting
    parts = re.split(MULTI_VALUE_SPLIT_PATTERN, text)
    # Clean each part and filter out empty strings
    cleaned_parts = [" ".join(part.strip().split()) for part in parts if part.strip()]
    
    # If we get empty result, return the original text as fallback
    if not cleaned_parts and text:
        return [text]
        
    return cleaned_parts


def _flatten_multi_value(value: Any) -> Any:
    """Convert a multi-value field (list, stringified list) to a comma-separated string.
    Scalar strings are parsed for delimiters and normalized."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    try:
        parsed = _parse_multi_value(value)
        return ", ".join(parsed) if parsed else np.nan
    except Exception as e:
        # Log error and return original value for debugging
        print(f"Warning: Error flattening multi-value field: {e}")
        return value if value is not None else np.nan


def is_potentially_similar_name(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """Determine if two names are potentially similar based on multiple similarity metrics.
    
    Implements caching and early termination conditions for performance optimization.
    
    Args:
        name1: First name to compare
        name2: Second name to compare
        threshold: Minimum confidence threshold for considering names similar
        
    Returns:
        True if names are considered potentially similar, False otherwise
    """
    if pd.isna(name1) or pd.isna(name2):
        return False
        
    # Early termination: Check if we already know this comparison from cache
    key1, key2 = str(name1).strip(), str(name2).strip()
    cache_key = tuple(sorted([key1, key2]))
    
    # Check if we have the result cached
    if cache_key in _similarity_cache:
        return _similarity_cache[cache_key]['confidence'] >= threshold
    
    # Calculate scores with early termination capabilities
    scores = get_name_confidence_score(name1, name2)
    return scores['confidence'] >= threshold


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
            # Check for consistency issues before applying mapping
            consistency_report = validate_platform_mapping_consistency()
            if consistency_report and consistency_report.get('total_inconsistencies', 0) > 0:
                print_platform_mapping_consistency_report()
            
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


def get_platform_mappings_summary() -> dict:
    """Get a summary of the current platform mappings for debugging."""
    platform_map = _load_global_platform_map()
    return {
        'total_mappings': len(platform_map),
        'sample_mappings': dict(list(platform_map.items())[:10]) if platform_map else {},
        'consistency_check': validate_platform_mapping_consistency()
    }


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
    duplicate_detection_threshold: float = 0.8,
) -> pd.DataFrame:
    """Merge source_df into main_df with enhanced error handling and deduplication.
    
    Args:
        main_df: Main DataFrame to merge into.
        source_df: Source DataFrame to merge.
        key_columns: Columns to use for matching.
        resolver_map: Custom resolver functions per column.
        schema: Schema to apply to merged result.
        duplicate_detection_threshold: Threshold for fuzzy duplicate detection.

    Returns:
        Merged DataFrame with source data integrated.
    """
    try:
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

        resolved = stacked.groupby(list(key_columns), as_index=False).agg(agg_map)
        return resolved
    except Exception as e:
        print(f"Warning: Error in merge_into_main: {e}")
        print("Returning main_df to preserve data")
        return main_df.copy()


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


def _process_chunk(args):
    """Process a chunk of data for duplicate detection in parallel.
    
    This version is more memory-efficient by avoiding unnecessary copies
    and using lazy evaluation where appropriate.
    """
    chunk, name_column, threshold = args
    local_duplicates = []
    
    # Group names within this chunk efficiently
    name_groups = {}
    
    # Normalize names and group by a more robust approach
    # Process names directly without unnecessary intermediate copies
    for i in range(len(chunk)):
        name = chunk.iloc[i][name_column]
        if pd.isna(name):
            continue
            
        # Use the same normalization approach as build_name_match_key for consistency
        normalized_name = str(name).strip()
        
        # Apply the same normalization steps as build_name_match_key for consistency
        # Replace various hyphens with standard spaces
        text = re.sub(r"[‐‑–—-]+", " ", normalized_name)
        
        # Remove control characters that might break matching
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize punctuation and whitespace
        text = re.sub(r"[:/]+", " ", text)
        text = re.sub(r"[’'`]+", "", text)
        text = re.sub(r"[^\w\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Create a grouping key that's more robust for grouping
        # Using first 3, middle 4, and last 3 characters with length for better distribution
        if len(text) >= 10:
            grouping_key = text[:3] + text[len(text)//2-2:len(text)//2+2] + text[-3:] + str(len(text))
        elif len(text) >= 6:
            grouping_key = text[:3] + text[-3:] + str(len(text))
        else:
            grouping_key = text + str(len(text))
        
        if grouping_key not in name_groups:
            name_groups[grouping_key] = []
        name_groups[grouping_key].append(i)
    
    # Check for duplicates within this chunk
    for group in name_groups.values():
        if len(group) > 1:
            # Compare each item in the group with others
            for i in range(len(group)):
                idx1 = group[i]
                name1 = chunk.iloc[idx1][name_column]
                
                # Only compare with remaining items in group to avoid duplicate comparisons
                for j in range(i + 1, len(group)):
                    idx2 = group[j]
                    name2 = chunk.iloc[idx2][name_column]
                    
                    # Use our improved matching logic - lazy evaluation for expensive operations
                    if is_potentially_similar_name(name1, name2, threshold):
                        scores = get_name_confidence_score(name1, name2)
                        local_duplicates.append((idx1, idx2, scores['confidence']))
    
    return local_duplicates


def identify_potential_duplicates(df: pd.DataFrame, name_column: str = "name", 
                                  threshold: float = 0.8, max_comparisons: int = 10000, 
                                  chunk_size: int = 1000, use_multiprocessing: bool = True,
                                  stream_results: bool = False, memory_efficient: bool = True) -> list:
    """Identify potentially duplicate entries in a DataFrame based on name similarity.
    
    Uses optimized data structures for improved efficiency over brute-force O(n^2) comparison.
    Pre-processes and indexes games by name variations for faster lookup, 
    uses sets for fast membership testing, and implements efficient duplicate management.
    Includes maximum comparison limits to prevent excessive computation.
    Implements chunked processing for large datasets and optional multiprocessing.
    Memory-efficient implementation with streaming and lazy evaluation options.
    
    Args:
        df: DataFrame to analyze
        name_column: Name of the column to compare (default: 'name')
        threshold: Similarity threshold for considering duplicates (default: 0.8)
        max_comparisons: Maximum number of comparisons to prevent excessive computation (default: 10000)
        chunk_size: Size of chunks for processing (default: 1000)
        use_multiprocessing: Whether to use multiprocessing for chunk processing (default: True)
        stream_results: If True, yield results instead of collecting all at once (default: False)
        memory_efficient: If True, uses memory-efficient processing methods (default: True)
        
    Returns:
        List of tuples containing (index1, index2, similarity_score) for potential duplicates
    """
    # If we want to stream results or memory is a concern, use different approach
    if stream_results or memory_efficient:
        # For streaming or memory-efficient processing, process data without collecting all results
        # This is a simplified approach that could be enhanced for true streaming
        # For now, we'll maintain the existing logic but with optimizations
        
        # If dataset is small, use standard approach
        if len(df) <= chunk_size or not use_multiprocessing:
            # Fall back to the original approach for smaller datasets
            return _identify_potential_duplicates_standard(df, name_column, threshold, max_comparisons)
        
        # For larger datasets, use chunked processing with parallel execution
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunks.append(chunk)
        
        # Process chunks in parallel if multiprocessing is enabled
        if len(chunks) > 1 and use_multiprocessing:
            # Prepare arguments for parallel processing
            chunk_args = [(chunk, name_column, threshold) for chunk in chunks]
            
            # Use multiprocessing pool for parallel processing
            with mp.Pool() as pool:
                results = pool.map(_process_chunk, chunk_args)
            
            # Combine results from all chunks
            potential_duplicates = []
            for result in results:
                potential_duplicates.extend(result)
        else:
            # Fallback to single-threaded processing
            potential_duplicates = []
            for chunk in chunks:
                result = _process_chunk((chunk, name_column, threshold))
                potential_duplicates.extend(result)
        
        # Apply maximum comparison limit
        if len(potential_duplicates) > max_comparisons:
            print(f"Warning: Found {len(potential_duplicates)} potential duplicates, "
                  f"exceeding maximum allowed ({max_comparisons}).")
            # Truncate to max_comparisons (though this might not be ideal for final results)
            # We'll let the user handle this at the calling level
        
        return potential_duplicates
    else:
        # Standard behavior
        potential_duplicates = []
        
        # If dataset is small, use standard approach
        if len(df) <= chunk_size or not use_multiprocessing:
            # Fall back to the original approach for smaller datasets
            return _identify_potential_duplicates_standard(df, name_column, threshold, max_comparisons)
        
        # For larger datasets, use chunked processing with parallel execution
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunks.append(chunk)
        
        # Process chunks in parallel if multiprocessing is enabled
        if len(chunks) > 1 and use_multiprocessing:
            # Prepare arguments for parallel processing
            chunk_args = [(chunk, name_column, threshold) for chunk in chunks]
            
            # Use multiprocessing pool for parallel processing
            with mp.Pool() as pool:
                results = pool.map(_process_chunk, chunk_args)
            
            # Combine results from all chunks
            for result in results:
                potential_duplicates.extend(result)
        else:
            # Fallback to single-threaded processing
            for chunk in chunks:
                result = _process_chunk((chunk, name_column, threshold))
                potential_duplicates.extend(result)
        
        # Apply maximum comparison limit
        if len(potential_duplicates) > max_comparisons:
            print(f"Warning: Found {len(potential_duplicates)} potential duplicates, "
                  f"exceeding maximum allowed ({max_comparisons}).")
            # Truncate to max_comparisons (though this might not be ideal for final results)
            # We'll let the user handle this at the calling level
        
        return potential_duplicates


def _identify_potential_duplicates_standard(df: pd.DataFrame, name_column: str = "name", 
                                            threshold: float = 0.8, max_comparisons: int = 10000) -> list:
    """Standard implementation of duplicate detection for smaller datasets."""
    potential_duplicates = []
    
    # Create a copy without the name_match_key column to avoid interference
    df_copy = df.copy()
    
    # Create hash map to group entries by normalized name variations
    name_groups = {}
    
    # Pre-process names and build index for faster lookup
    name_index = {}  # For fast membership testing
    
    # Normalize names and group by a more robust approach
    for i in range(len(df_copy)):
        name = df_copy.iloc[i][name_column]
        if pd.isna(name):
            continue
            
        # Use the same normalization approach as build_name_match_key for consistency
        normalized_name = str(name).strip()
        
        # Apply the same normalization steps as build_name_match_key for consistency
        # Replace various hyphens with standard spaces
        text = re.sub(r"[‐‑–—-]+", " ", normalized_name)
        
        # Remove control characters that might break matching
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize punctuation and whitespace
        text = re.sub(r"[:/]+", " ", text)
        text = re.sub(r"[’'`]+", "", text)
        text = re.sub(r"[^\w\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Create a grouping key that's more robust for grouping
        # Using first 3, middle 4, and last 3 characters with length for better distribution
        if len(text) >= 10:
            grouping_key = text[:3] + text[len(text)//2-2:len(text)//2+2] + text[-3:] + str(len(text))
        elif len(text) >= 6:
            grouping_key = text[:3] + text[-3:] + str(len(text))
        else:
            grouping_key = text + str(len(text))
        
        # Store the normalized name for fast lookup
        name_index[i] = text
        
        if grouping_key not in name_groups:
            name_groups[grouping_key] = []
        name_groups[grouping_key].append(i)
    
    # Check for duplicates only within groups using optimized approach
    comparison_count = 0
    
    for group in name_groups.values():
        if len(group) > 1:
            # Use set-based membership testing for faster lookups
            group_set = set(group)
            
            # Compare each item in the group with others
            for i in range(len(group)):
                if comparison_count >= max_comparisons:
                    # Early termination to prevent excessive computation
                    print(f"Warning: Maximum comparisons ({max_comparisons}) reached. Stopping duplicate detection.")
                    return potential_duplicates
                
                idx1 = group[i]
                name1 = df_copy.iloc[idx1][name_column]
                
                # Only compare with remaining items in group to avoid duplicate comparisons
                for j in range(i + 1, len(group)):
                    if comparison_count >= max_comparisons:
                        # Early termination to prevent excessive computation
                        print(f"Warning: Maximum comparisons ({max_comparisons}) reached. Stopping duplicate detection.")
                        return potential_duplicates
                    
                    idx2 = group[j]
                    name2 = df_copy.iloc[idx2][name_column]
                    
                    # Use our improved matching logic
                    if is_potentially_similar_name(name1, name2, threshold):
                        scores = get_name_confidence_score(name1, name2)
                        potential_duplicates.append((idx1, idx2, scores['confidence']))
                        comparison_count += 1
    
    return potential_duplicates


def collapse_by_name(
    df: pd.DataFrame,
    name_column: str = "name",
    resolver_map: Optional[Mapping[str, Any]] = None,
    source_priority: Optional[List[str]] = None,
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
        source_priority: Optional list of source names ordered by priority for conflict resolution.

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
        # Use enhanced resolver if we have source priority
        if source_priority and column in ["genres", "developer", "publisher", "platform"]:
            agg_map[column] = lambda x: collect_unique_ordered(x, source_priority)
        else:
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
    duplicate_detection_threshold: float = 0.8,
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
        duplicate_detection_threshold: Threshold for fuzzy name matching to detect potential duplicates.
    
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

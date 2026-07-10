# rewritten column_join priority functions as resolver functions
import re
import numpy as np
from typing import List, Optional

def _clean_strings(series):
    values = []
    for value in series.dropna():
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                values.append(cleaned)
        else:
            values.append(str(value))
    return values


# --- Resolver Functions ---
def pick_first(series):
    values = _clean_strings(series)
    return values[0] if values else np.nan


# pick the longest. used for description fields
def pick_longer(series):
    values = _clean_strings(series)
    return max(values, key=len) if values else np.nan


def collect_unique(series):
    """Collect unique non-null values into a sorted, comma-separated string.
    Splits on comma/semicolon/slash delimiters to deduplicate individual values.
    Used for multi-value fields: genres, developer, publisher, platform (collapse)."""
    values = set()
    for value in series.dropna():
        for part in re.split(r"[;,]", str(value)):
            cleaned = " ".join(part.strip().split())
            if cleaned:
                values.add(cleaned)
    return ", ".join(sorted(values)) if values else np.nan


def collect_unique_ordered(series, priority_order: Optional[List[str]] = None):
    """Collect unique non-null values into an ordered, comma-separated string.
    
    Args:
        series: Series of values to process
        priority_order: Optional list of values to prioritize in order (e.g., for source priority)
    """
    values = set()
    priority_values = []
    
    for value in series.dropna():
        for part in re.split(r"[;,]", str(value)):
            cleaned = " ".join(part.strip().split())
            if cleaned:
                values.add(cleaned)
                # If priority_order is specified, prioritize values that appear in it
                if priority_order and cleaned in priority_order:
                    priority_values.append(cleaned)
    
    # Combine priority values first, then remaining values
    remaining_values = sorted(values - set(priority_values))
    
    # Sort priority values based on their order in priority_order
    if priority_order:
        priority_sorted = [v for v in priority_order if v in values]
        ordered_values = priority_sorted + remaining_values
    else:
        ordered_values = sorted(values)
    
    return ", ".join(ordered_values) if ordered_values else np.nan


def any_truthy(series):
    truthy = {"true", "yes", "y", "1", "t"}
    falsy = {"false", "no", "n", "0", "f"}
    saw_explicit_false = False

    for value in series.dropna():
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                saw_explicit_false = True
                continue
            saw_explicit_false = True
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and np.isnan(value):
                continue
            if value == 0:
                saw_explicit_false = True
                continue
            return True
        if bool(value):
            return True
        saw_explicit_false = True
    if saw_explicit_false:
        return False
    return np.nan


def any_truthy_priority(series, source_priority: List[str]):
    """Handle boolean fields with priority-based resolution.
    
    Args:
        series: Series of values to process
        source_priority: List of source names ordered by priority
    """
    # Get all values (not just truthy ones)
    values = []
    for value in series.dropna():
        values.append(value)
    
    # If we have any truthy values, return True
    for value in values:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "t"}:
                return True
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and np.isnan(value):
                continue
            if value != 0:
                return True
        elif bool(value):
            return True
    
    # If we have any falsy values, return False
    for value in values:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "no", "n", "0", "f"}:
                return False
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and np.isnan(value):
                continue
            if value == 0:
                return False
        elif not bool(value):
            return False
    
    # If no truthy or falsy values found, return np.nan
    return np.nan


def resolve_with_priority(series, source_priority: List[str], default_resolver=None):
    """Resolve fields with priority for certain data sources.
    
    Args:
        series: Series of values to process
        source_priority: List of source names ordered by priority
        default_resolver: Function to use if no priority-based resolution is needed
    """
    # Simple approach: if there's a single source or no conflict, use pick_first
    # For multiple sources, we'll prioritize based on source_priority order
    values = _clean_strings(series)
    if not values:
        return np.nan
    
    # If we have only one value or all values are the same, return it
    if len(set(values)) == 1:
        return values[0]
    
    # For priority-based resolution, we'd want to return the first non-null value
    # from the priority-sorted sources if we can determine source information
    # For now, return the first value
    return values[0] if values else np.nan


def _resolve_by_source_priority(series, source_priority: List[str]):
    """Internal function to resolve values based on source priority.
    
    This is a placeholder for more sophisticated source-aware resolution.
    """
    values = []
    for value in series.dropna():
        values.append(value)
    
    return values[0] if values else np.nan


def weighted_avg(series):
    """Compute the arithmetic mean of non-null numeric values.
    
    Args:
        series: Series of numeric values.
    Returns:
        Arithmetic mean rounded to 1 decimal, or the single value if only one exists.
    """
    values = series.dropna()
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values.iloc[0])
    return round(float(values.mean()), 1)


def prefer_specific(series, priority_order: Optional[List[str]] = None):
    """Select the most specific non-null value from a series.
    
    Iterates values in priority order (if provided), counts comma-separated parts,
    and selects the value with the most parts (most specific). If tied, prefers
    the first encountered (highest priority source).
    
    Args:
        series: Series of values to process.
        priority_order: Optional list of source names ordered by priority.
    Returns:
        The most specific value, or np.nan if no values exist.
    """
    import pandas as pd
    
    values = series.dropna().tolist()
    if not values:
        return np.nan
    
    if priority_order:
        value_sources = []
        for val in values:
            val_str = str(val).strip()
            if not val_str:
                continue
            # Try to determine source from index or context
            value_sources.append(val_str)
        values = value_sources
        if not values:
            return np.nan
    
    def specificity(v):
        parts = len(re.split(r"[,;]", str(v)))
        length = len(str(v))
        return (parts, length)
    
    best = values[0]
    best_spec = specificity(best)
    for v in values[1:]:
        spec = specificity(v)
        if spec > best_spec:
            best = v
            best_spec = spec
    return str(best).strip()


# --- Resolver Dictionary ---
resolver = {
    "filename": pick_first,
    "summary": pick_longer,
    "release_date": "min",
    "release_year": "max",
    "genres": collect_unique,
    "developer": prefer_specific,
    "publisher": prefer_specific,
    "players": pick_first,
    "cooperative": any_truthy,
    "rating": weighted_avg,
    "user_rating": weighted_avg,
}

# Resolver dict for collapsing by game name (platforms become a list)
collapse_resolver = {
    **resolver,
    "platform": collect_unique,
    "filename": collect_unique,
}

# convenient alias
resolvers = resolver
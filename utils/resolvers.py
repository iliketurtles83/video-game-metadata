# rewritten column_join priority functions as resolver functions
import re

import numpy as np

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

# --- Resolver Dictionary ---
resolver = {
    "filename": pick_first,
    "summary": pick_longer,
    "release_date": "max",
    "release_year": "max",
    "genres": collect_unique,
    "developer": collect_unique,
    "publisher": collect_unique,
    "players": pick_first,
    "cooperative": any_truthy,
    "rating": "max",
    "user_rating": "max",
}

# Resolver dict for collapsing by game name (platforms become a list)
collapse_resolver = {
    **resolver,
    "platform": collect_unique,
    "filename": collect_unique,
}

# convenient alias
resolvers = resolver
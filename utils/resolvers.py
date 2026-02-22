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


def _flatten_genres(value):
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_genres(item)
    elif isinstance(value, str):
        for part in re.split(r"[;,]", value):
            cleaned = " ".join(part.strip().split())
            if cleaned:
                yield cleaned
    elif value is not None and not (isinstance(value, float) and np.isnan(value)):
        yield str(value)


# --- Resolver Functions ---
def pick_first(series):
    values = _clean_strings(series)
    return values[0] if values else np.nan

# pick the longest. used for description fields
def pick_longer(series):
    values = _clean_strings(series)
    return max(values, key=len) if values else np.nan

def merge_genres(series):
    """Merge genres from multiple sources: split on delimiters, combine, dedupe."""
    genres = set()
    for value in series.dropna():
        for genre in _flatten_genres(value):
            genres.add(genre)
    return ", ".join(sorted(genres)) if genres else np.nan


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
    "genres": merge_genres,
    "developer": pick_first,
    "publisher": pick_first,
    "players": "max",
    "cooperative": any_truthy,
    "rating": "max",
    "user_rating": "max",
}

# convenient alias
resolvers = resolver
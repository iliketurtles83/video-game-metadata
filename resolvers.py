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


def _canonical_genre_name(value):
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        return ""
    return cleaned.title()


# --- Resolver Functions ---
def pick_first(series):
    values = _clean_strings(series)
    return values[0] if values else np.nan

# pick the longest. used for description fields
def pick_longer(series):
    values = _clean_strings(series)
    return max(values, key=len) if values else np.nan

def merge_genres(series):
    canonical_genres = {}
    for value in series.dropna():
        for genre in _flatten_genres(value):
            canonical = _canonical_genre_name(genre)
            if not canonical:
                continue
            canonical_genres[canonical.casefold()] = canonical
    if not canonical_genres:
        return np.nan
    return ", ".join(sorted(canonical_genres.values()))


def any_truthy(series):
    truthy = {"true", "yes", "y", "1", "t"}
    falsy = {"false", "no", "n", "0", "f"}
    for value in series.dropna():
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                continue
            return True
        if isinstance(value, (int, float)):
            if value == 0 or (isinstance(value, float) and np.isnan(value)):
                continue
            return True
        if bool(value):
            return True
    return False

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
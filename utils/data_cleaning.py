"""
Data cleaning and enrichment functions.
Applied after merge pipeline to standardize, derive, and format data for output.
"""
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd





def parse_players(
    df: pd.DataFrame,
    players_str_col: str = "players",
) -> pd.DataFrame:
    """
    Parse player counts from strings like "1-2", "1-4", "2", "1-4 players", etc.
    Uses the maximum numeric token found; invalid/missing values return null.
    """
    out = df.copy()

    if players_str_col not in out.columns:
        return out

    def parse_players_value(players_str: Optional[str]) -> Optional[int]:
        if players_str is None or pd.isna(players_str):
            return None

        players_text = str(players_str).strip()
        numeric_tokens = re.findall(r"\d+", players_text)
        if not numeric_tokens:
            return None

        return max(int(token) for token in numeric_tokens)

    out[players_str_col] = out[players_str_col].apply(parse_players_value).astype("Int64")
    return out


def translate_genres(
    df: pd.DataFrame,
    genre_column: str = "genres",
    translation_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Translate genre terms using a provided mapping (e.g., Portuguese to English).
    Handles comma/semicolon/hyphen delimited lists and compound genre tokens.
    Runs before normalize_genres.
    """
    if translation_map is None or genre_column not in df.columns:
        return df.copy()

    out = df.copy()
    # Sort terms by length descending so longer compound terms match first
    sorted_terms = sorted(translation_map.keys(), key=len, reverse=True)

    def translate_genre_string(value):
        if pd.isna(value):
            return value

        res = str(value)
        for term in sorted_terms:
            en = translation_map[term]
            pattern = r'(?<![a-zA-Z0-9À-ÿ])' + re.escape(term) + r'(?![a-zA-Z0-9À-ÿ])'
            res = re.sub(pattern, en, res, flags=re.IGNORECASE)

        # Replace compound hyphens between genre words with comma
        res = res.replace('-', ', ')
        return res

    out[genre_column] = out[genre_column].apply(translate_genre_string)
    return out


def normalize_genres(
    df: pd.DataFrame,
    genre_column: str = "genres",
) -> pd.DataFrame:
    """
    Normalize genre strings: split on comma/semicolon/slash, apply title case, dedupe case-insensitively.
    """
    out = df.copy()

    def normalize_genre_string(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan

        canonical_genres = {}
        for part in re.split(r"[;,/]", str(value)):
            cleaned = " ".join(part.strip().split())
            if not cleaned:
                continue
            title_cased = cleaned.title()
            canonical_genres[title_cased.casefold()] = title_cased

        if not canonical_genres:
            return np.nan
        return ", ".join(sorted(canonical_genres.values()))

    if genre_column in out.columns:
        out[genre_column] = out[genre_column].apply(normalize_genre_string)

    return out


def clean_placeholder_dates(
    df: pd.DataFrame,
    date_column: str = "release_date",
    platform_column: str = "platform",
) -> pd.DataFrame:
    """
    Remove known sentinel/dummy release dates (e.g. 1980-01-01 or 1970-01-01 on modern platforms).
    """
    out = df.copy()
    if date_column not in out.columns:
        return out

    post_1980_platforms = {
        "Nintendo Switch", "Nintendo DS", "Nintendo 3DS", "Nintendo Wii",
        "Nintendo Wii U", "Sony PlayStation", "Sony PlayStation 2",
        "Sony PlayStation 3", "Sony PlayStation 4", "Sony PlayStation 5",
        "Sony PlayStation Vita", "Sony PSP", "Microsoft Xbox",
        "Microsoft Xbox 360", "Microsoft Xbox One", "Microsoft Xbox Series X/S",
        "Nintendo Game Boy Advance", "Nintendo 64", "Nintendo GameCube",
        "Sega Dreamcast", "Sega Saturn",
    }

    parsed = pd.to_datetime(out[date_column], errors="coerce")
    is_sentinel = parsed.isin([pd.Timestamp("1980-01-01"), pd.Timestamp("1970-01-01")])
    if platform_column in out.columns:
        mask = is_sentinel & out[platform_column].isin(post_1980_platforms)
    else:
        mask = is_sentinel

    # Return parsed datetimes with sentinels replaced by NaT
    out[date_column] = parsed.where(~mask, pd.NaT)
    return out


def clean_release_years(
    df: pd.DataFrame,
    year_column: str = "release_year",
    min_year: int = 1900,
    max_year: int = 2030,
) -> pd.DataFrame:
    """
    Clean invalid or malformed release years (e.g. truncated decade values like 19, 199, 200).
    """
    out = df.copy()
    if year_column not in out.columns:
        return out

    numeric_years = pd.to_numeric(out[year_column], errors="coerce")
    invalid_mask = numeric_years.notna() & ((numeric_years < min_year) | (numeric_years > max_year))
    out.loc[invalid_mask, year_column] = pd.NA
    return out


def derive_release_year(
    df: pd.DataFrame,
    date_column: str = "release_date",
    year_column: str = "release_year",
    reconcile_conflicts: bool = True,
) -> pd.DataFrame:
    """
    Fill missing release_year values from release_date where available,
    and optionally reconcile conflicts by aligning release_year to release_date.dt.year.
    """
    out = df.copy()

    if date_column not in out.columns or year_column not in out.columns:
        return out

    parsed_dates = pd.to_datetime(out[date_column], errors="coerce")
    has_date_mask = parsed_dates.notna()

    if reconcile_conflicts:
        out.loc[has_date_mask, year_column] = parsed_dates.loc[has_date_mask].dt.year.astype("Int64")
    else:
        missing_year_mask = out[year_column].isna()
        fill_mask = missing_year_mask & has_date_mask
        out.loc[fill_mask, year_column] = parsed_dates.loc[fill_mask].dt.year.astype("Int64")

    return out


def infer_cooperative_from_players(
    df: pd.DataFrame,
    players_col: str = "players",
    cooperative_col: str = "cooperative",
) -> pd.DataFrame:
    """
    Infer cooperative values from player count.
    Rule: if cooperative is missing and players == 1, set cooperative to False.
    """
    out = df.copy()

    if players_col not in out.columns or cooperative_col not in out.columns:
        return out

    missing_cooperative = out[cooperative_col].isna()
    single_player = pd.to_numeric(out[players_col], errors="coerce").eq(1)
    fill_mask = missing_cooperative & single_player

    out.loc[fill_mask, cooperative_col] = False
    return out


def normalize_release_date(
    df: pd.DataFrame,
    date_column: str = "release_date",
) -> pd.DataFrame:
    """
    Normalize release date values to pandas datetime (YYYY-MM-DD, no time component).
    Missing values remain as NaT.
    """
    out = df.copy()

    if date_column not in out.columns:
        return out

    parsed_dates = pd.to_datetime(out[date_column], errors="coerce")

    # Keep datetime64 dtype while dropping the time component
    out[date_column] = parsed_dates.dt.normalize()

    return out


def round_decimal_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    decimals: int = 1,
) -> pd.DataFrame:
    """
    Round numeric columns to a maximum number of decimal places.
    Used for presentation formatting (e.g., ratings: 8.5, not 8.523456).
    """
    out = df.copy()

    for column in columns:
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").round(decimals)

    return out


def normalize_rating_scales(
    df: pd.DataFrame,
    rating_col: str = "rating",
    user_rating_col: str = "user_rating",
    target_scale: float = 10.0,
) -> pd.DataFrame:
    """Normalize rating columns to a consistent 0-target_scale range.
    
    Heuristic:
    - Values > 10 and <= 100: assumed 0-100 scale, divide by (100/target_scale)
    - Values > 100: clip to target_scale (likely data errors)
    - Values between 0 and 1 (exclusive): assumed 0-1 scale, multiply by target_scale
    - Values between 0 and target_scale: keep as-is
    """
    out = df.copy()
    
    for col in [rating_col, user_rating_col]:
        if col not in out.columns:
            continue
        
        numeric = pd.to_numeric(out[col], errors="coerce").astype("float64")
        
        # 0-100 scale -> 0-10
        mask_100 = numeric > 10.0
        if mask_100.any():
            scaled = (numeric / (100.0 / target_scale)).astype("float64")
            numeric = numeric.where(~mask_100, scaled)
        
        # 0-1 scale -> 0-10
        mask_01 = (numeric > 0.0) & (numeric < 1.0)
        if mask_01.any():
            scaled = (numeric * target_scale).astype("float64")
            numeric = numeric.where(~mask_01, scaled)
        
        # Clip to valid range
        numeric = numeric.clip(lower=0.0, upper=float(target_scale))
        
        out[col] = numeric
    
    return out


def run_cleaning_pipeline(
    df: pd.DataFrame,
    normalize_genres_col: str = "genres",
    normalize_date_col: str = "release_date",
    parse_players_col: str = "players",
    cooperative_col: str = "cooperative",
    derive_year_col: str = "release_year",
    platform_col: str = "platform",
    round_columns: Sequence[str] = ("rating", "user_rating"),
    round_decimals: int = 1,
    genre_translation_map: Optional[Mapping[str, str]] = None,
    clean_placeholders: bool = True,
    clean_years: bool = True,
    reconcile_year_conflicts: bool = True,
    normalize_ratings: bool = True,
) -> pd.DataFrame:
    """
    Apply all cleaning steps in sequence.

    Args:
        df: Merged dataframe from run_merge_pipeline
        normalize_genres_col: Genre column name to normalize
        normalize_date_col: Date column name to normalize to YYYY-MM-DD
        parse_players_col: Players column name to parse to max integer players
        cooperative_col: Cooperative column name to infer from player count
        derive_year_col: Release year column to fill and align from release_date
        platform_col: Platform column name for platform-specific cleaning
        round_columns: Columns to round for display
        round_decimals: Number of decimal places to round to
        genre_translation_map: Optional dict to translate genre terms (e.g., Portuguese to English)
        clean_placeholders: Whether to clean dummy/sentinel dates (e.g. 1980-01-01 on modern platforms)
        clean_years: Whether to drop invalid truncated release years (< 1900 or > 2030)
        reconcile_year_conflicts: Whether to align release_year to release_date.dt.year where they conflict
        normalize_ratings: Whether to normalize rating scales to 0-10

    Returns:
        Cleaned dataframe ready for output
    """
    out = df.copy()
    out = translate_genres(out, genre_column=normalize_genres_col, translation_map=genre_translation_map)
    out = normalize_genres(out, genre_column=normalize_genres_col)
    out = normalize_release_date(out, date_column=normalize_date_col)
    if clean_placeholders:
        out = clean_placeholder_dates(out, date_column=normalize_date_col, platform_column=platform_col)
    if clean_years:
        out = clean_release_years(out, year_column=derive_year_col)
    out = parse_players(out, players_str_col=parse_players_col)
    out = infer_cooperative_from_players(out, players_col=parse_players_col, cooperative_col=cooperative_col)
    out = derive_release_year(
        out,
        date_column=normalize_date_col,
        year_column=derive_year_col,
        reconcile_conflicts=reconcile_year_conflicts,
    )
    
    if normalize_ratings:
        out = normalize_rating_scales(
            out, 
            rating_col="rating", 
            user_rating_col="user_rating"
        )
        
    out = round_decimal_columns(out, columns=round_columns, decimals=round_decimals)
    return out

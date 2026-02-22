"""
Data cleaning and enrichment functions.
Applied after merge pipeline to standardize, derive, and format data for output.
"""
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def translate_genres(
    df: pd.DataFrame,
    genre_column: str = "genres",
    translation_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Translate genre terms using a provided mapping (e.g., Portuguese to English).
    Handles comma/semicolon/hyphen delimited lists. Runs before normalize_genres.
    """
    if translation_map is None or genre_column not in df.columns:
        return df.copy()

    out = df.copy()

    def translate_genre_string(value):
        if pd.isna(value):
            return value

        # Split on delimiters
        parts = re.split(r'[;,\-]', str(value))
        translated_parts = []

        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            # Check if we have a translation, otherwise keep the original
            translated = translation_map.get(cleaned, cleaned)
            translated_parts.append(translated)

        return ', '.join(translated_parts) if translated_parts else value

    out[genre_column] = out[genre_column].apply(translate_genre_string)
    return out


def normalize_genres(
    df: pd.DataFrame,
    genre_column: str = "genres",
) -> pd.DataFrame:
    """
    Normalize genre strings: split on comma/semicolon, apply title case, dedupe case-insensitively.
    """
    out = df.copy()

    def normalize_genre_string(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan

        canonical_genres = {}
        for part in re.split(r"[;,]", str(value)):
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


def derive_release_year(
    df: pd.DataFrame,
    date_column: str = "release_date",
    year_column: str = "release_year",
) -> pd.DataFrame:
    """
    Fill missing release_year values from release_date where available.
    """
    out = df.copy()

    if date_column not in out.columns or year_column not in out.columns:
        return out

    missing_year_mask = out[year_column].isna()
    has_date_mask = out[date_column].notna()
    fill_mask = missing_year_mask & has_date_mask

    out.loc[fill_mask, year_column] = pd.to_datetime(
        out.loc[fill_mask, date_column]
    ).dt.year.astype("Int64")

    return out


def normalize_release_date(
    df: pd.DataFrame,
    date_column: str = "release_date",
) -> pd.DataFrame:
    """
    Normalize release date values to YYYY-MM-DD (drop time component).
    Missing values remain as NaN (not 'NaT' string).
    """
    out = df.copy()

    if date_column not in out.columns:
        return out

    parsed_dates = pd.to_datetime(out[date_column], errors="coerce")
    
    # Create a new series with formatted dates (strings) or NaN
    formatted_dates = parsed_dates.dt.strftime("%Y-%m-%d")
    formatted_dates = formatted_dates.where(parsed_dates.notna(), np.nan)
    
    out[date_column] = formatted_dates

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


def run_cleaning_pipeline(
    df: pd.DataFrame,
    normalize_genres_col: str = "genres",
    normalize_date_col: str = "release_date",
    derive_year_col: str = "release_year",
    round_columns: Sequence[str] = ("rating", "user_rating"),
    round_decimals: int = 2,
    genre_translation_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Apply all cleaning steps in sequence.

    Args:
        df: Merged dataframe from run_merge_pipeline
        normalize_genres_col: Genre column name to normalize
        normalize_date_col: Date column name to normalize to YYYY-MM-DD
        derive_year_col: Release year column to fill from release_date
        round_columns: Columns to round for display
        round_decimals: Number of decimal places to round to
        genre_translation_map: Optional dict to translate genre terms (e.g., Portuguese to English)

    Returns:
        Cleaned dataframe ready for output
    """
    out = df.copy()
    out = translate_genres(out, genre_column=normalize_genres_col, translation_map=genre_translation_map)
    out = normalize_genres(out, genre_column=normalize_genres_col)
    out = normalize_release_date(out, date_column=normalize_date_col)
    out = derive_release_year(out, year_column=derive_year_col)
    out = round_decimal_columns(out, columns=round_columns, decimals=round_decimals)
    return out

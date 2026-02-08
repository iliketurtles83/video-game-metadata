# rewritten column_join priority functions as resolver functions
import numpy as np
import pandas as pd

# --- Resolver Dictionary ---
resolver = {
    "developer": pick_first,
    "publisher": pick_first,
    "year": "max",          
    "genre": merge_genres,
    "multiplayer": "any",   
    "summary": pick_longer
}

# --- Resolver Functions ---
def pick_first(series):
    return series.dropna().iloc[0] if series.notna().any() else np.nan

# pick the longest. used for description fields
def pick_longer(series):
    if series.notna().any():
        return max(series.dropna(), key=len)
    return np.nan

def merge_genres(series):
    return ", ".join(sorted(set(series.dropna()))) if series.notna().any() else np.nan
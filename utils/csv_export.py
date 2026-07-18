"""CSV export utilities for the canonical schema."""
from pathlib import Path

import pandas as pd


def write_to_csv(
    df: pd.DataFrame,
    output_path: Path,
    schema: dict[str, str] | None = None,
) -> None:
    """Write a DataFrame to CSV with proper type coercion for canonical schema columns.

    Handles datetime, boolean, int, float, and string columns, converting
    NaN/NaT to empty strings.

    Args:
        df: DataFrame to write.
        output_path: Path to write the CSV file.
        schema: Canonical schema dict mapping column names to types.
                If None, uses basic coercion for known types.
    """
    if schema is None:
        schema = {}

    out = df.copy()

    for col, col_type in schema.items():
        if col not in out.columns:
            continue

        if col_type == 'datetime64[ns]':
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime('%Y-%m-%d')
            out[col] = out[col].fillna('')
        elif col_type == 'int64':
            out[col] = out[col].astype('Int64').astype(str).replace('nan', '')
        elif col_type == 'float64':
            out[col] = out[col].round(1).astype(str).replace('nan', '')
        elif col_type == 'boolean':
            out[col] = out[col].astype(str).replace('nan', '')
        elif col_type == 'string':
            out[col] = out[col].fillna('')

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

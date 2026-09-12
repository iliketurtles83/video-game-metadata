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
            out[col] = pd.to_numeric(out[col], errors='coerce').round().astype('Int64').astype(str).replace('<NA>', '').replace('nan', '')
        elif col_type == 'float64':
            out[col] = pd.to_numeric(out[col], errors='coerce').round(1).astype('Float64').astype(str).replace('<NA>', '').replace('nan', '')
        elif col_type == 'boolean':
            out[col] = out[col].astype(str).replace('<NA>', '').replace('nan', '')
        elif col_type == 'string':
            # Handle nullable integer columns (e.g., players after parse_players)
            if pd.api.types.is_integer_dtype(out[col]) or str(out[col].dtype).startswith('Int'):
                out[col] = pd.to_numeric(out[col], errors='coerce').round().astype('Int64').astype(str).replace('<NA>', '').replace('nan', '')
            else:
                out[col] = out[col].fillna('')

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)


def write_to_parquet(
    df: pd.DataFrame,
    output_path: Path | str,
    schema: dict[str, str] | None = None,
    compression: str = "snappy",
) -> None:
    """Write a DataFrame to Parquet format with proper schema types.

    Args:
        df: DataFrame to write.
        output_path: Path to write the Parquet file.
        schema: Canonical schema dict mapping column names to types.
        compression: Compression codec (default: 'snappy').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    if schema:
        for col, col_type in schema.items():
            if col not in out.columns:
                continue
            if col_type == "datetime64[ns]":
                out[col] = pd.to_datetime(out[col], errors="coerce")
            elif col_type == "int64":
                out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
            elif col_type == "float64":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Float64")
            elif col_type == "boolean":
                out[col] = out[col].astype("boolean")
            elif col_type == "string":
                out[col] = out[col].astype("string")

    out.to_parquet(output_path, index=False, compression=compression, engine="pyarrow")


def write_to_sqlite(
    df: pd.DataFrame,
    output_path: Path | str,
    table_name: str = "games",
    if_exists: str = "replace",
    create_indices: bool = True,
    schema: dict[str, str] | None = None,
) -> None:
    """Write DataFrame to SQLite database with index creation.

    Args:
        df: DataFrame to write.
        output_path: Path to write the SQLite database.
        table_name: Name of the table (default: 'games').
        if_exists: What to do if table exists ('replace', 'append', 'fail').
        create_indices: Whether to create indices on name, platform, release_year.
        schema: Optional schema dictionary for datetime formatting.
    """
    import sqlite3

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    if schema:
        for col, col_type in schema.items():
            if col not in out.columns:
                continue
            if col_type == "datetime64[ns]" and pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")

    with sqlite3.connect(output_path) as conn:
        out.to_sql(table_name, conn, if_exists=if_exists, index=False)
        if create_indices:
            cursor = conn.cursor()
            cols = out.columns.tolist()
            if "name" in cols:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name)")
            if "platform" in cols:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_platform ON {table_name}(platform)")
            if "name" in cols and "platform" in cols:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name_platform ON {table_name}(name, platform)")
            if "release_year" in cols:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_year ON {table_name}(release_year)")
            conn.commit()


def export_dataset(
    df: pd.DataFrame,
    base_path: Path | str,
    schema: dict[str, str] | None = None,
    formats: tuple[str, ...] = ("csv", "parquet", "sqlite"),
    table_name: str = "games",
) -> dict[str, Path]:
    """Export a DataFrame across multiple formats simultaneously.

    Args:
        df: DataFrame to export.
        base_path: Base path without extension (e.g. 'output/cleaned_games').
        schema: Canonical schema dict.
        formats: Tuple of formats to export ('csv', 'parquet', 'sqlite').
        table_name: Table name for sqlite export.

    Returns:
        Dict mapping format name to exported file Path.
    """
    base_path = Path(base_path)
    # Strip existing extension if passed
    if base_path.suffix in (".csv", ".parquet", ".db", ".sqlite", ".sqlite3"):
        base_path = base_path.with_suffix("")

    exported: dict[str, Path] = {}
    if "csv" in formats:
        csv_path = base_path.with_suffix(".csv")
        write_to_csv(df, csv_path, schema=schema)
        exported["csv"] = csv_path

    if "parquet" in formats:
        parquet_path = base_path.with_suffix(".parquet")
        write_to_parquet(df, parquet_path, schema=schema)
        exported["parquet"] = parquet_path

    if "sqlite" in formats:
        sqlite_path = base_path.with_suffix(".db")
        write_to_sqlite(df, sqlite_path, table_name=table_name, schema=schema)
        exported["sqlite"] = sqlite_path

    return exported

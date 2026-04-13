"""Unified gamelist.xml builder / updater.

Builds or updates gamelist.xml files by combining two possible inventory sources:
  1. Existing gamelist.xml  — preserves all original XML tags.
  2. ROM files on disk      — discovers new games not yet in the XML.

Source mode controls which sources are used:
  "both"       — merge XML inventory with file scan (default).
  "xml_only"   — use only the existing gamelist.xml entries.
  "files_only" — use only the scanned ROM files.

When both sources are active, XML entries take priority (preserving tags); files
found on disk that are not already in the XML are added as new entries.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
import xml.etree.ElementTree as ET

import pandas as pd

from utils.game_parser import (
    load_platform_mappings as load_folder_mappings,
    scan_platform_folder,
)
from utils.update_gamelist import (
    DEFAULT_MATCH_THRESHOLD,
    PATH_PREFIX,
    RESERVED_XML_TAGS,
    build_inventory_from_gamelist,
    generate_audit_report,
    is_missing,
    match_inventory_to_metadata,
    normalize_game_name,
    print_audit_report,
    reverse_gamelist_mapping,
    update_or_create_element,
)

SourceMode = Literal["both", "xml_only", "files_only"]

DEFAULT_GAMELIST_FILENAME = "gamelist.xml"
DEFAULT_OUTPUT_FILENAME = "gamelist_updated.xml"


# ------------------------------------------------------------------
# Metadata loader
# ------------------------------------------------------------------

def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load a video game metadata DataFrame from a pickle file.

    Args:
        metadata_path: Path to a ``.pkl`` file produced by the data-cleaning
                       pipeline (e.g. ``output/cleaned_df.pkl``).

    Returns:
        pandas DataFrame with game metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a DataFrame.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    data = pd.read_pickle(path)
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(data).__name__}")
    return data


def _resolve_game_df(
    game_df: Optional[pd.DataFrame],
    metadata_path: Optional[str | Path],
) -> Optional[pd.DataFrame]:
    """Return a metadata DataFrame from whichever source is provided.

    * If *game_df* is already set, return it as-is.
    * Otherwise load from *metadata_path* if given.
    * If neither is provided, return ``None`` (no enrichment).
    """
    if game_df is not None:
        return game_df
    if metadata_path is not None:
        return load_metadata(metadata_path)
    return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_xml_path(value: Any) -> Optional[str]:
    """Ensure a value has the standard './' prefix for gamelist paths."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith(PATH_PREFIX) else f"{PATH_PREFIX}{text}"


def _inventory_from_scanned_files(
    scanned_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Convert a game_parser scan DataFrame into the inventory dict format
    expected by ``match_inventory_to_metadata``."""
    if scanned_df is None or scanned_df.empty:
        return []

    inventory: List[Dict[str, Any]] = []
    for _, row in scanned_df.iterrows():
        raw_name = row.get("name")
        if is_missing(raw_name):
            continue
        xml_name = str(raw_name).strip()
        if not xml_name:
            continue

        inventory.append({
            "xml_name": xml_name,
            "path": _to_xml_path(row.get("filename")),
            "name_normalized": normalize_game_name(xml_name),
            "game_elem": None,
        })
    return inventory


def _merge_inventories(
    xml_inventory: List[Dict[str, Any]],
    file_inventory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge XML-sourced and file-sourced inventories.

    XML entries always take priority (they carry ``game_elem`` for tag
    preservation).  File entries whose normalized path already appears in the
    XML inventory are skipped; the rest are appended as new games.
    """
    # Build lookup of paths already covered by XML entries
    xml_paths: set[str] = set()
    for item in xml_inventory:
        path = item.get("path")
        if path:
            xml_paths.add(path.lower())

    merged = list(xml_inventory)
    for item in file_inventory:
        path = item.get("path")
        if path and path.lower() in xml_paths:
            continue
        merged.append(item)

    return merged


# ------------------------------------------------------------------
# XML writer
# ------------------------------------------------------------------

def _write_gamelist_xml(
    matched_games: List[Dict[str, Any]],
    output_path: Path,
    field_mapping: Dict[str, str],
) -> None:
    """Write matched game entries to a gamelist.xml file.

    If a game carries a ``game_elem`` from an existing XML parse, all original
    child elements are preserved and only mapped fields are updated.  Games
    without ``game_elem`` (file-scan-only) get new ``<game>`` elements.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("gameList")

    for item in matched_games:
        original_elem = item.get("game_elem")
        if original_elem is not None:
            game_elem = deepcopy(original_elem)
        else:
            game_elem = ET.Element("game")

        root.append(game_elem)

        # Always set name and path
        update_or_create_element(game_elem, "name", item["xml_name"])
        if item.get("path"):
            update_or_create_element(game_elem, "path", item["path"])

        # Merge metadata fields (skip reserved tags already handled above)
        metadata = item.get("metadata")
        if metadata is None:
            continue
        for df_column, xml_tag in field_mapping.items():
            if xml_tag in RESERVED_XML_TAGS:
                continue
            if df_column not in metadata.index:
                continue
            value = metadata[df_column]
            if not is_missing(value):
                update_or_create_element(game_elem, xml_tag, value)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------------
# Single-platform pipeline
# ------------------------------------------------------------------

def build_platform_gamelist(
    platform_folder: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    game_df: Optional[pd.DataFrame] = None,
    metadata_path: Optional[str | Path] = None,
    source: SourceMode = "both",
    strict_extensions: bool = True,
    threshold: int = DEFAULT_MATCH_THRESHOLD,
    field_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build or update a gamelist.xml for a single platform folder.

    Args:
        platform_folder: Path to the platform's game folder (e.g. ``~/roms/nes``).
        output_path:     Where to write the result XML.  Defaults to
                         ``<platform_folder>/gamelist_updated.xml``.
        game_df:         Optional metadata DataFrame for enrichment.
        metadata_path:   Path to a metadata ``.pkl`` file.  Used when
                         *game_df* is not provided.
        source:          ``"both"`` | ``"xml_only"`` | ``"files_only"``.
        strict_extensions: When scanning files, only keep known ROM extensions.
        threshold:       Fuzzy-match confidence threshold (0-100).
        field_mapping:   Column→XML-tag mapping override.
        verbose:         Print progress information.

    Returns:
        Dict with keys: success, output_path, platform_key, platform_name,
        inventory_count, matched_count, match_rate, audit.
    """
    folder = Path(platform_folder)
    if not folder.is_dir():
        return {"success": False, "error": f"Not a directory: {folder}"}

    game_df = _resolve_game_df(game_df, metadata_path)

    platform_key = folder.name.strip().lower()
    folder_mappings = load_folder_mappings()
    platform_name = folder_mappings.get(platform_key, folder.name)

    if output_path is None:
        output_path = folder / DEFAULT_OUTPUT_FILENAME
    output_path = Path(output_path)

    if field_mapping is None:
        field_mapping = reverse_gamelist_mapping

    if verbose:
        print(f"\n{'='*60}")
        print(f"PLATFORM: {platform_key} -> {platform_name}")
        print(f"  source={source}  strict_extensions={strict_extensions}")
        print(f"{'='*60}")

    # --- 1. Collect inventory from selected sources -----------------
    xml_path = folder / DEFAULT_GAMELIST_FILENAME
    xml_inventory: List[Dict[str, Any]] = []
    file_inventory: List[Dict[str, Any]] = []

    if source in ("both", "xml_only") and xml_path.exists():
        xml_inventory = build_inventory_from_gamelist(xml_path)
        if verbose:
            print(f"  XML inventory: {len(xml_inventory)} entries")

    if source in ("both", "files_only"):
        scanned_df = scan_platform_folder(
            platform_folder=folder,
            strict_extensions=strict_extensions,
        )
        file_inventory = _inventory_from_scanned_files(scanned_df)
        if verbose:
            print(f"  File inventory: {len(file_inventory)} entries")

    # --- 2. Merge ---------------------------------------------------
    if source == "xml_only":
        inventory = xml_inventory
    elif source == "files_only":
        inventory = file_inventory
    else:
        inventory = _merge_inventories(xml_inventory, file_inventory)

    if verbose:
        print(f"  Combined inventory: {len(inventory)} entries")

    if not inventory:
        _write_gamelist_xml([], output_path, field_mapping)
        return {
            "success": True,
            "output_path": str(output_path),
            "platform_key": platform_key,
            "platform_name": platform_name,
            "inventory_count": 0,
            "matched_count": 0,
            "match_rate": 0.0,
            "audit": generate_audit_report([]),
        }

    # --- 3. Match inventory to metadata -----------------------------
    if game_df is not None and not game_df.empty:
        matched_games = match_inventory_to_metadata(
            inventory=inventory,
            game_df=game_df,
            platform_name=platform_name,
            threshold=threshold,
        )
    else:
        matched_games = [
            {
                "xml_name": item["xml_name"],
                "path": item["path"],
                "game_elem": item["game_elem"],
                "match_name": None,
                "match_confidence": 0,
                "metadata": None,
                "match_type": None,
            }
            for item in inventory
        ]

    # --- 4. Write XML -----------------------------------------------
    _write_gamelist_xml(matched_games, output_path, field_mapping)

    audit = generate_audit_report(matched_games, threshold=threshold)
    matched_count = sum(1 for g in matched_games if g.get("match_name") is not None)

    if verbose:
        print(f"  Written: {output_path}")
        print_audit_report(audit, platform_name)

    return {
        "success": True,
        "output_path": str(output_path),
        "platform_key": platform_key,
        "platform_name": platform_name,
        "inventory_count": len(inventory),
        "matched_count": matched_count,
        "match_rate": audit["match_rate"],
        "audit": audit,
    }


# ------------------------------------------------------------------
# Multi-platform runner
# ------------------------------------------------------------------

def build_all_gamelists(
    games_root: str | Path,
    *,
    output_root: Optional[str | Path] = None,
    platform_keys: Optional[Sequence[str]] = None,
    game_df: Optional[pd.DataFrame] = None,
    metadata_path: Optional[str | Path] = None,
    source: SourceMode = "both",
    strict_extensions: bool = True,
    threshold: int = DEFAULT_MATCH_THRESHOLD,
    field_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Build / update gamelists for multiple platform folders.

    Args:
        games_root:      Root folder containing platform subfolders.
        output_root:     Root folder for output XMLs.  If None, each XML is
                         written inside its source platform folder.
        platform_keys:   Explicit list of subfolder names to process.
                         If None, every direct child directory is processed.
        game_df:         Optional metadata DataFrame for enrichment.
        metadata_path:   Path to a metadata ``.pkl`` file.  Used when
                         *game_df* is not provided.
        source:          ``"both"`` | ``"xml_only"`` | ``"files_only"``.
        strict_extensions: When scanning files, only keep known ROM extensions.
        threshold:       Fuzzy-match confidence threshold (0-100).
        field_mapping:   Column→XML-tag mapping override.
        verbose:         Print per-platform progress.

    Returns:
        List of result dicts (one per platform), same shape as
        ``build_platform_gamelist`` output.
    """
    root = Path(games_root)
    if not root.is_dir():
        print(f"Error: games_root is not a directory: {root}")
        return []

    game_df = _resolve_game_df(game_df, metadata_path)

    if platform_keys is not None:
        folders = [root / key for key in platform_keys]
    else:
        folders = sorted(p for p in root.iterdir() if p.is_dir())

    results: List[Dict[str, Any]] = []
    for folder in folders:
        if not folder.is_dir():
            if verbose:
                print(f"  Skipping {folder.name}: not a directory")
            continue

        out_path: Optional[Path] = None
        if output_root is not None:
            out_path = Path(output_root) / folder.name / DEFAULT_OUTPUT_FILENAME

        result = build_platform_gamelist(
            platform_folder=folder,
            output_path=out_path,
            game_df=game_df,
            source=source,
            strict_extensions=strict_extensions,
            threshold=threshold,
            field_mapping=field_mapping,
            verbose=verbose,
        )
        results.append(result)

    if verbose:
        total = len(results)
        ok = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"SUMMARY: {ok}/{total} platforms processed successfully")
        print(f"{'='*60}")

    return results

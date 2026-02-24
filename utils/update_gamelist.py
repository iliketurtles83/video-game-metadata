"""Gamelist XML update pipeline with metadata matching and field preservation.

This module provides a robust pipeline for updating RetroPie/EmulationStation gamelist.xml files
by matching existing inventory to external metadata sources and enriching them with additional fields
while preserving all original unmapped elements.
"""

from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import traceback
import json
import re
from rapidfuzz import fuzz

from utils.gamelist_parser import GAMELIST_COLUMN_MAP


# ============================================================
# Configuration Constants
# ============================================================

PLATFORM_MAPPINGS_PATH = 'utils/platform_mappings.json'
"""Path to JSON file mapping platform directory names to canonical names."""

GAME_FILE_EXTENSIONS = r'\.(zip|nes|sfc|smc|bin|iso|cue|gg|sms)$'
"""Regex pattern for common game file extensions to remove during normalization."""

DEFAULT_MATCH_THRESHOLD = 80
"""Default minimum confidence threshold (0-100) for considering a match valid."""

LOW_CONFIDENCE_THRESHOLD = 90
"""Threshold (0-100) for flagging matches as low confidence in audit reports."""

PATH_PREFIX = './'
"""Standard prefix for relative file paths in gamelist XML."""

RESERVED_XML_TAGS = {'name', 'path'}
"""XML tags that are handled specially and should not be overwritten via mapping."""

# ============================================================
# Module-level initialization
# ============================================================

reverse_gamelist_mapping: Dict[str, str] = {v: k for k, v in GAMELIST_COLUMN_MAP.items()}
"""Mapping from game_df column names to XML tag names."""

platform_mappings: Dict[str, str] = {}
"""Platform directory-to-canonical-name mappings, loaded from JSON."""

try:
    with open(PLATFORM_MAPPINGS_PATH, 'r') as f:
        platform_mappings = json.load(f)
except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
    print(f"Warning: Failed to load platform mappings from {PLATFORM_MAPPINGS_PATH}: {e}")
    platform_mappings = {}


# ============================================================
# Utility Functions
# ============================================================


def is_missing(value: Any) -> bool:
    """Check if a value is missing (None, NaN, or NaT).

    Args:
        value: Value to check.

    Returns:
        True if value is missing, False otherwise.
    """
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def update_or_create_element(parent_elem: ET.Element, element_name: str, text: Any) -> None:
    """Update existing XML element or create new one with given text.
    
    Args:
        parent_elem: Parent XML element to search/modify.
        element_name: Name of the child element to update or create.
        text: Text content to set on the element.
    """
    elem = parent_elem.find(element_name)
    if elem is None:
        elem = ET.SubElement(parent_elem, element_name)
    elem.text = str(text)


def update_gamelist_xml(
    xml_path: Union[str, Path],
    updates_by_name: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Update a gamelist.xml file with metadata keyed by <game><name>.

    Args:
        xml_path: Path to input gamelist.xml file.
        updates_by_name: Dict mapping game names to update field dicts.
                         Example: {'Game Name': {'desc': 'A description', 'rated': '8'}}
        output_path: Path to write updated gamelist.xml. If None, uses gamelist_test_updated.xml.

    Returns:
        Dict with 'input', 'output', and 'updated_games' counts.
        
    Raises:
        ValueError: If root tag is not 'gameList'.
    """
    xml_path = Path(xml_path)
    if output_path is None:
        output_path = xml_path.with_name("gamelist_test_updated.xml")
    else:
        output_path = Path(output_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "gameList":
        raise ValueError(f"Unexpected root tag: {root.tag}. Expected 'gameList'.")

    updates_casefold = {name.casefold(): values for name, values in updates_by_name.items()}

    updated_games = 0
    for game_elem in root.findall("game"):
        name_elem = game_elem.find("name")
        if name_elem is None or not name_elem.text:
            continue

        game_name = name_elem.text.strip()
        game_updates = updates_casefold.get(game_name.casefold())
        if not game_updates:
            continue

        for xml_tag, value in game_updates.items():
            if is_missing(value):
                continue
            update_or_create_element(game_elem, xml_tag, value)

        updated_games += 1

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "input": str(xml_path),
        "output": str(output_path),
        "updated_games": updated_games,
    }


# ============================================================
# PIPELINE A: Build Inventory from existing gamelist.xml
# ============================================================

def build_inventory_from_gamelist(xml_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract owned games from an existing gamelist.xml, including original XML element.
    
    Args:
        xml_path: Path to gamelist.xml file.
    
    Returns:
        List of game dicts with keys: 'path', 'xml_name', 'name_normalized', 'game_elem'.
        Returns empty list if file not found or parsing fails.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML at {xml_path}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error reading {xml_path}: {e}")
        return []
    
    inventory = []
    for game_elem in root.findall("game"):
        name_elem = game_elem.find("name")
        path_elem = game_elem.find("path")
        
        if name_elem is not None and name_elem.text and name_elem.text.strip():
            path_value = None
            if path_elem is not None and path_elem.text and path_elem.text.strip():
                path_value = path_elem.text.strip()

            name_text = name_elem.text.strip()
            inventory.append({
                "xml_name": name_text,
                "path": path_value,
                "name_normalized": normalize_game_name(name_text),
                "game_elem": game_elem,
            })
    
    return inventory


def normalize_game_name(name: str) -> str:
    """Normalize game name for matching: lowercase, remove suffixes, parentheticals, etc.
    
    Args:
        name: Raw game name to normalize.
        
    Returns:
        Normalized game name (lowercase, without extensions or region tags).
    """
    # Remove file extensions
    name = re.sub(GAME_FILE_EXTENSIONS, '', name, flags=re.IGNORECASE)
    
    # Remove region/version tags in brackets/parentheses
    name = re.sub(r'\s*[\[\(][^\]\)]*[\]\)]', '', name)
    
    # Remove "The" prefix
    name = re.sub(r'^the\s+', '', name, flags=re.IGNORECASE)
    
    # Strip and lowercase
    return name.strip().lower()


def extract_filename_from_path(path: Optional[str]) -> str:
    """Extract filename base from path (strip ./ prefix and extension).
    
    Args:
        path: File path (e.g., './game.nes' or 'game.nes').
        
    Returns:
        Filename without extension or prefix (e.g., 'game').
    """
    if path is None:
        return ""

    path = str(path)
    # Strip ./ prefix
    path = path.lstrip('./')
    # Remove extension
    if '.' in path:
        path = path.rsplit('.', 1)[0]
    return path.strip()


def build_path_from_filename(filename: Optional[str]) -> Optional[str]:
    """Build XML path from filename by ensuring a leading ./ prefix.
    
    Args:
        filename: Filename to convert (e.g., 'game.nes').
        
    Returns:
        XML path with ./ prefix (e.g., './game.nes'), or None if filename is empty.
    """
    if not filename:
        return None

    filename = str(filename).strip().lstrip('./')
    return f"{PATH_PREFIX}{filename}" if filename else None


# ============================================================
# PIPELINE B: Match inventory to game_df metadata
# ============================================================

def _build_unmatched_result(inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create list of unmatched game results from inventory.
    
    Args:
        inventory: List of inventory games.
        
    Returns:
        List of match results with all None/0 values (no matches found).
    """
    return [{
        'xml_name': item['xml_name'],
        'path': item['path'],
        'game_elem': item['game_elem'],
        'match_name': None,
        'match_confidence': 0,
        'metadata': None,
        'match_type': None,
    } for item in inventory]


def match_inventory_to_metadata(
    inventory: List[Dict[str, Any]],
    game_df: Optional[pd.DataFrame],
    platform_name: str,
    threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Match inventory games to game_df metadata using multi-strategy approach:
    1. Primary: Exact or fuzzy match on filename (if available in game_df)
    2. Fallback: Fuzzy match on normalized game name
    
    Returns: list of dicts with keys:
      - 'xml_name', 'path', 'game_elem', 'match_name', 'match_confidence', 'metadata' (Series or None),
        'match_type' ('filename_exact', 'filename_fuzzy', or 'name')
    """
    if not isinstance(inventory, list) or len(inventory) == 0:
        return []
    
    if game_df is None or game_df.empty:
        print(f"Warning: game_df is empty or None for platform '{platform_name}'")
        return _build_unmatched_result(inventory)
    
    # Filter game_df to this platform
    platform_games = game_df[game_df['platform'] == platform_name].copy()
    
    if len(platform_games) == 0:
        print(f"Warning: No games found in game_df for platform '{platform_name}'")
        return _build_unmatched_result(inventory)
    
    # Prepare matching data
    try:
        platform_games['name_normalized'] = platform_games['name'].apply(normalize_game_name)
    except Exception as e:
        print(f"Error: Failed to normalize game names for platform '{platform_name}': {e}")
        return []

    
    # Create lookup for filenames (only for non-null filenames)
    filename_lookup = {}
    for idx, row in platform_games.iterrows():
        if pd.notna(row['filename']):
            filename_key = row['filename'].lower().strip()
            if filename_key not in filename_lookup:
                filename_lookup[filename_key] = row
    
    matched = []
    for item in inventory:
        best_match = None
        best_score = 0
        best_row = None
        match_type = None
        
        # Strategy 1: Try exact/fuzzy match on filename
        path_value = item.get('path')
        path_filename = extract_filename_from_path(path_value).lower() if path_value else None
        
        # First try exact match on filename
        if path_filename and path_filename in filename_lookup:
            best_row = filename_lookup[path_filename]
            best_match = best_row['name']
            best_score = 100
            match_type = 'filename_exact'
        elif path_filename:
            # Try fuzzy match on filenames
            for fname_key, row in filename_lookup.items():
                score = fuzz.ratio(path_filename, fname_key)
                if score > best_score:
                    best_score = score
                    best_match = row['name']
                    best_row = row
                    match_type = 'filename_fuzzy'
        
        # Strategy 2: Fallback to fuzzy match on normalized names (only if filename match is weak)
        if best_score < threshold:
            for idx, row in platform_games.iterrows():
                score = fuzz.ratio(item['name_normalized'], row['name_normalized'])
                if score > best_score:
                    best_score = score
                    best_match = row['name']
                    best_row = row
                    match_type = 'name'
        
        matched.append({
            'xml_name': item['xml_name'],
            'path': item['path'],
            'game_elem': item['game_elem'],
            'match_name': best_match if best_score >= threshold else None,
            'match_confidence': best_score,
            'metadata': best_row if best_score >= threshold else None,
            'match_type': match_type,
        })
    
    return matched


# ============================================================
# PIPELINE C: Generate/Update gamelist.xml
# ============================================================

def generate_gamelist_xml(
    matched_games: List[Dict[str, Any]],
    output_path: Union[str, Path],
    field_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Update gamelist.xml from matched games, preserving unmapped fields.
    
    Uses original <game> elements from XML and updates/adds metadata fields.
    Preserves unmapped fields like <image>, <marquee>, <playcount>, <lastplayed>.
    
    Args:
        matched_games: List of matched game dicts from match_inventory_to_metadata()
                      Each should contain 'game_elem' (original XML element)
        output_path: Path to write the updated gamelist.xml
        field_mapping: Optional dict mapping game_df columns → XML tag names.
                      Defaults to reverse_gamelist_mapping (game_df columns → XML tags)
                      e.g., {'summary': 'desc', 'release_date': 'releasedate', ...}
    
    Raises:
        ValueError: If output_path parent directory cannot be created
    """
    
    output_path = Path(output_path)
    
    # Use default mapping if not provided (avoid mutable default argument issue)
    if field_mapping is None:
        field_mapping = reverse_gamelist_mapping
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    root = ET.Element("gameList")
    
    for item in matched_games:
        # Use original game element as base (preserves unmapped fields)
        original_game_elem = item.get('game_elem')
        if original_game_elem is None:
            continue
        
        # Deep copy original element to preserve all children and attributes
        game_elem = deepcopy(original_game_elem)
        root.append(game_elem)
        metadata = item.get('metadata')

        # Update path: keep existing if present, otherwise derive from metadata
        existing_path = item.get('path')
        if isinstance(existing_path, str):
            existing_path = existing_path.strip()
        
        if existing_path:
            final_path = existing_path
        else:
            filename_value = None
            if metadata is not None and 'filename' in metadata.index:
                candidate = metadata['filename']
                if not is_missing(candidate):
                    filename_value = candidate
            final_path = build_path_from_filename(filename_value) if filename_value is not None else None

        if final_path:
            update_or_create_element(game_elem, "path", final_path)
        
        # Update name (always use XML name)
        update_or_create_element(game_elem, "name", item['xml_name'])
        
        # Update metadata fields if match exists (skip name and path as handled above)
        if metadata is not None:
            for df_column, xml_tag in field_mapping.items():
                if xml_tag in RESERVED_XML_TAGS:
                    continue
                if df_column in metadata.index:
                    value = metadata[df_column]
                    if not is_missing(value):
                        update_or_create_element(game_elem, xml_tag, value)
    
    # Write with pretty formatting
    try:
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
    except IOError as e:
        print(f"Error: Failed to write gamelist to {output_path}: {e}")
        raise
    except Exception as e:
        print(f"Error: Unexpected error writing gamelist: {e}")
        raise
    
    return str(output_path)


# ============================================================
# PIPELINE D: Audit Report
# ============================================================

def generate_audit_report(
    matched_games: List[Dict[str, Any]],
    threshold: int = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Generate audit statistics for matched games.

    Args:
        matched_games: List of matched game dicts from match_inventory_to_metadata().
        threshold: Confidence level (0-100) below which matches flagged as low confidence.

    Returns:
        Dict with audit statistics: total_games, matched, unmatched, low_confidence_count,
        low_confidence_games, no_match_games, match_rate.
    """
    total = len(matched_games)
    matched = sum(1 for g in matched_games if g['match_name'] is not None)
    unmatched = total - matched
    
    low_confidence = [
        g
        for g in matched_games
        if g['match_name'] is not None and g['match_confidence'] < threshold
    ]
    
    no_match = [g for g in matched_games if g['match_name'] is None]
    
    return {
        'total_games': total,
        'matched': matched,
        'unmatched': unmatched,
        'low_confidence_count': len(low_confidence),
        'low_confidence_games': low_confidence,
        'no_match_games': no_match,
        'match_rate': matched / total if total > 0 else 0,
    }


def print_audit_report(report: Dict[str, Any], platform_name: str) -> None:
    """Pretty print audit report.

    Args:
        report: Audit report dict from generate_audit_report().
        platform_name: Platform name for report header.
    """
    print(f"\n{'='*60}")
    print(f"AUDIT REPORT: {platform_name}")
    print(f"{'='*60}")
    print(f"Total games in gamelist.xml: {report['total_games']}")
    print(f"Matched to metadata: {report['matched']} ({report['match_rate']*100:.1f}%)")
    print(f"Unmatched: {report['unmatched']}")
    print(f"Low confidence matches (<{LOW_CONFIDENCE_THRESHOLD}): {report['low_confidence_count']}")
    
    # Count matches by type
    filename_exact_count = sum(1 for g in report.get('low_confidence_games', []) if g.get('match_type') == 'filename_exact')
    filename_fuzzy_count = sum(1 for g in report.get('low_confidence_games', []) if g.get('match_type') == 'filename_fuzzy')
    name_match_count = sum(1 for g in report.get('low_confidence_games', []) if g.get('match_type') == 'name')
    
    if report['low_confidence_games']:
        print(f"\n--- Low Confidence Matches (sample) ---")
        for item in report['low_confidence_games'][:5]:
            match_type = item.get('match_type', 'unknown')
            print(f"  [{match_type:15}] {item['xml_name'][:40]:40} -> {item['match_name'][:35]:35} ({item['match_confidence']:.0f}%)")
    
    if report['no_match_games']:
        print(f"\n--- No Match Found (sample) ---")
        for item in report['no_match_games'][:5]:
            print(f"  {item['xml_name']}")
    
    print(f"{'='*60}\n")


# ============================================================
# HIGH-LEVEL PIPELINE: End-to-end update for one platform
# ============================================================

def update_platform_gamelist(
    platform_abbrev: str,
    game_df: pd.DataFrame,
    platform_mappings_dict: Optional[Dict[str, str]] = None,
    lists_base_path: str = "lists",
    threshold: int = DEFAULT_MATCH_THRESHOLD,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Complete pipeline to update gamelist.xml for a single platform.
    
    Args:
        platform_abbrev: Platform directory abbreviation (e.g., 'nes').
        game_df: DataFrame with game metadata.
        platform_mappings_dict: Dict mapping directory names to canonical platform names.
                               If None, uses module-level platform_mappings.
        lists_base_path: Base path to lists directory (default: 'lists').
        threshold: Minimum match confidence threshold (default: 80).
        verbose: Print progress and audit report (default: True).
    
    Returns:
        Dict with pipeline results:
            - 'platform_abbrev': Input platform abbreviation.
            - 'platform_name': Canonical platform name (if success=True).
            - 'success': True if processing completed successfully.
            - 'xml_input': Path to input gamelist.xml (if success=True).
            - 'xml_output': Path to updated gamelist.xml (if success=True).
            - 'inventory_count': Number of games in inventory (if success=True).
            - 'matched_count': Number of matched games (if success=True).
            - 'match_rate': Match success rate 0-1 (if success=True).
            - 'error': Error message if success=False.
    """
    # Use provided mappings or fall back to module-level ones
    pmappings = platform_mappings_dict if platform_mappings_dict is not None else platform_mappings
    
    xml_input = Path(lists_base_path) / platform_abbrev / "gamelist.xml"
    xml_output = Path(lists_base_path) / platform_abbrev / "gamelist_updated.xml"
    
    if not xml_input.exists():
        if verbose:
            print(f"⚠️  Skipping {platform_abbrev}: gamelist.xml not found at {xml_input}")
        return {
            'platform_abbrev': platform_abbrev,
            'success': False,
            'error': 'gamelist.xml not found'
        }
    
    platform_name = pmappings.get(platform_abbrev, platform_abbrev)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PLATFORM: {platform_abbrev} → {platform_name}")
        print(f"{'='*60}")
    
    try:
        # Step 1: Build inventory
        if verbose:
            print(f"[1/4] Building inventory...")
        try:
            inventory = build_inventory_from_gamelist(xml_input)
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to build inventory: {e}")
            raise
        if verbose:
            if len(inventory) == 0:
                print(f"      ⚠️  Found 0 games in inventory")
            else:
                print(f"      Found {len(inventory)} games")
        
        # Step 2: Match to metadata
        if verbose:
            print(f"[2/4] Matching to game_df metadata...")
        try:
            matched_games = match_inventory_to_metadata(
                inventory=inventory,
                game_df=game_df,
                platform_name=platform_name,
                threshold=threshold
            )
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to match games: {e}")
            raise
        matched_count = sum(1 for g in matched_games if g.get('match_name') is not None)
        if verbose:
            print(f"      Matched {matched_count}/{len(matched_games)} games")
        
        # Step 3: Generate updated XML
        if verbose:
            print(f"[3/4] Generating updated gamelist.xml...")
        try:
            output_file = generate_gamelist_xml(
                matched_games=matched_games,
                output_path=xml_output
            )
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to generate XML: {e}")
            raise
        if verbose:
            print(f"      Written to {xml_output.relative_to(Path(lists_base_path).parent)}")
        
        # Step 4: Audit
        if verbose:
            print(f"[4/4] Generating audit report...")
        try:
            audit = generate_audit_report(matched_games, threshold=threshold)
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to generate audit: {e}")
            raise
        
        if verbose:
            print_audit_report(audit, platform_name)
        
        return {
            'platform_abbrev': platform_abbrev,
            'platform_name': platform_name,
            'success': True,
            'xml_input': str(xml_input),
            'xml_output': str(xml_output),
            'inventory_count': len(inventory),
            'matched_count': matched_count,
            'match_rate': audit['match_rate'],
            'inventory': inventory,
            'matched_games': matched_games,
            'audit': audit,
        }
    
    except Exception as e:
        if verbose:
            print(f"❌ Error processing {platform_abbrev}: {e}")
        if verbose:
            traceback.print_exc()
        return {
            'platform_abbrev': platform_abbrev,
            'success': False,
            'error': str(e)
        }


def analyze_matching_breakdown(
    matched_games: List[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Analyze and print matching strategy breakdown.

    Args:
        matched_games: List from match_inventory_to_metadata().
        audit: Optional audit report dict (used only for improvement calculation).

    Returns:
        Dict with match type counts: filename_exact, filename_fuzzy, name_match,
        no_match, filename_based_total.
    """
    """
    Analyze and print matching strategy breakdown.
    
    Args:
        matched_games: List from match_inventory_to_metadata()
        audit: Optional audit report dict (used only if provided)
    
    Returns:
        dict with match type counts
    """
    if not matched_games or len(matched_games) == 0:
        print("No matched games to analyze")
        return {
            'filename_exact': 0,
            'filename_fuzzy': 0,
            'name_match': 0,
            'no_match': 0,
            'filename_based_total': 0,
        }
    
    filename_exact = sum(1 for g in matched_games if g.get('match_type') == 'filename_exact')
    filename_fuzzy = sum(1 for g in matched_games if g.get('match_type') == 'filename_fuzzy')
    name_match = sum(1 for g in matched_games if g.get('match_type') == 'name')
    no_match = sum(1 for g in matched_games if g.get('match_name') is None)
    
    print(f"\n{'='*60}")
    print(f"MATCHING STRATEGY BREAKDOWN")
    print(f"{'='*60}")
    print(f"\nMatching Type Distribution:")
    print(f"  Filename (exact):     {filename_exact:4d} ({filename_exact/len(matched_games)*100:5.1f}%)")
    print(f"  Filename (fuzzy):     {filename_fuzzy:4d} ({filename_fuzzy/len(matched_games)*100:5.1f}%)")
    print(f"  Name-based:           {name_match:4d} ({name_match/len(matched_games)*100:5.1f}%)")
    print(f"  No match:             {no_match:4d} ({no_match/len(matched_games)*100:5.1f}%)")
    print(f"                        -----")
    print(f"  Total:                {len(matched_games):4d}")
    
    filename_based_total = filename_exact + filename_fuzzy
    print(f"\nFilename-based matches: {filename_based_total:4d} ({filename_based_total/len(matched_games)*100:.1f}%)")
    
    if audit:
        improvement = filename_based_total - audit.get('matched', 0)
        if improvement > 0:
            print(f"✓ Filename strategy improved matches by {improvement} over name-based only")
    
    print(f"{'='*60}\n")
    
    return {
        'filename_exact': filename_exact,
        'filename_fuzzy': filename_fuzzy,
        'name_match': name_match,
        'no_match': no_match,
        'filename_based_total': filename_based_total,
    }
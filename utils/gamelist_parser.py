"""
Parser for gamelist.xml files from RetroPie/EmulationStation.
Extracts game metadata from XML files organized by platform.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import pandas as pd


GAMELIST_COLUMN_MAP = {
    "name": "name",
    "path": "filename",
    "desc": "summary",
    "releasedate": "release_date",
    "developer": "developer",
    "publisher": "publisher",
    "genre": "genres",
    "players": "players",
    "rating": "user_rating",
}


def _load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from file, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

    return data if isinstance(data, dict) else {}


def load_platform_mappings(
    mappings_file: str = "utils/platform_registry.json",
) -> dict[str, str]:
    """Load platform folder -> canonical platform name mappings from the unified registry.

    Flattens the registry into a single alias->canonical lookup dict.

    Args:
        mappings_file: Path to platform_registry.json file

    Returns:
        Dict mapping directory names (lowercased) to canonical platform names
    """
    raw = _load_json_file(mappings_file)
    flat = {}
    for canonical, data in raw.items():
        aliases = data.get("aliases", [])
        for alias in aliases:
            flat[str(alias).strip().lower()] = str(canonical).strip()
        # Also map the canonical name itself
        flat[str(canonical).strip().lower()] = str(canonical).strip()
    return flat


def parse_rating(rating_str: Optional[str]) -> Optional[float]:
    """
    Parse rating and normalize to 0-10 scale.
    Detects if value is already 0-10 or needs conversion from 0-1.
    """
    if not rating_str:
        return None

    try:
        value = float(rating_str)
    except (TypeError, ValueError):
        return None

    # If value is already in 0-10 range, return as-is
    if value > 1.0:
        return value

    # If value is 0-1 range, scale to 0-10
    if 0.0 <= value <= 1.0:
        return value * 10

    # Out of expected range, return as-is
    return value


def parse_gamelist_xml(xml_path: Path, platform: str) -> list[dict]:
    """
    Parse a single gamelist.xml file and extract game metadata.

    Args:
        xml_path: Path to gamelist.xml file
        platform: Canonical platform name

    Returns:
        List of game dictionaries
    """
    games = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for game in root.findall('game'):
            # Skip if no name
            name_elem = game.find('name')
            if name_elem is None or not name_elem.text:
                continue

            game_data: dict[str, Optional[object]] = {"platform": platform}

            for xml_tag, target_field in GAMELIST_COLUMN_MAP.items():
                elem = game.find(xml_tag)
                if elem is None or elem.text is None:
                    value = None
                else:
                    value = elem.text.strip()

                if xml_tag == "rating" and value:
                    value = parse_rating(str(value))

                game_data[target_field] = value

            games.append(game_data)

    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
    except Exception as e:
        print(f"Unexpected error parsing {xml_path}: {e}")

    return games


def load_all_gamelists(
    lists_dir: str = "lists",
    systems_file: str = "utils/platform_registry.json",
) -> pd.DataFrame:
    """
    Load and parse all gamelist.xml files from the lists directory.

    Args:
        lists_dir: Directory containing platform subdirectories
        systems_file: Path to gamelist_folder_mappings.json file

    Returns:
        DataFrame with all game data from gamelist.xml files
    """
    lists_path = Path(lists_dir)
    platform_mappings = load_platform_mappings(systems_file)

    all_games = []

    # Find all gamelist.xml files
    for xml_file in lists_path.rglob('gamelist.xml'):
        # Get platform directory name (parent of gamelist.xml)
        platform_dir = xml_file.parent.name

        # Map to canonical platform name
        platform_key = platform_dir.strip().lower()
        platform_name = platform_mappings.get(platform_key, platform_dir)

        # Parse the XML file
        games = parse_gamelist_xml(xml_file, platform_name)
        all_games.extend(games)

    # Convert to DataFrame
    df = pd.DataFrame(all_games)

    return df

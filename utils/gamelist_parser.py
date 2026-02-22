"""
Parser for gamelist.xml files from RetroPie/EmulationStation.
Extracts game metadata from XML files organized by platform.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import pandas as pd

gamelist_mapping = {
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


def load_platform_mappings(systems_file: str = "lists/systems.txt") -> dict[str, str]:
    """
    Load platform directory name to canonical platform name mappings.
    
    Args:
        systems_file: Path to systems.txt file with format "dirname: Platform Name"
    
    Returns:
        Dict mapping directory names to canonical platform names
    """
    mappings = {}
    systems_path = Path(systems_file)
    
    if not systems_path.exists():
        return mappings
    
    with open(systems_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            dirname, platform_name = line.split(':', 1)
            mappings[dirname.strip()] = platform_name.strip()
    
    return mappings


def clean_game_name(name: str) -> str:
    """
    Clean game name by removing region codes, version tags, etc.
    Examples:
        "Super Mario World (U)" -> "Super Mario World"
        "Zelda (E) [!]" -> "Zelda"
    """
    if not name:
        return name
    
    # Remove region codes: (U), (E), (J), (USA), etc.
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    # Remove tags: [!], [a], [b1], etc.
    name = re.sub(r'\s*\[[^\]]*\]\s*', ' ', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name.strip()



def parse_players(players_str: Optional[str]) -> Optional[int]:
    """
    Extract maximum players from strings like "1-2", "1-4", "2", etc.
    """
    if not players_str:
        return None
    
    players_str = str(players_str).strip()
    
    # Handle ranges like "1-2"
    if '-' in players_str:
        parts = players_str.split('-')
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            pass
    
    # Handle single numbers
    try:
        return int(players_str)
    except ValueError:
        pass
    
    return None


def parse_rating(rating_str: Optional[str]) -> Optional[float]:
    """
    Convert EmulationStation rating (0-1) to 0-10 scale.
    """
    if not rating_str:
        return None

    try:
        return float(rating_str) * 10
    except (TypeError, ValueError):
        return None


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

            for xml_tag, target_field in gamelist_mapping.items():
                elem = game.find(xml_tag)
                if elem is None or elem.text is None:
                    value = None
                else:
                    value = elem.text.strip()

                if xml_tag == "name" and value:
                    value = clean_game_name(str(value))
                elif xml_tag == "path" and value:
                    value = Path(str(value)).stem
                elif xml_tag == "players" and value:
                    value = parse_players(str(value))
                elif xml_tag == "rating" and value:
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
    systems_file: str = "lists/systems.txt",
) -> pd.DataFrame:
    """
    Load and parse all gamelist.xml files from the lists directory.
    
    Args:
        lists_dir: Directory containing platform subdirectories
        systems_file: Path to systems.txt platform mappings
    
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
        platform_name = platform_mappings.get(platform_dir, platform_dir.title())
        
        # Parse the XML file
        games = parse_gamelist_xml(xml_file, platform_name)
        all_games.extend(games)
        
        print(f"Loaded {len(games)} games from {platform_dir} ({platform_name})")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_games)
    
    print(f"\nTotal games loaded: {len(df)}")
    if not df.empty:
        print(f"Platforms: {df['platform'].nunique()}")
    
    return df

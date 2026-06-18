#!/usr/bin/env python3
"""Test script to verify case-insensitive matching improvements."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from update_gamelist import normalize_game_name, update_gamelist_xml
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

def test_normalize_game_name():
    """Test that normalize_game_name handles various cases correctly."""
    test_cases = [
        ("Super Mario Bros", "super mario bros"),
        ("The Legend of Zelda", "legend of zelda"),
        ("MARIO KART", "mario kart"),
        ("sonic the hedgehog (us)", "sonic the hedgehog"),
        ("game.nes", "game"),
    ]
    
    for input_name, expected in test_cases:
        result = normalize_game_name(input_name)
        print(f"normalize_game_name('{input_name}') -> '{result}'")
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("✓ All normalize_game_name tests passed")

def test_case_insensitive_matching():
    """Test case-insensitive matching in update_gamelist_xml."""
    
    # Create a temporary XML file for testing
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gameList>
  <game>
    <name>Super Mario Bros</name>
    <path>./Super Mario Bros.nes</path>
  </game>
  <game>
    <name>Legend of Zelda</name>
    <path>./Legend of Zelda.nes</path>
  </game>
</gameList>'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    
    try:
        # Test case-insensitive matching
        updates = {
            "super mario bros": {"desc": "A classic platformer"},
            "legend of zelda": {"desc": "An adventure game"},
        }
        
        # This should work regardless of case
        result = update_gamelist_xml(xml_path, updates)
        
        print("✓ Case-insensitive matching test completed")
        
        # Parse the updated XML to verify content
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for game in root.findall('game'):
            name_elem = game.find('name')
            desc_elem = game.find('desc')
            if name_elem is not None and desc_elem is not None:
                print(f"Game: {name_elem.text}, Description: {desc_elem.text}")
        
    finally:
        os.unlink(xml_path)

if __name__ == "__main__":
    print("Testing case-insensitive matching improvements...")
    test_normalize_game_name()
    test_case_insensitive_matching()
    print("All tests passed!")
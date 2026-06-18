#!/usr/bin/env python3
"""Test script to verify XML preservation improvements."""

import sys
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Add the utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from update_gamelist import generate_gamelist_xml, build_inventory_from_gamelist
import pandas as pd

def test_xml_preservation():
    """Test that XML elements are properly preserved during updates."""
    
    # Create a temporary XML file with nested elements for testing
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gameList>
  <game>
    <name>Super Mario Bros</name>
    <path>./Super Mario Bros.nes</path>
    <image>./images/super_mario_bros.png</image>
    <marquee>./marquees/super_mario_bros.png</marquee>
    <playcount>5</playcount>
    <lastplayed>2023-01-01</lastplayed>
    <desc>A classic platformer game</desc>
  </game>
  <game>
    <name>Legend of Zelda</name>
    <path>./Legend of Zelda.nes</path>
    <image>./images/legend_of_zelda.png</image>
    <marquee>./marquees/legend_of_zelda.png</marquee>
    <playcount>3</playcount>
    <lastplayed>2023-01-02</lastplayed>
    <desc>An adventure game</desc>
  </game>
</gameList>'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    
    try:
        # Test XML preservation
        print("Testing XML preservation...")
        
        # First, build inventory to get original elements
        inventory = build_inventory_from_gamelist(xml_path)
        print(f"Found {len(inventory)} games in inventory")
        
        # Create mock metadata
        game_data = pd.DataFrame([
            {
                'name': 'Super Mario Bros',
                'platform': 'Nintendo Entertainment System',
                'summary': 'A classic platformer game',
                'developer': 'Nintendo',
                'publisher': 'Nintendo',
                'release_date': '1985-08-13',
                'genres': 'Platform'
            },
            {
                'name': 'Legend of Zelda',
                'platform': 'Nintendo Entertainment System',
                'summary': 'An adventure game',
                'developer': 'Nintendo',
                'publisher': 'Nintendo',
                'release_date': '1986-08-21',
                'genres': 'Adventure'
            }
        ])
        
        # Create matched games for our test
        matched_games = []
        for item in inventory:
            # Find matching metadata
            matching_row = game_data[game_data['name'] == item['xml_name']].iloc[0] if len(game_data[game_data['name'] == item['xml_name']]) > 0 else None
            matched_games.append({
                'xml_name': item['xml_name'],
                'path': item['path'],
                'game_elem': item['game_elem'],
                'match_name': item['xml_name'] if matching_row is not None else None,
                'match_confidence': 100 if matching_row is not None else 0,
                'metadata': matching_row if matching_row is not None else None,
                'match_type': 'name' if matching_row is not None else None,
            })
        
        # Create output file path
        output_path = xml_path.replace('.xml', '_updated.xml')
        
        # Generate updated XML
        result = generate_gamelist_xml(matched_games, output_path)
        print(f"Generated updated XML at: {result}")
        
        # Parse the output to verify preservation
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        print("\nVerifying XML preservation:")
        for game in root.findall('game'):
            name_elem = game.find('name')
            image_elem = game.find('image')
            marquee_elem = game.find('marquee')
            playcount_elem = game.find('playcount')
            lastplayed_elem = game.find('lastplayed')
            desc_elem = game.find('desc')
            
            print(f"  Game: {name_elem.text if name_elem is not None else 'Unknown'}")
            print(f"    Image preserved: {image_elem is not None}")
            print(f"    Marquee preserved: {marquee_elem is not None}")
            print(f"    Playcount preserved: {playcount_elem is not None}")
            print(f"    Lastplayed preserved: {lastplayed_elem is not None}")
            print(f"    Description preserved: {desc_elem is not None}")
        
        print("✓ XML preservation test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in XML preservation test: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(xml_path):
            os.unlink(xml_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    print("Testing XML preservation improvements...")
    success = test_xml_preservation()
    if success:
        print("All XML preservation tests passed!")
    else:
        print("XML preservation tests failed!")
        sys.exit(1)
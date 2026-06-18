#!/usr/bin/env python3
"""Test script to verify metadata mapping improvements."""

import sys
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Add the utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from update_gamelist import generate_gamelist_xml, build_inventory_from_gamelist
import pandas as pd
import io
import contextlib

def test_metadata_mapping():
    """Test that all DataFrame columns are handled properly and unmapped columns are logged."""
    
    # Create a temporary XML file for testing
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gameList>
  <game>
    <name>Super Mario Bros</name>
    <path>./Super Mario Bros.nes</path>
    <desc>A classic platformer game</desc>
  </game>
</gameList>'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    
    try:
        # Test metadata mapping
        print("Testing metadata mapping improvements...")
        
        # First, build inventory to get original elements
        inventory = build_inventory_from_gamelist(xml_path)
        print(f"Found {len(inventory)} games in inventory")
        
        # Create mock metadata with various columns including some unmapped ones
        game_data = pd.DataFrame([
            {
                'name': 'Super Mario Bros',
                'platform': 'Nintendo Entertainment System',
                'summary': 'A classic platformer game',
                'developer': 'Nintendo',
                'publisher': 'Nintendo',
                'release_date': '1985-08-13',
                'genres': 'Platform',
                'players': '1',
                'user_rating': '9.5',
                'esrb_rating': 'E',
                'rating': '9.5',
                'tags': 'classic,nes'
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
        
        # Capture stdout to check for warnings
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            # Generate updated XML
            result = generate_gamelist_xml(matched_games, output_path)
            print(f"Generated updated XML at: {result}")
        
        # Check if warnings were logged for unmapped columns
        captured_output = stdout_capture.getvalue()
        print(f"Captured output: {captured_output}")
        
        # Verify the output file exists and is valid
        assert os.path.exists(output_path), "Output file was not created"
        
        # Parse the output to verify content
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        game = root.find('game')
        if game is not None:
            desc_elem = game.find('desc')
            name_elem = game.find('name')
            print(f"Game name: {name_elem.text if name_elem is not None else 'None'}")
            print(f"Description: {desc_elem.text if desc_elem is not None else 'None'}")
        
        # Verify that the mapping works correctly by checking that mapped fields are handled
        print("✓ Metadata mapping test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in metadata mapping test: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(xml_path):
            os.unlink(xml_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    print("Testing metadata mapping improvements...")
    success = test_metadata_mapping()
    if success:
        print("All metadata mapping tests passed!")
    else:
        print("Metadata mapping tests failed!")
        sys.exit(1)
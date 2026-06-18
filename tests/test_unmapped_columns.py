#!/usr/bin/env python3
"""Test script to verify unmapped column warning."""

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

def test_unmapped_columns_warning():
    """Test that unmapped columns are properly logged."""
    
    # Create a temporary XML file for testing
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gameList>
  <game>
    <name>Test Game</name>
    <path>./test.nes</path>
  </game>
</gameList>'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    
    try:
        # Test with unmapped columns
        print("Testing unmapped column warning...")
        
        # First, build inventory to get original elements
        inventory = build_inventory_from_gamelist(xml_path)
        
        # Create mock metadata with unmapped columns
        game_data = pd.DataFrame([
            {
                'name': 'Test Game',
                'platform': 'Test Platform',
                'summary': 'A test game',
                'developer': 'Test Dev',
                'publisher': 'Test Pub',
                'release_date': '2023-01-01',
                'genres': 'Test Genre',
                'players': '1',
                'user_rating': '8.5',
                'esrb_rating': 'T',
                'rating': '8.5',
                'tags': 'test,unmapped',  # This column should trigger a warning
                'custom_field': 'custom_value'  # This should also trigger a warning
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
        
        # Verify that warnings were logged for unmapped columns
        assert 'Warning:' in captured_output, "Expected warning message for unmapped columns"
        assert 'tags' in captured_output or 'custom_field' in captured_output, "Expected unmapped column names in warning"
        
        print("✓ Unmapped column warning test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in unmapped column warning test: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(xml_path):
            os.unlink(xml_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    print("Testing unmapped column warning...")
    success = test_unmapped_columns_warning()
    if success:
        print("All unmapped column tests passed!")
    else:
        print("Unmapped column tests failed!")
        sys.exit(1)
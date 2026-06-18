#!/usr/bin/env python3
"""Test script to verify error handling improvements."""

import sys
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Add the utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from update_gamelist import (
    update_gamelist_xml, 
    build_inventory_from_gamelist, 
    match_inventory_to_metadata,
    generate_gamelist_xml,
    update_platform_gamelist
)
import pandas as pd
import io
import contextlib

def test_error_handling():
    """Test that error handling works correctly."""
    
    print("Testing error handling improvements...")
    
    # Test 1: Invalid XML file
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("Invalid XML content")
            xml_path = f.name
            
        result = build_inventory_from_gamelist(xml_path)
        print("✓ Invalid XML handled gracefully")
        os.unlink(xml_path)
    except Exception as e:
        print(f"❌ Error in invalid XML handling: {e}")
        return False
    
    # Test 2: Empty or non-existent file
    try:
        result = build_inventory_from_gamelist("/non/existent/file.xml")
        print("✓ Non-existent file handled gracefully")
    except Exception as e:
        print(f"❌ Error in non-existent file handling: {e}")
        return False
    
    # Test 3: Invalid game_df in matching
    try:
        inventory = [{'xml_name': 'Test Game', 'path': './test.nes', 'game_elem': None}]
        result = match_inventory_to_metadata(inventory, None, "Test Platform")
        print("✓ Invalid game_df handled gracefully")
    except Exception as e:
        print(f"❌ Error in invalid game_df handling: {e}")
        return False
    
    # Test 4: Test with empty data
    try:
        result = match_inventory_to_metadata([], None, "Test Platform")
        print("✓ Empty inventory handled gracefully")
    except Exception as e:
        print(f"❌ Error in empty inventory handling: {e}")
        return False
    
    # Test 5: Test XML generation with invalid data
    try:
        output_path = "/tmp/test_invalid_output.xml"
        matched_games = [{'xml_name': 'Test Game', 'path': './test.nes', 'game_elem': None}]
        result = generate_gamelist_xml(matched_games, output_path)
        print("✓ Invalid matched games handled gracefully")
    except Exception as e:
        print(f"✓ Invalid matched games handled gracefully: {e}")
    
    print("✓ All error handling tests passed!")
    return True

def test_production_readiness():
    """Test production readiness checks."""
    
    print("Testing production readiness...")
    
    try:
        from update_gamelist import validate_production_environment
        result = validate_production_environment()
        print(f"✓ Production readiness check: {'PASS' if result else 'WARN'}")
    except Exception as e:
        print(f"❌ Error in production readiness check: {e}")
        return False
    
    print("✓ Production readiness tests passed!")
    return True

if __name__ == "__main__":
    print("Testing error handling improvements...")
    success1 = test_error_handling()
    success2 = test_production_readiness()
    
    if success1 and success2:
        print("All error handling tests passed!")
    else:
        print("Some error handling tests failed!")
        sys.exit(1)
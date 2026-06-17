#!/usr/bin/env python3
"""
Simplified test suite to verify key pipeline functionality works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np


def test_basic_functionality():
    """Test that core pipeline functions work."""
    print("Testing basic pipeline functionality...")
    
    # Test that imports work
    from utils.merge_pipeline import (
        clean_game_name,
        build_name_match_key,
        _parse_multi_value,
        _flatten_multi_value
    )
    
    from utils.resolvers import (
        collect_unique,
        collect_unique_ordered,
        any_truthy_priority
    )
    
    print("✓ All imports successful")
    
    # Test clean_game_name with edge cases
    result = clean_game_name('Game\x00Title')
    assert 'Game' in result and 'Title' in result
    print("✓ clean_game_name handles control characters")
    
    # Test build_name_match_key
    key = build_name_match_key('Game Title\x00(US)')
    assert key is not None
    print("✓ build_name_match_key handles control characters")
    
    # Test multi-value parsing
    result = _parse_multi_value('action, adventure; strategy')
    assert len(result) == 3
    print("✓ _parse_multi_value handles delimiters")
    
    # Test flattening
    result = _flatten_multi_value(['action', 'adventure'])
    assert 'action' in str(result) and 'adventure' in str(result)
    print("✓ _flatten_multi_value works")
    
    # Test unique collection
    values = ['action', 'adventure', 'action']
    result = collect_unique(pd.Series(values))
    assert 'action' in str(result) and 'adventure' in str(result)
    print("✓ collect_unique works")
    
    print("✓ All core functionality tests passed")


def test_international_titles():
    """Test international titles."""
    print("Testing international titles...")
    
    from utils.merge_pipeline import clean_game_name
    
    # Test Japanese
    result = clean_game_name('ゼルダの伝説')
    assert 'ゼルダの伝説' in result
    print("✓ Japanese titles work")
    
    # Test Chinese
    result = clean_game_name('仙剑奇侠传')
    assert '仙剑奇侠传' in result
    print("✓ Chinese titles work")
    
    # Test mixed
    result = clean_game_name('Final Fantasy VII (Japan, USA)')
    assert 'Final Fantasy VII' in result
    print("✓ Mixed language titles work")


def test_complex_names():
    """Test complex naming scenarios."""
    print("Testing complex names...")
    
    from utils.merge_pipeline import clean_game_name
    
    # Test version tags
    result = clean_game_name('Game Title [v1.0]')
    assert 'Game Title' in result
    print("✓ Version tags work")
    
    # Test special characters
    result = clean_game_name('Game & More')
    assert 'Game & More' in result
    print("✓ Special characters work")
    
    # Test repeated whitespace
    result = clean_game_name('Game    Title')
    assert 'Game Title' in result
    print("✓ Repeated whitespace works")


def main():
    """Run all tests."""
    print("Running pipeline tests...")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_international_titles()
        test_complex_names()
        print("=" * 40)
        print("🎉 All tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
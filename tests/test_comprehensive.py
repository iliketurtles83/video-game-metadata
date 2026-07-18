#!/usr/bin/env python3
"""
Comprehensive test suite for the video game metadata pipeline.
Tests edge cases, international titles, and complex naming conventions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
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


def test_edge_cases():
    """Test edge cases and problematic titles."""
    print("Testing edge cases...")
    
    # Test control characters
    result = clean_game_name('Game\x00Title\x01')
    assert 'Game' in result and 'Title' in result
    print("✓ Control characters handled")
    
    # Test various hyphens
    result = clean_game_name('Game-Title–Title—Title')
    assert 'Game' in result and 'Title' in result
    print("✓ Various hyphens handled")
    
    # Test special characters in name matching
    key = build_name_match_key('Game Title\x00(US)')
    assert 'Game Title US' in key
    print("✓ Special characters in name matching handled")
    
    # Test multi-value parsing with special characters
    result = _parse_multi_value('action\x00, adventure')
    assert 'action' in result and 'adventure' in result
    print("✓ Multi-value parsing with special characters works")
    
    # Test flattening with None
    result = _flatten_multi_value(None)
    assert pd.isna(result)
    print("✓ None handling works")


def test_international_titles():
    """Test international titles and non-ASCII characters."""
    print("Testing international titles...")
    
    # Test Japanese title
    result = clean_game_name('ゼルダの伝説')
    assert 'ゼルダの伝説' in result
    print("✓ Japanese titles work")
    
    # Test Chinese title
    result = clean_game_name('仙剑奇侠传')
    assert '仙剑奇侠传' in result
    print("✓ Chinese titles work")
    
    # Test Korean title
    result = clean_game_name('마리오 월드')
    assert '마리오 월드' in result
    print("✓ Korean titles work")
    
    # Test mixed languages
    result = clean_game_name('Final Fantasy VII (Japan, USA)')
    assert 'Final Fantasy VII' in result
    print("✓ Mixed language titles work")


def test_complex_naming():
    """Test complex naming conventions."""
    print("Testing complex naming conventions...")
    
    # Test Roman numerals
    result = clean_game_name('Super Mario Bros. II')
    assert 'Super Mario Bros' in result
    print("✓ Roman numerals handled")
    
    # Test version tags
    result = clean_game_name('Game Title [v1.0]')
    assert 'Game Title' in result
    print("✓ Version tags handled")
    
    # Test special characters
    result = clean_game_name('Game & More')
    assert 'Game & More' in result
    print("✓ Special characters handled")
    
    # Test repeated whitespace
    result = clean_game_name('Game    Title')
    assert 'Game Title' in result
    print("✓ Repeated whitespace handled")


def test_multi_value_handling():
    """Test multi-value field parsing and resolution."""
    print("Testing multi-value handling...")
    
    # Test unique collection
    values = ['action', 'adventure', 'action']
    result = collect_unique(pd.Series(values))
    assert 'action' in str(result)
    assert 'adventure' in str(result)
    print("✓ Unique collection works")
    
    # Test ordered unique collection
    values = ['action', 'adventure', 'action', 'rpg']
    priority_order = ['rpg', 'action']
    result = collect_unique_ordered(pd.Series(values), priority_order)
    assert 'rpg' in str(result)
    print("✓ Ordered unique collection works")
    
    # Test truthy resolution
    values = ['false', 'true', 'yes']
    result = any_truthy_priority(pd.Series(values), ['source1', 'source2'])
    assert result is True
    print("✓ Truthy resolution works")


def test_integration():
    """Test integration of pipeline components."""
    print("Testing integration...")
    
    # Test with problematic data
    test_data = pd.DataFrame({
        'name': ['Game Title (U)', 'Game\x00Title', 'Game Title\x00(US)'],
        'platform': ['Nintendo 64', 'Nintendo 64', 'Nintendo 64'],
        'genres': ['action, adventure', 'action', 'action, adventure']
    })
    
    # Test cleaning
    cleaned = test_data.copy()
    cleaned['name'] = cleaned['name'].apply(clean_game_name)
    
    # Test key building
    cleaned['name_key'] = cleaned['name'].apply(build_name_match_key)
    
    assert len(cleaned) == 3
    assert not cleaned['name_key'].isna().all()
    print("✓ Integration works with problematic data")
    
    # Test mixed language metadata
    test_data2 = pd.DataFrame({
        'name': ['Final Fantasy VII', 'ゼルダの伝説', '仙剑奇侠传'],
        'platform': ['PlayStation', 'Nintendo Switch', 'PC'],
        'genres': ['rpg', 'adventure', 'rpg']
    })
    
    cleaned2 = test_data2.copy()
    cleaned2['name'] = cleaned2['name'].apply(clean_game_name)
    
    assert len(cleaned2) == 3
    assert not cleaned2['name'].isna().any()
    print("✓ Mixed language integration works")


def run_all_tests():
    """Run all tests and report results."""
    print("Running comprehensive test suite...")
    print("=" * 50)
    
    try:
        test_edge_cases()
        test_international_titles()
        test_complex_naming()
        test_multi_value_handling()
        test_integration()
        print("=" * 50)
        print("🎉 All tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
#!/usr/bin/env python3
"""
Comprehensive test suite for the video game metadata pipeline.
Tests edge cases, international titles, and complex naming conventions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import pytest
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


class TestEdgeCaseHandling:
    """Test edge cases and problematic titles."""
    
    def test_clean_game_name_with_special_characters(self):
        """Test cleaning of names with special characters."""
        # Test control characters
        result = clean_game_name('Game\x00Title\x01')
        assert 'Game' in result and 'Title' in result
        
        # Test various hyphens
        result = clean_game_name('Game-Title–Title—Title')
        assert result == 'Game Title Title Title'
        
        # Test empty cleaning
        result = clean_game_name('   ')
        assert result == '   '  # Should preserve original
        
    def test_build_name_match_key_with_special_characters(self):
        """Test building match keys with special characters."""
        key = build_name_match_key('Game Title\x00(US)')
        assert 'game title' in key
        
        # Test with multiple hyphens
        key = build_name_match_key('Game-Title–Title')
        assert key is not None
        assert len(key) > 0
        
    def test_parse_multi_value_with_special_characters(self):
        """Test parsing multi-value fields with special characters."""
        # Test with control characters
        result = _parse_multi_value('action\x00, adventure')
        assert 'action' in result and 'adventure' in result
        
        # Test with various delimiters
        result = _parse_multi_value('action; adventure| strategy')
        assert len(result) == 3
        
    def test_flatten_multi_value_with_edge_cases(self):
        """Test flattening multi-value fields with edge cases."""
        # Test with None
        result = _flatten_multi_value(None)
        assert pd.isna(result)
        
        # Test with empty list
        result = _flatten_multi_value([])
        assert pd.isna(result)
        
        # Test with normal data
        result = _flatten_multi_value(['action', 'adventure'])
        assert str(result) == 'action, adventure'


class TestInternationalTitles:
    """Test international titles and non-ASCII characters."""
    
    def test_japanese_titles(self):
        """Test Japanese titles."""
        # Test basic Japanese title
        result = clean_game_name('ゼルダの伝説')
        assert 'ゼルダの伝説' in result
        
        # Test with romanization
        result = clean_game_name('Zelda no Densetsu (Japan)')
        assert 'Zelda no Densetsu' in result
        
    def test_chinese_titles(self):
        """Test Chinese titles."""
        result = clean_game_name('仙剑奇侠传')
        assert '仙剑奇侠传' in result
        
        # Test with pinyin
        result = clean_game_name('Xian Jian Qi Xia Zhuan (China)')
        assert 'Xian Jian Qi Xia Zhuan' in result
        
    def test_korean_titles(self):
        """Test Korean titles."""
        result = clean_game_name('마리오 월드')
        assert '마리오 월드' in result
        
        # Test with romanization
        result = clean_game_name('Mario World (Korea)')
        assert 'Mario World' in result
        
    def test_multilingual_titles(self):
        """Test titles with mixed languages."""
        result = clean_game_name('Final Fantasy VII (Japan, USA)')
        assert 'Final Fantasy VII' in result
        
        # Test with special characters
        result = clean_game_name('Mega Man 10 (USA, Europe, Japan)')
        assert 'Mega Man 10' in result


class TestComplexNamingConventions:
    """Test complex naming conventions and edge cases."""
    
    def test_sequels_and_versions(self):
        """Test sequel and version naming."""
        # Test Roman numerals
        result = clean_game_name('Super Mario Bros. II')
        assert 'Super Mario Bros' in result
        
        # Test version tags
        result = clean_game_name('Game Title [v1.0]')
        assert 'Game Title' in result
        
        # Test multiple version tags
        result = clean_game_name('Game Title [v2.0][!]')
        assert 'Game Title' in result
        
    def test_special_characters_in_names(self):
        """Test special characters in game names."""
        # Test ampersands
        result = clean_game_name('Game & More')
        assert 'Game & More' in result
        
        # Test quotes
        result = clean_game_name('"Game Title"')
        assert 'Game Title' in result
        
        # Test parentheses
        result = clean_game_name('Game (2023)')
        assert 'Game' in result
        
    def test_repeated_whitespace(self):
        """Test repeated whitespace handling."""
        result = clean_game_name('Game    Title')
        assert 'Game Title' in result
        
        result = clean_game_name('   Game   Title   ')
        assert 'Game Title' in result


class TestMultiValueFieldHandling:
    """Test multi-value field parsing and resolution."""
    
    def test_collect_unique(self):
        """Test unique collection."""
        # Test basic unique collection
        values = ['action', 'adventure', 'action']
        result = collect_unique(pd.Series(values))
        assert 'action' in str(result)
        assert 'adventure' in str(result)
        assert str(result).count('action') == 1  # Should be unique
        
    def test_collect_unique_ordered(self):
        """Test ordered unique collection."""
        values = ['action', 'adventure', 'action', 'rpg']
        priority_order = ['rpg', 'action']
        result = collect_unique_ordered(pd.Series(values), priority_order)
        # Should prioritize items in priority_order first
        assert 'rpg' in str(result)
        assert 'action' in str(result)
        assert 'adventure' in str(result)
        
    def test_any_truthy_priority(self):
        """Test truthy resolution with priority."""
        values = ['false', 'true', 'yes']
        result = any_truthy_priority(pd.Series(values), ['source1', 'source2'])
        assert result is True
        
        values = ['false', 'no', '0']
        result = any_truthy_priority(pd.Series(values), ['source1', 'source2'])
        assert result is False


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_pipeline_with_problematic_data(self):
        """Test the complete pipeline with problematic data."""
        # Create DataFrame with edge cases
        test_data = pd.DataFrame({
            'name': ['Game Title (U)', 'Game\x00Title', 'Game Title\x00(US)'],
            'platform': ['Nintendo 64', 'Nintendo 64', 'Nintendo 64'],
            'genres': ['action, adventure', 'action', 'action, adventure']
        })
        
        # Test that cleaning works
        cleaned = test_data.copy()
        cleaned['name'] = cleaned['name'].apply(clean_game_name)
        
        # Test that key building works
        cleaned['name_key'] = cleaned['name'].apply(build_name_match_key)
        
        assert len(cleaned) == 3
        assert not cleaned['name_key'].isna().all()
        
    def test_mixed_language_metadata(self):
        """Test metadata with mixed language titles."""
        # Test data with international titles
        test_data = pd.DataFrame({
            'name': ['Final Fantasy VII', 'ゼルダの伝説', '仙剑奇侠传'],
            'platform': ['PlayStation', 'Nintendo Switch', 'PC'],
            'genres': ['rpg', 'adventure', 'rpg']
        })
        
        # Test that cleaning works with mixed languages
        cleaned = test_data.copy()
        cleaned['name'] = cleaned['name'].apply(clean_game_name)
        
        assert len(cleaned) == 3
        assert not cleaned['name'].isna().any()


def run_all_tests():
    """Run all tests and report results."""
    print("Running comprehensive test suite...")
    
    # Create test instances
    edge_case_test = TestEdgeCaseHandling()
    international_test = TestInternationalTitles()
    naming_test = TestComplexNamingConventions()
    multi_value_test = TestMultiValueFieldHandling()
    integration_test = TestIntegration()
    
    tests = [
        ("Edge Case Handling", edge_case_test),
        ("International Titles", international_test),
        ("Complex Naming", naming_test),
        ("Multi-Value Handling", multi_value_test),
        ("Integration", integration_test)
    ]
    
    all_passed = True
    
    for test_name, test_instance in tests:
        print(f"\n--- {test_name} ---")
        try:
            # Run all methods that start with test_
            methods = [method for method in dir(test_instance) if method.startswith('test_')]
            for method_name in methods:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
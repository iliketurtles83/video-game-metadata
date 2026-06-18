"""
Comprehensive performance test for metadata matching optimizations.
"""
import time
import pandas as pd
from utils.update_gamelist import match_inventory_to_metadata, normalize_game_name

def test_large_dataset_performance():
    """Test performance with larger datasets"""
    
    # Create larger test data
    inventory = [
        {
            'xml_name': f'Game {i}',
            'path': f'/roms/nes/game_{i}.nes',
            'game_elem': None,
            'name_normalized': normalize_game_name(f'Game {i}')
        }
        for i in range(1000)
    ]
    
    # Create even larger game dataset
    game_df = pd.DataFrame({
        'name': [f'Game {i}' for i in range(2000)],
        'platform': ['Nintendo Entertainment System'] * 2000,
        'filename': [f'game_{i}.nes' for i in range(2000)]
    })
    
    start_time = time.time()
    result = match_inventory_to_metadata(inventory, game_df, 'Nintendo Entertainment System')
    end_time = time.time()
    
    print(f"Processed {len(inventory)} games in {end_time - start_time:.2f} seconds")
    print(f"Average time per game: {(end_time - start_time) / len(inventory) * 1000:.2f} ms")
    
    # Verify results
    matched_count = len([r for r in result if r['match_name'] is not None])
    print(f"Matched {matched_count} games")
    
    # Test with different platform name to make sure we get partial matches
    game_df2 = pd.DataFrame({
        'name': [f'Game {i}' for i in range(1000)],
        'platform': ['Nintendo Entertainment System'] * 1000,
        'filename': [f'game_{i}.nes' for i in range(1000)]
    })
    
    # Create inventory with some names that won't match exactly
    inventory2 = [
        {
            'xml_name': f'Game {i}',
            'path': f'/roms/nes/game_{i}.nes',
            'game_elem': None,
            'name_normalized': normalize_game_name(f'Game {i}')
        }
        for i in range(100)
    ]
    
    start_time = time.time()
    result2 = match_inventory_to_metadata(inventory2, game_df2, 'Nintendo Entertainment System')
    end_time = time.time()
    
    print(f"\nProcessed {len(inventory2)} games in {end_time - start_time:.2f} seconds")
    print(f"Average time per game: {(end_time - start_time) / len(inventory2) * 1000:.2f} ms")
    
    # Verify results
    matched_count2 = len([r for r in result2 if r['match_name'] is not None])
    print(f"Matched {matched_count2} games")

if __name__ == "__main__":
    test_large_dataset_performance()
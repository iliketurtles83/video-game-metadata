"""
Performance test for metadata matching optimizations.
"""
import time
import pandas as pd
from utils.update_gamelist import match_inventory_to_metadata, normalize_game_name

def test_performance():
    """Test performance improvements for large datasets"""
    
    # Create test data with proper name_normalized field
    inventory = [
        {
            'xml_name': f'Game {i}',
            'path': f'/roms/nes/game_{i}.nes',
            'game_elem': None,
            'name_normalized': normalize_game_name(f'Game {i}')
        }
        for i in range(100)
    ]
    
    # Create larger game dataset
    game_df = pd.DataFrame({
        'name': [f'Game {i}' for i in range(500)],
        'platform': ['Nintendo Entertainment System'] * 500,
        'filename': [f'game_{i}.nes' for i in range(500)]
    })
    
    start_time = time.time()
    result = match_inventory_to_metadata(inventory, game_df, 'Nintendo Entertainment System')
    end_time = time.time()
    
    print(f"Processed {len(inventory)} games in {end_time - start_time:.2f} seconds")
    print(f"Average time per game: {(end_time - start_time) / len(inventory) * 1000:.2f} ms")
    
    # Verify results
    print(f"Matched {len([r for r in result if r['match_name'] is not None])} games")
    
if __name__ == "__main__":
    test_performance()
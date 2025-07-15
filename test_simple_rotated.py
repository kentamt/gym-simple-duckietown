#!/usr/bin/env python3
"""
Simple test for rotated tiles functionality.
"""

import sys
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.rendering.tile_image_manager import TileImageManager


def test_rotated_tiles_simple():
    """Simple test of rotated tiles."""
    print("=== Simple Rotated Tiles Test ===")
    
    # Test tile manager
    tile_manager = TileImageManager()
    print(f"Total tile mappings: {len(tile_manager.get_tile_mappings())}")
    
    # Test loading specific tiles
    test_tiles = [0, 1, 2, 3, 4, 10, 11, 20, 24, 30]
    
    print("\nTesting tile loading:")
    for tile_id in test_tiles:
        image = tile_manager.get_tile_image(tile_id, (64, 64))
        status = "✓" if image else "✗"
        tile_name = tile_manager.get_tile_mappings().get(tile_id, "unknown")
        print(f"  Tile {tile_id} ({tile_name}): {status}")
    
    # Create simple map with rotated tiles
    simple_layout = [
        [3, 10, 3],   # Grass, vertical road, grass
        [11, 30, 11], # Horizontal road, intersection, horizontal road  
        [3, 12, 3]    # Grass, vertical road, grass
    ]
    
    print(f"\nCreating 3x3 map with rotated tiles:")
    map_instance = create_map_from_array(simple_layout)
    print(f"Map created: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    
    return map_instance


if __name__ == "__main__":
    test_rotated_tiles_simple()
    print("Simple rotated tiles test completed!")
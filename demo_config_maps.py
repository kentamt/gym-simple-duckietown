#!/usr/bin/env python3
"""
Simple demonstration of config file-based map layouts.
"""

import sys
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array, create_map_from_file

def demo_array_maps():
    """Demonstrate creating maps from 2D arrays."""
    print("=== Demo: Array-Based Maps ===")
    
    # Your example layout as requested
    layout = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0], 
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    print("Creating map from 2D array:")
    print("Layout:", layout)
    
    map_instance = create_map_from_array(layout)
    print(f"\nResult:")
    print(map_instance)
    
    return map_instance

def demo_file_maps():
    """Demonstrate loading maps from config files."""
    print("\n=== Demo: File-Based Maps ===")
    
    # Load one of the example maps
    map_file = "duckietown_simulator/assets/maps/simple_room.json"
    
    print(f"Loading map from: {map_file}")
    map_instance = create_map_from_file(map_file)
    
    print(f"Result:")
    print(map_instance)
    
    return map_instance

def show_available_maps():
    """Show all available example maps."""
    print("\n=== Available Map Files ===")
    
    import os
    maps_dir = "duckietown_simulator/assets/maps"
    
    if os.path.exists(maps_dir):
        map_files = [f for f in os.listdir(maps_dir) if f.endswith('.json')]
        for map_file in sorted(map_files):
            print(f"  - {map_file}")
    else:
        print("  No maps directory found")

if __name__ == "__main__":
    print("Config File-Based Map Layout Demo")
    print("=" * 40)
    
    # Demo array creation 
    array_map = demo_array_maps()
    
    # Demo file loading
    file_map = demo_file_maps()
    
    # Show available maps
    show_available_maps()
    
    print("\n" + "=" * 40)
    print("Usage Examples:")
    print()
    print("1. Create from 2D array:")
    print("   layout = [[0,0,0],[0,1,0],[0,0,0]]")
    print("   map = create_map_from_array(layout)")
    print()
    print("2. Load from config file:")
    print("   map = create_map_from_file('path/to/map.json')")
    print()
    print("3. Tile types correspond to images:")
    print("   0 = empty.png, 1 = obstacle.png, 2 = road.png")
    print("   3 = grass.png, 4 = intersection.png")
#!/usr/bin/env python3
"""
Test script for config file-based map layouts.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.world.map import (
    create_map_from_array, 
    create_map_from_file,
    create_map_from_config
)
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def test_array_based_maps():
    """Test creating maps from 2D arrays."""
    print("=== Testing Array-Based Maps ===")
    
    # Example 1: Simple room as requested by user
    simple_room = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    print("\n1. Simple Room Layout:")
    map1 = create_map_from_array(simple_room)
    print(map1)
    
    # Example 2: More complex layout
    complex_layout = [
        [3, 3, 3, 3, 3, 3, 3],
        [3, 2, 2, 4, 2, 2, 3],
        [3, 2, 1, 1, 1, 2, 3],
        [3, 4, 1, 0, 1, 4, 3],
        [3, 2, 1, 1, 1, 2, 3],
        [3, 2, 2, 4, 2, 2, 3],
        [3, 3, 3, 3, 3, 3, 3]
    ]
    
    print("\n2. Complex Layout with Multiple Tile Types:")
    map2 = create_map_from_array(complex_layout)
    print(map2)
    
    return map1, map2


def test_file_based_maps():
    """Test creating maps from JSON config files."""
    print("\n=== Testing File-Based Maps ===")
    
    # Test each example map file
    map_files = [
        "duckietown_simulator/assets/maps/simple_room.json",
        "duckietown_simulator/assets/maps/race_track.json", 
        "duckietown_simulator/assets/maps/intersection_demo.json",
        "duckietown_simulator/assets/maps/complex_maze.json"
    ]
    
    loaded_maps = {}
    
    for map_file in map_files:
        try:
            print(f"\nLoading {map_file}:")
            map_instance = create_map_from_file(map_file)
            print(map_instance)
            
            # Store for visualization test
            map_name = map_file.split('/')[-1].replace('.json', '')
            loaded_maps[map_name] = map_instance
            
        except Exception as e:
            print(f"Error loading {map_file}: {e}")
    
    return loaded_maps


def test_map_operations():
    """Test various map operations with config-loaded maps."""
    print("\n=== Testing Map Operations ===")
    
    # Create a test layout
    test_layout = [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 2, 2, 0, 1],
        [1, 2, 4, 4, 2, 1],
        [1, 2, 4, 4, 2, 1], 
        [1, 0, 2, 2, 0, 1],
        [1, 1, 1, 1, 1, 1]
    ]
    
    map_instance = create_map_from_array(test_layout)
    
    print("Testing map operations:")
    
    # Test position queries
    test_positions = [(1.5, 1.5), (2.5, 2.5), (0.5, 0.5)]
    for x, y in test_positions:
        tile_coords = map_instance.get_tile_at_position(x, y)
        if tile_coords[0] >= 0:
            tile_type = map_instance.get_tile_type(tile_coords[0], tile_coords[1])
            center = map_instance.get_tile_center(tile_coords[0], tile_coords[1])
            print(f"  Position ({x}, {y}) -> Tile {tile_coords} -> Type {tile_type} -> Center {center}")
        else:
            print(f"  Position ({x}, {y}) -> Out of bounds")
    
    # Test bounds checking
    bounds = map_instance.get_map_boundaries()
    print(f"  Map boundaries: {bounds}")
    
    # Test save and reload
    print("\nTesting save/reload:")
    save_file = "test_map_save.json"
    map_instance.save_layout_to_file(save_file)
    
    # Reload and verify
    reloaded_map = create_map_from_file(save_file)
    print("  Original and reloaded maps match:", np.array_equal(map_instance.tiles, reloaded_map.tiles))
    
    return map_instance


def visualize_config_map(map_name: str = "simple_room"):
    """Visualize a map loaded from config file."""
    print(f"\n=== Visualizing {map_name} Map ===")
    
    try:
        # Load the map
        map_file = f"duckietown_simulator/assets/maps/{map_name}.json"
        map_instance = create_map_from_file(map_file)
        
        # Add some robots
        robots = {
            "robot1": create_duckiebot(x=1.0, y=1.0, theta=0.0),
            "robot2": create_duckiebot(x=2.0, y=2.0, theta=np.pi/2),
        }
        
        # Create renderer
        config = RenderConfig(
            width=1000, height=800, fps=60,
            use_tile_images=True,
            show_grid=True,
            show_robot_ids=True
        )
        
        renderer = create_pygame_renderer(map_instance, config)
        renderer.set_robots(robots)
        
        print(f"Map loaded: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
        print("Controls:")
        print("  I: Toggle Images/Colors")
        print("  G: Toggle Grid") 
        print("  SPACE: Pause/Resume")
        print("  R: Reset Camera")
        print("  ESC: Quit")
        print("\nPygame window should open...")
        
        # Simple robot movement
        step = 0
        dt = 0.016
        
        try:
            while renderer.render():
                if not renderer.paused:
                    # Simple circular movement for demonstration
                    t = step * dt
                    
                    # Robot 1: circular motion
                    center_x, center_y = 2.0, 2.0
                    radius = 1.0
                    new_x = center_x + radius * np.cos(t * 0.5)
                    new_y = center_y + radius * np.sin(t * 0.5)
                    new_theta = t * 0.5 + np.pi/2
                    robots["robot1"].pose[0] = new_x
                    robots["robot1"].pose[1] = new_y
                    robots["robot1"].pose[2] = new_theta
                    
                    # Robot 2: back and forth
                    new_x2 = 1.5 + 0.8 * np.sin(t * 0.8)
                    new_y2 = 1.5 + 0.3 * np.cos(t * 0.8)
                    new_theta2 = t * 0.8
                    robots["robot2"].pose[0] = new_x2
                    robots["robot2"].pose[1] = new_y2
                    robots["robot2"].pose[2] = new_theta2
                    
                    step += 1
                
                renderer.run_simulation_step()
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            renderer.cleanup()
    
    except Exception as e:
        print(f"Error visualizing map: {e}")


def demonstrate_custom_layout():
    """Demonstrate creating and using a custom layout."""
    print("\n=== Custom Layout Demonstration ===")
    
    # Create a custom layout representing a parking lot
    parking_lot = [
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 2, 2, 2, 4, 4, 2, 2, 2, 3],
        [3, 2, 1, 0, 2, 2, 0, 1, 2, 3],
        [3, 2, 1, 0, 2, 2, 0, 1, 2, 3],
        [3, 2, 2, 2, 2, 2, 2, 2, 2, 3],
        [3, 2, 1, 0, 2, 2, 0, 1, 2, 3],
        [3, 2, 1, 0, 2, 2, 0, 1, 2, 3],
        [3, 2, 2, 2, 4, 4, 2, 2, 2, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]
    
    print("Creating parking lot layout:")
    map_instance = create_map_from_array(parking_lot)
    print(map_instance)
    
    # Save it as a config file for future use
    map_instance.save_layout_to_file("duckietown_simulator/assets/maps/parking_lot.json")
    
    print("Saved parking lot layout to file!")
    
    return map_instance


if __name__ == "__main__":
    print("Config File-Based Map Layout Test")
    print("=" * 50)
    
    # Test all functionality
    array_maps = test_array_based_maps()
    file_maps = test_file_based_maps()
    test_map = test_map_operations()
    custom_map = demonstrate_custom_layout()
    
    # Show available maps
    print(f"\n=== Available Maps ===")
    print("File-based maps:")
    for name in file_maps.keys():
        print(f"  - {name}")
    
    print("\nArray-based maps created in this session:")
    print("  - simple_room (from array)")
    print("  - complex_layout (from array)")
    print("  - parking_lot (custom)")
    
    # Offer interactive visualization
    print("\n" + "=" * 50)
    
    # Try to visualize one map
    try:
        visualize_config_map("simple_room")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\nConfig-based map loading test completed!")
    print("\nYou can now:")
    print("1. Create maps using 2D arrays: create_map_from_array(layout)")
    print("2. Load maps from JSON files: create_map_from_file('path/to/file.json')")
    print("3. Save current maps: map.save_layout_to_file('filename.json')")
    print("4. Edit JSON files to customize layouts")
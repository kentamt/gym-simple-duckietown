#!/usr/bin/env python3
"""
Test script for rotated road tiles functionality.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array, create_map_from_file
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def test_basic_rotated_tiles():
    """Test basic rotated tile functionality."""
    print("=== Testing Basic Rotated Tiles ===")
    
    # Test layout with various directional road types
    # 10-13: straight roads, 20-27: curves, 30-33: intersections
    layout = [
        [3, 3, 10, 3, 3],    # Vertical straight road in middle
        [3, 20, 30, 24, 3],  # Left curve, intersection, right curve
        [11, 11, 11, 11, 11], # Horizontal straight road
        [3, 22, 31, 26, 3],  # Different rotations
        [3, 3, 12, 3, 3]     # Another vertical road
    ]
    
    print("Creating map with rotated road tiles:")
    print("Layout uses:")
    print("  10,12: Vertical straight roads")
    print("  11: Horizontal straight roads") 
    print("  20,22: Left curves (different rotations)")
    print("  24,26: Right curves (different rotations)")
    print("  30,31: Intersections (different rotations)")
    print("  3: Grass background")
    
    map_instance = create_map_from_array(layout)
    print(f"\nMap created: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    
    return map_instance


def test_tile_manager_rotated():
    """Test the tile manager with rotated tiles."""
    print("\n=== Testing Tile Manager with Rotated Tiles ===")
    
    from duckietown_simulator.rendering.tile_image_manager import TileImageManager
    
    tile_manager = TileImageManager()
    
    print("Testing tile loading:")
    
    # Test basic tiles
    for tile_type in [0, 1, 2, 3, 4]:
        image = tile_manager.get_tile_image(tile_type, (64, 64))
        status = "✓" if image else "✗"
        print(f"  Tile {tile_type}: {status}")
    
    # Test directional tiles
    directional_tiles = [10, 11, 20, 21, 22, 23, 24, 25, 30, 31]
    print("\nTesting directional tiles:")
    
    for tile_type in directional_tiles:
        image = tile_manager.get_tile_image(tile_type, (64, 64))
        status = "✓" if image else "✗"
        print(f"  Tile {tile_type}: {status}")
    
    # Show all mappings
    print(f"\nTotal tile mappings loaded: {len(tile_manager.get_tile_mappings())}")
    
    cache_info = tile_manager.get_cache_info()
    print(f"Cache info: {cache_info}")


def create_road_circuit_map():
    """Create a map demonstrating various road types in a circuit."""
    print("\n=== Creating Road Circuit Demo ===")
    
    # Create a circuit using directional road tiles
    circuit = [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],    # Grass border
        [3, 20, 10, 10, 10, 10, 24, 3, 3],  # Top: left curve, straights, right curve
        [3, 11, 3, 3, 3, 3, 11, 3, 3],    # Sides: horizontal roads
        [3, 11, 3, 30, 10, 31, 11, 3, 3],  # Middle: roads with intersections
        [3, 11, 3, 3, 3, 3, 11, 3, 3],    # Sides: horizontal roads
        [3, 22, 10, 10, 10, 10, 26, 3, 3],  # Bottom: curves and straights
        [3, 3, 3, 3, 3, 3, 3, 3, 3]     # Grass border
    ]
    
    print("Circuit layout:")
    print("  Uses curves (20,22,24,26), straights (10,11), intersections (30,31)")
    
    map_instance = create_map_from_array(circuit)
    return map_instance


def visualize_rotated_tiles(map_instance = None):
    """Visualize rotated tiles in pygame."""
    print("\n=== Visualizing Rotated Tiles ===")
    
    if map_instance is None:
        map_instance = test_basic_rotated_tiles()
    
    # Add robots
    robots = {
        "robot1": create_duckiebot(x=1.5, y=1.5, theta=0.0),
        "robot2": create_duckiebot(x=3.5, y=2.5, theta=np.pi/2),
    }
    
    # Create renderer
    config = RenderConfig(
        width=1200, height=900, fps=60,
        use_tile_images=True,
        show_grid=True,
        show_robot_ids=True
    )
    
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print("Controls:")
    print("  I: Toggle Images/Colors")
    print("  G: Toggle Grid")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  ESC: Quit")
    print("\nPygame window should open showing rotated road tiles...")
    
    # Simple robot movement
    step = 0
    dt = 0.016
    
    try:
        while renderer.render():
            if not renderer.paused:
                t = step * dt
                
                # Robot 1: Follow road pattern
                robots["robot1"].pose[0] = 2.5 + 1.5 * np.cos(t * 0.3)
                robots["robot1"].pose[1] = 2.5 + 1.0 * np.sin(t * 0.3)
                robots["robot1"].pose[2] = t * 0.3
                
                # Robot 2: Different pattern
                robots["robot2"].pose[0] = 1.5 + 0.5 * np.sin(t * 0.5)
                robots["robot2"].pose[1] = 3.5 + 0.3 * np.cos(t * 0.5)
                robots["robot2"].pose[2] = t * 0.5 + np.pi/2
                
                step += 1
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()


def create_directional_demo_map():
    """Create a comprehensive demo map showing all directional tile types."""
    
    demo_map_data = {
        "layout": [
            [3, 10, 3, 11, 11, 11, 3, 10, 3],
            [20, 30, 24, 3, 3, 3, 20, 30, 24],
            [3, 10, 3, 11, 11, 11, 3, 10, 3],
            [11, 11, 11, 30, 10, 31, 11, 11, 11],
            [3, 12, 3, 11, 11, 11, 3, 12, 3],
            [22, 31, 26, 3, 3, 3, 22, 31, 26],
            [3, 12, 3, 11, 11, 11, 3, 12, 3]
        ],
        "tile_types": {
            "0": "empty",
            "1": "obstacle",
            "2": "road",
            "3": "grass",
            "4": "intersection",
            "10": "road_straight_vertical",
            "11": "road_straight_horizontal", 
            "12": "road_straight_vertical",
            "20": "road_curve_left_0",
            "22": "road_curve_left_180",
            "24": "road_curve_right_0",
            "26": "road_curve_right_180",
            "30": "road_intersection_0",
            "31": "road_intersection_90"
        },
        "metadata": {
            "name": "Directional Roads Demo",
            "description": "Comprehensive demo of all directional road tile types",
            "width_tiles": 9,
            "height_tiles": 7,
            "tile_size": 0.61
        }
    }
    
    # Save demo map
    import json
    map_file = "duckietown_simulator/assets/maps/directional_demo.json"
    with open(map_file, 'w') as f:
        json.dump(demo_map_data, f, indent=2)
    
    print(f"Created directional demo map: {map_file}")
    
    return create_map_from_file(map_file)


if __name__ == "__main__":
    print("Rotated Road Tiles Test")
    print("=" * 40)
    
    # Test tile manager functionality
    test_tile_manager_rotated()
    
    # Test basic rotated tiles
    basic_map = test_basic_rotated_tiles()
    
    # Create road circuit
    circuit_map = create_road_circuit_map()
    
    # Create comprehensive demo map
    demo_map = create_directional_demo_map()
    
    print("\n" + "=" * 40)
    print("Available test maps:")
    print("1. Basic rotated tiles test")
    print("2. Road circuit demo")
    print("3. Comprehensive directional demo")
    
    # Visualize the comprehensive demo
    try:
        visualize_rotated_tiles(demo_map)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n" + "=" * 40)
    print("Rotated tiles test completed!")
    print("\nDirectional tile types now available:")
    print("Basic: 0-4 (empty, obstacle, road, grass, intersection)")
    print("Straight roads: 10-13 (vertical/horizontal)")
    print("Left curves: 20-23 (4 rotations)")
    print("Right curves: 24-27 (4 rotations)")  
    print("Intersections: 30-33 (4 rotations)")
    print("\nUse these numbers in your map layouts for proper road orientation!")
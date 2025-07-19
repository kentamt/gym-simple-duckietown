#!/usr/bin/env python3
"""
Demo of rotated road tiles system.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def create_demo_road_network():
    """Create a demo road network showing various rotated tiles."""
    
    # Create a road network with proper orientations
    # 10,12: Vertical roads (North-South)
    # 11,13: Horizontal roads (East-West)  
    # 20-23: Left curves (different rotations)
    # 24-27: Right curves (different rotations)
    # 30-33: Intersections (different rotations)
    # 3: Grass background
    
    road_network = [
        [20, 11, 11, 24],
        [10, 3,  3,  10],
        [30, 11, 11, 30],
        [10, 3,  3,  10],
        [26, 11, 11, 27]
    ]
    
    print("Creating road network with rotated tiles:")
    print("  Vertical roads: 10, 12")
    print("  Horizontal roads: 11, 13")
    print("  Left curves: 20-23 (N→W, E→N, S→E, W→S)")
    print("  Right curves: 24-27 (N→E, E→S, S→W, W→N)")
    print("  Intersections: 30-33 (4 rotations)")
    
    return create_map_from_array(road_network)


def create_simple_circuit():
    """Create a simple racing circuit using rotated tiles."""
    
    # Simple oval track
    circuit = [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 20, 11, 11, 11, 11, 24, 3, 3], # Top: left curve, straights, right curve
        [3, 10, 3, 3, 3, 3, 10, 3, 3],     # Sides
        [3, 10, 3, 3, 3, 3, 10, 3, 3],     # Sides
        [3, 22, 11, 11, 11, 11, 26, 3, 3], # Bottom: curves and straights
        [3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]
    
    print("Creating simple racing circuit:")
    print("  Oval track using curves and straight roads")
    
    return create_map_from_array(circuit)


def visualize_rotated_roads():
    """Visualize the rotated roads demo."""
    print("\n=== Rotated Roads Visualization ===")
    
    # Create the demo map
    map_instance = create_demo_road_network()
    
    # Add robots
    robots = {
        "robot1": create_duckiebot(x=2.0, y=2.0, theta=0.0),
        "robot2": create_duckiebot(x=6.0, y=4.0, theta=np.pi/2),
        "robot3": create_duckiebot(x=4.0, y=6.0, theta=np.pi),
    }
    
    # Create renderer
    config = RenderConfig(
        width=1400, height=1000, fps=60,
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
    print("\nDemo shows rotated road tiles with proper orientations!")
    
    # Simple robot movement
    step = 0
    dt = 0.016
    
    try:
        while renderer.render():
            if not renderer.paused:
                t = step * dt
                
                # Robot 1: Circle around center
                center_x, center_y = 5.5, 4.5
                radius = 2.0
                robots["robot1"].pose[0] = center_x + radius * np.cos(t * 0.2)
                robots["robot1"].pose[1] = center_y + radius * np.sin(t * 0.2)
                robots["robot1"].pose[2] = t * 0.2 + np.pi/2
                
                # Robot 2: Move along roads
                robots["robot2"].pose[0] = 3.0 + 2.0 * np.sin(t * 0.3)
                robots["robot2"].pose[1] = 4.0 + 1.5 * np.cos(t * 0.3)
                robots["robot2"].pose[2] = t * 0.3
                
                # Robot 3: Different pattern
                robots["robot3"].pose[0] = 7.0 + 1.0 * np.cos(t * 0.4)
                robots["robot3"].pose[1] = 6.0 + 1.0 * np.sin(t * 0.4)
                robots["robot3"].pose[2] = -t * 0.4
                
                step += 1
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()


def demonstrate_tile_types():
    """Demonstrate all available tile types."""
    print("\n=== Available Tile Types ===")
    
    tile_descriptions = {
        # Basic tiles (0-9)
        0: "Empty/Floor",
        1: "Obstacle/Wall", 
        2: "Basic Road",
        3: "Grass",
        4: "Basic Intersection",
        
        # Straight roads (10-19)
        10: "Straight Road (Vertical N-S)",
        11: "Straight Road (Horizontal E-W)",
        12: "Straight Road (Vertical N-S)",
        13: "Straight Road (Horizontal E-W)",
        
        # Left curves (20-29)
        20: "Left Curve (North → West)",
        21: "Left Curve (East → North)",
        22: "Left Curve (South → East)",
        23: "Left Curve (West → South)",
        
        # Right curves (24-27 within 20-29 range)
        24: "Right Curve (North → East)",
        25: "Right Curve (East → South)",
        26: "Right Curve (South → West)",
        27: "Right Curve (West → North)",
        
        # Intersections (30-39)
        30: "Intersection (0° rotation)",
        31: "Intersection (90° rotation)",
        32: "Intersection (180° rotation)",
        33: "Intersection (270° rotation)",
    }
    
    print("Tile Type Reference:")
    for tile_id, description in tile_descriptions.items():
        print(f"  {tile_id:2d}: {description}")
    
    print("\nExample layouts:")
    print("Simple crossroad:")
    print("  [[3,10,3],")
    print("   [11,30,11],") 
    print("   [3,12,3]]")
    
    print("\nLeft turn:")
    print("  [[3,10,3],")
    print("   [3,20,11],")
    print("   [3,3,3]]")


if __name__ == "__main__":
    print("Rotated Road Tiles Demo")
    print("=" * 40)
    
    # Show available tile types
    demonstrate_tile_types()
    
    # Create demo maps
    road_network = create_demo_road_network()
    circuit = create_simple_circuit()
    
    print(f"\nDemo maps created:")
    print(f"  Road network: {road_network.width_tiles}x{road_network.height_tiles}")
    print(f"  Circuit: {circuit.width_tiles}x{circuit.height_tiles}")
    
    # Run visualization
    try:
        visualize_rotated_roads()
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n" + "=" * 40)
    print("Rotated roads demo completed!")
    print("\nKey Features:")
    print("✓ Rotated road tiles (0°, 90°, 180°, 270°)")
    print("✓ Directional curves and intersections")
    print("✓ Proper road orientation in layouts")
    print("✓ Easy-to-use tile numbering system")
    print("✓ Full integration with existing renderer")
    print("\nYou can now create realistic road networks using numbered tiles!")
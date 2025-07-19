#!/usr/bin/env python3
"""
Debug script to investigate pygame scaling and coordinate issues.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def debug_scaling():
    """Debug the scaling and coordinate system."""
    print("=== Debugging Pygame Scaling ===")
    
    # Create a small map for easier analysis
    map_instance = create_map_from_config(4, 4, "straight")
    
    print(f"Map info:")
    print(f"  Tiles: {map_instance.width_tiles}x{map_instance.height_tiles}")
    print(f"  Tile size: {map_instance.tile_size}m")
    print(f"  Map size: {map_instance.width_meters:.2f}m x {map_instance.height_meters:.2f}m")
    
    # Create one robot in a known position
    robot = create_duckiebot(x=1.0, y=1.0, theta=0.0)
    print(f"\nRobot info:")
    print(f"  Position: ({robot.x:.3f}, {robot.y:.3f})")
    print(f"  Collision radius: {robot.collision_radius:.3f}m")
    print(f"  Wheelbase: {robot.config.wheelbase:.3f}m")
    
    # Calculate expected sizes
    print(f"\nExpected relative sizes:")
    print(f"  Tile size: {map_instance.tile_size:.3f}m (61cm)")
    print(f"  Robot collision radius: {robot.collision_radius:.3f}m (5cm)")
    print(f"  Robot size as % of tile: {(robot.collision_radius * 2 / map_instance.tile_size) * 100:.1f}%")
    print(f"  Robot should be about 1/6 the size of a tile")
    
    # Create renderer
    config = RenderConfig(width=800, height=600, fps=30)
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots({"robot1": robot})
    
    print(f"\nRenderer info:")
    print(f"  Screen size: {config.width}x{config.height}")
    print(f"  Initial zoom: {renderer.zoom:.3f}")
    
    # Calculate screen coordinates
    screen_pos = renderer.world_to_screen(robot.x, robot.y)
    print(f"  Robot screen position: {screen_pos}")
    
    # Calculate screen sizes
    tile_screen_size = renderer.world_distance_to_screen(map_instance.tile_size)
    robot_collision_screen_size = renderer.world_distance_to_screen(robot.collision_radius * 2)
    
    print(f"  Tile screen size: {tile_screen_size} pixels")
    print(f"  Robot collision screen size: {robot_collision_screen_size} pixels")
    
    # Test coordinate conversion
    print(f"\nTesting coordinate conversion:")
    test_positions = [
        (0.0, 0.0),  # Map corner
        (map_instance.width_meters/2, map_instance.height_meters/2),  # Map center
        (map_instance.width_meters, map_instance.height_meters),  # Map opposite corner
    ]
    
    for world_x, world_y in test_positions:
        screen_x, screen_y = renderer.world_to_screen(world_x, world_y)
        back_x, back_y = renderer.screen_to_world(screen_x, screen_y)
        print(f"  World ({world_x:.2f}, {world_y:.2f}) -> Screen ({screen_x}, {screen_y}) -> World ({back_x:.2f}, {back_y:.2f})")
    
    # Run a quick visualization
    print(f"\nStarting debug visualization...")
    print(f"Watch the robot size relative to tiles. It should be much smaller!")
    print(f"Controls: SPACE=pause, R=reset camera, ESC=quit")
    
    step = 0
    try:
        while renderer.render() and step < 300:  # Auto-quit after 300 frames
            step += 1
            
            # Don't move the robot, just observe the scaling
            if step % 60 == 0:
                print(f"  Frame {step}: Robot at ({robot.x:.3f}, {robot.y:.3f})")
            
            renderer.run_simulation_step(update_trajectories=False)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("Debug scaling completed")


if __name__ == "__main__":
    debug_scaling()
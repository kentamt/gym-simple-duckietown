#!/usr/bin/env python3
"""
Quick verification of robot size in pygame renderer.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def verify_robot_size():
    """Verify robot appears at correct size."""
    print("=== Verifying Robot Size ===")
    
    # Create map and robot
    map_instance = create_map_from_config(6, 6, "straight")
    robot = create_duckiebot(x=1.8, y=1.8, theta=0.0)  # Center of a tile
    
    # Create renderer
    config = RenderConfig(width=800, height=600, fps=30)
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots({"robot1": robot})
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles, {map_instance.width_meters:.2f}m total")
    print(f"Robot at tile center: ({robot.x:.2f}, {robot.y:.2f})")
    print(f"Tile size: {map_instance.tile_size:.3f}m ({map_instance.tile_size*100:.0f}cm)")
    print(f"Robot collision radius: {robot.collision_radius:.3f}m ({robot.collision_radius*100:.0f}cm)")
    print(f"Robot should be {(robot.collision_radius*2/map_instance.tile_size)*100:.1f}% of tile size")
    
    # Calculate screen sizes
    tile_pixels = renderer.world_distance_to_screen(map_instance.tile_size)
    robot_pixels = renderer.world_distance_to_screen(robot.collision_radius * 2)
    robot_body_length = renderer.world_distance_to_screen(0.18)  # 18cm
    robot_body_width = renderer.world_distance_to_screen(0.10)   # 10cm
    
    print(f"\nScreen rendering sizes:")
    print(f"  Tile: {tile_pixels} pixels")
    print(f"  Robot collision circle: {robot_pixels} pixels")
    print(f"  Robot body: {robot_body_length}x{robot_body_width} pixels")
    print(f"  Robot body as % of tile: {(robot_body_length/tile_pixels)*100:.1f}% length, {(robot_body_width/tile_pixels)*100:.1f}% width")
    
    print(f"\nRunning visual test for 3 seconds...")
    print(f"You should see:")
    print(f"  - 6x6 grid of light gray tiles")
    print(f"  - Small colored rectangle (robot) in center area")
    print(f"  - Robot should be much smaller than tile")
    print(f"  - Orange circle around robot (collision radius)")
    
    frame_count = 0
    max_frames = 90  # 3 seconds at 30 FPS
    
    try:
        while renderer.render() and frame_count < max_frames:
            frame_count += 1
            
            if frame_count % 30 == 0:
                seconds = frame_count // 30
                print(f"  {seconds}/3 seconds elapsed...")
            
            renderer.run_simulation_step(update_trajectories=False)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("Size verification completed!")
    print("\nExpected results:")
    print("  ✓ Robot should appear as small rectangle, much smaller than grid tiles")
    print("  ✓ Robot body should be about 30% of tile length, 16% of tile width")
    print("  ✓ Collision circle should be about 16% of tile size")


if __name__ == "__main__":
    verify_robot_size()
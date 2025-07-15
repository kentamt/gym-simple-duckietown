#!/usr/bin/env python3
"""
Test script for image-based tile rendering in pygame.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.world.collision_detection import create_collision_detector
from duckietown_simulator.world.obstacles import ObstacleConfig, ObstacleType
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def test_image_tiles():
    """Test pygame renderer with image-based tiles."""
    print("=== Testing Image-Based Tile Rendering ===")
    print("Controls:")
    print("  I: Toggle between Images and Colors")
    print("  G: Toggle Grid")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  Mouse: Pan and Zoom")
    print("  ESC: Quit")
    print("\nStarting image tile test...")
    
    # Create a more interesting map layout
    map_instance = create_map_from_config(8, 6, "loop")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Add some obstacles
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=2.0, y=2.0,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.3,
        name="circle_obstacle"
    ))
    
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=4.5, y=3.0,
        obstacle_type=ObstacleType.RECTANGLE,
        width=0.8, height=0.4, rotation=np.pi/6,
        name="rect_obstacle"
    ))
    
    # Create robots
    robots = {
        "robot1": create_duckiebot(x=1.0, y=1.0, theta=0.0),
        "robot2": create_duckiebot(x=2.5, y=3.5, theta=np.pi/2),
        "robot3": create_duckiebot(x=4.0, y=1.5, theta=np.pi),
    }
    
    # Create renderer with image support enabled
    config = RenderConfig(
        width=1200, height=900, fps=60,
        use_tile_images=True,  # Enable image rendering
        show_grid=True,
        show_collision_circles=True,
        show_robot_ids=True
    )
    
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(detector.obstacle_manager)
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print(f"Tile images loaded: {renderer.tile_manager is not None}")
    
    if renderer.tile_manager:
        cache_info = renderer.tile_manager.get_cache_info()
        print(f"Image cache: {cache_info}")
        print(f"Available images: {renderer.tile_manager.list_available_images()}")
        print(f"Tile mappings: {renderer.tile_manager.get_tile_mappings()}")
    
    # Simulation parameters
    step = 0
    collision_count = 0
    dt = 0.05
    
    print("\nPygame window should open showing image-based tiles!")
    print("Try pressing 'I' to toggle between images and solid colors.")
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Move robots in patterns
                actions = {
                    "robot1": np.array([3.0, 3.5]),    # Slight right curve
                    "robot2": np.array([2.5, 4.0]),    # Left curve
                    "robot3": np.array([4.0, 2.5]),    # Right curve
                }
                
                # Apply actions
                for robot_id, action in actions.items():
                    robots[robot_id].step(action, dt)
                
                # Check collisions
                collisions = detector.check_all_collisions(robots)
                renderer.set_collision_results(collisions)
                
                if collisions:
                    collision_count += len(collisions)
                
                step += 1
                
                # Reset robots if they go out of bounds
                for robot_id, robot in robots.items():
                    if not map_instance.is_position_in_bounds(robot.x, robot.y):
                        if robot_id == "robot1":
                            robot.reset(x=1.0, y=1.0, theta=0.0)
                        elif robot_id == "robot2":
                            robot.reset(x=2.5, y=3.5, theta=np.pi/2)
                        elif robot_id == "robot3":
                            robot.reset(x=4.0, y=1.5, theta=np.pi)
                
                # Print status every 120 frames (2 seconds)
                if step % 120 == 0:
                    mode = "images" if config.use_tile_images else "colors"
                    print(f"Step: {step}, Mode: {mode}, Collisions: {collision_count}")
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print(f"Image tile test completed. Total collisions: {collision_count}")


def test_tile_image_features():
    """Test specific tile image features."""
    print("\n=== Testing Tile Image Features ===")
    
    # Test the tile manager directly
    from duckietown_simulator.rendering.tile_image_manager import TileImageManager
    
    tile_manager = TileImageManager()
    
    print("Tile Image Manager Test:")
    print(f"  Available images: {tile_manager.list_available_images()}")
    print(f"  Tile mappings: {tile_manager.get_tile_mappings()}")
    
    # Test image loading
    for tile_type in [0, 1, 2]:
        image = tile_manager.get_tile_image(tile_type, (64, 64))
        if image:
            print(f"  Tile type {tile_type}: {image.get_size()} pixels")
        else:
            print(f"  Tile type {tile_type}: Failed to load")
    
    # Test cache
    cache_info = tile_manager.get_cache_info()
    print(f"  Cache info: {cache_info}")
    
    # Test fallback for unknown tile type
    fallback = tile_manager.get_tile_image(99, (32, 32))
    if fallback:
        print(f"  Fallback tile: {fallback.get_size()} pixels")
    
    print("Tile image features test completed!")


if __name__ == "__main__":
    print("Image-Based Tile Rendering Test")
    print("=" * 40)
    
    # Test tile manager features first
    test_tile_image_features()
    
    # Then test the full rendering
    test_image_tiles()
    
    print("\n" + "=" * 40)
    print("All image tile tests completed!")
    print("\nTo use your own tile images:")
    print("1. Replace images in duckietown_simulator/assets/tiles/")
    print("2. Update tile_mapping.ini to map tile types to your images")
    print("3. Images should be square (recommended: 256x256 pixels)")
    print("4. Supported formats: PNG, JPG, BMP, GIF")
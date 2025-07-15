#!/usr/bin/env python3
"""
Simple test for pygame renderer - just basic functionality.
"""

import sys
import numpy as np
import time
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.world.collision_detection import create_collision_detector
from duckietown_simulator.world.obstacles import ObstacleConfig, ObstacleType
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def main():
    """Simple pygame renderer test."""
    print("=== Simple Pygame Renderer Test ===")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  T: Toggle Trajectories")
    print("  C: Clear Trajectories")
    print("  G: Toggle Grid")
    print("  Mouse: Pan and Zoom")
    print("  ESC: Quit")
    print("\nStarting simple rendering test...")
    
    # Create map and collision detector
    map_instance = create_map_from_config(6, 6, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Add some obstacles
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=2.0, y=2.0,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.3,
        name="obstacle_1"
    ))
    
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=4.0, y=3.0,
        obstacle_type=ObstacleType.RECTANGLE,
        width=0.6, height=0.4, rotation=np.pi/4,
        name="obstacle_2"
    ))
    
    # Create robots
    robots = {
        "robot1": create_duckiebot(x=1.0, y=1.0, theta=0.0),
        "robot2": create_duckiebot(x=3.0, y=1.0, theta=np.pi/2),
        "robot3": create_duckiebot(x=1.0, y=3.0, theta=np.pi),
    }
    
    # Create renderer
    config = RenderConfig(width=1000, height=800, fps=60)
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(detector.obstacle_manager)
    
    # Simple animation loop
    step = 0
    collision_count = 0
    dt = 0.05
    
    print("Pygame window should open now. Close it or press ESC to exit.")
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Simple robot movement
                actions = {
                    "robot1": np.array([3.0, 3.0]),    # Forward
                    "robot2": np.array([2.0, 4.0]),    # Curve right
                    "robot3": np.array([4.0, 2.0]),    # Curve left
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
                            robot.reset(x=3.0, y=1.0, theta=np.pi/2)
                        elif robot_id == "robot3":
                            robot.reset(x=1.0, y=3.0, theta=np.pi)
                
                # Print status every 60 frames
                if step % 60 == 0:
                    print(f"Step: {step}, Collisions: {collision_count}, "
                          f"Robot positions: "
                          f"R1({robots['robot1'].x:.1f},{robots['robot1'].y:.1f}) "
                          f"R2({robots['robot2'].x:.1f},{robots['robot2'].y:.1f}) "
                          f"R3({robots['robot3'].x:.1f},{robots['robot3'].y:.1f})")
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print(f"Simple rendering test completed. Total collisions detected: {collision_count}")


if __name__ == "__main__":
    # Check if pygame is available
    try:
        import pygame
        print("Pygame is available, running simple test...")
        main()
    except ImportError:
        print("Pygame is not installed. Please install it with: pip install pygame")
        sys.exit(1)
    
    print("Simple pygame renderer test completed!")
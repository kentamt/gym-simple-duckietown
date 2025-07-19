#!/usr/bin/env python3
"""
Test script for visualizing collision detection during movement with image tiles.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.world.collision_detection import create_collision_detector
from duckietown_simulator.world.obstacles import ObstacleConfig, ObstacleType
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def test_collision_during_movement():
    """Test collision detection visualization during robot movement."""
    print("=== Testing Collision Detection During Movement ===")
    print("Controls:")
    print("  I: Toggle between Images and Colors")
    print("  G: Toggle Grid")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  Mouse: Pan and Zoom")
    print("  ESC: Quit")
    print("\nThis test shows robots moving and colliding with obstacles and each other.")
    print("Watch for red X markers at collision points!\n")
    
    # Create map with interesting layout
    map_instance = create_map_from_config(10, 8, "loop")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=True)
    
    # Add strategic obstacles to create collision scenarios
    obstacles = [
        ObstacleConfig(
            x=3.0, y=2.0,
            obstacle_type=ObstacleType.CIRCLE,
            radius=0.4,
            name="central_circle"
        ),
        ObstacleConfig(
            x=5.5, y=4.0,
            obstacle_type=ObstacleType.RECTANGLE,
            width=0.6, height=1.2, rotation=np.pi/4,
            name="diagonal_barrier"
        ),
        ObstacleConfig(
            x=7.0, y=1.5,
            obstacle_type=ObstacleType.CIRCLE,
            radius=0.3,
            name="corner_obstacle"
        ),
        ObstacleConfig(
            x=2.5, y=5.0,
            obstacle_type=ObstacleType.RECTANGLE,
            width=1.0, height=0.3, rotation=0,
            name="horizontal_wall"
        )
    ]
    
    for obstacle in obstacles:
        detector.obstacle_manager.add_obstacle(obstacle)
    
    # Create robots with collision-prone paths
    robots = {
        "robot1": create_duckiebot(x=1.0, y=1.0, theta=0.0),
        "robot2": create_duckiebot(x=8.0, y=1.0, theta=np.pi),
        "robot3": create_duckiebot(x=1.0, y=6.0, theta=np.pi/2),
        "robot4": create_duckiebot(x=8.0, y=6.0, theta=-np.pi/2),
        "robot5": create_duckiebot(x=4.5, y=3.5, theta=np.pi/4),
    }
    
    # Create renderer with image tiles
    config = RenderConfig(
        width=1400, height=1000, fps=60,
        use_tile_images=True,
        show_grid=True,
        show_collision_circles=True,
        show_robot_ids=True
    )
    
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(detector.obstacle_manager)
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print(f"Robots: {len(robots)}")
    print(f"Obstacles: {len(obstacles)}")
    
    if renderer.tile_manager:
        cache_info = renderer.tile_manager.get_cache_info()
        print(f"Tile images loaded: {cache_info}")
    
    # Movement patterns designed to cause collisions
    movement_patterns = {
        "robot1": {"base_speed": 2.0, "curve": 0.5, "direction": 1},
        "robot2": {"base_speed": 2.5, "curve": -0.3, "direction": -1},
        "robot3": {"base_speed": 1.8, "curve": 0.7, "direction": 1},
        "robot4": {"base_speed": 2.2, "curve": -0.6, "direction": -1},
        "robot5": {"base_speed": 1.5, "curve": 0.0, "direction": 1},
    }
    
    # Simulation state
    step = 0
    collision_count = 0
    total_collisions = 0
    dt = 0.016  # 60 FPS
    
    print("\nStarting collision visualization...")
    print("Robots will move in patterns designed to cause collisions!")
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Generate movement actions
                actions = {}
                for robot_id, robot in robots.items():
                    pattern = movement_patterns[robot_id]
                    
                    # Calculate time-based movement
                    t = step * dt
                    base_speed = pattern["base_speed"]
                    curve = pattern["curve"]
                    direction = pattern["direction"]
                    
                    # Add some variation to make movements more interesting
                    speed_var = 0.5 * np.sin(t * 2.0 + hash(robot_id) % 100 * 0.1)
                    curve_var = 0.3 * np.cos(t * 1.5 + hash(robot_id) % 100 * 0.05)
                    
                    left_speed = base_speed + direction * (curve + curve_var) + speed_var
                    right_speed = base_speed - direction * (curve + curve_var) + speed_var
                    
                    actions[robot_id] = np.array([left_speed, right_speed])
                
                # Apply actions
                for robot_id, action in actions.items():
                    robots[robot_id].step(action, dt)
                
                # Check for collisions
                collisions = detector.check_all_collisions(robots)
                renderer.set_collision_results(collisions)
                
                # Track collision statistics
                if collisions:
                    collision_count += len(collisions)
                    total_collisions += len(collisions)
                    
                    # Print collision details
                    for collision in collisions:
                        if collision.collision_type == "robot_robot":
                            print(f"Robot-Robot collision: {collision.robot_id} vs {collision.other_robot_id} at ({collision.collision_point[0]:.2f}, {collision.collision_point[1]:.2f})")
                        elif collision.collision_type == "robot_obstacle":
                            print(f"Robot-Obstacle collision: {collision.robot_id} vs {collision.obstacle_name} at ({collision.collision_point[0]:.2f}, {collision.collision_point[1]:.2f})")
                        elif collision.collision_type == "robot_boundary":
                            print(f"Robot-Boundary collision: {collision.robot_id} at ({collision.collision_point[0]:.2f}, {collision.collision_point[1]:.2f})")
                
                step += 1
                
                # Reset robots if they go way out of bounds
                for robot_id, robot in robots.items():
                    if (robot.x < -1.0 or robot.x > map_instance.width_meters + 1.0 or
                        robot.y < -1.0 or robot.y > map_instance.height_meters + 1.0):
                        # Reset to a safe position
                        safe_positions = {
                            "robot1": (1.0, 1.0, 0.0),
                            "robot2": (8.0, 1.0, np.pi),
                            "robot3": (1.0, 6.0, np.pi/2),
                            "robot4": (8.0, 6.0, -np.pi/2),
                            "robot5": (4.5, 3.5, np.pi/4),
                        }
                        pos = safe_positions.get(robot_id, (2.0, 2.0, 0.0))
                        robot.reset(x=pos[0], y=pos[1], theta=pos[2])
                        print(f"Reset {robot_id} to safe position")
                
                # Print status every 3 seconds
                if step % 180 == 0:
                    mode = "images" if config.use_tile_images else "colors"
                    print(f"Step: {step}, Mode: {mode}, Active collisions: {len(collisions)}, Total collisions: {total_collisions}")
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print(f"\nCollision test completed!")
    print(f"Total collisions detected: {total_collisions}")
    print(f"Steps simulated: {step}")
    if step > 0:
        print(f"Average collisions per second: {total_collisions / (step * dt):.2f}")


def test_collision_scenarios():
    """Test specific collision scenarios."""
    print("\n=== Testing Specific Collision Scenarios ===")
    
    # Test head-on collision
    print("\n1. Head-on collision test:")
    map_instance = create_map_from_config(5, 3, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Two robots facing each other
    robot1 = create_duckiebot(x=1.0, y=1.5, theta=0.0)
    robot2 = create_duckiebot(x=4.0, y=1.5, theta=np.pi)
    
    robots = {"robot1": robot1, "robot2": robot2}
    
    # Simulate head-on collision
    for i in range(50):
        robot1.step(np.array([2.0, 2.0]), 0.02)  # Move right
        robot2.step(np.array([2.0, 2.0]), 0.02)  # Move left
        
        collisions = detector.check_all_collisions(robots)
        if collisions:
            print(f"  Head-on collision at step {i}")
            print(f"  Robot1 position: ({robot1.x:.2f}, {robot1.y:.2f})")
            print(f"  Robot2 position: ({robot2.x:.2f}, {robot2.y:.2f})")
            print(f"  Collision point: ({collisions[0].collision_point[0]:.2f}, {collisions[0].collision_point[1]:.2f})")
            break
    
    # Test robot-obstacle collision
    print("\n2. Robot-obstacle collision test:")
    detector_obs = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=True)
    detector_obs.obstacle_manager.add_obstacle(ObstacleConfig(
        x=2.5, y=1.5,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.5,
        name="test_obstacle"
    ))
    
    robot3 = create_duckiebot(x=1.0, y=1.5, theta=0.0)
    robots_obs = {"robot3": robot3}
    
    for i in range(30):
        robot3.step(np.array([1.5, 1.5]), 0.02)  # Move toward obstacle
        
        collisions = detector_obs.check_all_collisions(robots_obs)
        if collisions:
            print(f"  Robot-obstacle collision at step {i}")
            print(f"  Robot position: ({robot3.x:.2f}, {robot3.y:.2f})")
            print(f"  Collision point: ({collisions[0].collision_point[0]:.2f}, {collisions[0].collision_point[1]:.2f})")
            break
    
    print("Collision scenarios test completed!")


if __name__ == "__main__":
    print("Collision Detection Visualization Test")
    print("=" * 50)
    
    # Test specific collision scenarios first
    test_collision_scenarios()
    
    # Then run the full interactive visualization
    test_collision_during_movement()
    
    print("\n" + "=" * 50)
    print("All collision visualization tests completed!")
    print("\nFeatures demonstrated:")
    print("✓ Real-time collision detection")
    print("✓ Visual collision markers (red X)")
    print("✓ Robot-robot collisions")
    print("✓ Robot-obstacle collisions")
    print("✓ Image-based tile rendering")
    print("✓ Interactive controls (pan, zoom, pause)")
    print("✓ Collision statistics")
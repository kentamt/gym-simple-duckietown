#!/usr/bin/env python3
"""
Test script for pygame real-time renderer.
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
from duckietown_simulator.robot.pid_controller import (
    WaypointFollowPIDController, create_default_pid_config, create_waypoint_trajectory
)


def test_basic_pygame_rendering():
    """Test basic pygame rendering with static robots."""
    print("=== Testing Basic Pygame Rendering ===")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  T: Toggle Trajectories")
    print("  C: Clear Trajectories")
    print("  G: Toggle Grid")
    print("  Mouse: Pan and Zoom")
    print("  ESC: Quit")
    print("\nStarting basic rendering test...")
    
    # Create map and robots
    map_instance = create_map_from_config(6, 6, "straight")
    
    robots = {
        "robot1": create_duckiebot(x=1.0, y=1.0, theta=0.0),
        "robot2": create_duckiebot(x=3.0, y=3.0, theta=np.pi/2),
        "robot3": create_duckiebot(x=2.0, y=4.0, theta=np.pi),
    }
    
    # Create renderer
    config = RenderConfig(width=1000, height=800, fps=60)
    renderer = create_pygame_renderer(map_instance, config)
    
    # Set robots
    renderer.set_robots(robots)
    
    # Simple animation loop
    step = 0
    try:
        while renderer.render():
            # Simple robot movement
            if not renderer.paused:
                # Move robots in simple patterns
                robots["robot1"].step(np.array([2.0, 2.0]), 0.05)  # Forward
                robots["robot2"].step(np.array([1.0, 3.0]), 0.05)  # Turn left
                robots["robot3"].step(np.array([3.0, 1.0]), 0.05)  # Turn right
                
                step += 1
                
                # Reset robots if they go too far
                if step % 200 == 0:
                    robots["robot1"].reset(x=1.0, y=1.0, theta=0.0)
                    robots["robot2"].reset(x=3.0, y=3.0, theta=np.pi/2)
                    robots["robot3"].reset(x=2.0, y=4.0, theta=np.pi)
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("Basic rendering test completed")


def test_collision_detection_rendering():
    """Test pygame rendering with collision detection."""
    print("\n=== Testing Collision Detection Rendering ===")
    print("Watch for collision markers (red X) and collision circles!")
    print("This test will run for about 30 seconds or until you close the window.")
    
    # Create map with obstacles
    map_instance = create_map_from_config(8, 8, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Add some obstacles
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=2.5, y=2.5,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.4,
        name="obstacle_1"
    ))
    
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=4.0, y=2.0,
        obstacle_type=ObstacleType.RECTANGLE,
        width=0.8, height=0.5, rotation=np.pi/6,
        name="obstacle_2"
    ))
    
    # Create robots
    robots = {
        "robot1": create_duckiebot(x=0.5, y=2.5, theta=0.0),
        "robot2": create_duckiebot(x=3.5, y=1.5, theta=np.pi/3),
        "robot3": create_duckiebot(x=1.0, y=3.5, theta=np.pi/2),
        "robot4": create_duckiebot(x=3.0, y=3.5, theta=-np.pi/4),
    }
    
    # Create renderer
    config = RenderConfig(width=1200, height=900, fps=60)
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(detector.obstacle_manager)
    
    # Simulation parameters
    dt = 0.05
    step = 0
    collision_count = 0
    start_time = time.time()
    
    try:
        while renderer.render():
            current_time = time.time()
            
            # Run for 30 seconds
            if current_time - start_time > 30:
                break
            
            if not renderer.paused:
                # Move robots with different control strategies
                actions = {
                    "robot1": np.array([4.0, 4.0]),    # Straight forward
                    "robot2": np.array([3.0, 5.0]),    # Slight curve right
                    "robot3": np.array([5.0, 3.0]),    # Slight curve left
                    "robot4": np.array([3.5, 3.5]),    # Straight forward
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
                
                # Reset robots periodically to keep them in bounds
                if step % 300 == 0:
                    robots["robot1"].reset(x=0.5, y=2.5, theta=0.0)
                    robots["robot2"].reset(x=3.5, y=1.5, theta=np.pi/3)
                    robots["robot3"].reset(x=1.0, y=3.5, theta=np.pi/2)
                    robots["robot4"].reset(x=3.0, y=3.5, theta=-np.pi/4)
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print(f"Collision detection test completed. Total collisions detected: {collision_count}")


def test_pid_control_rendering():
    """Test pygame rendering with PID controlled robots."""
    print("\n=== Testing PID Control Rendering ===")
    print("Watch robots follow predefined waypoint trajectories!")
    print("Green dots = waypoints, Colored paths = robot trajectories")
    
    # Create map
    map_instance = create_map_from_config(10, 10, "straight")
    
    # Create robots with PID controllers
    robots = {}
    controllers = {}
    waypoint_sets = {}
    
    # Robot 1: Square trajectory
    robots["robot1"] = create_duckiebot(x=1.0, y=1.0, theta=0.0)
    controllers["robot1"] = WaypointFollowPIDController(create_default_pid_config())
    waypoint_sets["robot1"] = create_waypoint_trajectory(
        "square", map_instance.width_meters, map_instance.height_meters, margin=1.0
    )
    controllers["robot1"].set_waypoints(waypoint_sets["robot1"])
    
    # Robot 2: Spiral trajectory
    robots["robot2"] = create_duckiebot(x=2.0, y=2.0, theta=0.0)
    controllers["robot2"] = WaypointFollowPIDController(create_default_pid_config())
    waypoint_sets["robot2"] = create_waypoint_trajectory(
        "spiral", map_instance.width_meters, map_instance.height_meters, margin=1.0
    )
    controllers["robot2"].set_waypoints(waypoint_sets["robot2"])
    
    # Robot 3: Line trajectory
    robots["robot3"] = create_duckiebot(x=1.0, y=5.0, theta=0.0)
    controllers["robot3"] = WaypointFollowPIDController(create_default_pid_config())
    waypoint_sets["robot3"] = create_waypoint_trajectory(
        "line", map_instance.width_meters, map_instance.height_meters, margin=1.0
    )
    controllers["robot3"].set_waypoints(waypoint_sets["robot3"])
    
    # Create renderer
    config = RenderConfig(width=1200, height=900, fps=60, show_collision_circles=False)
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    
    # Custom waypoint rendering (we'll draw waypoints manually)
    import pygame
    
    dt = 0.05
    step = 0
    completed_robots = set()
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Update each robot with PID control
                for robot_id, robot in robots.items():
                    if robot_id not in completed_robots:
                        controller = controllers[robot_id]
                        
                        # Compute PID control
                        linear_vel, angular_vel, info = controller.compute_control(
                            robot.x, robot.y, robot.theta, dt
                        )
                        
                        # Convert to wheel speeds
                        omega_l, omega_r = controller.body_vel_to_wheel_speeds(
                            linear_vel, angular_vel, robot.config.wheelbase, robot.config.wheel_radius
                        )
                        
                        # Apply control
                        action = np.array([omega_l, omega_r])
                        robot.step(action, dt)
                        
                        # Check if completed
                        if info['status'] == 'completed':
                            completed_robots.add(robot_id)
                            print(f"{robot_id} completed its trajectory!")
                
                step += 1
                
                # Reset if all robots completed
                if len(completed_robots) == len(robots):
                    print("All robots completed! Restarting...")
                    completed_robots.clear()
                    for robot_id, controller in controllers.items():
                        waypoints = waypoint_sets[robot_id]
                        robots[robot_id].reset(x=waypoints[0][0], y=waypoints[0][1], theta=0.0)
                        controller.set_waypoints(waypoints)
            
            # Draw waypoints on top of everything else
            for robot_id, waypoints in waypoint_sets.items():
                for i, (wx, wy) in enumerate(waypoints):
                    screen_pos = renderer.world_to_screen(wx, wy)
                    if (0 <= screen_pos[0] < config.width and 0 <= screen_pos[1] < config.height):
                        pygame.draw.circle(renderer.screen, (0, 255, 0), screen_pos, 5)
                        if i == 0:  # Mark start waypoint
                            pygame.draw.circle(renderer.screen, (255, 255, 255), screen_pos, 3)
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("PID control rendering test completed")


def test_interactive_features():
    """Test interactive features of pygame renderer."""
    print("\n=== Testing Interactive Features ===")
    print("Interactive controls test:")
    print("  Mouse wheel: Zoom in/out")
    print("  Mouse drag: Pan around")
    print("  SPACE: Pause/Resume simulation")
    print("  T: Toggle trajectory display")
    print("  C: Clear all trajectories")
    print("  G: Toggle grid display")
    print("  R: Reset camera to default view")
    print("  ESC: Exit")
    print("\nTry all the controls!")
    
    # Create a more complex scene
    map_instance = create_map_from_config(12, 8, "loop")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters)
    
    # Add random obstacles
    detector.obstacle_manager.create_random_obstacles(
        map_instance.width_meters, map_instance.height_meters, 
        num_obstacles=5, min_size=0.2, max_size=0.5
    )
    
    # Create many robots for stress testing
    robots = {}
    for i in range(6):
        angle = i * 2 * np.pi / 6
        x = map_instance.width_meters/2 + 1.5 * np.cos(angle)
        y = map_instance.height_meters/2 + 1.5 * np.sin(angle)
        robots[f"robot{i+1}"] = create_duckiebot(x=x, y=y, theta=angle)
    
    # Create renderer with custom config
    config = RenderConfig(
        width=1400, height=1000, fps=60,
        show_grid=True, show_coordinates=True,
        show_collision_circles=True, show_robot_ids=True
    )
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(detector.obstacle_manager)
    
    # Simulation loop
    dt = 0.033  # ~30 FPS simulation
    step = 0
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Move robots in circular patterns
                for i, (robot_id, robot) in enumerate(robots.items()):
                    # Different movement patterns for each robot
                    if i % 3 == 0:
                        action = np.array([3.0, 4.0])  # Curve right
                    elif i % 3 == 1:
                        action = np.array([4.0, 3.0])  # Curve left
                    else:
                        action = np.array([3.5, 3.5])  # Straight
                    
                    robot.step(action, dt)
                
                # Check collisions every few steps
                if step % 3 == 0:
                    collisions = detector.check_all_collisions(robots)
                    renderer.set_collision_results(collisions)
                
                step += 1
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("Interactive features test completed")


if __name__ == "__main__":
    print("Pygame Renderer Test Suite")
    print("=" * 50)
    
    # Check if pygame is available
    try:
        import pygame
        print("Pygame is available, running tests...")
    except ImportError:
        print("Pygame is not installed. Please install it with: pip install pygame")
        sys.exit(1)
    
    # Run tests
    test_basic_pygame_rendering()
    test_collision_detection_rendering()
    test_pid_control_rendering()
    test_interactive_features()
    
    print("\n" + "=" * 50)
    print("All pygame renderer tests completed!")
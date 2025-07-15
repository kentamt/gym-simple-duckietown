#!/usr/bin/env python3
"""
Test script to demonstrate the Duckiebot with random control inputs.
"""

import sys
import numpy as np
import time
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot, RobotConfig, Duckiebot
from duckietown_simulator.robot.kinematics import create_duckiebot_kinematics
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.rendering.visualizer import create_visualizer
from duckietown_simulator.robot.pid_controller import (
    WaypointFollowPIDController, create_default_pid_config, create_waypoint_trajectory
)


def test_kinematics():
    """Test the differential drive kinematics."""
    print("=== Testing Differential Drive Kinematics ===")
    
    kinematics = create_duckiebot_kinematics()
    
    # Test wheel speeds to body velocities
    test_cases = [
        (5.0, 5.0),    # Straight forward
        (-5.0, -5.0),  # Straight backward
        (5.0, -5.0),   # Turn left
        (-5.0, 5.0),   # Turn right
        (10.0, 5.0),   # Curve right
        (0.0, 0.0)     # Stationary
    ]
    
    print("\nWheel speeds to body velocities:")
    for omega_l, omega_r in test_cases:
        linear_vel, angular_vel = kinematics.wheel_speeds_to_body_vel(omega_l, omega_r)
        print(f"  ωL={omega_l:5.1f}, ωR={omega_r:5.1f} -> v={linear_vel:6.3f} m/s, ω={angular_vel:6.3f} rad/s")
    
    # Test pose integration
    print("\nPose integration test:")
    pose = np.array([0.0, 0.0, 0.0])
    dt = 0.1
    
    for i in range(5):
        omega_l, omega_r = 5.0, 5.0  # Straight motion
        pose = kinematics.integrate_pose(pose, omega_l, omega_r, dt)
        print(f"  Step {i+1}: pose = ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})")
    
    # Test velocity limits
    limits = kinematics.get_velocity_limits()
    print(f"\nVelocity limits:")
    for key, value in limits.items():
        print(f"  {key}: {value:.3f}")


def test_duckiebot_basic():
    """Test basic Duckiebot functionality."""
    print("\n=== Testing Basic Duckiebot Functionality ===")
    
    # Create robot
    robot = create_duckiebot(x=1.0, y=1.0, theta=0.0)
    print(f"Created robot: {robot}")
    
    # Test state access
    state = robot.get_state_dict()
    print(f"Initial state: {state}")
    
    # Test a few control steps
    print("\nTesting control steps:")
    for i in range(3):
        action = np.array([5.0, 5.0])  # Straight forward
        step_info = robot.step(action, dt=0.1)
        print(f"  Step {i+1}: {robot}")
        print(f"    Distance: {step_info['distance_step']:.3f}m")
    
    # Test reset
    robot.reset(x=0.0, y=0.0, theta=0.0)
    print(f"After reset: {robot}")


def test_random_control():
    """Test Duckiebot with random control inputs."""
    print("\n=== Testing Random Control ===")
    
    # Create robot at center of 5x5 map
    robot = create_duckiebot(x=1.5, y=1.5, theta=0.0)
    
    # Create map for bounds checking
    map_instance = create_map_from_config(5, 5, "straight")
    
    print(f"Starting position: {robot}")
    print(f"Map size: {map_instance.width_meters:.2f}m x {map_instance.height_meters:.2f}m")
    
    # Run random control for several steps
    np.random.seed(42)  # For reproducible results
    dt = 0.05
    
    print("\nRunning random control simulation:")
    for step in range(20):
        # Generate random wheel velocities
        max_speed = robot.config.max_wheel_speed
        omega_l = np.random.uniform(-max_speed, max_speed)
        omega_r = np.random.uniform(-max_speed, max_speed)
        
        action = np.array([omega_l, omega_r])
        step_info = robot.step(action, dt)
        
        # Check if robot is still within map bounds
        in_bounds = map_instance.is_position_in_bounds(robot.x, robot.y)
        
        if step % 5 == 0:  # Print every 5th step
            print(f"  Step {step:2d}: pos=({robot.x:.3f}, {robot.y:.3f}), "
                  f"θ={robot.theta:.3f}, v={robot.linear_velocity:.3f}, "
                  f"in_bounds={in_bounds}")
        
        # Stop if robot goes out of bounds
        if not in_bounds:
            print(f"  Robot went out of bounds at step {step}")
            break
    
    print(f"Final position: {robot}")
    print(f"Total distance traveled: {robot.total_distance:.3f}m")


def test_collision_detection():
    """Test collision detection functionality."""
    print("\n=== Testing Collision Detection ===")
    
    robot = create_duckiebot(x=1.0, y=1.0, theta=0.0)
    
    # Get collision points
    collision_points = robot.get_collision_points()
    print(f"Collision points shape: {collision_points.shape}")
    print(f"Collision points:")
    for i, point in enumerate(collision_points):
        print(f"  Point {i}: ({point[0]:.3f}, {point[1]:.3f})")
    
    # Test forward point
    forward_point = robot.get_forward_point(distance=0.1)
    print(f"Forward point (0.1m): ({forward_point[0]:.3f}, {forward_point[1]:.3f})")
    
    # Test collision state
    robot.set_collision_state(True)
    print(f"Collision state: {robot.is_collided}")
    
    robot.set_collision_state(False)
    print(f"Collision state after reset: {robot.is_collided}")


def test_integrated_simulation():
    """Test integrated simulation with map and robot."""
    print("\n=== Testing Integrated Simulation ===")
    
    # Create map and robot
    map_instance = create_map_from_config(5, 5, "loop")
    robot = create_duckiebot(x=2.0, y=2.0, theta=0.0)  # Start in middle
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print(f"Robot starting at: ({robot.x:.3f}, {robot.y:.3f})")
    
    # Run simulation with random controls
    np.random.seed(123)
    dt = 1.0
    max_steps = 100
    
    trajectory = []
    
    for step in range(max_steps):
        # Store trajectory
        trajectory.append((robot.x, robot.y, robot.theta))
        
        # Generate random action
        max_speed = robot.config.max_wheel_speed * 0.5  # Reduce speed
        omega_l = np.random.uniform(-max_speed, max_speed)
        omega_r = np.random.uniform(-max_speed, max_speed)
        
        action = np.array([omega_l, omega_r])
        step_info = robot.step(action, dt)
        
        # Check map bounds
        in_bounds = map_instance.is_position_in_bounds(robot.x, robot.y)
        
        # Get current tile
        tile_row, tile_col = map_instance.get_tile_at_position(robot.x, robot.y)
        
        if step % 10 == 0:
            print(f"  Step {step:2d}: pos=({robot.x:.3f}, {robot.y:.3f}), "
                  f"tile=({tile_row}, {tile_col}), in_bounds={in_bounds}")
        
        if not in_bounds:
            print(f"  Simulation ended: robot out of bounds at step {step}")
            break
    
    print(f"Simulation completed. Total steps: {len(trajectory)}")
    print(f"Final position: ({robot.x:.3f}, {robot.y:.3f})")
    print(f"Total distance: {robot.total_distance:.3f}m")
    
    # Show some trajectory points
    print("\nTrajectory sample (every 10th point):")
    for i in range(0, len(trajectory), 10):
        x, y, theta = trajectory[i]
        print(f"  Step {i:2d}: ({x:.3f}, {y:.3f}, {theta:.3f})")
    
    # Create and show visualization
    print("\nCreating visualization...")
    visualizer = create_visualizer(map_instance, figsize=(12, 10))
    
    # Draw map
    visualizer.draw_map()
    
    # Draw trajectory
    visualizer.draw_trajectory(trajectory)
    
    # Draw final robot position
    visualizer.draw_robot(robot, show_collision_circle=True)
    
    # Save the visualization
    visualizer.save("duckiebot_simulation.png")
    
    # Show the plot
    print("Displaying visualization (close window to continue)...")
    visualizer.show(block=False)
    
    # Keep plot open for a few seconds
    import matplotlib.pyplot as plt
    plt.pause(3.0)
    plt.close()


def test_pid_controller():
    """Test PID controller with predetermined waypoints."""
    print("\n=== Testing PID Controller ===")
    
    # Create map and robot
    map_instance = create_map_from_config(8, 8, "straight")
    robot = create_duckiebot(x=1.0, y=1.0, theta=0.0)
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print(f"Map size: {map_instance.width_meters:.2f}m x {map_instance.height_meters:.2f}m")
    print(f"Robot starting at: ({robot.x:.3f}, {robot.y:.3f})")
    
    # Create PID controller
    pid_config = create_default_pid_config()
    controller = WaypointFollowPIDController(pid_config)
    
    # Test different trajectory patterns
    patterns = ["square", "line", "spiral"]
    
    for pattern in patterns:
        print(f"\n--- Testing {pattern} trajectory ---")
        
        # Create waypoints
        waypoints = create_waypoint_trajectory(
            pattern, map_instance.width_meters, map_instance.height_meters, margin=0.8
        )
        
        print(f"Created {len(waypoints)} waypoints:")
        for i, (x, y) in enumerate(waypoints):
            print(f"  Waypoint {i}: ({x:.3f}, {y:.3f})")
        
        # Set waypoints in controller
        controller.set_waypoints(waypoints)
        
        # Reset robot to start position
        robot.reset(x=waypoints[0][0], y=waypoints[0][1], theta=0.0)
        
        # Run simulation
        dt = 0.1
        max_steps = 500
        trajectory = []
        
        for step in range(max_steps):
            # Store trajectory
            trajectory.append((robot.x, robot.y, robot.theta))
            
            # Compute PID control
            linear_vel, angular_vel, info = controller.compute_control(
                robot.x, robot.y, robot.theta, dt
            )
            
            # Convert to wheel speeds
            omega_l, omega_r = controller.body_vel_to_wheel_speeds(
                linear_vel, angular_vel, robot.config.wheelbase, robot.config.wheel_radius
            )
            
            # Apply control action
            action = np.array([omega_l, omega_r])
            step_info = robot.step(action, dt)
            
            # Check progress
            if step % 50 == 0:
                progress = controller.get_progress()
                print(f"  Step {step:3d}: waypoint {info['waypoint_idx']}/{len(waypoints)-1}, "
                      f"distance: {info.get('distance_to_target', 0):.3f}m, "
                      f"status: {info['status']}")
            
            # Check if completed
            if info['status'] == 'completed':
                print(f"  Completed {pattern} trajectory at step {step}")
                break
            
            # Safety check - stop if robot goes out of bounds
            if not map_instance.is_position_in_bounds(robot.x, robot.y):
                print(f"  Robot went out of bounds at step {step}")
                break
        
        print(f"Final position: ({robot.x:.3f}, {robot.y:.3f})")
        print(f"Total distance: {robot.total_distance:.3f}m")
        
        # Create visualization for this pattern
        print(f"Creating visualization for {pattern} trajectory...")
        visualizer = create_visualizer(map_instance, figsize=(12, 10))
        
        # Draw map
        visualizer.draw_map()
        
        # Draw waypoints
        for i, (wx, wy) in enumerate(waypoints):
            if i == 0:
                visualizer.ax.plot(wx, wy, 'go', markersize=10, label='Start Waypoint')
            elif i == len(waypoints) - 1:
                visualizer.ax.plot(wx, wy, 'ro', markersize=10, label='End Waypoint')
            else:
                visualizer.ax.plot(wx, wy, 'yo', markersize=8, label='Waypoint' if i == 1 else '')
            
            # Add waypoint numbers
            visualizer.ax.text(wx, wy + 0.1, f'{i}', ha='center', va='bottom', 
                             fontweight='bold', fontsize=10)
        
        # Draw planned path (connecting waypoints)
        waypoint_x = [w[0] for w in waypoints]
        waypoint_y = [w[1] for w in waypoints]
        visualizer.ax.plot(waypoint_x, waypoint_y, 'g--', linewidth=2, alpha=0.7, 
                          label='Planned Path')
        
        # Draw actual trajectory
        visualizer.draw_trajectory(trajectory)
        
        # Draw final robot position
        visualizer.draw_robot(robot, show_collision_circle=True)
        
        # Save visualization
        filename = f"pid_controller_{pattern}_trajectory.png"
        visualizer.save(filename)
        
        print(f"Visualization saved to {filename}")
        
        # Show plot briefly
        visualizer.show(block=False)
        import matplotlib.pyplot as plt
        plt.pause(2.0)
        plt.close()
        
        print(f"--- {pattern} trajectory test completed ---\n")


def test_pid_tuning():
    """Test PID controller with different gain settings."""
    print("\n=== Testing PID Tuning ===")
    
    map_instance = create_map_from_config(6, 6, "straight")
    
    # Test different PID gains
    test_configs = [
        {"name": "Conservative", "distance_kp": 0.5, "heading_kp": 1.0},
        {"name": "Aggressive", "distance_kp": 2.0, "heading_kp": 4.0},
        {"name": "High Damping", "distance_kp": 1.0, "distance_kd": 0.5, "heading_kp": 2.0, "heading_kd": 0.3},
    ]
    
    # Simple waypoint trajectory
    waypoints = [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]
    
    for config_params in test_configs:
        print(f"\nTesting {config_params['name']} PID settings:")
        
        # Create custom PID config
        from duckietown_simulator.robot.pid_controller import PIDConfig, PIDGains
        
        distance_gains = PIDGains(
            kp=config_params.get('distance_kp', 1.0),
            ki=config_params.get('distance_ki', 0.0),
            kd=config_params.get('distance_kd', 0.0)
        )
        
        heading_gains = PIDGains(
            kp=config_params.get('heading_kp', 2.0),
            ki=config_params.get('heading_ki', 0.0),
            kd=config_params.get('heading_kd', 0.0)
        )
        
        pid_config = PIDConfig(
            distance_gains=distance_gains,
            heading_gains=heading_gains,
            max_linear_velocity=0.3,
            max_angular_velocity=2.0
        )
        
        # Create robot and controller
        robot = create_duckiebot(x=waypoints[0][0], y=waypoints[0][1], theta=0.0)
        controller = WaypointFollowPIDController(pid_config)
        controller.set_waypoints(waypoints)
        
        # Run simulation
        dt = 0.1
        trajectory = []
        start_time = 0
        
        for step in range(300):
            trajectory.append((robot.x, robot.y, robot.theta))
            
            linear_vel, angular_vel, info = controller.compute_control(
                robot.x, robot.y, robot.theta, dt
            )
            
            omega_l, omega_r = controller.body_vel_to_wheel_speeds(
                linear_vel, angular_vel, robot.config.wheelbase, robot.config.wheel_radius
            )
            
            robot.step(np.array([omega_l, omega_r]), dt)
            
            if info['status'] == 'completed':
                completion_time = step * dt
                print(f"  Completed in {completion_time:.1f}s ({step} steps)")
                print(f"  Total distance: {robot.total_distance:.3f}m")
                break
        else:
            print(f"  Did not complete within time limit")
            print(f"  Total distance: {robot.total_distance:.3f}m")


if __name__ == "__main__":
    test_kinematics()
    test_duckiebot_basic()
    test_random_control()
    test_collision_detection()
    test_integrated_simulation()
    test_pid_controller()
    test_pid_tuning()
    
    print("\n=== All Duckiebot tests completed successfully! ===")
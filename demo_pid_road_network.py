#!/usr/bin/env python3
"""
Demo of PID-controlled robots on a road network with rotated tiles.
"""

import sys
import numpy as np
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.robot.pid_controller import WaypointFollowPIDController, PIDConfig, PIDGains
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def create_road_network_map():
    """Create the specific road network requested by user."""
    
    road_network = [
        [9, 2, 2, 8],  # Top: left curve, horizontal roads, right curve
        [3, 0,  0, 3],  # Vertical roads with grass in middle
        [14, 2, 2, 12],  # Middle: intersections with horizontal road
        [3, 0,  0, 3],  # Vertical roads with grass in middle
        [7, 2, 2, 4]   # Bottom: right curve, horizontal roads, left curve
    ]
    
    print("Creating road network:")
    print("  Layout: 4x5 tiles with curves, straights, and intersections")
    print("  Road types: curves (20,24,26,27), verticals (10), horizontals (11), intersections (30)")
    
    return create_map_from_array(road_network)


def create_robot_trajectories(map_instance):
    """Create fixed trajectories for 3 robots following the road network."""
    
    tile_size = map_instance.tile_size
    
    # Robot 1: Clockwise outer loop
    trajectory_1 = [
        # Start at top-left, go around clockwise
        (0.5 * tile_size, 0.5 * tile_size),   # Top-left curve
        (1.0 * tile_size, 0.5 * tile_size),  # Top-left curve
        (1.5 * tile_size, 0.5 * tile_size),   # Top horizontal
        (2.5 * tile_size, 0.5 * tile_size),   # Top horizontal
        # (3.5 * tile_size, 0.5 * tile_size),   # Top-right curve
        (3.5 * tile_size, 1.5 * tile_size),   # Right vertical
        (3.5 * tile_size, 2.5 * tile_size),   # Right intersection
        (3.5 * tile_size, 3.5 * tile_size),   # Right vertical
        # (3.5 * tile_size, 4.5 * tile_size),   # Bottom-right curve
        (2.5 * tile_size, 4.5 * tile_size),   # Bottom horizontal
        (1.5 * tile_size, 4.5 * tile_size),   # Bottom horizontal
        # (0.5 * tile_size, 4.5 * tile_size),   # Bottom-left curve
        (0.5 * tile_size, 3.5 * tile_size),   # Left vertical
        (0.5 * tile_size, 2.5 * tile_size),   # Left intersection
        (0.5 * tile_size, 1.5 * tile_size),   # Left vertical
    ]

    # Robot 2: Horizontal figure-8 through intersections
    trajectory_2 = [
        (0.5 * tile_size, 2.5 * tile_size),   # Left intersection
        (1.5 * tile_size, 2.5 * tile_size),   # Middle horizontal
        (2.5 * tile_size, 2.5 * tile_size),   # Middle horizontal
        (3.5 * tile_size, 2.5 * tile_size),   # Right intersection
        (2.5 * tile_size, 2.5 * tile_size),   # Back middle horizontal
        (1.5 * tile_size, 2.5 * tile_size),   # Back middle horizontal
    ]
    
    # Robot 3: Vertical movement through intersections
    trajectory_3 = [
        (2.0 * tile_size, 0.5 * tile_size),   # Start at top middle
        (2.0 * tile_size, 1.5 * tile_size),   # Move down
        (2.0 * tile_size, 2.5 * tile_size),   # Middle intersection
        (2.0 * tile_size, 3.5 * tile_size),   # Move down
        (2.0 * tile_size, 4.5 * tile_size),   # Bottom
        (2.0 * tile_size, 3.5 * tile_size),   # Move back up
        (2.0 * tile_size, 2.5 * tile_size),   # Middle intersection
        (2.0 * tile_size, 1.5 * tile_size),   # Move up
    ]
    
    return {
        "robot1": trajectory_1,
        # "robot2": trajectory_2,
        # "robot3": trajectory_3
    }


def setup_pid_controllers():
    """Setup PID controllers for the robots."""
    
    # Create PID controllers with different parameters for variety
    controllers = {}

    # Robot 1: Smooth, steady movement
    config1 = PIDConfig(
        distance_gains=PIDGains(kp=0.5, ki=0.0, kd=0.0),
        heading_gains=PIDGains(kp=1.0, ki=0.0, kd=0.0),
        max_linear_velocity=2.5,
        max_angular_velocity=4.0,
        position_tolerance=0.20
    )
    controllers["robot1"] = WaypointFollowPIDController(config1)
    
    # Robot 2: More aggressive, faster
    config2 = PIDConfig(
        distance_gains=PIDGains(kp=1.0, ki=0.0, kd=0.0),
        heading_gains=PIDGains(kp=2.0, ki=0.0, kd=0.0),
        max_linear_velocity=2.0,
        max_angular_velocity=2.5,
        position_tolerance=0.12
    )
    controllers["robot2"] = WaypointFollowPIDController(config2)
    
    # Robot 3: Precise, slower
    config3 = PIDConfig(
        distance_gains=PIDGains(kp=1.0, ki=0.0, kd=0.0),
        heading_gains=PIDGains(kp=2.0, ki=0.0, kd=0.0),
        max_linear_velocity=1.0,
        max_angular_velocity=1.5,
        position_tolerance=0.18
    )
    controllers["robot3"] = WaypointFollowPIDController(config3)
    
    return controllers


def run_pid_demo():
    """Run the PID controller demo with rotated road network."""
    print("=== PID Controller Demo on Road Network ===")
    
    # Create map
    map_instance = create_road_network_map()
    
    # Create robots
    robots = {
        "robot1": create_duckiebot(x=0.5 * map_instance.tile_size, y=0.5 * map_instance.tile_size, theta=0.0),
        # "robot2": create_duckiebot(x=0.5 * map_instance.tile_size, y=2.5 * map_instance.tile_size, theta=0.0),
        # "robot3": create_duckiebot(x=2.0 * map_instance.tile_size, y=0.5 * map_instance.tile_size, theta=np.pi/2),
    }
    
    # Setup trajectories and controllers
    trajectories = create_robot_trajectories(map_instance)
    controllers = setup_pid_controllers()
    
    # Initialize controllers with trajectories
    for robot_id in robots.keys():
        controllers[robot_id].set_waypoints(trajectories[robot_id])
    
    # Create renderer
    config = RenderConfig(
        width=1200, height=1000, fps=60,
        use_tile_images=True,
        show_grid=True,
        show_robot_ids=True
    )
    
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    
    # Set planned trajectories for visualization
    renderer.planned_trajectories = trajectories
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print(f"Robots: {len(robots)} with PID controllers")
    print("\nTrajectories:")
    print(f"  Robot1: Clockwise outer loop ({len(trajectories['robot1'])} waypoints)")
    # print(f"  Robot2: Horizontal figure-8 ({len(trajectories['robot2'])} waypoints)")
    # print(f"  Robot3: Vertical movement ({len(trajectories['robot3'])} waypoints)")
    
    print("\nControls:")
    print("  I: Toggle Images/Colors")
    print("  G: Toggle Grid")
    print("  T: Toggle Trajectories")
    print("  P: Toggle Planned Paths")
    print("  SPACE: Pause/Resume")
    print("  R: Reset Camera")
    print("  ESC: Quit")
    print("\nPID controllers will guide robots along their planned trajectories!")
    print("Planned paths are shown as dashed lines with waypoint markers.")
    
    # Simulation parameters
    dt = 0.016  # 60 FPS
    step = 0
    status_interval = 180  # Print status every 3 seconds
    
    # Trajectory visualization data
    robot_paths = {robot_id: [] for robot_id in robots.keys()}
    
    try:
        while renderer.render():
            if not renderer.paused:
                # Update each robot with PID control
                for robot_id, robot in robots.items():
                    controller = controllers[robot_id]
                    
                    # Compute control action using PID
                    linear_vel, angular_vel, info = controller.compute_control(
                        robot.x, robot.y, robot.theta, dt
                    )
                    
                    # Convert to wheel speeds
                    omega_l, omega_r = controller.body_vel_to_wheel_speeds(
                        linear_vel, angular_vel, 
                        robot.config.wheelbase, robot.config.wheel_radius
                    )
                    
                    # Apply control action to robot
                    control_action = np.array([omega_l, omega_r])
                    robot.step(control_action, dt)
                    
                    # Record path for visualization
                    robot_paths[robot_id].append((robot.x, robot.y))
                    if len(robot_paths[robot_id]) > 300:  # Limit path length
                        robot_paths[robot_id].pop(0)
                
                # Update renderer trajectories
                renderer.trajectories = robot_paths
                
                step += 1
                
                # Print status periodically
                if step % status_interval == 0:
                    print(f"Step: {step}")
                    for robot_id, controller in controllers.items():
                        progress_info = controller.get_progress()
                        print(f"  {robot_id}: waypoint {progress_info['current_waypoint']}/{progress_info['total_waypoints']}")
            
            renderer.run_simulation_step()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("PID demo completed!")
    
    # Final statistics
    print("\nFinal Statistics:")
    for robot_id, controller in controllers.items():
        progress = controller.get_progress()
        print(f"  {robot_id}:")
        print(f"    Waypoints completed: {progress['current_waypoint']}/{progress['total_waypoints']}")
        print(f"    Progress: {progress['progress_ratio']*100:.1f}%")
        print(f"    Completed: {progress['completed']}")


def test_trajectories():
    """Test trajectory creation without visualization."""
    print("\n=== Testing Trajectory Creation ===")
    
    map_instance = create_road_network_map()
    trajectories = create_robot_trajectories(map_instance)
    
    for robot_id, trajectory in trajectories.items():
        print(f"\n{robot_id} trajectory:")
        print(f"  Waypoints: {len(trajectory)}")
        print(f"  Start: ({trajectory[0][0]:.2f}, {trajectory[0][1]:.2f})")
        print(f"  End: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f})")
        
        # Calculate total path length
        total_length = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        print(f"  Total path length: {total_length:.2f}m")


if __name__ == "__main__":
    print("PID-Controlled Robots on Road Network Demo")
    print("=" * 50)
    
    # Test trajectory creation
    test_trajectories()
    
    # Run the main demo
    try:
        run_pid_demo()
    except Exception as e:
        print(f"Demo error: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey features demonstrated:")
    print("✓ Rotated road tiles (curves, straights, intersections)")
    print("✓ PID waypoint following controllers") 
    print("✓ Multiple robots with different trajectories")
    print("✓ Real-time trajectory visualization")
    print("✓ Interactive controls and status monitoring")
    print("✓ Realistic robot movement on road network")
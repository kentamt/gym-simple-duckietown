#!/usr/bin/env python3
"""
Demo of PID-controlled robots on a road network with rotated tiles.
Can load trajectories from trajectory editor files.
"""

import sys
import numpy as np
import json
import os
import argparse
import pygame
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.robot.pid_controller import WaypointFollowPIDController, PIDConfig, PIDGains
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig
from duckietown_simulator.world.collision_detection import CollisionDetector, CollisionResult
from duckietown_simulator.world.obstacles import ObstacleConfig, ObstacleType


def load_trajectory_from_file(filename, interpolate=True, interpolation_method='linear', **interpolation_kwargs):
    """
    Load a trajectory from a JSON file created by the trajectory editor.
    
    Args:
        filename: Path to the trajectory JSON file
        interpolate: Whether to interpolate waypoints for smoother movement
        interpolation_method: Method to use ('linear' or 'spline')
        **interpolation_kwargs: Additional parameters for interpolation
        
    Returns:
        List of (x, y) tuples representing waypoints
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Trajectory file not found: {filename}")
    
    try:
        with open(filename, 'r') as f:
            trajectory_data = json.load(f)
        
        # Extract waypoints from trajectory editor format
        original_waypoints = []
        for wp in trajectory_data["waypoints"]:
            original_waypoints.append((wp["x"], wp["y"]))
        
        # Apply interpolation if requested
        if interpolate and len(original_waypoints) > 1:
            waypoints = interpolate_trajectory(original_waypoints, interpolation_method, **interpolation_kwargs)
            print(f"Loaded trajectory '{trajectory_data.get('name', 'unnamed')}' with {len(original_waypoints)} original waypoints, interpolated to {len(waypoints)} waypoints from {filename}")
        else:
            waypoints = original_waypoints
            print(f"Loaded trajectory '{trajectory_data.get('name', 'unnamed')}' with {len(waypoints)} waypoints from {filename}")
        
        return waypoints
        
    except Exception as e:
        raise ValueError(f"Error loading trajectory from {filename}: {e}")


def find_trajectory_files():
    """Find all trajectory JSON files in the current directory."""
    trajectory_files = []
    for file in os.listdir('.'):
        if file.endswith('.json') and file != 'test_map_save.json':
            # Check if it's a trajectory file by looking for trajectory structure
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                if 'waypoints' in data and isinstance(data['waypoints'], list):
                    trajectory_files.append(file)
            except:
                pass
    return trajectory_files


def interpolate_waypoints(waypoints, interpolation_distance=0.1):
    """
    Interpolate additional waypoints between existing ones for smoother trajectories.
    
    Args:
        waypoints: List of (x, y) tuples representing original waypoints
        interpolation_distance: Target distance between interpolated points in meters
        
    Returns:
        List of (x, y) tuples with interpolated waypoints
    """
    if len(waypoints) < 2:
        return waypoints
    
    interpolated = [waypoints[0]]  # Start with first waypoint
    
    for i in range(1, len(waypoints)):
        prev_point = waypoints[i-1]
        curr_point = waypoints[i]
        
        # Calculate distance between consecutive waypoints
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        segment_length = np.sqrt(dx*dx + dy*dy)
        
        # Determine number of interpolated points needed
        if segment_length > interpolation_distance:
            num_points = int(np.ceil(segment_length / interpolation_distance))
            
            # Create interpolated points along the line segment
            for j in range(1, num_points):
                t = j / num_points  # Parameter from 0 to 1
                interp_x = prev_point[0] + t * dx
                interp_y = prev_point[1] + t * dy
                interpolated.append((interp_x, interp_y))
        
        # Add the current waypoint
        interpolated.append(curr_point)
    
    return interpolated


def smooth_trajectory_spline(waypoints, num_points=None):
    """
    Create a smooth spline-interpolated trajectory from waypoints.
    
    Args:
        waypoints: List of (x, y) tuples representing original waypoints
        num_points: Number of points in output trajectory (if None, uses 3x original count)
        
    Returns:
        List of (x, y) tuples representing smooth trajectory
    """
    if len(waypoints) < 3:
        return interpolate_waypoints(waypoints)
    
    # Extract x and y coordinates
    x_coords = [wp[0] for wp in waypoints]
    y_coords = [wp[1] for wp in waypoints]
    
    # Create parameter array (cumulative distance along path)
    distances = [0.0]
    for i in range(1, len(waypoints)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distances.append(distances[-1] + np.sqrt(dx*dx + dy*dy))
    
    # Determine output points
    if num_points is None:
        num_points = len(waypoints) * 3
    
    # Create parameter values for interpolation
    t_original = np.array(distances)
    t_new = np.linspace(0, distances[-1], num_points)
    
    # Simple linear interpolation (can be replaced with spline if scipy available)
    x_interp = np.interp(t_new, t_original, x_coords)
    y_interp = np.interp(t_new, t_original, y_coords)
    
    return list(zip(x_interp, y_interp))


def interpolate_trajectory(waypoints, method='linear', **kwargs):
    """
    Main function to interpolate trajectories with different methods.
    
    Args:
        waypoints: List of (x, y) tuples representing original waypoints
        method: Interpolation method ('linear', 'spline')
        **kwargs: Additional parameters for interpolation methods
        
    Returns:
        List of (x, y) tuples representing interpolated trajectory
    """
    if method == 'linear':
        distance = kwargs.get('interpolation_distance', 0.1)
        return interpolate_waypoints(waypoints, distance)
    elif method == 'spline':
        num_points = kwargs.get('num_points', None)
        return smooth_trajectory_spline(waypoints, num_points)
    else:
        print(f"Unknown interpolation method: {method}, using linear")
        return interpolate_waypoints(waypoints)


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


def create_robot_trajectories(map_instance, trajectory_files=None, interpolation_config=None):
    """
    Create trajectories for robots. Can use saved trajectories or default ones.
    
    Args:
        map_instance: Map instance for tile size reference
        trajectory_files: Dict mapping robot_id to trajectory filename, or None for defaults
        interpolation_config: Dict with interpolation settings or None for defaults
        
    Returns:
        Dict mapping robot_id to list of (x, y) waypoints
    """
    
    # Default interpolation configuration
    if interpolation_config is None:
        interpolation_config = {
            'interpolate': True,
            'method': 'linear',
            'interpolation_distance': 0.15  # 15cm between waypoints
        }
    
    # If trajectory files are provided, load them
    if trajectory_files:
        trajectories = {}
        for robot_id, filename in trajectory_files.items():
            try:
                if interpolation_config['interpolate']:
                    method = interpolation_config.get('method', 'linear')
                    if method == 'linear':
                        kwargs = {'interpolation_distance': interpolation_config.get('interpolation_distance', 0.15)}
                    else:  # spline
                        kwargs = {'num_points': interpolation_config.get('num_points', None)}
                    
                    trajectories[robot_id] = load_trajectory_from_file(
                        filename, 
                        interpolate=True,
                        interpolation_method=method,
                        **kwargs
                    )
                else:
                    trajectories[robot_id] = load_trajectory_from_file(filename, interpolate=False)
            except Exception as e:
                print(f"Error loading trajectory for {robot_id}: {e}")
                print(f"Falling back to default trajectory for {robot_id}")
                trajectories[robot_id] = create_default_trajectory(map_instance, robot_id)
        return trajectories
    
    # Otherwise use default trajectories
    return create_default_trajectories(map_instance)


def create_default_trajectories(map_instance):
    """Create the original default trajectories."""
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


def create_default_trajectory(map_instance, robot_id):
    """Create a single default trajectory for a specific robot."""
    default_trajectories = create_default_trajectories(map_instance)
    return default_trajectories.get(robot_id, default_trajectories["robot1"])


def setup_pid_controllers():
    """Setup PID controllers for the robots."""
    
    # Create PID controllers with different parameters for variety
    controllers = {}

    # Robot 1: Smooth, steady movement
    config1 = PIDConfig(
        distance_gains=PIDGains(kp=0.5, ki=0.5, kd=0.2),
        heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
        max_linear_velocity=2.5,
        max_angular_velocity=4.0,
        position_tolerance=0.10
    )
    controllers["robot1"] = WaypointFollowPIDController(config1)
    
    # Robot 2: More aggressive, faster
    config2 = PIDConfig(
        distance_gains=PIDGains(kp=0.5, ki=0.5, kd=0.2),
        heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
        max_linear_velocity=2.0,
        max_angular_velocity=4.5,
        position_tolerance=0.10
    )
    controllers["robot2"] = WaypointFollowPIDController(config2)
    
    # Robot 3: Precise, slower
    config3 = PIDConfig(
        distance_gains=PIDGains(kp=0.5, ki=0.5, kd=0.2),
        heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
        max_linear_velocity=2.0,
        max_angular_velocity=4.5,
        position_tolerance=0.10
    )
    controllers["robot3"] = WaypointFollowPIDController(config3)
    
    return controllers


def run_pid_demo(trajectory_files=None, interpolation_config=None):
    """
    Run the PID controller demo with rotated road network.
    
    Args:
        trajectory_files: Dict mapping robot_id to trajectory filename, or None for defaults
        interpolation_config: Dict with interpolation settings or None for defaults
    """
    print("=== PID Controller Demo on Road Network ===")
    
    # Create map
    map_instance = create_road_network_map()
    
    # Initialize collision detector
    collision_detector = CollisionDetector(
        map_width=map_instance.width_meters,
        map_height=map_instance.height_meters
    )
    
    # Add some test obstacles to make collisions more likely
    # Add a circular obstacle in the center of the map
    center_x = map_instance.width_meters / 2
    center_y = map_instance.height_meters / 2
    obstacle_config = ObstacleConfig(
        x=center_x,
        y=center_y,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.3,  # 30cm radius obstacle
        name="center_obstacle"
    )

    # collision_detector.obstacle_manager.add_obstacle(obstacle_config)

    # Setup trajectories (either from files or defaults)
    trajectories = create_robot_trajectories(map_instance, trajectory_files, interpolation_config)
    
    # Create robots at starting positions of their trajectories
    robots = {}
    for robot_id, trajectory in trajectories.items():
        start_x, start_y = trajectory[0]  # Use first waypoint as starting position
        robots[robot_id] = create_duckiebot(x=start_x, y=start_y, theta=0.0)
    
    # Setup controllers
    controllers = setup_pid_controllers()
    
    # Initialize controllers with trajectories
    for robot_id in robots.keys():
        controllers[robot_id].set_waypoints(trajectories[robot_id])
    
    # Create renderer
    config = RenderConfig(
        width=1200, height=1000, fps=120,
        use_tile_images=True,
        show_grid=True,
        show_robot_ids=True
    )
    
    renderer = create_pygame_renderer(map_instance, config)
    renderer.set_robots(robots)
    renderer.set_obstacle_manager(collision_detector.obstacle_manager)
    
    # Set planned trajectories for visualization
    renderer.planned_trajectories = trajectories
    
    # Initialize collision tracking
    collision_results = []
    
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
    
    # Speed tracking data
    robot_speeds = {robot_id: {'linear': 0.0, 'angular': 0.0, 'total': 0.0} for robot_id in robots.keys()}
    
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
                    
                    # Track robot speeds
                    robot_speeds[robot_id]['linear'] = abs(linear_vel)
                    robot_speeds[robot_id]['angular'] = abs(angular_vel)
                    robot_speeds[robot_id]['total'] = np.sqrt(linear_vel**2 + angular_vel**2)
                    
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
                
                # Check for collisions after all robots have moved
                collision_results = collision_detector.check_all_collisions(robots)
                
                # Handle collision responses
                robots_in_collision = set()
                for collision in collision_results:
                    if collision.is_colliding:
                        print(f"COLLISION: {collision.collision_type} - {collision.robot_id} "
                              f"{'with ' + collision.other_robot_id if collision.other_robot_id else ''}"
                              f"{'with ' + collision.obstacle_name if collision.obstacle_name else ''}")
                        
                        robots_in_collision.add(collision.robot_id)
                        if collision.other_robot_id:
                            robots_in_collision.add(collision.other_robot_id)
                
                # Apply collision response: stop robots in collision temporarily
                for robot_id in robots.keys():
                    if robot_id in controllers:
                        controller = controllers[robot_id]
                        if robot_id in robots_in_collision:
                            # Stop the robot temporarily during collision
                            robot_speeds[robot_id]['linear'] = 0.0
                            robot_speeds[robot_id]['angular'] = 0.0
                            robot_speeds[robot_id]['total'] = 0.0
                        # Note: We're not applying the zero speeds to the actual control
                        # This is just for display. In a real system, you'd implement
                        # proper collision avoidance in the controller
                
                # Update renderer with trajectories and collision results
                renderer.trajectories = robot_paths
                renderer.set_collision_results(collision_results)
                
                step += 1
                
                # Print status periodically
                if step % status_interval == 0:
                    print(f"Step: {step}, Collisions: {len([c for c in collision_results if c.is_colliding])}")
                    for robot_id in robots.keys():
                        if robot_id in controllers and robot_id in robot_speeds:
                            controller = controllers[robot_id]
                            progress_info = controller.get_progress()
                            speed_info = robot_speeds[robot_id]
                            print(f"  {robot_id}: waypoint {progress_info['current_waypoint']}/{progress_info['total_waypoints']}, "
                                  f"speed: {speed_info['linear']:.2f}m/s linear, {speed_info['angular']:.2f}rad/s angular")
            
            # Draw speed and collision overlays on screen
            if hasattr(renderer, 'screen') and hasattr(renderer, 'font'):
                y_offset = 10
                
                # Draw collision summary
                active_collisions = len([c for c in collision_results if c.is_colliding])
                collision_text = f"Collisions: {active_collisions}"
                collision_color = (255, 0, 0) if active_collisions > 0 else (0, 255, 0)
                text_surface = renderer.font.render(collision_text, True, collision_color)
                
                # Add background for better readability
                text_rect = text_surface.get_rect()
                background_rect = text_rect.copy()
                background_rect.width += 10
                background_rect.height += 4
                background_rect.x = 10
                background_rect.y = y_offset
                
                pygame.draw.rect(renderer.screen, (0, 0, 0, 180), background_rect)
                renderer.screen.blit(text_surface, (15, y_offset + 2))
                y_offset += 30
                
                # Draw robot speeds
                for robot_id, speed_info in robot_speeds.items():
                    robot = robots.get(robot_id)
                    if robot:
                        # Check if robot is in collision
                        robot_in_collision = any(c.is_colliding and c.robot_id == robot_id for c in collision_results)
                        
                        # Create speed text with collision indicator
                        speed_text = f"{robot_id}: {speed_info['linear']:.2f}m/s"
                        if robot_in_collision:
                            speed_text += " [COLLISION]"
                        
                        text_color = (255, 255, 0) if robot_in_collision else (255, 255, 255)
                        text_surface = renderer.font.render(speed_text, True, text_color)
                        
                        # Add background for better readability
                        text_rect = text_surface.get_rect()
                        background_rect = text_rect.copy()
                        background_rect.width += 10
                        background_rect.height += 4
                        background_rect.x = 10
                        background_rect.y = y_offset
                        
                        pygame.draw.rect(renderer.screen, (0, 0, 0, 180), background_rect)
                        renderer.screen.blit(text_surface, (15, y_offset + 2))
                        
                        y_offset += 30
            
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


def test_trajectories(trajectory_files=None, interpolation_config=None):
    """Test trajectory creation without visualization."""
    print("\n=== Testing Trajectory Creation ===")
    
    map_instance = create_road_network_map()
    trajectories = create_robot_trajectories(map_instance, trajectory_files, interpolation_config)
    
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PID-controlled robots demo with support for custom trajectories and interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_pid_road_network.py                    # Use default trajectories
  python demo_pid_road_network.py --trajectory my_trajectory.json  # Single robot with custom trajectory
  python demo_pid_road_network.py --trajectory my_trajectory.json --no-interpolation  # No interpolation
  python demo_pid_road_network.py --trajectory my_trajectory.json --interpolation-method spline  # Smooth spline
  python demo_pid_road_network.py --list-trajectories             # Show available trajectory files
  
Trajectory files should be JSON files created by the trajectory editor.
        """
    )
    
    parser.add_argument('--trajectory', '-t', 
                        help='Trajectory file to load for robot1 (JSON format from trajectory editor)')
    parser.add_argument('--robot2-trajectory', 
                        help='Trajectory file for robot2 (enables robot2)')
    parser.add_argument('--robot3-trajectory', 
                        help='Trajectory file for robot3 (enables robot3)')
    parser.add_argument('--list-trajectories', action='store_true',
                        help='List available trajectory files and exit')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test trajectory creation, do not run simulation')
    
    # Interpolation options
    parser.add_argument('--no-interpolation', action='store_true',
                        help='Disable waypoint interpolation (use original waypoints only)')
    parser.add_argument('--interpolation-method', choices=['linear', 'spline'], default='linear',
                        help='Interpolation method (default: linear)')
    parser.add_argument('--interpolation-distance', type=float, default=0.15,
                        help='Target distance between interpolated waypoints in meters (default: 0.15)')
    parser.add_argument('--spline-points', type=int,
                        help='Number of points for spline interpolation (default: 3x original)')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("PID-Controlled Robots on Road Network Demo")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # List available trajectory files if requested
    if args.list_trajectories:
        trajectory_files = find_trajectory_files()
        if trajectory_files:
            print("Available trajectory files:")
            for i, filename in enumerate(trajectory_files, 1):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    name = data.get('name', 'unnamed')
                    waypoint_count = len(data.get('waypoints', []))
                    print(f"  {i}. {filename} - '{name}' ({waypoint_count} waypoints)")
                except:
                    print(f"  {i}. {filename} - (error reading file)")
        else:
            print("No trajectory files found in current directory.")
            print("Create trajectories using: python trajectory_editor.py")
        exit(0)
    
    # Build trajectory files dict from arguments
    trajectory_files = None
    if args.trajectory or args.robot2_trajectory or args.robot3_trajectory:
        trajectory_files = {}
        if args.trajectory:
            trajectory_files['robot1'] = args.trajectory
        if args.robot2_trajectory:
            trajectory_files['robot2'] = args.robot2_trajectory
        if args.robot3_trajectory:
            trajectory_files['robot3'] = args.robot3_trajectory
    
    # Build interpolation configuration from arguments
    interpolation_config = {
        'interpolate': not args.no_interpolation,
        'method': args.interpolation_method,
        'interpolation_distance': args.interpolation_distance
    }
    if args.spline_points:
        interpolation_config['num_points'] = args.spline_points
    
    # Test trajectory creation
    if args.test_only:
        test_trajectories(trajectory_files, interpolation_config)
        exit(0)
    
    # Show loaded trajectory info
    if trajectory_files:
        print("Loading custom trajectories:")
        for robot_id, filename in trajectory_files.items():
            print(f"  {robot_id}: {filename}")
        print(f"Interpolation: {'enabled' if interpolation_config['interpolate'] else 'disabled'}")
        if interpolation_config['interpolate']:
            print(f"  Method: {interpolation_config['method']}")
            if interpolation_config['method'] == 'linear':
                print(f"  Distance: {interpolation_config['interpolation_distance']:.2f}m")
            elif 'num_points' in interpolation_config:
                print(f"  Points: {interpolation_config['num_points']}")
    else:
        print("Using default trajectories")
    
    test_trajectories(trajectory_files, interpolation_config)
    
    # Run the main demo
    try:
        run_pid_demo(trajectory_files, interpolation_config)
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey features demonstrated:")
    print("✓ Rotated road tiles (curves, straights, intersections)")
    print("✓ PID waypoint following controllers") 
    print("✓ Multiple robots with different trajectories")
    print("✓ Real-time trajectory visualization")
    print("✓ Interactive controls and status monitoring")
    print("✓ Realistic robot movement on road network")
    if trajectory_files:
        print("✓ Custom trajectories from trajectory editor")
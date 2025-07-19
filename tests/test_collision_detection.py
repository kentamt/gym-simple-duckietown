#!/usr/bin/env python3
"""
Test script for collision detection system.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('.')

from duckietown_simulator.robot.duckiebot import create_duckiebot, Duckiebot
from duckietown_simulator.world.map import create_map_from_config
from duckietown_simulator.world.collision_detection import create_collision_detector
from duckietown_simulator.world.obstacles import ObstacleManager, ObstacleConfig, ObstacleType
from duckietown_simulator.rendering.visualizer import create_visualizer
from duckietown_simulator.utils.geometry import Point, Circle, Rectangle


def test_geometry_primitives():
    """Test basic geometry collision functions."""
    print("=== Testing Geometry Primitives ===")
    
    # Test circle-circle collision
    from duckietown_simulator.utils.geometry import circle_circle_collision, Circle, Point
    
    circle1 = Circle(Point(0, 0), 1.0)
    circle2 = Circle(Point(1.5, 0), 1.0)
    circle3 = Circle(Point(3.0, 0), 1.0)
    
    print(f"Circle1 vs Circle2 (overlapping): {circle_circle_collision(circle1, circle2)}")
    print(f"Circle1 vs Circle3 (separate): {circle_circle_collision(circle1, circle3)}")
    
    # Test point in rectangle
    from duckietown_simulator.utils.geometry import point_in_rectangle, Rectangle
    
    rect = Rectangle(Point(0, 0), 2.0, 1.0, 0.0)
    test_points = [
        Point(0, 0),    # Center
        Point(0.9, 0.4),  # Inside
        Point(1.1, 0.6),  # Outside
        Point(-1.1, 0),   # Outside left
    ]
    
    print("\nPoint in rectangle tests:")
    for i, point in enumerate(test_points):
        inside = point_in_rectangle(point, rect)
        print(f"  Point {i} ({point.x}, {point.y}): {'Inside' if inside else 'Outside'}")
    
    # Test circle-rectangle collision
    from duckietown_simulator.utils.geometry import circle_rectangle_collision
    
    circle = Circle(Point(1.2, 0), 0.5)
    collision = circle_rectangle_collision(circle, rect)
    print(f"\nCircle-Rectangle collision: {collision}")


def test_obstacle_system():
    """Test obstacle creation and management."""
    print("\n=== Testing Obstacle System ===")
    
    # Create obstacle manager
    manager = ObstacleManager()
    
    # Add various obstacles
    circle_config = ObstacleConfig(
        x=2.0, y=2.0,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.5,
        name="test_circle"
    )
    
    rect_config = ObstacleConfig(
        x=4.0, y=3.0,
        obstacle_type=ObstacleType.RECTANGLE,
        width=1.0, height=0.5, rotation=np.pi/4,
        name="test_rectangle"
    )
    
    obstacle1 = manager.add_obstacle(circle_config)
    obstacle2 = manager.add_obstacle(rect_config)
    
    print(f"Created obstacles: {manager.count_obstacles()}")
    print(f"Circle obstacle: {obstacle1}")
    print(f"Rectangle obstacle: {obstacle2}")
    
    # Test bounds
    bounds1 = obstacle1.get_bounds()
    bounds2 = obstacle2.get_bounds()
    print(f"Circle bounds: {bounds1}")
    print(f"Rectangle bounds: {bounds2}")
    
    # Test point containment
    test_points = [(2.0, 2.0), (4.0, 3.0), (0.0, 0.0)]
    for x, y in test_points:
        in_circle = obstacle1.contains_point(x, y)
        in_rect = obstacle2.contains_point(x, y)
        print(f"Point ({x}, {y}): Circle={in_circle}, Rectangle={in_rect}")


def test_single_robot_collision():
    """Test collision detection with single robot."""
    print("\n=== Testing Single Robot Collision ===")
    
    # Create map and collision detector
    map_instance = create_map_from_config(6, 6, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters)
    
    # Create robot
    robot = create_duckiebot(x=2.0, y=2.0, theta=0.0)
    
    print(f"Map size: {map_instance.width_meters:.2f}m x {map_instance.height_meters:.2f}m")
    print(f"Robot at: ({robot.x:.2f}, {robot.y:.2f})")
    
    # Test boundary collision at different positions
    test_positions = [
        (2.0, 2.0),    # Safe center
        (-0.1, 2.0),   # Left boundary
        (map_instance.width_meters + 0.1, 2.0),  # Right boundary
        (2.0, -0.1),   # Bottom boundary
        (2.0, map_instance.height_meters + 0.1)  # Top boundary
    ]
    
    print("\nBoundary collision tests:")
    for x, y in test_positions:
        robot.reset(x=x, y=y, theta=0.0)
        collision = detector.check_robot_boundary_collision(robot, "test_robot")
        print(f"  Position ({x:.1f}, {y:.1f}): {'COLLISION' if collision.is_colliding else 'Safe'}")
    
    # Test obstacle collision
    obstacle_collisions = detector.check_robot_obstacle_collision(robot, "test_robot")
    print(f"\nObstacle collisions: {len(obstacle_collisions)}")
    for collision in obstacle_collisions:
        print(f"  Collision with {collision.obstacle_name}")


def test_multi_robot_collision():
    """Test collision detection with multiple robots."""
    print("\n=== Testing Multi-Robot Collision ===")
    
    # Create map and detector
    map_instance = create_map_from_config(8, 8, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Create multiple robots
    robots = {
        "robot1": create_duckiebot(x=2.0, y=2.0, theta=0.0),
        "robot2": create_duckiebot(x=2.1, y=2.0, theta=0.0),  # Very close
        "robot3": create_duckiebot(x=4.0, y=4.0, theta=0.0),  # Far away
    }
    
    print(f"Created {len(robots)} robots")
    for robot_id, robot in robots.items():
        print(f"  {robot_id}: ({robot.x:.2f}, {robot.y:.2f})")
    
    # Check all collisions
    all_collisions = detector.check_all_collisions(robots)
    
    print(f"\nDetected {len(all_collisions)} collisions:")
    for collision in all_collisions:
        if collision.collision_type == "robot_robot":
            print(f"  Robot collision: {collision.robot_id} vs {collision.other_robot_id}")
            print(f"    Penetration: {collision.penetration_depth:.3f}m")
        elif collision.collision_type == "robot_obstacle":
            print(f"  Obstacle collision: {collision.robot_id} vs {collision.obstacle_name}")
        elif collision.collision_type == "robot_boundary":
            print(f"  Boundary collision: {collision.robot_id}")
    
    # Test safe spawn positions
    print("\nTesting safe spawn positions:")
    for i in range(5):
        safe_pos = detector.find_safe_spawn_position(0.05, robots)
        if safe_pos:
            print(f"  Safe position {i+1}: ({safe_pos[0]:.2f}, {safe_pos[1]:.2f})")
        else:
            print(f"  Could not find safe position {i+1}")


def test_collision_during_movement():
    """Test collision detection during robot movement."""
    print("\n=== Testing Collision During Movement ===")
    
    # Create map with obstacles
    map_instance = create_map_from_config(8, 8, "straight")
    detector = create_collision_detector(map_instance.width_meters, map_instance.height_meters, with_obstacles=False)
    
    # Add some custom obstacles for more interesting scenario
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=2.5, y=2.5,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.3,
        name="obstacle_1"
    ))
    
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=4.0, y=2.0,
        obstacle_type=ObstacleType.RECTANGLE,
        width=0.6, height=0.4, rotation=np.pi/6,
        name="obstacle_2"
    ))
    
    detector.obstacle_manager.add_obstacle(ObstacleConfig(
        x=1.5, y=4.0,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.25,
        name="obstacle_3"
    ))
    
    # Create multiple robots in different scenarios
    robots = {
        "robot1": create_duckiebot(x=0.5, y=2.5, theta=0.0),      # Moving towards obstacle_1
        "robot2": create_duckiebot(x=3.5, y=1.5, theta=np.pi/3), # Moving towards obstacle_2
        "robot3": create_duckiebot(x=1.0, y=3.5, theta=np.pi/2), # Moving towards obstacle_3
        "robot4": create_duckiebot(x=3.0, y=3.5, theta=-np.pi/4), # Moving towards robot1 area
    }
    
    print(f"Simulating {len(robots)} robots moving with collision detection...")
    print(f"Map size: {map_instance.width_meters:.1f}m x {map_instance.height_meters:.1f}m")
    
    # Simulation parameters
    dt = 0.05
    max_steps = 200
    collision_count = 0
    
    # Track trajectories and collisions
    trajectories = {robot_id: [] for robot_id in robots.keys()}
    collision_events = []
    collision_frames = []  # Store collision info for each frame
    
    for step in range(max_steps):
        # Store current positions
        for robot_id, robot in robots.items():
            trajectories[robot_id].append((robot.x, robot.y, robot.theta))
        
        # Move robots with different control strategies
        actions = {
            "robot1": np.array([4.0, 4.0]),    # Straight forward
            "robot2": np.array([3.0, 4.0]),    # Slight curve right
            "robot3": np.array([4.0, 3.0]),    # Slight curve left
            "robot4": np.array([3.5, 3.5]),    # Straight forward
        }
        
        # Apply actions
        for robot_id, action in actions.items():
            robots[robot_id].step(action, dt)
        
        # Check collisions
        collisions = detector.check_all_collisions(robots)
        collision_frames.append({
            'step': step,
            'time': step * dt,
            'collisions': collisions.copy(),
            'robot_positions': {rid: (r.x, r.y, r.theta) for rid, r in robots.items()}
        })
        
        if collisions:
            collision_count += len(collisions)
            collision_events.extend(collisions)
            
            if step % 20 == 0:
                print(f"  Step {step:3d} (t={step*dt:.1f}s): {len(collisions)} collisions")
                for collision in collisions:
                    if collision.collision_type == "robot_robot":
                        print(f"    Robot-Robot: {collision.robot_id} ↔ {collision.other_robot_id}")
                        print(f"      Position: ({collision.collision_point[0]:.2f}, {collision.collision_point[1]:.2f})")
                        print(f"      Penetration: {collision.penetration_depth:.3f}m")
                    elif collision.collision_type == "robot_obstacle":
                        print(f"    Robot-Obstacle: {collision.robot_id} ↔ {collision.obstacle_name}")
                    elif collision.collision_type == "robot_boundary":
                        print(f"    Boundary: {collision.robot_id}")
        
        # Stop if too many robots go out of bounds
        robots_in_bounds = sum(1 for robot in robots.values() 
                             if map_instance.is_position_in_bounds(robot.x, robot.y))
        if robots_in_bounds < len(robots) // 2:
            print(f"  Simulation stopped at step {step}: too many robots out of bounds")
            break
    
    print(f"\nSimulation completed:")
    print(f"  Total simulation time: {len(trajectories['robot1']) * dt:.1f}s")
    print(f"  Total collision events: {collision_count}")
    print(f"  Final robot positions:")
    for robot_id, robot in robots.items():
        in_bounds = map_instance.is_position_in_bounds(robot.x, robot.y)
        status = "✓" if in_bounds else "✗ (out of bounds)"
        print(f"    {robot_id}: ({robot.x:.2f}, {robot.y:.2f}) {status}")
    
    # Get collision statistics
    stats = detector.get_collision_statistics()
    print(f"  Collision breakdown: {stats['collision_breakdown']}")
    
    return map_instance, detector, trajectories, robots, collision_frames


def test_collision_visualization():
    """Test visualization of collision detection."""
    print("\n=== Testing Collision Visualization ===")
    
    # Run movement test to get data
    map_instance, detector, trajectories, robots, collision_frames = test_collision_during_movement()
    
    # Create enhanced visualizer
    visualizer = create_visualizer(map_instance, figsize=(14, 12))
    
    # Draw map
    visualizer.draw_map()
    
    # Draw obstacles with labels
    obstacle_patches = []
    for i, obstacle in enumerate(detector.obstacle_manager.get_all_obstacles()):
        viz_data = obstacle.get_visualization_data()
        
        if viz_data['type'] == 'circle':
            circle = plt.Circle(
                (viz_data['x'], viz_data['y']), 
                viz_data['radius'],
                facecolor=viz_data['color'],
                alpha=0.6,
                edgecolor='black',
                linewidth=2,
                label='Obstacles' if i == 0 else ''
            )
            visualizer.ax.add_patch(circle)
            obstacle_patches.append(circle)
            
            # Add obstacle label
            visualizer.ax.text(viz_data['x'], viz_data['y'], 
                             viz_data['name'].replace('obstacle_', 'O'),
                             ha='center', va='center', fontweight='bold', 
                             color='white', fontsize=10)
        
        elif viz_data['type'] == 'rectangle':
            # Draw rectangle using corners
            corners = viz_data['corners']
            corners.append(corners[0])  # Close the shape
            
            x_coords = [c[0] for c in corners]
            y_coords = [c[1] for c in corners]
            
            visualizer.ax.plot(x_coords, y_coords, 
                             color='black', linewidth=2)
            visualizer.ax.fill(x_coords, y_coords, 
                             color=viz_data['color'], alpha=0.6,
                             label='Rectangle Obstacles' if i == 1 else '')
            
            # Add obstacle label
            visualizer.ax.text(viz_data['x'], viz_data['y'], 
                             viz_data['name'].replace('obstacle_', 'O'),
                             ha='center', va='center', fontweight='bold', 
                             color='white', fontsize=10)
    
    # Draw robot trajectories with different styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors
    linestyles = ['-', '--', '-.', ':']
    
    for i, (robot_id, trajectory) in enumerate(trajectories.items()):
        if trajectory:
            x_coords = [pos[0] for pos in trajectory]
            y_coords = [pos[1] for pos in trajectory]
            
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            # Draw trajectory
            visualizer.ax.plot(x_coords, y_coords, 
                             color=color, linewidth=2.5, linestyle=linestyle,
                             alpha=0.8, label=f'{robot_id} path')
            
            # Mark start position
            visualizer.ax.plot(x_coords[0], y_coords[0], 
                             'o', color=color, markersize=12, 
                             markeredgecolor='black', markeredgewidth=2,
                             label=f'{robot_id} start' if i == 0 else '')
            
            # Mark end position
            visualizer.ax.plot(x_coords[-1], y_coords[-1], 
                             's', color=color, markersize=12,
                             markeredgecolor='black', markeredgewidth=2,
                             label=f'{robot_id} end' if i == 0 else '')
            
            # Add robot ID at start
            visualizer.ax.text(x_coords[0], y_coords[0] + 0.15, robot_id,
                             ha='center', va='bottom', fontweight='bold',
                             fontsize=9, color=color)
    
    # Draw final robot positions with collision circles
    for i, (robot_id, robot) in enumerate(robots.items()):
        color = colors[i % len(colors)]
        
        # Custom robot drawing with color coding
        visualizer.draw_robot(robot, show_collision_circle=True)
        
        # Override robot color in visualization
        robot_patches = visualizer.ax.patches[-2:]  # Get last two patches (collision circle and robot)
        if len(robot_patches) >= 1:
            robot_patches[-1].set_facecolor(color)  # Robot body
            robot_patches[-1].set_alpha(0.8)
    
    # Mark collision points
    collision_points = []
    for frame in collision_frames:
        for collision in frame['collisions']:
            if collision.collision_point:
                collision_points.append(collision.collision_point)
    
    if collision_points:
        # Remove duplicates (approximate)
        unique_points = []
        for point in collision_points:
            is_duplicate = any(
                abs(point[0] - up[0]) < 0.1 and abs(point[1] - up[1]) < 0.1 
                for up in unique_points
            )
            if not is_duplicate:
                unique_points.append(point)
        
        # Draw collision markers
        for point in unique_points:
            visualizer.ax.plot(point[0], point[1], 'X', 
                             color='red', markersize=15, markeredgewidth=3,
                             markeredgecolor='darkred',
                             label='Collision Points' if point == unique_points[0] else '')
    
    # Add detailed title and information
    total_collisions = sum(len(frame['collisions']) for frame in collision_frames)
    simulation_time = len(collision_frames) * 0.05
    
    title = f'Multi-Robot Collision Detection Simulation\n'
    title += f'4 Robots, 3 Obstacles, {total_collisions} Collisions in {simulation_time:.1f}s'
    visualizer.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics box
    stats_text = f'Simulation Statistics:\n'
    stats_text += f'• Total Time: {simulation_time:.1f}s\n'
    stats_text += f'• Total Collisions: {total_collisions}\n'
    stats_text += f'• Collision Rate: {total_collisions/simulation_time:.1f}/s'
    
    visualizer.ax.text(0.02, 0.98, stats_text,
                      transform=visualizer.ax.transAxes,
                      verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                      fontsize=10, fontfamily='monospace')
    
    # Improve legend
    visualizer.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                        frameon=True, fancybox=True, shadow=True)
    
    # Add grid for better readability
    visualizer.ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save and show
    visualizer.save("collision_detection_movement_visualization.png", dpi=300)
    print("Enhanced visualization saved to collision_detection_movement_visualization.png")
    
    visualizer.show(block=False)
    plt.pause(5.0)  # Show longer for better viewing
    plt.close()
    
    # Create a collision timeline plot
    create_collision_timeline(collision_frames)


def create_collision_timeline(collision_frames):
    """Create a timeline visualization of collisions."""
    print("\nCreating collision timeline visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Extract timeline data
    times = [frame['time'] for frame in collision_frames]
    collision_counts = [len(frame['collisions']) for frame in collision_frames]
    
    # Plot 1: Collision count over time
    ax1.plot(times, collision_counts, 'r-', linewidth=2, label='Collisions per timestep')
    ax1.fill_between(times, collision_counts, alpha=0.3, color='red')
    ax1.set_ylabel('Number of Collisions', fontsize=12)
    ax1.set_title('Collision Detection Timeline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Collision types over time
    collision_types = {'robot_robot': [], 'robot_obstacle': [], 'robot_boundary': []}
    
    for frame in collision_frames:
        type_counts = {'robot_robot': 0, 'robot_obstacle': 0, 'robot_boundary': 0}
        for collision in frame['collisions']:
            type_counts[collision.collision_type] += 1
        
        for collision_type in collision_types:
            collision_types[collision_type].append(type_counts[collision_type])
    
    # Stacked area plot
    ax2.stackplot(times, 
                  collision_types['robot_robot'],
                  collision_types['robot_obstacle'], 
                  collision_types['robot_boundary'],
                  labels=['Robot-Robot', 'Robot-Obstacle', 'Robot-Boundary'],
                  colors=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                  alpha=0.8)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Collision Count by Type', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('collision_timeline.png', dpi=300, bbox_inches='tight')
    print("Collision timeline saved to collision_timeline.png")
    
    plt.show(block=False)
    plt.pause(3.0)
    plt.close()


if __name__ == "__main__":
    test_geometry_primitives()
    test_obstacle_system()
    test_single_robot_collision()
    test_multi_robot_collision()
    test_collision_visualization()
    
    print("\n=== All collision detection tests completed successfully! ===")
#!/usr/bin/env python3
"""
Demo script for discrete action Duckietown environment.

This demo shows how to use discrete actions (STOP, FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD)
with the PID controller to achieve smooth, realistic robot motion.
"""

import numpy as np
import time
import math
from duckietown_simulator.environment.discrete_action_env import create_discrete_duckietown_env
from duckietown_simulator.environment.reward_functions import get_reward_function
from duckietown_simulator.robot.discrete_action_mapper import DiscreteAction


def demo_discrete_actions():
    """Demonstrate discrete actions with PID control."""
    print("🤖 Discrete Action Duckietown Demo")
    print("=" * 50)
    
    # Define custom road network
    road_network = [
        [9, 2, 2, 8],   # Top: left curve, horizontal roads, right curve
        [3, 0, 0, 3],   # Vertical roads with grass in middle
        [14, 2, 2, 12], # Middle: intersections with horizontal road
        [3, 0, 0, 3],   # Vertical roads with grass in middle
        [7, 2, 2, 4]    # Bottom: right curve, horizontal roads, left curve
    ]
    
    # Create discrete action environment
    env = create_discrete_duckietown_env(
        map_config={"layout": road_network},
        reward_function=get_reward_function('lane_following'),
        render_mode="human",  # Use pygame rendering if available
        max_steps=10000,
        forward_distance=0.2,  # 0.2 meter forward steps
        turn_radius=0.18,       # 0.2 meter turn radius
        turn_angle=math.pi/2   # 90-degree turns
    )
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.get_action_meanings()}")
    print("\nDiscrete Action Configuration:")
    
    config_info = env.get_discrete_action_info()
    for key, value in config_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Starting demo...")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial robot position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    print(f"Initial robot heading: {math.degrees(info['robot_state']['theta']):.1f}°")
    
    # Demo sequence: square pattern using discrete actions
    action_sequence = [
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.FORWARD, "Move forward"),
        (DiscreteAction.TURN_LEFT, "Turn left 90°"),
        (DiscreteAction.TURN_LEFT, "Turn left 90°"),
    ]
    
    print(f"\nExecuting square pattern using discrete actions...")
    print("Press Ctrl+C to stop demo early\n")
    
    try:
        step_count = 0
        action_index = 0
        
        for episode_step in range(env.env.max_steps):
            # Get current action
            current_action, action_description = action_sequence[action_index % len(action_sequence)]

            # Execute action
            obs, reward, terminated, truncated, info = env.step(current_action)
            step_count += 1
            
            # Print step information
            robot_state = info['robot_state']
            discrete_info = info['discrete_action']
            
            print(f"Step {step_count:3d}: {action_description}")
            print(f"  Position: ({robot_state['x']:.3f}, {robot_state['y']:.3f})")
            print(f"  Heading: {math.degrees(robot_state['theta']):.1f}°")
            print(f"  Action: {discrete_info['action_name']}")
            print(f"  Control: v={discrete_info['linear_velocity']:.3f} m/s, ω={discrete_info['angular_velocity']:.3f} rad/s")
            print(f"  Status: {discrete_info['control_status']}")
            print(f"  Reward: {reward:.3f}")
            
            # Check if action is completed
            if discrete_info['action_completed']:
                action_index += 1
                print(f"  → Action completed! Moving to next action.\n")
            else:
                print()
            
            # Render environment
            env.render()
            
            # Check for episode end
            if terminated or truncated:
                print(f"\nEpisode ended after {step_count} steps!")
                break
            
            # Small delay for visualization
            time.sleep(1./30.)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    # Final statistics
    final_robot_state = env.env.robot.get_state_dict()
    print(f"\nFinal robot position: ({final_robot_state['x']:.3f}, {final_robot_state['y']:.3f})")
    print(f"Final robot heading: {math.degrees(final_robot_state['theta']):.1f}°")
    print(f"Total steps: {step_count}")
    
    env.close()
    print("\nDiscrete action demo completed!")


def demo_action_visualization():
    """Demonstrate action waypoint visualization."""
    print("\n" + "=" * 50)
    print("🎯 Action Waypoint Visualization")
    print("=" * 50)

    # Create environment (no rendering needed for this demo)
    env = create_discrete_duckietown_env(
        map_config={"width": 3, "height": 3, "track_type": "straight"},
        render_mode=None,
        turn_radius=0.8,
        turn_angle=math.pi/4  # 45-degree turns
    )
    
    # Reset and get initial position
    obs, info = env.reset()
    
    # Show waypoints for all actions from current position
    print("Current robot state:")
    robot_state = info['robot_state']
    print(f"  Position: ({robot_state['x']:.3f}, {robot_state['y']:.3f})")
    print(f"  Heading: {math.degrees(robot_state['theta']):.1f}°")
    
    print("\nPossible waypoints for each action:")
    waypoints = env.visualize_actions()
    
    for action_name, (x, y) in waypoints.items():
        print(f"  {action_name:10s}: ({x:6.3f}, {y:6.3f})")
    
    env.close()


def demo_manual_control():
    """Demo manual control using keyboard input."""
    print("\n" + "=" * 50)
    print("🎮 Manual Control Demo")
    print("=" * 50)
    print("Controls:")
    print("  0 - STOP")
    print("  1 - FORWARD") 
    print("  2 - TURN_LEFT")
    print("  3 - TURN_RIGHT")
    print("  4 - BACKWARD")
    print("  q - Quit")
    print()
    
    # Create environment
    env = create_discrete_duckietown_env(
        map_config={"width": 4, "height": 4, "track_type": "loop"},
        render_mode="human",
        forward_distance=0.8,
        turn_radius=0.6,
        turn_angle=math.pi/3  # 60-degree turns
    )
    
    obs, info = env.reset()
    print("Environment ready! Enter actions (0-4) or 'q' to quit:")
    
    try:
        step_count = 0
        while True:
            # Get user input
            try:
                user_input = input(f"Step {step_count + 1}: ").strip().lower()
                
                if user_input == 'q':
                    break
                
                action = int(user_input)
                if not (0 <= action <= 4):
                    print("Invalid action! Use 0-4.")
                    continue
                    
            except ValueError:
                print("Invalid input! Enter a number 0-4 or 'q'.")
                continue
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Show results
            robot_state = info['robot_state']
            discrete_info = info['discrete_action']
            
            print(f"  Action: {discrete_info['action_name']}")
            print(f"  Position: ({robot_state['x']:.3f}, {robot_state['y']:.3f})")
            print(f"  Reward: {reward:.3f}")
            
            if discrete_info['action_completed']:
                print("  Status: Action completed!")
            
            env.render()
            
            if terminated or truncated:
                print("Episode ended!")
                break
                
    except KeyboardInterrupt:
        print("\nManual control stopped by user")
    
    env.close()


def main():
    """Run all demos."""
    print("🚗 Duckietown Discrete Action System")
    print("=" * 60)
    
    # Run main demo
    demo_discrete_actions()
    
    # Run visualization demo  
    # demo_action_visualization()
    
    # Ask if user wants manual control
    # try:
    #     response = input("\nWould you like to try manual control? (y/n): ").strip().lower()
    #     if response in ['y', 'yes']:
    #         demo_manual_control()
    # except KeyboardInterrupt:
    #     pass
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("\nThe discrete action system provides:")
    print("• Simple, interpretable actions for RL")
    print("• Smooth motion via PID control") 
    print("• Configurable movement distances and turn angles")
    print("• Realistic robot dynamics")


if __name__ == "__main__":
    main()
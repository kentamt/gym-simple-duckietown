#!/usr/bin/env python3
"""
Example usage of the Duckietown Gym interface.

This script shows how to use the Duckietown environment for:
- Basic RL training loop
- Different environment configurations
- Custom reward functions
- Rendering
"""

import numpy as np
import sys
sys.path.append('.')

from duckietown_simulator.environment import DuckietownEnv, make_env
from duckietown_simulator.environment.reward_functions import get_reward_function


def basic_usage_example():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Create environment
    env = DuckietownEnv()
    
    # Reset environment
    observation, info = env.reset()
    
    # Run episode
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: reward={reward:.3f}")
        
        # Check if episode is done
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()
    print()


def different_environments_example():
    """Example showing different environment configurations."""
    print("=== Different Environments Example ===")
    
    # Different predefined environments
    environments = [
        ("default", "Default 5x5 straight track"),
        ("loop", "6x5 loop track"),
        ("small", "3x3 small track"),
        ("large", "8x8 large loop"),
    ]
    
    for env_name, description in environments:
        print(f"Testing {env_name}: {description}")
        
        env = make_env(env_name)
        obs, info = env.reset()
        
        # Run a few steps
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Map size: {env.map.width_meters:.1f}x{env.map.height_meters:.1f}m")
        print(f"  10-step reward: {total_reward:.3f}")
        
        env.close()
    print()


def custom_reward_example():
    """Example with custom reward functions."""
    print("=== Custom Reward Functions Example ===")
    
    reward_functions = [
        ("lane_following", "Lane following task"),
        ("exploration", "Exploration task"),
        ("racing", "Racing task"),
        ("sparse", "Sparse rewards"),
    ]
    
    for reward_name, description in reward_functions:
        print(f"Testing {reward_name}: {description}")
        
        env = DuckietownEnv(
            reward_function=get_reward_function(reward_name),
            max_steps=50
        )
        
        obs, info = env.reset()
        total_reward = 0
        
        # Run forward motion test
        for _ in range(20):
            # Forward action
            action = np.array([2.0, 2.0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Forward motion reward: {total_reward:.3f}")
        env.close()
    print()


def custom_map_example():
    """Example with custom map."""
    print("=== Custom Map Example ===")
    
    # Define custom maze-like layout
    maze_layout = [
        [1, 1, 1, 1, 1, 1, 1],  # Walls
        [1, 2, 2, 1, 2, 2, 1],  # Roads and walls
        [1, 2, 1, 1, 1, 2, 1],  # Maze structure
        [1, 2, 2, 2, 2, 2, 1],  # Open area
        [1, 1, 1, 2, 1, 1, 1],  # Narrow passage
        [1, 2, 2, 2, 2, 2, 1],  # Open area
        [1, 1, 1, 1, 1, 1, 1],  # Walls
    ]
    
    env = DuckietownEnv(
        map_config={"layout": maze_layout},
        reward_function=get_reward_function('exploration')
    )
    
    print(f"Custom maze: {env.map.width_tiles}x{env.map.height_tiles} tiles")
    print("Map layout:")
    print(env.map)
    
    # Test navigation in maze
    obs, info = env.reset()
    print(f"Starting position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    
    env.close()
    print()


def navigation_task_example():
    """Example of navigation task with target."""
    print("=== Navigation Task Example ===")
    
    # Target position (top-right corner)
    target_x, target_y = 4.0, 4.0
    
    env = DuckietownEnv(
        reward_function=get_reward_function('navigation', target_position=(target_x, target_y)),
        max_steps=200
    )
    
    obs, info = env.reset()
    start_x = info['robot_state']['x']
    start_y = info['robot_state']['y']
    
    print(f"Start: ({start_x:.3f}, {start_y:.3f})")
    print(f"Target: ({target_x:.3f}, {target_y:.3f})")
    
    total_reward = 0
    min_distance = float('inf')
    
    for step in range(100):
        # Simple navigation policy: move towards target
        current_x = info['robot_state']['x']
        current_y = info['robot_state']['y']
        current_theta = info['robot_state']['theta']
        
        # Calculate direction to target
        dx = target_x - current_x
        dy = target_y - current_y
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = target_angle - current_theta
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Simple control: turn towards target, then move forward
        if abs(angle_diff) > 0.2:  # Need to turn
            if angle_diff > 0:
                action = np.array([1.0, -1.0])  # Turn left
            else:
                action = np.array([-1.0, 1.0])  # Turn right
        else:
            action = np.array([2.0, 2.0])  # Move forward
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Track closest approach
        distance = np.sqrt(dx**2 + dy**2)
        min_distance = min(min_distance, distance)
        
        if step % 20 == 0:
            print(f"Step {step}: pos=({current_x:.3f}, {current_y:.3f}), "
                  f"distance={distance:.3f}, reward={total_reward:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"Final distance to target: {distance:.3f}")
    print(f"Closest approach: {min_distance:.3f}")
    print(f"Total reward: {total_reward:.3f}")
    
    env.close()
    print()


def training_loop_example():
    """Example of basic training loop structure."""
    print("=== Training Loop Example ===")
    
    env = DuckietownEnv(
        reward_function=get_reward_function('lane_following'),
        max_steps=200
    )
    
    # Simple policy: mostly forward with some exploration
    def simple_policy(obs):
        """Simple policy for demonstration."""
        # Extract position and angle from observation
        x, y, theta, lin_vel, ang_vel, left_wheel, right_wheel = obs
        
        # Base forward motion
        base_action = np.array([2.0, 2.0])
        
        # Add some noise for exploration
        noise = np.random.normal(0, 0.3, 2)
        action = base_action + noise
        
        # Clip to valid range
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        return action
    
    # Run multiple episodes
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action = simple_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: {step_count} steps, reward: {episode_reward:.3f}")
        
        if info['collision']:
            print("  Ended in collision")
        elif step_count >= env.max_steps:
            print("  Ended due to time limit")
    
    env.close()
    print()


def main():
    """Run all examples."""
    print("=== Duckietown Gym Interface Examples ===\n")
    
    basic_usage_example()
    different_environments_example()
    custom_reward_example()
    custom_map_example()
    navigation_task_example()
    training_loop_example()
    
    print("=== All examples completed! ===")


if __name__ == "__main__":
    main()
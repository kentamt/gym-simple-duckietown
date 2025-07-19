#!/usr/bin/env python3
"""
Test script for the Duckietown Gym interface.

This script demonstrates how to use the Duckietown environment
with the OpenAI Gym interface.
"""

import numpy as np
import sys
import time
sys.path.append('.')

# Import the environment
from duckietown_simulator.environment import DuckietownEnv, make_env
from duckietown_simulator.environment.reward_functions import get_reward_function


def test_basic_env():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")
    
    # Create environment
    env = DuckietownEnv()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    for step in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Robot position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("Basic test completed!\n")


def test_make_env():
    """Test environment factory function."""
    print("Testing environment factory...")
    
    # Test different map configurations
    env1 = make_env("default")
    env2 = make_env("loop")
    env3 = make_env("small")
    
    print(f"Default env action space: {env1.action_space}")
    print(f"Loop env action space: {env2.action_space}")  
    print(f"Small env action space: {env3.action_space}")
    
    # Quick test
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    obs3, _ = env3.reset()
    
    print(f"Default map size: {env1.map.width_meters:.1f}x{env1.map.height_meters:.1f}m")
    print(f"Loop map size: {env2.map.width_meters:.1f}x{env2.map.height_meters:.1f}m")
    print(f"Small map size: {env3.map.width_meters:.1f}x{env3.map.height_meters:.1f}m")
    
    env1.close()
    env2.close()
    env3.close()
    print("Factory test completed!\n")


def test_reward_functions():
    """Test different reward functions."""
    print("Testing reward functions...")
    
    # Test with lane following reward
    env = DuckietownEnv(
        reward_function=get_reward_function('lane_following')
    )
    
    obs, _ = env.reset()
    
    # Test forward motion (should get positive reward)
    forward_action = np.array([2.0, 2.0])  # Both wheels forward
    obs, reward, _, _, info = env.step(forward_action)
    print(f"Forward motion reward: {reward:.3f}")
    
    # Test turning motion (should get lower reward)
    turn_action = np.array([2.0, -2.0])  # Turn in place
    obs, reward, _, _, info = env.step(turn_action)
    print(f"Turning motion reward: {reward:.3f}")
    
    env.close()
    
    # Test with racing reward
    env = DuckietownEnv(
        reward_function=get_reward_function('racing')
    )
    
    obs, _ = env.reset()
    
    # Test high speed forward motion
    fast_action = np.array([5.0, 5.0])  # High speed
    obs, reward, _, _, info = env.step(fast_action)
    print(f"Racing (high speed) reward: {reward:.3f}")
    
    env.close()
    print("Reward function test completed!\n")


def test_custom_map():
    """Test with custom map configuration."""
    print("Testing custom map...")
    
    # Create custom map layout
    custom_layout = [
        [1, 1, 1, 1, 1],  # Wall
        [1, 2, 2, 2, 1],  # Road with walls
        [1, 2, 0, 2, 1],  # Road with empty space
        [1, 2, 2, 2, 1],  # Road
        [1, 1, 1, 1, 1],  # Wall
    ]
    
    map_config = {
        "layout": custom_layout,
        "tile_size": 0.61
    }
    
    env = DuckietownEnv(map_config=map_config)
    
    print(f"Custom map tiles: {env.map.width_tiles}x{env.map.height_tiles}")
    print(f"Custom map size: {env.map.width_meters:.1f}x{env.map.height_meters:.1f}m")
    
    obs, info = env.reset()
    print(f"Starting position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    
    env.close()
    print("Custom map test completed!\n")


def test_collision_detection():
    """Test collision detection."""
    print("Testing collision detection...")
    
    # Create small environment for easier collision testing
    env = make_env("small")
    
    obs, info = env.reset()
    print(f"Starting at: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    
    # Move towards boundary to trigger collision
    for step in range(20):
        # Move diagonally towards corner (should hit boundary)
        action = np.array([3.0, 2.0])  # Slightly curved path
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos_x = info['robot_state']['x']
        pos_y = info['robot_state']['y']
        collision = info['collision']
        
        print(f"Step {step + 1}: pos=({pos_x:.3f}, {pos_y:.3f}), collision={collision}, reward={reward:.2f}")
        
        if collision:
            print("Collision detected!")
            break
        
        if terminated or truncated:
            break
    
    env.close()
    print("Collision test completed!\n")


def run_simple_agent():
    """Run a simple agent for demonstration."""
    print("Running simple agent demonstration...")
    
    env = DuckietownEnv(
        reward_function=get_reward_function('lane_following'),
        max_steps=100
    )
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    
    print("Running simple forward-moving agent...")
    
    while True:
        # Simple policy: mostly move forward with small random variations
        base_speed = 2.0
        noise = np.random.normal(0, 0.1, 2)  # Small noise
        action = np.array([base_speed, base_speed]) + noise
        
        # Clip to action space bounds
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step_count % 20 == 0:
            pos_x = info['robot_state']['x']
            pos_y = info['robot_state']['y']
            collision = info['collision']
            print(f"Step {step_count}: pos=({pos_x:.3f}, {pos_y:.3f}), "
                  f"reward={reward:.3f}, total_reward={total_reward:.3f}, collision={collision}")
        
        if terminated or truncated:
            print(f"Episode finished after {step_count} steps")
            print(f"Final total reward: {total_reward:.3f}")
            print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            break
    
    env.close()
    print("Simple agent demonstration completed!\n")


def main():
    """Run all tests."""
    print("=== Duckietown Gym Interface Test ===\n")
    
    try:
        test_basic_env()
        test_make_env()
        test_reward_functions()
        test_custom_map()
        test_collision_detection()
        run_simple_agent()
        
        print("=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
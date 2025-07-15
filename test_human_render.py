#!/usr/bin/env python3
"""
Test script for human rendering mode.

This script demonstrates the visual rendering capabilities of the Duckietown environment.
"""

import numpy as np
import time
import sys
sys.path.append('.')

from duckietown_simulator.environment import DuckietownEnv, make_env
from duckietown_simulator.environment.reward_functions import get_reward_function


def test_human_render_basic():
    """Test basic human rendering with manual control."""
    print("Testing human rendering mode...")
    print("Control the robot with WASD keys (if implemented) or watch autonomous movement")
    print("Press ESC to quit, SPACE to pause")
    
    # Create environment with human rendering
    env = DuckietownEnv(
        map_config={"width": 5, "height": 5, "track_type": "straight"},
        reward_function=get_reward_function('lane_following'),
        render_mode="human",
        max_steps=1000
    )
    
    obs, info = env.reset()
    step_count = 0
    total_reward = 0
    
    print(f"Starting at position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    
    try:
        while step_count < 500:  # Run for 500 steps max
            # Simple forward motion with small random steering
            base_speed = 2.0
            steering_noise = np.random.normal(0, 0.2)
            action = np.array([base_speed + steering_noise, base_speed - steering_noise])
            
            # Clip to valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render the environment
            env.render()
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                pos_x = info['robot_state']['x']
                pos_y = info['robot_state']['y']
                collision = info['collision']
                print(f"Step {step_count}: pos=({pos_x:.3f}, {pos_y:.3f}), "
                      f"reward={reward:.3f}, total={total_reward:.3f}, collision={collision}")
            
            # Small delay to make it watchable
            time.sleep(0.05)  # 20 FPS
            
            if terminated or truncated:
                print("Episode ended!")
                break
        
        print(f"Test completed after {step_count} steps")
        print(f"Total reward: {total_reward:.3f}")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def test_human_render_navigation():
    """Test human rendering with navigation task."""
    print("\nTesting navigation task with human rendering...")
    print("Watch the robot try to navigate to target position")
    
    # Target in corner
    target_x, target_y = 4.0, 4.0
    
    env = DuckietownEnv(
        map_config={"width": 6, "height": 6, "track_type": "straight"},
        reward_function=get_reward_function('navigation', target_position=(target_x, target_y)),
        render_mode="human",
        max_steps=300
    )
    
    obs, info = env.reset()
    start_x = info['robot_state']['x']
    start_y = info['robot_state']['y']
    
    print(f"Start: ({start_x:.3f}, {start_y:.3f})")
    print(f"Target: ({target_x:.3f}, {target_y:.3f})")
    
    step_count = 0
    total_reward = 0
    min_distance = float('inf')
    
    try:
        while step_count < 300:
            # Simple navigation: head towards target
            current_x = info['robot_state']['x']
            current_y = info['robot_state']['y']
            current_theta = info['robot_state']['theta']
            
            # Calculate direction to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)
            min_distance = min(min_distance, distance)
            
            target_angle = np.arctan2(dy, dx)
            angle_diff = target_angle - current_theta
            
            # Normalize angle difference
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Simple control policy
            if abs(angle_diff) > 0.3:  # Need to turn
                if angle_diff > 0:
                    action = np.array([1.5, -1.0])  # Turn left
                else:
                    action = np.array([-1.0, 1.5])  # Turn right
            else:
                # Move forward with proportional speed based on distance
                speed = min(3.0, distance * 2.0)
                action = np.array([speed, speed])
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            
            # Status update
            if step_count % 30 == 0:
                print(f"Step {step_count}: distance={distance:.3f}, reward={total_reward:.3f}")
            
            # Small delay
            time.sleep(0.05)
            
            # Stop if we reach the target
            if distance < 0.3:
                print("Target reached!")
                break
            
            if terminated or truncated:
                break
        
        print(f"Navigation test completed after {step_count} steps")
        print(f"Final distance: {distance:.3f}")
        print(f"Closest approach: {min_distance:.3f}")
        print(f"Total reward: {total_reward:.3f}")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def test_rgb_array_mode():
    """Test RGB array rendering mode."""
    print("\nTesting RGB array rendering mode...")
    
    env = DuckietownEnv(
        map_config={"width": 3, "height": 3, "track_type": "straight"},
        render_mode="rgb_array",
        max_steps=50
    )
    
    obs, info = env.reset()
    
    try:
        for step in range(10):
            action = np.array([2.0, 2.0])  # Move forward
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get RGB array
            rgb_array = env.render()
            
            if rgb_array is not None:
                print(f"Step {step}: RGB array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
                print(f"  Min/Max values: {rgb_array.min()}/{rgb_array.max()}")
            else:
                print(f"Step {step}: No RGB array returned")
            
            if terminated or truncated:
                break
        
        print("RGB array test completed successfully")
        
    except Exception as e:
        print(f"Error during RGB array test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    """Run rendering tests."""
    print("=== Duckietown Human Rendering Test ===\n")
    
    try:
        # Check if pygame is available
        import pygame
        print(f"Pygame version: {pygame.version.ver}")
        
        test_human_render_basic()
        test_human_render_navigation()
        test_rgb_array_mode()
        
        print("\n=== All rendering tests completed! ===")
        
    except ImportError:
        print("Error: pygame is not installed. Install with: pip install pygame")
        return 1
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
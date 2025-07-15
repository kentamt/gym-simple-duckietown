#!/usr/bin/env python3
"""
Test script for simple text-based rendering.

This demonstrates the fallback renderer when pygame is not available.
"""

import numpy as np
import time
import sys
sys.path.append('.')

from duckietown_simulator.environment import DuckietownEnv
from duckietown_simulator.environment.reward_functions import get_reward_function


def test_simple_human_render():
    """Test simple text-based human rendering."""
    print("Testing simple text-based rendering...")
    print("This will show ASCII art visualization in the terminal")
    
    # Create environment with human rendering
    env = DuckietownEnv(
        map_config={"width": 5, "height": 5, "track_type": "straight"},
        reward_function=get_reward_function('lane_following'),
        render_mode="human",
        max_steps=100
    )
    
    obs, info = env.reset()
    step_count = 0
    total_reward = 0
    
    try:
        while step_count < 20:  # Run for 20 steps for demonstration
            # Simple forward motion with small turns
            base_speed = 2.0
            steering = 0.5 * np.sin(step_count * 0.3)  # Gentle S-curve
            action = np.array([base_speed + steering, base_speed - steering])
            
            # Clip to valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render the environment
            env.render()
            
            # Wait a bit so we can see the changes
            time.sleep(1.0)
            
            if terminated or truncated:
                print("Episode ended!")
                break
        
        print(f"\nDemo completed after {step_count} steps")
        print(f"Total reward: {total_reward:.3f}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def test_navigation_simple():
    """Test navigation with simple rendering."""
    print("\n" + "="*60)
    print("Testing navigation task with simple rendering...")
    
    # Create a simple 3x3 environment for easier visualization
    env = DuckietownEnv(
        map_config={"width": 3, "height": 3, "track_type": "straight"},
        reward_function=get_reward_function('exploration'),
        render_mode="human",
        max_steps=50
    )
    
    obs, info = env.reset()
    
    try:
        for step in range(15):
            # Simple forward motion
            action = np.array([1.5, 1.5])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            # Wait to see changes
            time.sleep(0.8)
            
            if terminated or truncated:
                break
        
        print(f"\nNavigation demo completed")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def test_matplotlib_render():
    """Test matplotlib rendering if available."""
    print("\n" + "="*60)
    print("Testing matplotlib rendering...")
    
    try:
        import matplotlib.pyplot as plt
        print("Matplotlib available - testing visual render")
        
        from duckietown_simulator.rendering.simple_renderer import MatplotlibRenderer
        from duckietown_simulator.world.map import create_map_from_config
        from duckietown_simulator.robot.duckiebot import create_duckiebot
        
        # Create map and robot
        map_instance = create_map_from_config(4, 4, "straight")
        robot = create_duckiebot(2.0, 2.0, 0.5)
        
        # Create renderer
        renderer = MatplotlibRenderer(map_instance)
        renderer.set_robots({"robot": robot})
        
        # Render and save
        renderer.render(save_path="test_render.png")
        print("Saved matplotlib render to test_render.png")
        
        renderer.close()
        
    except ImportError:
        print("Matplotlib not available - skipping visual test")
    except Exception as e:
        print(f"Error with matplotlib test: {e}")


def main():
    """Run rendering tests."""
    print("=== Duckietown Simple Rendering Test ===\n")
    
    try:
        test_simple_human_render()
        test_navigation_simple()
        test_matplotlib_render()
        
        print("\n=== All simple rendering tests completed! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Demo of human rendering for Duckietown environment.

This script demonstrates both text-based and visual rendering modes.
"""

import numpy as np
import sys
import time
sys.path.append('.')

from duckietown_simulator.environment import DuckietownEnv
from duckietown_simulator.environment.reward_functions import get_reward_function


def demo_text_rendering():
    """Demonstrate text-based rendering."""
    print("=== Text-Based Rendering Demo ===")
    print("This shows ASCII art visualization of the robot moving through the map.")
    print()
    
    # Create environment with text rendering
    env = DuckietownEnv(
        map_config={"width": 4, "height": 4, "track_type": "straight"},
        reward_function=get_reward_function('lane_following'),
        # render_mode="acii",
        max_steps=50
    )
    
    obs, info = env.reset()
    
    print(f"Initial robot position: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
    print("Press Ctrl+C to stop the demo early\n")
    
    try:
        for step in range(100):  # Show 8 steps
            # Simple movement pattern
            if step < 4:
                action = np.array([2.0, 2.0])  # Move forward
            else:
                action = np.array([1.0, 3.0])  # Turn slightly
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render current state
            print(f"\n--- Step {step + 1} ---")
            env.render()
            print(f"Action: Left={action[0]:.1f}, Right={action[1]:.1f}")
            print(f"Reward: {reward:.3f}")
            
            if terminated or truncated:
                print("Episode ended!")
                break
            
            # Pause to make it readable
            print("\n(Waiting 0.1 seconds...)")
            time.sleep(0.01)
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    env.close()
    print("\nText rendering demo completed!")


def demo_matplotlib_rendering():
    """Demonstrate matplotlib rendering if available."""
    print("\n=== Matplotlib Rendering Demo ===")
    
    try:
        import matplotlib.pyplot as plt
        
        from duckietown_simulator.rendering.simple_renderer import MatplotlibRenderer
        from duckietown_simulator.world.map import create_map_from_config
        from duckietown_simulator.robot.duckiebot import create_duckiebot
        
        print("Creating visual render with matplotlib...")
        
        # Create a map and robot
        map_instance = create_map_from_config(5, 5, "loop")
        robot = create_duckiebot(1.5, 1.5, np.pi/4)  # 45 degree angle
        
        # Move robot to show trajectory
        positions = [
            (1.5, 1.5, np.pi/4),
            (2.0, 2.0, np.pi/2),
            (2.5, 2.8, 3*np.pi/4),
            (2.8, 3.2, np.pi),
        ]
        
        renderer = MatplotlibRenderer(map_instance, figsize=(10, 8))
        
        for i, (x, y, theta) in enumerate(positions):
            robot.reset(x, y, theta)
            renderer.set_robots({"robot": robot})
            
            filename = f"duckietown_render_step_{i+1}.png"
            renderer.render(save_path=filename)
            print(f"Saved visualization to {filename}")
        
        renderer.close()
        print("Matplotlib rendering demo completed!")
        
    except ImportError:
        print("Matplotlib not available - skipping visual demo")
        print("Install matplotlib with: pip install matplotlib")
    except Exception as e:
        print(f"Error in matplotlib demo: {e}")


def demo_pygame_info():
    """Show information about pygame rendering."""
    print("\n=== Pygame Rendering Info ===")
    
    try:
        import pygame
        print(f"✅ Pygame is available (version {pygame.version.ver})")
        print("Full interactive rendering with real-time visualization is available!")
        print("Run with pygame installed for the complete experience.")
    except ImportError:
        print("❌ Pygame not available")
        print("Install pygame for full interactive rendering: pip install pygame")
        print("Features with pygame:")
        print("  - Real-time interactive visualization")
        print("  - Mouse control (pan, zoom)")
        print("  - Keyboard shortcuts")
        print("  - Collision visualization")
        print("  - Robot trajectory tracking")
        print("  - Smooth animations")


def main():
    """Run rendering demos."""
    print("🤖 Duckietown Environment Rendering Demo")
    print("=" * 50)
    
    demo_text_rendering()
    # demo_matplotlib_rendering()
    # demo_pygame_info()
    
    print("\n" + "=" * 50)
    print("📚 Rendering Options Summary:")
    print("1. Text Mode: Always available, shows ASCII art map")
    print("2. Matplotlib: Visual plots (install matplotlib)")
    print("3. Pygame: Full interactive real-time rendering (install pygame)")
    print("\nFor the best experience, install both matplotlib and pygame!")


if __name__ == "__main__":
    main()
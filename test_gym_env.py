#!/usr/bin/env python3
"""
Test script for the PID Road Network Gym Environment.
Demonstrates the discrete action space {STOP, GO}.
"""

import numpy as np
from gym_pid_road_network import make_env


def test_stop_go_actions():
    """Test the STOP and GO actions."""
    print("Testing PID Road Network Gym Environment")
    print("=" * 50)
    
    # Create environment
    env = make_env(trajectory_file="trajectory_1.json", render_mode="human")
    
    print(f"Action space: {env.action_space}")
    print(f"  0: STOP (zero wheel speeds)")
    print(f"  1: GO (follow PID trajectory)")
    print()
    print(f"Observation space: {env.observation_space}")
    print(f"  [x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress]")
    print()
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info['waypoint_progress']}")
    print()
    
    # Test sequence: GO, STOP, GO pattern
    action_sequence = [
        (1, "GO", 200),    # GO for 20 steps
        (0, "STOP", 200),  # STOP for 10 steps
        (1, "GO", 200),    # GO for 20 steps
        (0, "STOP", 100),  # STOP for 10 steps
        (1, "GO", 500),    # GO for 50 steps
    ]
    
    total_steps = 0
    
    for action, action_name, num_steps in action_sequence:
        print(f"\n--- {action_name} for {num_steps} steps ---")
        
        for step in range(num_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render environment
            if not env.render():
                print("Window closed!")
                return
            
            # Print status every 10 steps
            if step % 10 == 0:
                print(f"  Step {total_steps}: Action={action_name}, Reward={reward:.2f}, "
                      f"Speed={info['robot_speeds']['linear']:.2f}m/s, "
                      f"Progress={info['waypoint_progress']['progress_ratio']*100:.1f}%, "
                      f"Collisions={info['collisions']}")
            
            total_steps += 1
            
            # Check if episode ended
            if terminated or truncated:
                print(f"\nEpisode ended at step {total_steps}!")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Final progress: {info['waypoint_progress']['progress_ratio']*100:.1f}%")
                env.close()
                return
    
    print(f"\nTest completed! Total steps: {total_steps}")
    env.close()


def test_random_policy():
    """Test with random STOP/GO actions."""
    print("\n" + "=" * 50)
    print("Testing Random STOP/GO Policy")
    print("=" * 50)
    
    env = make_env(trajectory_file="trajectory_1.json", render_mode="human")
    
    obs, info = env.reset()
    
    for step in range(100):
        # Random action: 80% GO, 20% STOP
        action = np.random.choice([0, 1], p=[0.2, 0.8])
        action_name = "STOP" if action == 0 else "GO"
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if not env.render():
            break
        
        if step % 20 == 0:
            print(f"Step {step}: {action_name}, Reward={reward:.2f}, "
                  f"Speed={info['robot_speeds']['linear']:.2f}m/s, "
                  f"Progress={info['waypoint_progress']['progress_ratio']*100:.1f}%")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}!")
            break
    
    env.close()


if __name__ == "__main__":
    # Test deterministic STOP/GO pattern
    test_stop_go_actions()
    
    # Test random policy
    # test_random_policy()
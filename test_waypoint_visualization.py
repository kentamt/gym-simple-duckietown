#!/usr/bin/env python3
"""
Quick test for waypoint visualization.
"""

import numpy as np
import math
from duckietown_simulator.environment.discrete_action_env import create_discrete_duckietown_env
from duckietown_simulator.environment.reward_functions import get_reward_function
from duckietown_simulator.robot.discrete_action_mapper import DiscreteAction

def test_waypoint_visualization():
    """Test discrete actions with waypoint visualization."""
    print("🎯 Testing Waypoint Visualization")
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
        render_mode="human",  # Use pygame rendering
        max_steps=50,
        forward_distance=0.8,  # Smaller steps for better visualization
        turn_radius=0.6,       # Smaller turn radius
        turn_angle=math.pi/4   # 45-degree turns
    )
    
    print("Environment created with pygame visualization!")
    print("🎯 Yellow crosshair shows current waypoint target")
    print("🤖 Robot will follow discrete actions to reach waypoints")
    print("⌨️  Press 'W' to toggle waypoint display on/off")
    print("📍 Watch the waypoint target move as robot executes actions")
    print("\nStarting demo...")
    
    # Reset environment
    obs, info = env.reset()
    
    # Simple test sequence
    actions = [
        DiscreteAction.FORWARD,
        DiscreteAction.TURN_LEFT,
        DiscreteAction.FORWARD,
        DiscreteAction.TURN_RIGHT,
        DiscreteAction.FORWARD,
        DiscreteAction.STOP
    ]
    
    try:
        step_count = 0
        action_index = 0
        
        for _ in range(50):  # Limited steps for testing
            # Get current action
            if action_index < len(actions):
                current_action = actions[action_index]
            else:
                current_action = DiscreteAction.STOP
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(current_action)
            step_count += 1
            
            # Print step information
            discrete_info = info['discrete_action']
            print(f"Step {step_count}: {discrete_info['action_name']} - Status: {discrete_info['control_status']}")
            
            # Move to next action when current one completes
            if discrete_info['action_completed'] and action_index < len(actions) - 1:
                action_index += 1
                print(f"  → Action completed! Next: {actions[action_index].name}")
            
            # Render environment
            env.render()
            
            # Check for episode end
            if terminated or truncated:
                print(f"Episode ended after {step_count} steps!")
                break
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    env.close()
    print("\nWaypoint visualization test completed!")

if __name__ == "__main__":
    test_waypoint_visualization()
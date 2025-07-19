#!/usr/bin/env python3
"""
Test script for the Multi-Agent PID Road Network Gym Environment.
Demonstrates multi-agent discrete action space {STOP, GO}.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path to import from examples
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples.multi_agent_gym_env import make_multi_agent_env


def test_multi_agent_basic():
    """Test basic multi-agent functionality."""
    print("Testing Multi-Agent PID Road Network Gym Environment")
    print("=" * 60)
    
    # Create environment with 2 agents
    env = make_multi_agent_env(
        num_agents=3,
        trajectory_files=["../data/exp_traj_1.json",
                          "../data/exp_traj_2.json",
                          "../data/exp_traj_3.json",
                          ],  # Trajectory files from data directory
        render_mode='human'  # Use human rendering for visualization
    )
    
    print(f"Agent IDs: {env.agent_ids}")
    print(f"Action spaces: {env.action_space}")
    print(f"Number of agents: {env.num_agents}")
    print()
    
    # Reset environment
    obs, infos = env.reset()
    print(f"Initial observations shape: {list(obs.keys())} -> {[obs[k].shape for k in obs.keys()]}")
    print(f"Observation details for robot1: {obs['robot1']}")
    print()
    
    # Test coordinated actions
    action_sequences = [
        ("All GO", {agent_id: 1 for agent_id in env.agent_ids}, 10000),
        ("robot1 STOP, robot2 GO", {"robot1": 0, "robot2": 1}, 50),
        ("Alternating", {"robot1": 1, "robot2": 0}, 50),
        ("All STOP", {agent_id: 0 for agent_id in env.agent_ids}, 30),
        ("Random actions", "random", 70),
    ]
    
    total_steps = 0
    
    for seq_name, actions, num_steps in action_sequences:
        print(f"\n--- {seq_name} for {num_steps} steps ---")
        
        for step in range(num_steps):
            if actions == "random":
                # Random actions for each agent
                actions_dict = {agent_id: np.random.choice([0, 1]) for agent_id in env.agent_ids}
            else:
                actions_dict = actions
            
            obs, rewards, terminated, truncated, infos = env.step(actions_dict)
            
            # Render environment
            if not env.render():
                print("Window closed!")
                return
            
            # Print status every 25 steps
            if step % 25 == 0:
                print(f"  Step {total_steps}:")
                for agent_id in env.agent_ids:
                    action_name = "STOP" if actions_dict[agent_id] == 0 else "GO"
                    print(f"    {agent_id}: Action={action_name}, Reward={rewards[agent_id]:.2f}, "
                          f"Progress={infos[agent_id]['waypoint_progress']['progress_ratio']*100:.1f}%, "
                          f"Collisions={infos[agent_id]['collisions']}")
                    # show robots position and speed
                    print(f"    {agent_id} Position: ({obs[agent_id][0]:.2f}, {obs[agent_id][1]:.2f}), "
                          f"Speed: {infos[agent_id]['robot_speeds']['linear']:.2f} m/s")
            
            total_steps += 1
            
            # Check if any agent completed or episode ended
            if any(terminated.values()) or any(truncated.values()):
                print(f"\nEpisode ended at step {total_steps}!")
                for agent_id in env.agent_ids:
                    print(f"  {agent_id}: Terminated={terminated[agent_id]}, Truncated={truncated[agent_id]}")
                env.close()
                return
            
            # time.sleep(0.03)  # Slow down for better visualization
    
    print(f"\nTest completed! Total steps: {total_steps}")
    env.close()


def test_collision_scenarios():
    """Test collision detection in multi-agent environment."""
    print("\n" + "=" * 60)
    print("Testing Multi-Agent Collision Detection")
    print("=" * 60)
    
    # Create environment with 3 agents using different trajectories
    env = make_multi_agent_env(
        num_agents=3,
        trajectory_files={"robot1": "data/trajectory_1.json", "robot2": "data/trajectory_2.json"},
        render_mode=None
    )
    
    obs, infos = env.reset()
    
    # Run scenario where agents might collide
    for step in range(150):
        # Policy: robot1 always GO, robot2 and robot3 alternate to create potential collisions
        actions = {
            "robot1": 1,  # Always GO
            "robot2": 1 if step < 75 else step % 2,  # GO then alternate
            "robot3": 1 if step < 50 else 0  # GO then STOP
        }
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        if not env.render():
            break
        
        # Print collision information
        if step % 30 == 0:
            print(f"\nStep {step} Collision Status:")
            for agent_id in env.agent_ids:
                collisions = infos[agent_id]['collisions']
                collision_details = infos[agent_id]['collision_details']
                action_name = "STOP" if actions[agent_id] == 0 else "GO"
                print(f"  {agent_id}: {action_name}, Collisions={collisions}, Reward={rewards[agent_id]:.2f}")
                if collision_details:
                    for detail in collision_details:
                        print(f"    - {detail['type']} with {detail.get('other_robot_id', 'obstacle')}")
        
        if any(terminated.values()) or any(truncated.values()):
            print(f"\nEpisode ended at step {step}!")
            break
        
        # time.sleep(0.05)
    
    env.close()


def test_observation_spaces():
    """Test observation space dimensions and content."""
    print("\n" + "=" * 60)
    print("Testing Multi-Agent Observation Spaces")
    print("=" * 60)
    
    env = make_multi_agent_env(num_agents=2, render_mode=None)  # No rendering for this test
    
    obs, infos = env.reset()
    
    print("Observation space analysis:")
    for agent_id in env.agent_ids:
        print(f"\n{agent_id}:")
        print(f"  Observation shape: {obs[agent_id].shape}")
        print(f"  Expected: [x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress, other_agents...]")
        print(f"  Actual values: {obs[agent_id]}")
        
        # Verify observation bounds
        obs_space = env.observation_space[agent_id]
        print(f"  Low bounds: {obs_space.low}")
        print(f"  High bounds: {obs_space.high}")
        
        # Check if observation is within bounds
        within_bounds = np.all(obs[agent_id] >= obs_space.low) and np.all(obs[agent_id] <= obs_space.high)
        print(f"  Within bounds: {within_bounds}")
    
    # Test a few steps
    for step in range(5):
        actions = {agent_id: 1 for agent_id in env.agent_ids}  # All GO
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        print(f"\nStep {step}:")
        for agent_id in env.agent_ids:
            within_bounds = np.all(obs[agent_id] >= env.observation_space[agent_id].low) and \
                           np.all(obs[agent_id] <= env.observation_space[agent_id].high)
            print(f"  {agent_id} obs within bounds: {within_bounds}")
    
    env.close()


if __name__ == "__main__":
    # Test basic multi-agent functionality
    test_multi_agent_basic()
    
    # Test collision scenarios
    test_collision_scenarios()
    
    # Test observation spaces
    test_observation_spaces()
    
    print("\nAll multi-agent tests completed!")
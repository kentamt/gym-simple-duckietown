#!/usr/bin/env python3
"""
Multi-Agent Gym environment for PID road network with discrete action spaces.
Each agent has its own action space: {STOP, GO} and trajectory to follow.
"""

import sys
import numpy as np
import json
import os
import argparse
import pygame
from typing import Dict, List, Tuple, Any, Optional
sys.path.append('.')

# Try to import gymnasium, fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_VERSION = "gym"
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is installed. Please install one of them.")

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.robot.pid_controller import WaypointFollowPIDController, PIDConfig, PIDGains
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig
from duckietown_simulator.world.collision_detection import CollisionDetector, CollisionResult
from duckietown_simulator.world.obstacles import ObstacleConfig, ObstacleType

# Import trajectory utilities from the demos directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from demos.demo_pid_road_network import (
    load_trajectory_from_file, create_robot_trajectories, create_road_network_map,
    interpolate_trajectory
)


class MultiAgentPIDRoadNetworkEnv(gym.Env):
    """
    Multi-Agent Gym environment for PID-controlled robots on road network.
    
    Each agent has a discrete action space: {0: STOP, 1: GO}
    - STOP (0): Zero wheel speeds, robot stops
    - GO (1): Follow PID trajectory toward next waypoint
    
    Supports variable number of agents, each with their own trajectory.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, num_agents=2, trajectory_files=None, map_config=None, render_mode=None):
        """
        Initialize the multi-agent environment.
        
        Args:
            num_agents: Number of agents (robots) in the environment
            trajectory_files: Dict mapping agent_id to trajectory file, or list of files
            map_config: Map configuration (optional, defaults to road network)
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Store config
        self.num_agents = num_agents
        self.trajectory_files = trajectory_files
        self.map_config = map_config
        self.render_mode = render_mode
        
        # Create agent IDs
        self.agent_ids = [f"robot{i+1}" for i in range(num_agents)]
        
        # Action spaces: each agent has {STOP, GO}
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(2) for agent_id in self.agent_ids
        })
        
        # Observation spaces: each agent observes own state + other agents' positions
        # Own state: [x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress]
        # Others: [other_x, other_y, other_collision] for each other agent
        obs_dim_self = 8  # Own state
        obs_dim_others = 3 * (num_agents - 1)  # Other agents' states
        total_obs_dim = obs_dim_self + obs_dim_others
        
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=np.concatenate([
                    np.array([-10.0, -10.0, -np.pi, -5.0, -5.0, 0.0, 0.0, 0.0]),  # Own state
                    np.tile(np.array([-10.0, -10.0, 0.0]), num_agents - 1)  # Others
                ]),
                high=np.concatenate([
                    np.array([10.0, 10.0, np.pi, 5.0, 5.0, 10.0, 1.0, 1.0]),  # Own state
                    np.tile(np.array([10.0, 10.0, 1.0]), num_agents - 1)  # Others
                ]),
                dtype=np.float32
            ) for agent_id in self.agent_ids
        })
        
        # Environment state
        self.map_instance = None
        self.robots = {}  # Dict of robot_id -> robot
        self.controllers = {}  # Dict of robot_id -> controller
        self.trajectories = {}  # Dict of robot_id -> trajectory
        self.collision_detector = None
        self.renderer = None
        
        # Simulation state
        self.dt = 0.016  # 60 FPS
        self.step_count = 0
        self.max_steps = 2000
        self.collision_results = []
        self.robot_speeds = {}  # Dict of robot_id -> speed info
        
        # Initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize map, robots, and other components."""
        # Create map
        self.map_instance = create_road_network_map()
        
        # Initialize collision detector
        self.collision_detector = CollisionDetector(
            map_width=self.map_instance.width_meters,
            map_height=self.map_instance.height_meters
        )
        
        # Load trajectories for each agent
        self._load_trajectories()
        
        # Create robots and controllers
        self._create_robots_and_controllers()
        
        # Initialize renderer if needed
        if self.render_mode == "human":
            self._initialize_renderer()
    
    def _load_trajectories(self):
        """Load trajectories for each agent."""
        if self.trajectory_files:
            if isinstance(self.trajectory_files, dict):
                # Dict mapping agent_id to file
                trajectory_files_dict = self.trajectory_files
            elif isinstance(self.trajectory_files, list):
                # List of files, assign to agents in order
                trajectory_files_dict = {}
                for i, agent_id in enumerate(self.agent_ids):
                    if i < len(self.trajectory_files):
                        trajectory_files_dict[agent_id] = self.trajectory_files[i]
            else:
                trajectory_files_dict = {}
            
            # Load trajectories
            self.trajectories = create_robot_trajectories(
                self.map_instance, 
                trajectory_files_dict
            )
        else:
            # Use default trajectories
            default_trajectories = create_robot_trajectories(self.map_instance)
            
            # Assign trajectories to agents (duplicate if needed)
            self.trajectories = {}
            for i, agent_id in enumerate(self.agent_ids):
                if i == 0:
                    self.trajectories[agent_id] = default_trajectories['robot1']
                else:
                    # Create offset trajectories for other agents
                    base_trajectory = default_trajectories['robot1']
                    offset_x = 0.5 * i  # Offset by 0.5m for each additional robot
                    offset_trajectory = [(x + offset_x, y) for x, y in base_trajectory]
                    self.trajectories[agent_id] = offset_trajectory
        
        # Ensure all agents have trajectories (fallback for missing ones)
        if len(self.trajectories) < self.num_agents:
            default_trajectories = create_robot_trajectories(self.map_instance)
            base_trajectory = default_trajectories['robot1']
            
            for i, agent_id in enumerate(self.agent_ids):
                if agent_id not in self.trajectories:
                    # Create offset trajectory for missing agent
                    offset_x = 0.5 * i
                    offset_trajectory = [(x + offset_x, y) for x, y in base_trajectory]
                    self.trajectories[agent_id] = offset_trajectory
    
    def _create_robots_and_controllers(self):
        """Create robots and PID controllers for each agent."""
        self.robots = {}
        self.controllers = {}
        self.robot_speeds = {}
        
        for agent_id in self.agent_ids:
            # Get starting position from trajectory
            start_x, start_y = self.trajectories[agent_id][0]
            
            # Create robot
            self.robots[agent_id] = create_duckiebot(x=start_x, y=start_y, theta=0.0)
            
            # Create PID controller
            pid_config = PIDConfig(
                distance_gains=PIDGains(kp=0.5, ki=0.0, kd=0.0),
                heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
                max_linear_velocity=2.5,
                max_angular_velocity=4.0,
                position_tolerance=0.1
            )
            controller = WaypointFollowPIDController(pid_config)
            controller.set_waypoints(self.trajectories[agent_id])
            self.controllers[agent_id] = controller
            
            # Initialize speed tracking
            self.robot_speeds[agent_id] = {'linear': 0.0, 'angular': 0.0, 'total': 0.0}
    
    def _initialize_renderer(self):
        """Initialize pygame renderer for visualization."""
        config = RenderConfig(
            width=1400, height=900, fps=120,
            use_tile_images=True,
            show_grid=True,
            show_robot_ids=True
        )
        
        self.renderer = create_pygame_renderer(self.map_instance, config)
        self.renderer.set_robots(self.robots)
        self.renderer.planned_trajectories = self.trajectories
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset all robots
        for agent_id in self.agent_ids:
            start_x, start_y = self.trajectories[agent_id][0]
            self.robots[agent_id].reset(x=start_x, y=start_y, theta=0.0)
            
            # Reset controller
            self.controllers[agent_id].reset_controllers()
            self.controllers[agent_id].set_waypoints(self.trajectories[agent_id])
            
            # Reset speed tracking
            self.robot_speeds[agent_id] = {'linear': 0.0, 'angular': 0.0, 'total': 0.0}
        
        # Reset simulation state
        self.step_count = 0
        self.collision_results = []
        
        # Get initial observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos
    
    def step(self, actions):
        """Execute one step in the environment with multi-agent actions."""
        # Validate actions
        assert isinstance(actions, dict), "Actions must be a dictionary"
        for agent_id in self.agent_ids:
            assert agent_id in actions, f"Missing action for agent {agent_id}"
            assert self.action_space[agent_id].contains(actions[agent_id]), f"Invalid action for {agent_id}: {actions[agent_id]}"
        
        # Apply actions for each agent
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            robot = self.robots[agent_id]
            controller = self.controllers[agent_id]
            
            if action == 0:  # STOP
                # Zero wheel speeds
                linear_vel = 0.0
                angular_vel = 0.0
                omega_l = 0.0
                omega_r = 0.0
            else:  # GO (action == 1)
                # Use PID controller to compute velocities
                linear_vel, angular_vel, info = controller.compute_control(
                    robot.x, robot.y, robot.theta, self.dt
                )
                
                # Convert to wheel speeds
                omega_l, omega_r = controller.body_vel_to_wheel_speeds(
                    linear_vel, angular_vel,
                    robot.config.wheelbase, robot.config.wheel_radius
                )
            
            # Update robot speeds for tracking
            self.robot_speeds[agent_id]['linear'] = abs(linear_vel)
            self.robot_speeds[agent_id]['angular'] = abs(angular_vel)
            self.robot_speeds[agent_id]['total'] = np.sqrt(linear_vel**2 + angular_vel**2)
            
            # Apply control action to robot
            control_action = np.array([omega_l, omega_r])
            robot.step(control_action, self.dt)
        
        # Check for collisions
        self.collision_results = self.collision_detector.check_all_collisions(self.robots)
        
        # Calculate rewards for each agent
        rewards = self._calculate_rewards(actions)
        
        # Check if episode is done for each agent
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        
        # Get observations and info
        observations = self._get_observations()
        infos = self._get_infos()
        
        self.step_count += 1
        
        return observations, rewards, terminated, truncated, infos
    
    def _calculate_rewards(self, actions):
        """Calculate rewards for each agent."""
        rewards = {}
        
        # Get agents involved in collisions
        colliding_agents = set()
        for collision in self.collision_results:
            if collision.is_colliding:
                colliding_agents.add(collision.robot_id)
                if collision.other_robot_id:
                    colliding_agents.add(collision.other_robot_id)
        
        for agent_id in self.agent_ids:
            reward = 0.0
            action = actions[agent_id]
            
            # Base reward for making progress
            progress_info = self.controllers[agent_id].get_progress()
            
            # Reward for progress along trajectory
            if action == 1:  # GO action
                reward += 1.0  # Base reward for moving
                
                # Bonus for reaching waypoints or completing trajectory
                if progress_info['completed']:
                    reward += 100.0  # Large bonus for completing trajectory
            
            # Penalty for collisions
            if agent_id in colliding_agents:
                reward -= 10.0  # Penalty for being in collision
            
            # Small penalty for stopping when not in collision
            if action == 0 and agent_id not in colliding_agents:
                reward -= 0.1  # Small penalty for unnecessary stopping
            
            # Bonus for avoiding collisions while others are colliding
            if action == 0 and agent_id not in colliding_agents and len(colliding_agents) > 0:
                reward += 2.0  # Reward for smart collision avoidance
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _get_terminated(self):
        """Check if episode should terminate for each agent."""
        terminated = {}
        
        for agent_id in self.agent_ids:
            terminated[agent_id] = False
            
            # Terminate if trajectory is completed
            progress_info = self.controllers[agent_id].get_progress()
            if progress_info['completed']:
                terminated[agent_id] = True
                continue
            
            # Terminate if robot goes out of bounds
            robot = self.robots[agent_id]
            if (robot.x < 0 or robot.x > self.map_instance.width_meters or
                robot.y < 0 or robot.y > self.map_instance.height_meters):
                terminated[agent_id] = True
        
        return terminated
    
    def _get_truncated(self):
        """Check if episode should truncate (time limit)."""
        truncated_all = self.step_count >= self.max_steps
        return {agent_id: truncated_all for agent_id in self.agent_ids}
    
    def _get_observations(self):
        """Get observations for each agent."""
        observations = {}
        
        for agent_id in self.agent_ids:
            robot = self.robots[agent_id]
            controller = self.controllers[agent_id]
            
            # Own state
            progress_info = controller.get_progress()
            current_waypoint_idx = progress_info['current_waypoint']
            
            if current_waypoint_idx < len(self.trajectories[agent_id]):
                next_waypoint = self.trajectories[agent_id][current_waypoint_idx]
                dist_to_waypoint = np.sqrt(
                    (robot.x - next_waypoint[0])**2 + 
                    (robot.y - next_waypoint[1])**2
                )
            else:
                dist_to_waypoint = 0.0
            
            # Check collision status for this agent
            collision_status = 1.0 if any(
                c.is_colliding and c.robot_id == agent_id for c in self.collision_results
            ) else 0.0
            
            # Own state: [x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress]
            own_obs = np.array([
                robot.x,
                robot.y,
                robot.theta,
                self.robot_speeds[agent_id]['linear'],
                self.robot_speeds[agent_id]['angular'],
                dist_to_waypoint,
                collision_status,
                progress_info['progress_ratio']
            ], dtype=np.float32)
            
            # Other agents' states: [x, y, collision] for each other agent
            other_obs = []
            for other_agent_id in self.agent_ids:
                if other_agent_id != agent_id:
                    other_robot = self.robots[other_agent_id]
                    other_collision = 1.0 if any(
                        c.is_colliding and c.robot_id == other_agent_id for c in self.collision_results
                    ) else 0.0
                    
                    other_obs.extend([
                        other_robot.x,
                        other_robot.y,
                        other_collision
                    ])
            
            # Combine own and others' observations
            full_obs = np.concatenate([own_obs, np.array(other_obs, dtype=np.float32)])
            observations[agent_id] = full_obs
        
        return observations
    
    def _get_infos(self):
        """Get info dictionaries for each agent."""
        infos = {}
        
        for agent_id in self.agent_ids:
            progress_info = self.controllers[agent_id].get_progress()
            robot = self.robots[agent_id]
            
            infos[agent_id] = {
                'step_count': self.step_count,
                'robot_position': (robot.x, robot.y),
                'robot_theta': robot.theta,
                'waypoint_progress': progress_info,
                'collisions': len([c for c in self.collision_results if c.is_colliding and c.robot_id == agent_id]),
                'collision_details': [
                    {
                        'type': c.collision_type,
                        'robot_id': c.robot_id,
                        'other_robot_id': c.other_robot_id,
                        'obstacle_name': c.obstacle_name
                    } for c in self.collision_results if c.is_colliding and c.robot_id == agent_id
                ],
                'robot_speeds': self.robot_speeds[agent_id].copy(),
                'all_agents_positions': {aid: (self.robots[aid].x, self.robots[aid].y) for aid in self.agent_ids}
            }
        
        return infos
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.renderer is None:
                self._initialize_renderer()
            
            # Update renderer with current state
            self.renderer.set_robots(self.robots)
            self.renderer.set_collision_results(self.collision_results)
            
            # Render frame
            if not self.renderer.render():
                return False  # Window closed
            
            # Draw additional overlays
            self._draw_overlays()
            
            return True
        
        elif self.render_mode == "rgb_array":
            # TODO: Implement rgb_array rendering
            return np.zeros((900, 1400, 3), dtype=np.uint8)
    
    def _draw_overlays(self):
        """Draw multi-agent information overlays."""
        if self.renderer and hasattr(self.renderer, 'screen') and hasattr(self.renderer, 'font'):
            y_offset = 10
            
            # Draw collision summary
            active_collisions = len([c for c in self.collision_results if c.is_colliding])
            collision_text = f"Total Collisions: {active_collisions}"
            collision_color = (255, 0, 0) if active_collisions > 0 else (0, 255, 0)
            text_surface = self.renderer.font.render(collision_text, True, collision_color)
            
            background_rect = (10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 4)
            pygame.draw.rect(self.renderer.screen, (0, 0, 0, 180), background_rect)
            self.renderer.screen.blit(text_surface, (15, y_offset + 2))
            y_offset += 30
            
            # Draw agent information
            for agent_id in self.agent_ids:
                # Check if agent is in collision
                agent_in_collision = any(c.is_colliding and c.robot_id == agent_id for c in self.collision_results)
                
                # Create agent status text
                progress_info = self.controllers[agent_id].get_progress()
                status_text = f"{agent_id}: {self.robot_speeds[agent_id]['linear']:.2f}m/s, {progress_info['progress_ratio']*100:.0f}%"
                if agent_in_collision:
                    status_text += " [COLLISION]"
                
                text_color = (255, 255, 0) if agent_in_collision else (255, 255, 255)
                text_surface = self.renderer.font.render(status_text, True, text_color)
                
                background_rect = (10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 4)
                pygame.draw.rect(self.renderer.screen, (0, 0, 0, 180), background_rect)
                self.renderer.screen.blit(text_surface, (15, y_offset + 2))
                
                y_offset += 25
    
    def close(self):
        """Clean up environment resources."""
        if self.renderer:
            self.renderer.cleanup()
            self.renderer = None


def make_multi_agent_env(num_agents=2, trajectory_files=None, render_mode="human"):
    """Factory function to create the multi-agent environment."""
    return MultiAgentPIDRoadNetworkEnv(
        num_agents=num_agents,
        trajectory_files=trajectory_files,
        render_mode=render_mode
    )


if __name__ == "__main__":
    """Demo of the multi-agent gym environment."""
    import time
    
    print("Creating Multi-Agent PID Road Network Gym Environment...")
    
    # Create environment with 3 agents
    env = make_multi_agent_env(
        num_agents=3, 
        trajectory_files=["trajectory_1.json", "trajectory_2.json"],
        render_mode="human"
    )
    
    print(f"Agent IDs: {env.agent_ids}")
    print(f"Action spaces: {env.action_space}")
    print(f"Observation space (robot1): {env.observation_space['robot1']}")
    
    # Reset environment
    obs, infos = env.reset()
    print(f"Initial observations shape: {list(obs.keys())} -> {[obs[k].shape for k in obs.keys()]}")
    
    # Run demo with different agent policies
    for step in range(300):
        # Different policies for each agent
        actions = {}
        
        if step < 100:
            # All agents GO
            actions = {agent_id: 1 for agent_id in env.agent_ids}
        elif step < 150:
            # robot1 STOP, others GO (collision avoidance scenario)
            actions = {'robot1': 0, 'robot2': 1, 'robot3': 1}
        elif step < 200:
            # Alternating pattern
            actions = {agent_id: step % 2 for agent_id in env.agent_ids}
        else:
            # Random actions
            actions = {agent_id: np.random.choice([0, 1]) for agent_id in env.agent_ids}
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Render
        if not env.render():
            break
        
        # Print status every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step}:")
            for agent_id in env.agent_ids:
                action_name = "STOP" if actions[agent_id] == 0 else "GO"
                print(f"  {agent_id}: Action={action_name}, Reward={rewards[agent_id]:.2f}, "
                      f"Progress={infos[agent_id]['waypoint_progress']['progress_ratio']*100:.1f}%, "
                      f"Collisions={infos[agent_id]['collisions']}")
        
        # Check if any agent completed
        if any(terminated.values()) or any(truncated.values()):
            print(f"\nEpisode ended at step {step}!")
            for agent_id in env.agent_ids:
                print(f"  {agent_id}: Terminated={terminated[agent_id]}, Truncated={truncated[agent_id]}")
            break
        
        # time.sleep(0.02)  # Slow down for visualization
    
    env.close()
    print("Multi-agent demo completed!")
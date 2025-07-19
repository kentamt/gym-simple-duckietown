#!/usr/bin/env python3
"""
Gym environment wrapper for PID road network demo with discrete action space.
Action space: {STOP, GO} - controls whether the robot follows its trajectory or stops.
"""

import sys
import numpy as np
import json
import os
import argparse
import pygame
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


class PIDRoadNetworkEnv(gym.Env):
    """
    Gym environment for PID-controlled robot on road network with discrete actions.
    
    Action Space:
        Discrete(2): {0: STOP, 1: GO}
        - STOP (0): Zero wheel speeds, robot stops
        - GO (1): Follow PID trajectory toward next waypoint
    
    Observation Space:
        Box containing:
        - Robot position (x, y)
        - Robot orientation (theta)
        - Robot velocity (linear, angular)
        - Distance to next waypoint
        - Collision status
        - Progress along trajectory
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}
    
    def __init__(self, trajectory_file=None, map_config=None, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            trajectory_file: Path to trajectory JSON file (optional)
            map_config: Map configuration (optional, defaults to road network)
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Store config
        self.trajectory_file = trajectory_file
        self.map_config = map_config
        self.render_mode = render_mode
        
        # Action space: {STOP, GO}
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -np.pi, -5.0, -5.0, 0.0, 0.0, 0.0]),
            high=np.array([10.0, 10.0, np.pi, 5.0, 5.0, 10.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Environment state
        self.map_instance = None
        self.robot = None
        self.controller = None
        self.collision_detector = None
        self.trajectory = None
        self.renderer = None
        
        # Simulation state
        self.dt = 0.016  # 60 FPS
        self.step_count = 0
        self.max_steps = 1000
        self.collision_results = []
        self.robot_speeds = {'linear': 0.0, 'angular': 0.0, 'total': 0.0}
        
        # Initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize map, robot, and other components."""
        # Create map
        self.map_instance = create_road_network_map()
        
        # Initialize collision detector
        self.collision_detector = CollisionDetector(
            map_width=self.map_instance.width_meters,
            map_height=self.map_instance.height_meters
        )
        
        # Load trajectory
        if self.trajectory_file and os.path.exists(self.trajectory_file):
            trajectories = create_robot_trajectories(
                self.map_instance, 
                {'robot1': self.trajectory_file}
            )
            self.trajectory = trajectories['robot1']
        else:
            # Use default trajectory
            trajectories = create_robot_trajectories(self.map_instance)
            self.trajectory = trajectories['robot1']
        
        # Create robot at starting position
        start_x, start_y = self.trajectory[0]
        self.robot = create_duckiebot(x=start_x, y=start_y, theta=0.0)
        
        # Create PID controller
        pid_config = PIDConfig(
            distance_gains=PIDGains(kp=0.5, ki=0.0, kd=0.0),
            heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
            max_linear_velocity=2.5,
            max_angular_velocity=4.0,
            position_tolerance=0.10
        )
        self.controller = WaypointFollowPIDController(pid_config)
        self.controller.set_waypoints(self.trajectory)
        
        # Initialize renderer if needed
        if self.render_mode == "human":
            self._initialize_renderer()
    
    def _initialize_renderer(self):
        """Initialize pygame renderer for visualization."""
        config = RenderConfig(
            width=1200, height=800, fps=60,
            use_tile_images=True,
            show_grid=True,
            show_robot_ids=True
        )
        
        self.renderer = create_pygame_renderer(self.map_instance, config)
        self.renderer.set_robots({'robot1': self.robot})
        self.renderer.planned_trajectories = {'robot1': self.trajectory}
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset robot to starting position
        start_x, start_y = self.trajectory[0]
        self.robot.reset(x=start_x, y=start_y, theta=0.0)
        
        # Reset controller
        self.controller.reset_controllers()
        self.controller.set_waypoints(self.trajectory)
        
        # Reset simulation state
        self.step_count = 0
        self.collision_results = []
        self.robot_speeds = {'linear': 0.0, 'angular': 0.0, 'total': 0.0}
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Apply action
        if action == 0:  # STOP
            # Zero wheel speeds
            linear_vel = 0.0
            angular_vel = 0.0
            omega_l = 0.0
            omega_r = 0.0
        else:  # GO (action == 1)
            # Use PID controller to compute velocities
            linear_vel, angular_vel, info = self.controller.compute_control(
                self.robot.x, self.robot.y, self.robot.theta, self.dt
            )
            
            # Convert to wheel speeds
            omega_l, omega_r = self.controller.body_vel_to_wheel_speeds(
                linear_vel, angular_vel,
                self.robot.config.wheelbase, self.robot.config.wheel_radius
            )
        
        # Update robot speeds for tracking
        self.robot_speeds['linear'] = abs(linear_vel)
        self.robot_speeds['angular'] = abs(angular_vel)
        self.robot_speeds['total'] = np.sqrt(linear_vel**2 + angular_vel**2)
        
        # Apply control action to robot
        control_action = np.array([omega_l, omega_r])
        self.robot.step(control_action, self.dt)
        
        # Check for collisions
        self.collision_results = self.collision_detector.check_all_collisions({'robot1': self.robot})
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        self.step_count += 1
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, action):
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Base reward for making progress
        progress_info = self.controller.get_progress()
        progress_ratio = progress_info['progress_ratio']
        
        # Reward for progress along trajectory
        if action == 1:  # GO action
            reward += 1.0  # Base reward for moving
            
            # Bonus for reaching waypoints
            if progress_info['completed']:
                reward += 100.0  # Large bonus for completing trajectory
        
        # Penalty for collisions
        collision_penalty = 0.0
        for collision in self.collision_results:
            if collision.is_colliding:
                collision_penalty -= 10.0  # Penalty per collision
        
        reward += collision_penalty
        
        # Small penalty for stopping when not necessary
        if action == 0 and len([c for c in self.collision_results if c.is_colliding]) == 0:
            reward -= 0.1  # Small penalty for unnecessary stopping
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate."""
        # Terminate if trajectory is completed
        progress_info = self.controller.get_progress()
        if progress_info['completed']:
            return True
        
        # Terminate if robot goes out of bounds
        if (self.robot.x < 0 or self.robot.x > self.map_instance.width_meters or
            self.robot.y < 0 or self.robot.y > self.map_instance.height_meters):
            return True
        
        return False
    
    def _get_observation(self):
        """Get current observation."""
        # Calculate distance to next waypoint
        progress_info = self.controller.get_progress()
        current_waypoint_idx = progress_info['current_waypoint']
        
        if current_waypoint_idx < len(self.trajectory):
            next_waypoint = self.trajectory[current_waypoint_idx]
            dist_to_waypoint = np.sqrt(
                (self.robot.x - next_waypoint[0])**2 + 
                (self.robot.y - next_waypoint[1])**2
            )
        else:
            dist_to_waypoint = 0.0
        
        # Check collision status
        collision_status = 1.0 if any(c.is_colliding for c in self.collision_results) else 0.0
        
        # Progress ratio
        progress_ratio = progress_info['progress_ratio']
        
        observation = np.array([
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            self.robot_speeds['linear'],
            self.robot_speeds['angular'],
            dist_to_waypoint,
            collision_status,
            progress_ratio
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self):
        """Get additional info dict."""
        progress_info = self.controller.get_progress()
        
        return {
            'step_count': self.step_count,
            'robot_position': (self.robot.x, self.robot.y),
            'robot_theta': self.robot.theta,
            'waypoint_progress': progress_info,
            'collisions': len([c for c in self.collision_results if c.is_colliding]),
            'collision_details': [
                {
                    'type': c.collision_type,
                    'robot_id': c.robot_id,
                    'other_robot_id': c.other_robot_id,
                    'obstacle_name': c.obstacle_name
                } for c in self.collision_results if c.is_colliding
            ],
            'robot_speeds': self.robot_speeds.copy()
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.renderer is None:
                self._initialize_renderer()
            
            # Update renderer with current state
            self.renderer.set_robots({'robot1': self.robot})
            self.renderer.set_collision_results(self.collision_results)
            
            # Render frame
            if not self.renderer.render():
                return False  # Window closed
            
            # Draw additional overlays
            self._draw_overlays()
            
            return True
        
        elif self.render_mode == "rgb_array":
            # TODO: Implement rgb_array rendering
            return np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def _draw_overlays(self):
        """Draw additional information overlays."""
        if self.renderer and hasattr(self.renderer, 'screen') and hasattr(self.renderer, 'font'):
            y_offset = 10
            
            # Draw collision status
            active_collisions = len([c for c in self.collision_results if c.is_colliding])
            collision_text = f"Collisions: {active_collisions}"
            collision_color = (255, 0, 0) if active_collisions > 0 else (0, 255, 0)
            text_surface = self.renderer.font.render(collision_text, True, collision_color)
            
            background_rect = (10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 4)
            pygame.draw.rect(self.renderer.screen, (0, 0, 0, 180), background_rect)
            self.renderer.screen.blit(text_surface, (15, y_offset + 2))
            y_offset += 30
            
            # Draw robot speed and action state
            speed_text = f"Speed: {self.robot_speeds['linear']:.2f}m/s"
            text_surface = self.renderer.font.render(speed_text, True, (255, 255, 255))
            
            background_rect = (10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 4)
            pygame.draw.rect(self.renderer.screen, (0, 0, 0, 180), background_rect)
            self.renderer.screen.blit(text_surface, (15, y_offset + 2))
            y_offset += 30
            
            # Draw progress
            progress_info = self.controller.get_progress()
            progress_text = f"Progress: {progress_info['progress_ratio']*100:.1f}%"
            text_surface = self.renderer.font.render(progress_text, True, (255, 255, 255))
            
            background_rect = (10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 4)
            pygame.draw.rect(self.renderer.screen, (0, 0, 0, 180), background_rect)
            self.renderer.screen.blit(text_surface, (15, y_offset + 2))
    
    def close(self):
        """Clean up environment resources."""
        if self.renderer:
            self.renderer.cleanup()
            self.renderer = None


def make_env(trajectory_file=None, render_mode="human"):
    """Factory function to create the environment."""
    return PIDRoadNetworkEnv(trajectory_file=trajectory_file, render_mode=render_mode)


if __name__ == "__main__":
    """Demo of the gym environment."""
    import time
    
    print("Creating PID Road Network Gym Environment...")
    
    # Create environment
    env = make_env(trajectory_file="trajectory_1.json", render_mode="human")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a simple demo
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    # Simple policy: GO for 100 steps, then alternate STOP/GO
    for step in range(200):
        if step < 100:
            action = 1  # GO
        else:
            action = step % 2  # Alternate STOP/GO
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        if not env.render():
            break
        
        # Print status every 50 steps
        if step % 50 == 0:
            action_name = "STOP" if action == 0 else "GO"
            print(f"Step {step}: Action={action_name}, Reward={reward:.2f}, "
                  f"Collisions={info['collisions']}, Progress={info['waypoint_progress']['progress_ratio']*100:.1f}%")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"Episode ended at step {step}!")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break
        
        # time.sleep(0.05)  # Slow down for visualization
    
    env.close()
    print("Demo completed!")
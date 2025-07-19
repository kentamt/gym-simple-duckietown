import numpy as np
from typing import Dict, Any, Tuple, Optional

# Try to import gymnasium, fall back to mock
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Mock gym for compatibility
    class Wrapper:
        def __init__(self, env):
            self.env = env
            self.action_space = getattr(env, 'action_space', None)
            self.observation_space = getattr(env, 'observation_space', None)
        
        def step(self, action):
            return self.env.step(action)
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)
        
        def render(self):
            return self.env.render()
        
        def close(self):
            return self.env.close()
    
    class Discrete:
        def __init__(self, n):
            self.n = n
        
        def sample(self):
            return np.random.randint(0, self.n)
        
        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= x < self.n
    
    class spaces:
        Discrete = Discrete
    
    class gym:
        Wrapper = Wrapper
        spaces = spaces
from ..environment.duckietown_env import DuckietownEnv
from ..robot.discrete_action_mapper import DiscreteActionMapper, DiscreteActionController, DiscreteAction
from ..robot.pid_controller import WaypointFollowPIDController, PIDConfig, PIDGains


class DiscreteActionDuckietownEnv(gym.Wrapper):
    """
    Gym wrapper that converts continuous Duckietown environment to discrete actions.
    
    This wrapper uses a PID controller to smoothly execute discrete actions,
    providing a more interpretable action space for reinforcement learning
    while maintaining realistic robot dynamics.
    """
    
    def __init__(self, 
                 env: DuckietownEnv,
                 forward_distance: float = 1.0,
                 turn_radius: float = 1.0, 
                 turn_angle: float = np.pi/2,
                 pid_config: Optional[PIDConfig] = None):
        """
        Initialize discrete action environment wrapper.
        
        Args:
            env: Base DuckietownEnv instance
            forward_distance: Distance to move forward (meters)
            turn_radius: Radius for turning arcs (meters)
            turn_angle: Angle to turn in radians (default: 90 degrees)
            pid_config: PID controller configuration
        """
        super().__init__(env)
        
        # Store parameters
        self.forward_distance = forward_distance
        self.turn_radius = turn_radius
        self.turn_angle = turn_angle
        
        # Create discrete action mapper
        self.action_mapper = DiscreteActionMapper(
            forward_distance=forward_distance,
            turn_radius=turn_radius,
            turn_angle=turn_angle
        )
        
        # Create PID controller
        if pid_config is None:
            pid_config = PIDConfig(
                distance_gains=PIDGains(kp=1.0, ki=0.0, kd=0.1),
                heading_gains=PIDGains(kp=2.0, ki=0.0, kd=0.2),
                max_linear_velocity=0.5,
                max_angular_velocity=2.0,
                position_tolerance=0.15,
                heading_tolerance=0.1
            )
        
        self.pid_controller = WaypointFollowPIDController(pid_config)
        
        # Create discrete action controller
        self.discrete_controller = DiscreteActionController(
            self.pid_controller, 
            self.action_mapper
        )
        
        # Override action space to discrete
        self.action_space = gym.spaces.Discrete(self.action_mapper.get_action_space_size())
        
        # Store current action info
        self.current_action = None
        self.action_start_time = 0
        self.max_action_duration = 10.0  # Max seconds to complete an action
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and discrete action controller."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset discrete action controller
        self.discrete_controller.reset()
        self.current_action = None
        self.action_start_time = 0
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute discrete action using PID controller.
        
        Args:
            action: Discrete action (0-4)
            
        Returns:
            Standard gym step returns: (obs, reward, terminated, truncated, info)
        """
        # Validate action
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be integer, got {type(action)}")
        if not (0 <= action < self.action_space.n):
            raise ValueError(f"Action {action} not in valid range [0, {self.action_space.n})")
        
        # Get current robot state
        robot_state = self.env.robot.get_state_dict()
        robot_x = robot_state['x']
        robot_y = robot_state['y']
        robot_theta = robot_state['theta']
        
        # If new action, set up waypoint
        if self.current_action != action or self.discrete_controller.is_action_completed():
            self.discrete_controller.execute_action(action, robot_x, robot_y, robot_theta)
            self.current_action = action
            self.action_start_time = 0
        
        # Execute action using PID control
        dt = 1.0 / 30  # Use 30 FPS frame rate
        
        # Get control from discrete controller
        linear_vel, angular_vel, control_info = self.discrete_controller.compute_control(
            robot_x, robot_y, robot_theta, dt
        )
        
        # Convert to wheel speeds
        wheel_speeds = self.pid_controller.body_vel_to_wheel_speeds(
            linear_vel, angular_vel,
            self.env.robot.config.wheelbase,
            self.env.robot.config.wheel_radius
        )
        
        # Execute in base environment (convert to numpy array)
        obs, reward, terminated, truncated, info = self.env.step(np.array(wheel_speeds))
        
        # Add waypoint visualization to renderer if available
        if hasattr(self.env, 'renderer') and self.env.renderer is not None:
            if hasattr(self.env.renderer, 'set_current_waypoint'):
                # Show current waypoint target
                current_waypoint = self.discrete_controller.current_waypoint
                if current_waypoint is not None:
                    self.env.renderer.set_current_waypoint(current_waypoint)
        
        # Add discrete action information to info
        info['discrete_action'] = {
            'action': action,
            'action_name': DiscreteAction(action).name,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel,
            'control_status': control_info.get('status', 'unknown'),
            'action_completed': self.discrete_controller.is_action_completed()
        }
        
        # Track action duration
        self.action_start_time += dt
        if self.action_start_time > self.max_action_duration:
            info['discrete_action']['timeout'] = True
            # Force completion of stuck actions
            self.discrete_controller.action_completed = True
        
        return obs, reward, terminated, truncated, info
    
    def get_action_meanings(self) -> list:
        """Get human-readable action names."""
        return self.action_mapper.get_action_names()
    
    def visualize_actions(self) -> dict:
        """
        Visualize all possible actions from current robot position.
        
        Returns:
            Dictionary mapping action names to waypoints
        """
        robot_state = self.env.robot.get_state_dict()
        return self.action_mapper.visualize_action_waypoints(
            robot_state['x'], 
            robot_state['y'], 
            robot_state['theta']
        )
    
    def get_discrete_action_info(self) -> dict:
        """Get information about the discrete action configuration."""
        return {
            'action_space_size': self.action_space.n,
            'action_names': self.get_action_meanings(),
            'forward_distance': self.forward_distance,
            'turn_radius': self.turn_radius,
            'turn_angle_degrees': np.degrees(self.turn_angle),
            'pid_config': {
                'max_linear_velocity': self.pid_controller.config.max_linear_velocity,
                'max_angular_velocity': self.pid_controller.config.max_angular_velocity,
                'position_tolerance': self.pid_controller.config.position_tolerance,
                'heading_tolerance': self.pid_controller.config.heading_tolerance
            }
        }


def create_discrete_duckietown_env(map_config: dict = None,
                                  reward_function=None,
                                  render_mode: str = "human",
                                  max_steps: int = 200,
                                  forward_distance: float = 1.0,
                                  turn_radius: float = 1.0,
                                  turn_angle: float = np.pi/2,
                                  pid_config: Optional[PIDConfig] = None) -> DiscreteActionDuckietownEnv:
    """
    Convenience function to create a discrete action Duckietown environment.
    
    Args:
        map_config: Map configuration for DuckietownEnv
        reward_function: Reward function for DuckietownEnv
        render_mode: Render mode ("human", "rgb_array", None)
        max_steps: Maximum steps per episode
        forward_distance: Distance to move forward (meters)
        turn_radius: Radius for turning arcs (meters)
        turn_angle: Angle to turn in radians
        pid_config: PID controller configuration
        
    Returns:
        DiscreteActionDuckietownEnv instance
    """
    # Default map configuration
    if map_config is None:
        map_config = {
            "width": 5,
            "height": 5,
            "track_type": "loop"
        }
    
    # Create base environment
    base_env = DuckietownEnv(
        map_config=map_config,
        reward_function=reward_function,
        render_mode=render_mode,
        max_steps=max_steps
    )
    
    # Wrap with discrete action wrapper
    discrete_env = DiscreteActionDuckietownEnv(
        env=base_env,
        forward_distance=forward_distance,
        turn_radius=turn_radius,
        turn_angle=turn_angle,
        pid_config=pid_config
    )
    
    return discrete_env
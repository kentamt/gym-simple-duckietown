import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import warnings

# Simple spaces implementation for gym compatibility
class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.shape = shape or self.low.shape
        self.dtype = dtype
    
    def sample(self):
        return np.random.uniform(self.low, self.high, size=self.shape)
    
    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)

class spaces:
    Box = Box

from ..robot.duckiebot import Duckiebot, RobotConfig, create_duckiebot
from ..world.map import Map, MapConfig, create_map_from_array, create_map_from_file
from ..world.collision_detection import CollisionDetector

# Try to import pygame renderer, fall back to simple renderer
try:
    from ..rendering.pygame_renderer import PygameRenderer, RenderConfig
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    from ..rendering.simple_renderer import create_fallback_renderer


class DuckietownEnv:
    """
    OpenAI Gym environment for Duckietown simulation.
    
    This environment simulates a Duckiebot robot navigating on a tile-based
    Duckietown map. The robot uses differential drive kinematics and can
    interact with the environment through wheel velocity commands.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        map_config: Optional[Union[Dict, str, MapConfig]] = None,
        robot_config: Optional[Union[Dict, RobotConfig]] = None,
        max_steps: int = 500,
        reward_function: Optional[callable] = None,
        render_mode: Optional[str] = None,
        dt: float = 0.05,
    ):
        """
        Initialize the Duckietown environment.
        
        Args:
            map_config: Map configuration (dict, file path, or MapConfig object)
            robot_config: Robot configuration (dict or RobotConfig object)
            max_steps: Maximum steps per episode
            reward_function: Custom reward function (state, action, next_state) -> reward
            render_mode: Rendering mode ("human" or "rgb_array")
            dt: Simulation time step in seconds
        """
        # super().__init__()  # Not needed without gym.Env
        
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        
        # Initialize map
        self._setup_map(map_config)
        
        # Initialize robot
        self._setup_robot(robot_config)
        
        # Initialize collision detector
        self.collision_detector = CollisionDetector(
            map_width=self.map.width_meters,
            map_height=self.map.height_meters
        )
        
        # Initialize renderer if needed
        self.renderer = None
        self.renderer_type = None
        if render_mode is not None:
            self._setup_renderer()
        
        # Set up reward function
        self.reward_function = reward_function or self._default_reward_function
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Environment state
        self.done = False
        self.info = {}
        
    def _setup_map(self, map_config):
        """Setup the map from configuration."""
        if map_config is None:
            # Default 5x5 straight track
            config = MapConfig(width=5, height=5)
            self.map = Map(config)
            self.map.create_straight_track()
        elif isinstance(map_config, str):
            # Load from file
            self.map = create_map_from_file(map_config)
        elif isinstance(map_config, dict):
            if "layout" in map_config:
                # Create from array layout
                self.map = create_map_from_array(
                    map_config["layout"], 
                    map_config.get("tile_size", 0.61)
                )
            else:
                # Create from config dict
                track_type = map_config.get("track_type", "straight")
                map_config_copy = {k: v for k, v in map_config.items() if k != "track_type"}
                config = MapConfig(**map_config_copy)
                self.map = Map(config)
                if track_type == "straight":
                    self.map.create_straight_track()
                elif track_type == "loop":
                    self.map.create_loop_track()
        elif isinstance(map_config, MapConfig):
            self.map = Map(map_config)
            self.map.create_straight_track()
        else:
            raise ValueError("Invalid map_config type")
    
    def _setup_robot(self, robot_config):
        """Setup the robot from configuration."""
        if robot_config is None:
            # Default robot at center of map
            center_x = self.map.width_meters / 2
            center_y = self.map.height_meters / 2
            self.robot = create_duckiebot(center_x, center_y, 0.0)
        elif isinstance(robot_config, dict):
            config = RobotConfig(**robot_config)
            self.robot = Duckiebot(config)
        elif isinstance(robot_config, RobotConfig):
            self.robot = Duckiebot(robot_config)
        else:
            raise ValueError("Invalid robot_config type")
        
        # Store initial robot state for reset
        self.initial_robot_state = {
            'x': self.robot.x,
            'y': self.robot.y,
            'theta': self.robot.theta
        }
    
    def _setup_renderer(self):
        """Setup the renderer for visualization."""
        if PYGAME_AVAILABLE:
            render_config = RenderConfig(
                width=800,
                height=600,
                fps=30,
                show_grid=True,
                show_coordinates=True,
                show_collision_circles=True,
                use_tile_images=True
            )
            self.renderer = PygameRenderer(self.map, render_config)
            self.renderer_type = "pygame"
        else:
            # Use fallback renderer
            self.renderer = create_fallback_renderer(self.map, prefer_matplotlib=True)
            self.renderer_type = "fallback"
            print("Note: Using fallback renderer. Install pygame for full visual rendering.")
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [left_wheel_velocity, right_wheel_velocity]
        max_wheel_speed = self.robot.config.max_wheel_speed
        self.action_space = spaces.Box(
            low=-max_wheel_speed,
            high=max_wheel_speed,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [x, y, theta, linear_vel, angular_vel, left_wheel_vel, right_wheel_vel]
        # Plus additional sensor data could be added here
        obs_low = np.array([
            0.0,  # x (map bounds)
            0.0,  # y (map bounds)
            -np.pi,  # theta
            -10.0,  # linear velocity
            -10.0,  # angular velocity
            -max_wheel_speed,  # left wheel velocity
            -max_wheel_speed,  # right wheel velocity
        ], dtype=np.float32)
        
        obs_high = np.array([
            self.map.width_meters,   # x
            self.map.height_meters,  # y
            np.pi,   # theta
            10.0,    # linear velocity
            10.0,    # angular velocity
            max_wheel_speed,   # left wheel velocity
            max_wheel_speed,   # right wheel velocity
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            self.robot.linear_velocity,
            self.robot.angular_velocity,
            self.robot.omega_l,
            self.robot.omega_r,
        ], dtype=np.float32)
    
    def _default_reward_function(self, state: Dict, action: np.ndarray, next_state: Dict) -> float:
        """
        Default reward function encouraging forward movement and penalizing collisions.
        
        Args:
            state: Previous state dictionary
            action: Action taken
            next_state: New state dictionary
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Reward for forward movement
        distance_moved = next_state.get('distance_step', 0.0)
        reward += distance_moved * 10.0  # Scale factor
        
        # Penalty for collision
        if next_state.get('collision', False):
            reward -= 100.0
        
        # Small penalty for excessive turning (encourage straight movement)
        angular_vel = abs(next_state.get('angular_velocity', 0.0))
        reward -= angular_vel * 0.1
        
        # Penalty for going backwards
        if next_state.get('linear_velocity', 0.0) < 0:
            reward -= 1.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def _check_collision(self) -> bool:
        """Check if robot has collided with obstacles or boundaries."""
        # Check map boundaries using collision detector
        boundary_collision = self.collision_detector.check_robot_boundary_collision(self.robot)
        if boundary_collision.is_colliding:
            return True
        
        # Check obstacle collisions (returns a list)
        obstacle_collisions = self.collision_detector.check_robot_obstacle_collision(self.robot)
        if any(collision.is_colliding for collision in obstacle_collisions):
            return True
        
        return False
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Check collision
        if self.robot.is_collided:
            return True
        
        # Check max steps
        if self.step_count >= self.max_steps:
            return True
        
        # Add custom termination conditions here if needed
        
        return False
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take [left_wheel_vel, right_wheel_vel]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            warnings.warn(
                "You are calling 'step()' even though this environment has already returned "
                "done = True. You should always call 'reset()' once you receive 'done = True' "
                "-- any further steps are undefined behavior."
            )
            return self._get_observation(), 0.0, True, False, {}
        
        # Store previous state
        prev_state = {
            'x': self.robot.x,
            'y': self.robot.y,
            'theta': self.robot.theta,
            'linear_velocity': self.robot.linear_velocity,
            'angular_velocity': self.robot.angular_velocity,
        }
        
        # Execute action
        step_info = self.robot.step(action, self.dt)
        
        # Check for collision
        collision = self._check_collision()
        self.robot.set_collision_state(collision)
        
        # Get new state
        current_state = {
            'x': self.robot.x,
            'y': self.robot.y,
            'theta': self.robot.theta,
            'linear_velocity': self.robot.linear_velocity,
            'angular_velocity': self.robot.angular_velocity,
            'distance_step': step_info['distance_step'],
            'collision': collision,
        }
        
        # Calculate reward
        reward = self.reward_function(prev_state, action, current_state)
        
        # Update step count
        self.step_count += 1
        
        # Check if done
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps
        self.done = terminated or truncated
        
        # Update info
        self.info = {
            'step_count': self.step_count,
            'total_distance': self.robot.total_distance,
            'collision': collision,
            'robot_state': self.robot.get_state_dict(),
            'action': action.copy(),
        }
        
        return self._get_observation(), reward, terminated, truncated, self.info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation, info
        """
        # super().reset(seed=seed)  # Not needed without gym.Env
        if seed is not None:
            np.random.seed(seed)
        
        # Reset robot to initial position
        if options and 'robot_position' in options:
            pos = options['robot_position']
            self.robot.reset(pos.get('x'), pos.get('y'), pos.get('theta'))
        else:
            self.robot.reset(
                self.initial_robot_state['x'],
                self.initial_robot_state['y'], 
                self.initial_robot_state['theta']
            )
        
        # Reset environment state
        self.step_count = 0
        self.done = False
        self.info = {
            'step_count': 0,
            'total_distance': 0.0,
            'collision': False,
            'robot_state': self.robot.get_state_dict(),
        }
        
        return self._get_observation(), self.info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.renderer is None:
            self._setup_renderer()
        
        # Update renderer with current robot state
        robots = {"robot": self.robot}
        self.renderer.set_robots(robots)
        
        # Render the current state
        if self.render_mode == "human":
            if self.renderer_type == "pygame":
                # Pygame renderer
                should_continue = self.renderer.render()
                if not should_continue:
                    self.close()
            else:
                # Fallback renderer
                self.renderer.render()
            return None
            
        elif self.render_mode == "rgb_array":
            if self.renderer_type == "pygame":
                # For pygame, render to surface and convert to array
                self.renderer.render()
                import pygame
                rgb_array = pygame.surfarray.array3d(self.renderer.screen)
                rgb_array = rgb_array.transpose([1, 0, 2])  # pygame uses (width, height, channels)
                return rgb_array
            else:
                # For fallback renderer, we can't provide rgb_array
                print("Warning: rgb_array mode not supported with fallback renderer")
                return None
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            if self.renderer_type == "pygame":
                import pygame
                pygame.quit()
            else:
                self.renderer.close()
            self.renderer = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete environment state."""
        return {
            'robot_state': self.robot.get_state_dict(),
            'map_state': {
                'width_tiles': self.map.width_tiles,
                'height_tiles': self.map.height_tiles,
                'tile_size': self.map.tile_size,
                'layout': self.map.tiles.tolist(),
            },
            'step_count': self.step_count,
            'done': self.done,
        }
    
    def set_reward_function(self, reward_function: callable):
        """Set custom reward function."""
        self.reward_function = reward_function


def make_env(
    map_name: str = "default",
    render_mode: Optional[str] = None,
    **kwargs
) -> DuckietownEnv:
    """
    Factory function to create DuckietownEnv instances.
    
    Args:
        map_name: Name of predefined map or path to map file
        render_mode: Rendering mode
        **kwargs: Additional arguments for environment
        
    Returns:
        DuckietownEnv instance
    """
    # Define some predefined maps
    predefined_maps = {
        "default": {"width": 5, "height": 5, "track_type": "straight"},
        "loop": {"width": 6, "height": 5, "track_type": "loop"},
        "small": {"width": 3, "height": 3, "track_type": "straight"},
        "large": {"width": 8, "height": 8, "track_type": "loop"},
    }
    
    if map_name in predefined_maps:
        map_config = predefined_maps[map_name]
    else:
        # Assume it's a file path
        map_config = map_name
    
    return DuckietownEnv(
        map_config=map_config,
        render_mode=render_mode,
        **kwargs
    )
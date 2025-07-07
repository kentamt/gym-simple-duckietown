import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .kinematics import DifferentialDriveKinematics, DifferentialDriveParams


@dataclass
class RobotConfig:
    """Configuration parameters for the Duckiebot."""
    wheelbase: float = 0.102  # Distance between wheels (m)
    wheel_radius: float = 0.0318  # Wheel radius (m)
    max_wheel_speed: float = 10.0  # Maximum wheel speed (rad/s)
    collision_radius: float = 0.05  # Robot collision radius (m)
    initial_x: float = 0.0  # Initial x position (m)
    initial_y: float = 0.0  # Initial y position (m)
    initial_theta: float = 0.0  # Initial orientation (rad)
    
    def __post_init__(self):
        if self.wheelbase <= 0:
            raise ValueError("Wheelbase must be positive")
        if self.wheel_radius <= 0:
            raise ValueError("Wheel radius must be positive")
        if self.max_wheel_speed <= 0:
            raise ValueError("Max wheel speed must be positive")
        if self.collision_radius <= 0:
            raise ValueError("Collision radius must be positive")


class Duckiebot:
    """
    Duckiebot robot implementation with differential drive kinematics.
    
    This class represents a Duckiebot robot that can move on the tile-based map
    using differential drive kinematics similar to gym-duckietown.
    """
    
    def __init__(self, config: RobotConfig):
        """
        Initialize the Duckiebot.
        
        Args:
            config: Robot configuration parameters
        """
        self.config = config
        
        # Initialize kinematics
        kinematics_params = DifferentialDriveParams(
            wheelbase=config.wheelbase,
            wheel_radius=config.wheel_radius,
            max_wheel_speed=config.max_wheel_speed
        )
        self.kinematics = DifferentialDriveKinematics(kinematics_params)
        
        # Robot state
        self.pose = np.array([
            config.initial_x,
            config.initial_y,
            config.initial_theta
        ])
        
        # Robot velocities (body frame)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Wheel velocities
        self.omega_l = 0.0  # Left wheel angular velocity
        self.omega_r = 0.0  # Right wheel angular velocity
        
        # Collision properties
        self.collision_radius = config.collision_radius
        
        # State tracking
        self.is_collided = False
        self.total_distance = 0.0
        self.step_count = 0
        
        # Previous pose for distance calculation
        self._prev_pose = self.pose.copy()
    
    @property
    def x(self) -> float:
        """Get robot x position."""
        return self.pose[0]
    
    @property
    def y(self) -> float:
        """Get robot y position."""
        return self.pose[1]
    
    @property
    def theta(self) -> float:
        """Get robot orientation."""
        return self.pose[2]
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get robot position as (x, y) tuple."""
        return (self.x, self.y)
    
    def reset(self, x: Optional[float] = None, y: Optional[float] = None, 
              theta: Optional[float] = None):
        """
        Reset the robot to initial or specified pose.
        
        Args:
            x: X position (uses initial if None)
            y: Y position (uses initial if None)
            theta: Orientation (uses initial if None)
        """
        self.pose = np.array([
            x if x is not None else self.config.initial_x,
            y if y is not None else self.config.initial_y,
            theta if theta is not None else self.config.initial_theta
        ])
        
        # Reset velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.omega_l = 0.0
        self.omega_r = 0.0
        
        # Reset state
        self.is_collided = False
        self.total_distance = 0.0
        self.step_count = 0
        self._prev_pose = self.pose.copy()
    
    def step(self, action: np.ndarray, dt: float = 0.05) -> Dict[str, Any]:
        """
        Update robot state given control action.
        
        Args:
            action: Control action [omega_l, omega_r] (wheel angular velocities)
            dt: Time step (seconds)
            
        Returns:
            Dictionary containing step information
        """
        if len(action) != 2:
            raise ValueError("Action must be [omega_l, omega_r]")
        
        # Store previous pose for distance calculation
        self._prev_pose = self.pose.copy()
        
        # Extract wheel velocities
        omega_l, omega_r = action
        
        # Clamp wheel velocities
        omega_l = np.clip(omega_l, -self.config.max_wheel_speed, self.config.max_wheel_speed)
        omega_r = np.clip(omega_r, -self.config.max_wheel_speed, self.config.max_wheel_speed)
        
        # Store current wheel velocities
        self.omega_l = omega_l
        self.omega_r = omega_r
        
        # Update pose using kinematics
        new_pose, body_velocities = self.kinematics.forward_kinematics(
            self.pose, omega_l, omega_r, dt
        )
        
        # Update robot state
        self.pose = new_pose
        self.linear_velocity = body_velocities[0]
        self.angular_velocity = body_velocities[1]
        
        # Calculate distance traveled
        distance_step = np.linalg.norm(self.pose[:2] - self._prev_pose[:2])
        self.total_distance += distance_step
        self.step_count += 1
        
        # Return step information
        return {
            'pose': self.pose.copy(),
            'linear_velocity': self.linear_velocity,
            'angular_velocity': self.angular_velocity,
            'wheel_velocities': np.array([omega_l, omega_r]),
            'distance_step': distance_step,
            'total_distance': self.total_distance,
            'step_count': self.step_count
        }
    
    def get_collision_points(self) -> np.ndarray:
        """
        Get collision points around the robot for collision detection.
        
        Returns:
            Array of collision points in world coordinates
        """
        # Generate points around the robot's collision circle
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        
        collision_points = []
        for angle in angles:
            # Local coordinates
            local_x = self.collision_radius * np.cos(angle)
            local_y = self.collision_radius * np.sin(angle)
            
            # Transform to world coordinates
            world_x = self.x + local_x * np.cos(self.theta) - local_y * np.sin(self.theta)
            world_y = self.y + local_x * np.sin(self.theta) + local_y * np.cos(self.theta)
            
            collision_points.append([world_x, world_y])
        
        return np.array(collision_points)
    
    def get_forward_point(self, distance: float = 0.1) -> Tuple[float, float]:
        """
        Get a point in front of the robot.
        
        Args:
            distance: Distance in front of robot (meters)
            
        Returns:
            (x, y) coordinates of point in front of robot
        """
        front_x = self.x + distance * np.cos(self.theta)
        front_y = self.y + distance * np.sin(self.theta)
        return (front_x, front_y)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get complete robot state as dictionary.
        
        Returns:
            Dictionary containing all robot state information
        """
        return {
            'pose': self.pose.copy(),
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'linear_velocity': self.linear_velocity,
            'angular_velocity': self.angular_velocity,
            'wheel_velocities': np.array([self.omega_l, self.omega_r]),
            'collision_radius': self.collision_radius,
            'is_collided': self.is_collided,
            'total_distance': self.total_distance,
            'step_count': self.step_count
        }
    
    def set_collision_state(self, collided: bool):
        """Set the collision state of the robot."""
        self.is_collided = collided
    
    def get_velocity_limits(self) -> Dict[str, float]:
        """Get velocity limits for the robot."""
        return self.kinematics.get_velocity_limits()
    
    def is_action_valid(self, action: np.ndarray) -> bool:
        """Check if action is within valid range."""
        if len(action) != 2:
            return False
        omega_l, omega_r = action
        return self.kinematics.is_action_valid(omega_l, omega_r)
    
    def __str__(self) -> str:
        """String representation of the robot."""
        return (f"Duckiebot at ({self.x:.3f}, {self.y:.3f}, {self.theta:.3f}), "
                f"v={self.linear_velocity:.3f} m/s, �={self.angular_velocity:.3f} rad/s")


def create_duckiebot(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Duckiebot:
    """
    Create a Duckiebot with default configuration.
    
    Args:
        x: Initial x position (meters)
        y: Initial y position (meters)
        theta: Initial orientation (radians)
        
    Returns:
        Duckiebot instance
    """
    config = RobotConfig(
        initial_x=x,
        initial_y=y,
        initial_theta=theta
    )
    return Duckiebot(config)


def create_custom_duckiebot(config: RobotConfig) -> Duckiebot:
    """
    Create a Duckiebot with custom configuration.
    
    Args:
        config: Robot configuration
        
    Returns:
        Duckiebot instance
    """
    return Duckiebot(config)
import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class DifferentialDriveParams:
    """Parameters for differential drive kinematics."""
    wheelbase: float = 0.102  # Distance between wheels in meters (10.2cm for Duckiebot)
    wheel_radius: float = 0.0318  # Wheel radius in meters (3.18cm for Duckiebot)
    max_wheel_speed: float = 10.0  # Maximum wheel speed in rad/s
    
    def __post_init__(self):
        if self.wheelbase <= 0:
            raise ValueError("Wheelbase must be positive")
        if self.wheel_radius <= 0:
            raise ValueError("Wheel radius must be positive")
        if self.max_wheel_speed <= 0:
            raise ValueError("Max wheel speed must be positive")


class DifferentialDriveKinematics:
    """
    Differential drive kinematics implementation following gym-duckietown approach.
    
    This class handles the conversion between wheel velocities and robot pose changes
    for a differential drive robot like the Duckiebot.
    """
    
    def __init__(self, params: DifferentialDriveParams):
        """
        Initialize the kinematics model.
        
        Args:
            params: Differential drive parameters
        """
        self.params = params
        self.wheelbase = params.wheelbase
        self.wheel_radius = params.wheel_radius
        self.max_wheel_speed = params.max_wheel_speed
    
    def wheel_speeds_to_body_vel(self, omega_l: float, omega_r: float) -> Tuple[float, float]:
        """
        Convert wheel angular velocities to body frame linear and angular velocities.
        
        Args:
            omega_l: Left wheel angular velocity (rad/s)
            omega_r: Right wheel angular velocity (rad/s)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity) in body frame
            - linear_velocity: Forward velocity in m/s
            - angular_velocity: Angular velocity in rad/s (positive = counterclockwise)
        """
        # Clamp wheel speeds to maximum
        omega_l = np.clip(omega_l, -self.max_wheel_speed, self.max_wheel_speed)
        omega_r = np.clip(omega_r, -self.max_wheel_speed, self.max_wheel_speed)
        
        # Convert wheel angular velocities to linear velocities
        v_l = omega_l * self.wheel_radius
        v_r = omega_r * self.wheel_radius
        
        # Calculate body frame velocities
        linear_velocity = (v_l + v_r) / 2.0
        angular_velocity = (v_r - v_l) / self.wheelbase
        
        return linear_velocity, angular_velocity
    
    def body_vel_to_wheel_speeds(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """
        Convert body frame velocities to wheel angular velocities.
        
        Args:
            linear_vel: Forward velocity in m/s
            angular_vel: Angular velocity in rad/s (positive = counterclockwise)
            
        Returns:
            Tuple of (omega_l, omega_r) wheel angular velocities in rad/s
        """
        # Calculate wheel linear velocities
        v_l = linear_vel - (angular_vel * self.wheelbase / 2.0)
        v_r = linear_vel + (angular_vel * self.wheelbase / 2.0)
        
        # Convert to wheel angular velocities
        omega_l = v_l / self.wheel_radius
        omega_r = v_r / self.wheel_radius
        
        # Clamp to maximum speeds
        omega_l = np.clip(omega_l, -self.max_wheel_speed, self.max_wheel_speed)
        omega_r = np.clip(omega_r, -self.max_wheel_speed, self.max_wheel_speed)
        
        return omega_l, omega_r
    
    def integrate_pose(self, pose: np.ndarray, omega_l: float, omega_r: float, dt: float) -> np.ndarray:
        """
        Integrate robot pose given wheel velocities and time step.
        
        Args:
            pose: Current pose [x, y, theta] in world frame
            omega_l: Left wheel angular velocity (rad/s)
            omega_r: Right wheel angular velocity (rad/s)
            dt: Time step (seconds)
            
        Returns:
            New pose [x, y, theta] after integration
        """
        if len(pose) != 3:
            raise ValueError("Pose must be [x, y, theta]")
        
        x, y, theta = pose
        
        # Get body frame velocities
        v, omega = self.wheel_speeds_to_body_vel(omega_l, omega_r)
        
        # Handle the case where angular velocity is very small (straight line motion)
        if abs(omega) < 1e-6:
            # Straight line motion
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0.0
        else:
            # Circular motion
            # Calculate instantaneous center of rotation
            R = v / omega  # Radius of curvature
            
            # Calculate change in pose
            dtheta = omega * dt
            dx = R * (np.sin(theta + dtheta) - np.sin(theta))
            dy = R * (-np.cos(theta + dtheta) + np.cos(theta))
        
        # Update pose
        new_pose = np.array([
            x + dx,
            y + dy,
            self._normalize_angle(theta + dtheta)
        ])
        
        return new_pose
    
    def forward_kinematics(self, pose: np.ndarray, omega_l: float, omega_r: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics including pose and body velocities.
        
        Args:
            pose: Current pose [x, y, theta]
            omega_l: Left wheel angular velocity (rad/s)
            omega_r: Right wheel angular velocity (rad/s)
            dt: Time step (seconds)
            
        Returns:
            Tuple of (new_pose, body_velocities)
            - new_pose: [x, y, theta] after integration
            - body_velocities: [linear_vel, angular_vel] in body frame
        """
        new_pose = self.integrate_pose(pose, omega_l, omega_r, dt)
        body_velocities = np.array(self.wheel_speeds_to_body_vel(omega_l, omega_r))
        
        return new_pose, body_velocities
    
    def get_max_linear_velocity(self) -> float:
        """Get maximum possible linear velocity."""
        return self.max_wheel_speed * self.wheel_radius
    
    def get_max_angular_velocity(self) -> float:
        """Get maximum possible angular velocity."""
        max_wheel_vel = self.max_wheel_speed * self.wheel_radius
        return 2 * max_wheel_vel / self.wheelbase
    
    def is_action_valid(self, omega_l: float, omega_r: float) -> bool:
        """Check if wheel velocities are within valid range."""
        return (abs(omega_l) <= self.max_wheel_speed and 
                abs(omega_r) <= self.max_wheel_speed)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_velocity_limits(self) -> dict:
        """Get velocity limits for the robot."""
        return {
            'max_linear_velocity': self.get_max_linear_velocity(),
            'max_angular_velocity': self.get_max_angular_velocity(),
            'max_wheel_speed': self.max_wheel_speed,
            'wheelbase': self.wheelbase,
            'wheel_radius': self.wheel_radius
        }


def create_duckiebot_kinematics() -> DifferentialDriveKinematics:
    """
    Create a kinematics model with default Duckiebot parameters.
    
    Returns:
        DifferentialDriveKinematics instance configured for Duckiebot
    """
    params = DifferentialDriveParams(
        wheelbase=0.102,    # 10.2cm wheelbase
        wheel_radius=0.0318,  # 3.18cm wheel radius
        max_wheel_speed=10.0  # 10 rad/s max speed
    )
    return DifferentialDriveKinematics(params)


def create_custom_kinematics(wheelbase: float, wheel_radius: float, max_wheel_speed: float) -> DifferentialDriveKinematics:
    """
    Create a kinematics model with custom parameters.
    
    Args:
        wheelbase: Distance between wheels (meters)
        wheel_radius: Wheel radius (meters)
        max_wheel_speed: Maximum wheel speed (rad/s)
        
    Returns:
        DifferentialDriveKinematics instance
    """
    params = DifferentialDriveParams(
        wheelbase=wheelbase,
        wheel_radius=wheel_radius,
        max_wheel_speed=max_wheel_speed
    )
    return DifferentialDriveKinematics(params)
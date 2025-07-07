import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain


@dataclass
class PIDConfig:
    """Configuration for PID controllers."""
    # Distance PID gains
    distance_gains: PIDGains = None
    # Heading PID gains
    heading_gains: PIDGains = None
    # Output limits
    max_linear_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 2.0  # rad/s
    # Tolerance for waypoint reaching
    position_tolerance: float = 0.1  # meters
    heading_tolerance: float = 0.1  # radians
    
    def __post_init__(self):
        if self.distance_gains is None:
            self.distance_gains = PIDGains(kp=1.0, ki=0.0, kd=0.1)
        if self.heading_gains is None:
            self.heading_gains = PIDGains(kp=2.0, ki=0.0, kd=0.2)


class PIDController:
    """
    PID controller for single variable control.
    """
    
    def __init__(self, gains: PIDGains, output_limits: Tuple[float, float] = None):
        """
        Initialize PID controller.
        
        Args:
            gains: PID gains (kp, ki, kd)
            output_limits: (min, max) output limits
        """
        self.gains = gains
        self.output_limits = output_limits
        
        # State variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.first_run = True
        
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller with new error.
        
        Args:
            error: Current error value
            dt: Time step (seconds)
            
        Returns:
            Control output
        """
        if self.first_run:
            self.prev_error = error
            self.first_run = False
        
        # Proportional term
        proportional = self.gains.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.gains.ki * self.integral
        
        # Derivative term
        derivative = 0.0
        if dt > 0:
            derivative = self.gains.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update previous error
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.first_run = True


class WaypointFollowPIDController:
    """
    PID-based waypoint following controller for differential drive robot.
    
    Uses separate PID controllers for distance and heading control.
    """
    
    def __init__(self, config: PIDConfig):
        """
        Initialize waypoint following controller.
        
        Args:
            config: PID configuration
        """
        self.config = config
        
        # Initialize PID controllers
        self.distance_pid = PIDController(
            config.distance_gains,
            output_limits=(-config.max_linear_velocity, config.max_linear_velocity)
        )
        
        self.heading_pid = PIDController(
            config.heading_gains,
            output_limits=(-config.max_angular_velocity, config.max_angular_velocity)
        )
        
        # Waypoint management
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        
    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """
        Set list of waypoints to follow.
        
        Args:
            waypoints: List of (x, y) waypoint coordinates
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.reset_controllers()
    
    def reset_controllers(self):
        """Reset PID controller states."""
        self.distance_pid.reset()
        self.heading_pid.reset()
    
    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get current target waypoint."""
        if (self.current_waypoint_idx < len(self.waypoints)):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def advance_waypoint(self):
        """Advance to next waypoint."""
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1
            self.waypoint_reached = False
            self.reset_controllers()
            return True
        return False
    
    def is_waypoint_reached(self, robot_x: float, robot_y: float) -> bool:
        """
        Check if current waypoint is reached.
        
        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            
        Returns:
            True if waypoint is reached
        """
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return True
        
        distance = math.sqrt((robot_x - waypoint[0])**2 + (robot_y - waypoint[1])**2)
        return distance < self.config.position_tolerance
    
    def compute_control(self, robot_x: float, robot_y: float, robot_theta: float, 
                       dt: float) -> Tuple[float, float, dict]:
        """
        Compute control commands for waypoint following.
        
        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            robot_theta: Robot orientation (radians)
            dt: Time step (seconds)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity, info_dict)
        """
        # Get current waypoint
        waypoint = self.get_current_waypoint()
        
        if waypoint is None:
            return 0.0, 0.0, {"status": "completed", "waypoint_idx": self.current_waypoint_idx}
        
        target_x, target_y = waypoint
        
        # Calculate distance and angle to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        
        # Calculate heading error (normalize to [-pi, pi])
        heading_error = angle_to_target - robot_theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Check if waypoint is reached
        if self.is_waypoint_reached(robot_x, robot_y):
            self.waypoint_reached = True
            advanced = self.advance_waypoint()
            if not advanced:
                return 0.0, 0.0, {
                    "status": "completed",
                    "waypoint_idx": self.current_waypoint_idx,
                    "distance_to_target": distance_to_target
                }
            else:
                # Recalculate for new waypoint
                return self.compute_control(robot_x, robot_y, robot_theta, dt)
        
        # Compute PID outputs
        linear_velocity = self.distance_pid.update(distance_to_target, dt)
        angular_velocity = self.heading_pid.update(heading_error, dt)
        
        # Reduce linear velocity when heading error is large
        heading_factor = max(0.1, 1.0 - abs(heading_error) / math.pi)
        linear_velocity *= heading_factor
        
        # Apply limits
        linear_velocity = np.clip(linear_velocity, 0.0, self.config.max_linear_velocity)
        angular_velocity = np.clip(angular_velocity, 
                                 -self.config.max_angular_velocity, 
                                 self.config.max_angular_velocity)
        
        info = {
            "status": "following",
            "waypoint_idx": self.current_waypoint_idx,
            "target": waypoint,
            "distance_to_target": distance_to_target,
            "heading_error": heading_error,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity
        }
        
        return linear_velocity, angular_velocity, info
    
    def body_vel_to_wheel_speeds(self, linear_vel: float, angular_vel: float, 
                                wheelbase: float, wheel_radius: float) -> Tuple[float, float]:
        """
        Convert body velocities to wheel speeds.
        
        Args:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
            wheelbase: Distance between wheels (m)
            wheel_radius: Wheel radius (m)
            
        Returns:
            Tuple of (omega_l, omega_r) wheel angular velocities
        """
        # Calculate wheel linear velocities
        v_l = linear_vel - (angular_vel * wheelbase / 2.0)
        v_r = linear_vel + (angular_vel * wheelbase / 2.0)
        
        # Convert to angular velocities
        omega_l = v_l / wheel_radius
        omega_r = v_r / wheel_radius
        
        return omega_l, omega_r
    
    def get_progress(self) -> dict:
        """Get progress information."""
        return {
            "current_waypoint": self.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "progress_ratio": self.current_waypoint_idx / max(1, len(self.waypoints)),
            "completed": self.current_waypoint_idx >= len(self.waypoints)
        }


def create_default_pid_config() -> PIDConfig:
    """Create default PID configuration for Duckiebot."""
    return PIDConfig(
        distance_gains=PIDGains(kp=1.0, ki=0.1, kd=0.05),
        heading_gains=PIDGains(kp=2.0, ki=0.0, kd=0.1),
        max_linear_velocity=0.3,
        max_angular_velocity=2.0,
        position_tolerance=0.15,
        heading_tolerance=0.2
    )


def create_waypoint_trajectory(pattern: str, map_width: float, map_height: float, 
                             margin: float = 0.5) -> List[Tuple[float, float]]:
    """
    Create predetermined waypoint trajectories.
    
    Args:
        pattern: Pattern type ("square", "figure8", "spiral", "line")
        map_width: Map width in meters
        map_height: Map height in meters
        margin: Margin from map edges in meters
        
    Returns:
        List of (x, y) waypoints
    """
    center_x = map_width / 2
    center_y = map_height / 2
    
    if pattern == "square":
        # Square trajectory
        size = min(map_width, map_height) - 2 * margin
        half_size = size / 2
        waypoints = [
            (center_x - half_size, center_y - half_size),  # Bottom-left
            (center_x + half_size, center_y - half_size),  # Bottom-right
            (center_x + half_size, center_y + half_size),  # Top-right
            (center_x - half_size, center_y + half_size),  # Top-left
            (center_x - half_size, center_y - half_size)   # Back to start
        ]
    
    elif pattern == "figure8":
        # Figure-8 trajectory
        radius = min(map_width, map_height) / 4 - margin
        waypoints = []
        for i in range(16):
            t = i * 2 * math.pi / 16
            if i < 8:
                # First circle
                x = center_x + radius * math.cos(t)
                y = center_y + radius * math.sin(t)
            else:
                # Second circle
                x = center_x - radius * math.cos(t)
                y = center_y + radius * math.sin(t)
            waypoints.append((x, y))
    
    elif pattern == "spiral":
        # Spiral trajectory
        max_radius = min(map_width, map_height) / 2 - margin
        waypoints = []
        for i in range(20):
            t = i * 4 * math.pi / 20
            radius = max_radius * (i / 20)
            x = center_x + radius * math.cos(t)
            y = center_y + radius * math.sin(t)
            waypoints.append((x, y))
    
    elif pattern == "line":
        # Simple line trajectory
        start_x = margin
        end_x = map_width - margin
        y = center_y
        waypoints = [
            (start_x, y),
            (end_x, y),
            (start_x, y)
        ]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return waypoints
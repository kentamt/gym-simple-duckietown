import numpy as np
import math
from typing import Tuple, List
from enum import IntEnum


class DiscreteAction(IntEnum):
    """Discrete action space for robot control."""
    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    BACKWARD = 4


class DiscreteActionMapper:
    """
    Maps discrete actions to waypoints using arc-based trajectory generation.
    
    This mapper generates smooth waypoints that the PID controller can follow,
    enabling discrete RL actions while maintaining realistic robot motion.
    """
    
    def __init__(self, 
                 forward_distance: float = 0.2,
                 turn_radius: float = 0.2,
                 turn_angle: float = math.pi/2):
        """
        Initialize the discrete action mapper.
        
        Args:
            forward_distance: Distance to move forward (meters)
            turn_radius: Radius for turning arcs (meters)
            turn_angle: Angle to turn in radians (default: 90 degrees)
        """
        self.forward_distance = forward_distance
        self.turn_radius = turn_radius
        self.turn_angle = turn_angle
        
    def action_to_waypoint(self, 
                          action: int, 
                          robot_x: float, 
                          robot_y: float, 
                          robot_theta: float) -> Tuple[float, float]:
        """
        Convert discrete action to target waypoint.
        
        Args:
            action: Discrete action (0-4)
            robot_x: Current robot x position (meters)
            robot_y: Current robot y position (meters)
            robot_theta: Current robot orientation (radians)
            
        Returns:
            Target waypoint (x, y) in meters
        """
        action = DiscreteAction(action)
        
        if action == DiscreteAction.STOP:
            # Stay at current position
            return (robot_x, robot_y)
            
        elif action == DiscreteAction.FORWARD:
            # Move forward in current direction
            target_x = robot_x + self.forward_distance * math.cos(robot_theta)
            target_y = robot_y + self.forward_distance * math.sin(robot_theta)
            return (target_x, target_y)
            
        elif action == DiscreteAction.TURN_LEFT:
            # Generate left turn waypoint using circular arc
            return self._generate_turn_left_waypoint(robot_x, robot_y, robot_theta)
            
        elif action == DiscreteAction.TURN_RIGHT:
            # Generate right turn waypoint using circular arc
            return self._generate_turn_right_waypoint(robot_x, robot_y, robot_theta)
            
        elif action == DiscreteAction.BACKWARD:
            # Move backward in opposite direction
            target_x = robot_x - self.forward_distance * math.cos(robot_theta)
            target_y = robot_y - self.forward_distance * math.sin(robot_theta)
            return (target_x, target_y)
            
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _generate_turn_left_waypoint(self, 
                                   robot_x: float, 
                                   robot_y: float, 
                                   robot_theta: float) -> Tuple[float, float]:
        """
        Generate waypoint for left turn using circular arc.
        
        The robot follows a circular arc to the left. The center of the circle
        is positioned perpendicular to the robot's current heading.
        
        Args:
            robot_x: Current robot x position
            robot_y: Current robot y position
            robot_theta: Current robot orientation (radians)
            
        Returns:
            Target waypoint (x, y) after left turn
        """
        # Calculate circle center for left turn
        # Center is to the left of the robot (perpendicular to heading)
        center_x = robot_x - self.turn_radius * math.sin(robot_theta)
        center_y = robot_y + self.turn_radius * math.cos(robot_theta)
        
        # Calculate target angle after turning left
        target_theta = robot_theta + self.turn_angle
        
        # Calculate target position on the circle
        target_x = center_x + self.turn_radius * math.sin(target_theta)
        target_y = center_y - self.turn_radius * math.cos(target_theta)
        
        return (target_x, target_y)
    
    def _generate_turn_right_waypoint(self, 
                                    robot_x: float, 
                                    robot_y: float, 
                                    robot_theta: float) -> Tuple[float, float]:
        """
        Generate waypoint for right turn using circular arc.
        
        The robot follows a circular arc to the right. The center of the circle
        is positioned perpendicular to the robot's current heading.
        
        Args:
            robot_x: Current robot x position
            robot_y: Current robot y position
            robot_theta: Current robot orientation (radians)
            
        Returns:
            Target waypoint (x, y) after right turn
        """
        # Calculate circle center for right turn
        # Center is to the right of the robot (perpendicular to heading)
        center_x = robot_x + self.turn_radius * math.sin(robot_theta)
        center_y = robot_y - self.turn_radius * math.cos(robot_theta)
        
        # Calculate target angle after turning right
        target_theta = robot_theta - self.turn_angle
        
        # Calculate target position on the circle
        target_x = center_x - self.turn_radius * math.sin(target_theta)
        target_y = center_y + self.turn_radius * math.cos(target_theta)
        
        return (target_x, target_y)
    
    def get_action_space_size(self) -> int:
        """Get the size of the discrete action space."""
        return len(DiscreteAction)
    
    def get_action_names(self) -> List[str]:
        """Get human-readable names for actions."""
        return [action.name for action in DiscreteAction]
    
    def visualize_action_waypoints(self, 
                                 robot_x: float, 
                                 robot_y: float, 
                                 robot_theta: float) -> dict:
        """
        Generate all possible waypoints for visualization.
        
        Args:
            robot_x: Current robot x position
            robot_y: Current robot y position
            robot_theta: Current robot orientation
            
        Returns:
            Dictionary mapping action names to waypoints
        """
        waypoints = {}
        for action in DiscreteAction:
            waypoint = self.action_to_waypoint(action, robot_x, robot_y, robot_theta)
            waypoints[action.name] = waypoint
        return waypoints


class DiscreteActionController:
    """
    Controller that combines discrete actions with PID control.
    
    This controller takes discrete actions and uses the PID controller
    to smoothly execute them by generating appropriate waypoints.
    """
    
    def __init__(self, pid_controller, action_mapper: DiscreteActionMapper):
        """
        Initialize the discrete action controller.
        
        Args:
            pid_controller: WaypointFollowPIDController instance
            action_mapper: DiscreteActionMapper instance
        """
        self.pid_controller = pid_controller
        self.action_mapper = action_mapper
        self.current_waypoint = None
        self.action_completed = True
        
    def execute_action(self, 
                      action: int, 
                      robot_x: float, 
                      robot_y: float, 
                      robot_theta: float) -> None:
        """
        Execute a discrete action by setting appropriate waypoint.
        
        Args:
            action: Discrete action (0-4)
            robot_x: Current robot x position
            robot_y: Current robot y position
            robot_theta: Current robot orientation
        """
        # Generate waypoint for the action
        waypoint = self.action_mapper.action_to_waypoint(action, robot_x, robot_y, robot_theta)
        
        # Set waypoint for PID controller
        self.pid_controller.set_waypoints([waypoint])
        self.current_waypoint = waypoint
        self.action_completed = False
        
    def compute_control(self, 
                       robot_x: float, 
                       robot_y: float, 
                       robot_theta: float, 
                       dt: float) -> Tuple[float, float, dict]:
        """
        Compute control outputs using PID controller.
        
        Args:
            robot_x: Current robot x position
            robot_y: Current robot y position
            robot_theta: Current robot orientation
            dt: Time step
            
        Returns:
            Tuple of (linear_velocity, angular_velocity, info)
        """
        if self.current_waypoint is None:
            return 0.0, 0.0, {"status": "no_waypoint"}
        
        # Get control from PID controller
        linear_vel, angular_vel, info = self.pid_controller.compute_control(
            robot_x, robot_y, robot_theta, dt
        )
        
        # Check if action is completed
        if info.get("status") == "completed":
            self.action_completed = True
            
        return linear_vel, angular_vel, info
    
    def is_action_completed(self) -> bool:
        """Check if current action is completed."""
        return self.action_completed
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.pid_controller.reset_controllers()
        self.current_waypoint = None
        self.action_completed = True
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass

from ..robot.duckiebot import Duckiebot
from ..utils.geometry import (
    get_robot_collision_shape, circle_circle_collision, 
    circle_rectangle_collision, check_collision_with_map_boundaries,
    Circle, Rectangle
)
from .obstacles import ObstacleManager, Obstacle


@dataclass
class CollisionResult:
    """Result of collision detection."""
    is_colliding: bool
    collision_type: str  # "robot_robot", "robot_obstacle", "robot_boundary"
    robot_id: Optional[str] = None
    other_robot_id: Optional[str] = None
    obstacle_name: Optional[str] = None
    collision_point: Optional[Tuple[float, float]] = None
    penetration_depth: Optional[float] = None


class CollisionDetector:
    """
    Collision detection system for the Duckietown simulator.
    
    Handles detection of:
    - Robot vs Robot collisions
    - Robot vs Static Obstacle collisions
    - Robot vs Map Boundary collisions
    """
    
    def __init__(self, map_width: float, map_height: float):
        """
        Initialize collision detector.
        
        Args:
            map_width: Width of the map in meters
            map_height: Height of the map in meters
        """
        self.map_width = map_width
        self.map_height = map_height
        self.obstacle_manager = ObstacleManager()
        
        # Collision tracking
        self.collision_history: List[CollisionResult] = []
        self.active_collisions: Set[Tuple[str, str]] = set()  # Track ongoing collisions
        
    def set_obstacle_manager(self, obstacle_manager: ObstacleManager):
        """Set the obstacle manager."""
        self.obstacle_manager = obstacle_manager
    
    def check_robot_robot_collision(self, robot1: Duckiebot, robot2: Duckiebot,
                                  robot1_id: str = "robot1", robot2_id: str = "robot2") -> CollisionResult:
        """
        Check collision between two robots.
        
        Args:
            robot1: First robot
            robot2: Second robot
            robot1_id: ID of first robot
            robot2_id: ID of second robot
            
        Returns:
            CollisionResult indicating if collision occurred
        """
        # Get collision shapes
        shape1 = get_robot_collision_shape(robot1.x, robot1.y, robot1.theta, robot1.collision_radius)
        shape2 = get_robot_collision_shape(robot2.x, robot2.y, robot2.theta, robot2.collision_radius)
        
        # Check collision
        is_colliding = circle_circle_collision(shape1, shape2)
        
        if is_colliding:
            # Calculate collision details
            distance = shape1.center.distance_to(shape2.center)
            penetration = (shape1.radius + shape2.radius) - distance
            collision_point = (
                (robot1.x + robot2.x) / 2,
                (robot1.y + robot2.y) / 2
            )
            
            return CollisionResult(
                is_colliding=True,
                collision_type="robot_robot",
                robot_id=robot1_id,
                other_robot_id=robot2_id,
                collision_point=collision_point,
                penetration_depth=penetration
            )
        
        return CollisionResult(is_colliding=False, collision_type="robot_robot")
    
    def check_robot_obstacle_collision(self, robot: Duckiebot, 
                                     robot_id: str = "robot") -> List[CollisionResult]:
        """
        Check collision between robot and all obstacles.
        
        Args:
            robot: Robot to check
            robot_id: ID of the robot
            
        Returns:
            List of CollisionResults for each obstacle collision
        """
        collisions = []
        
        # Get robot collision shape
        robot_shape = get_robot_collision_shape(robot.x, robot.y, robot.theta, robot.collision_radius)
        
        # Check against all obstacles
        for obstacle in self.obstacle_manager.get_all_obstacles():
            is_colliding = False
            
            if isinstance(obstacle.collision_shape, Circle):
                is_colliding = circle_circle_collision(robot_shape, obstacle.collision_shape)
            elif isinstance(obstacle.collision_shape, Rectangle):
                is_colliding = circle_rectangle_collision(robot_shape, obstacle.collision_shape)
            
            if is_colliding:
                # Calculate collision point (approximate)
                collision_point = (
                    (robot.x + obstacle.x) / 2,
                    (robot.y + obstacle.y) / 2
                )
                
                collision = CollisionResult(
                    is_colliding=True,
                    collision_type="robot_obstacle",
                    robot_id=robot_id,
                    obstacle_name=obstacle.name,
                    collision_point=collision_point
                )
                collisions.append(collision)
        
        return collisions
    
    def check_robot_boundary_collision(self, robot: Duckiebot,
                                     robot_id: str = "robot") -> CollisionResult:
        """
        Check collision between robot and map boundaries.
        
        Args:
            robot: Robot to check
            robot_id: ID of the robot
            
        Returns:
            CollisionResult indicating boundary collision
        """
        is_colliding = check_collision_with_map_boundaries(
            robot.x, robot.y, robot.collision_radius,
            self.map_width, self.map_height
        )
        
        if is_colliding:
            # Determine which boundary
            collision_point = None
            if robot.x - robot.collision_radius < 0:
                collision_point = (0, robot.y)
            elif robot.x + robot.collision_radius > self.map_width:
                collision_point = (self.map_width, robot.y)
            elif robot.y - robot.collision_radius < 0:
                collision_point = (robot.x, 0)
            elif robot.y + robot.collision_radius > self.map_height:
                collision_point = (robot.x, self.map_height)
            
            return CollisionResult(
                is_colliding=True,
                collision_type="robot_boundary",
                robot_id=robot_id,
                collision_point=collision_point
            )
        
        return CollisionResult(is_colliding=False, collision_type="robot_boundary")
    
    def check_all_collisions(self, robots: Dict[str, Duckiebot]) -> List[CollisionResult]:
        """
        Check all possible collisions for given robots.
        
        Args:
            robots: Dictionary of robot_id -> Duckiebot
            
        Returns:
            List of all detected collisions
        """
        all_collisions = []
        robot_ids = list(robots.keys())
        
        # Check robot-robot collisions
        for i, robot1_id in enumerate(robot_ids):
            for j, robot2_id in enumerate(robot_ids[i+1:], i+1):
                collision = self.check_robot_robot_collision(
                    robots[robot1_id], robots[robot2_id], robot1_id, robot2_id
                )
                if collision.is_colliding:
                    all_collisions.append(collision)
        
        # Check robot-obstacle and robot-boundary collisions
        for robot_id, robot in robots.items():
            # Obstacle collisions
            obstacle_collisions = self.check_robot_obstacle_collision(robot, robot_id)
            all_collisions.extend(obstacle_collisions)
            
            # Boundary collisions
            boundary_collision = self.check_robot_boundary_collision(robot, robot_id)
            if boundary_collision.is_colliding:
                all_collisions.append(boundary_collision)
        
        # Update collision tracking
        self._update_collision_tracking(all_collisions)
        
        return all_collisions
    
    def _update_collision_tracking(self, collisions: List[CollisionResult]):
        """Update collision tracking and history."""
        # Add to history
        self.collision_history.extend(collisions)
        
        # Update active collisions
        current_collisions = set()
        for collision in collisions:
            if collision.collision_type == "robot_robot":
                key = tuple(sorted([collision.robot_id, collision.other_robot_id]))
                current_collisions.add(key)
        
        self.active_collisions = current_collisions
    
    def get_collision_summary(self) -> Dict[str, int]:
        """Get summary of collision counts."""
        summary = {
            "robot_robot": 0,
            "robot_obstacle": 0,
            "robot_boundary": 0,
            "total": 0
        }
        
        for collision in self.collision_history:
            summary[collision.collision_type] += 1
            summary["total"] += 1
        
        return summary
    
    def clear_collision_history(self):
        """Clear collision history."""
        self.collision_history.clear()
        self.active_collisions.clear()
    
    def is_position_collision_free(self, x: float, y: float, 
                                 collision_radius: float,
                                 exclude_robots: Set[str] = None) -> bool:
        """
        Check if a position would be collision-free.
        
        Args:
            x: X position to check
            y: Y position to check
            collision_radius: Collision radius
            exclude_robots: Set of robot IDs to exclude from check
            
        Returns:
            True if position is collision-free
        """
        # Check boundary collision
        if check_collision_with_map_boundaries(x, y, collision_radius, 
                                             self.map_width, self.map_height):
            return False
        
        # Check obstacle collisions
        test_shape = get_robot_collision_shape(x, y, 0, collision_radius)
        
        for obstacle in self.obstacle_manager.get_all_obstacles():
            if isinstance(obstacle.collision_shape, Circle):
                if circle_circle_collision(test_shape, obstacle.collision_shape):
                    return False
            elif isinstance(obstacle.collision_shape, Rectangle):
                if circle_rectangle_collision(test_shape, obstacle.collision_shape):
                    return False
        
        return True
    
    def find_safe_spawn_position(self, collision_radius: float,
                                robots: Dict[str, Duckiebot] = None,
                                max_attempts: int = 100) -> Optional[Tuple[float, float]]:
        """
        Find a safe position to spawn a robot.
        
        Args:
            collision_radius: Collision radius of robot to spawn
            robots: Existing robots to avoid
            max_attempts: Maximum attempts to find safe position
            
        Returns:
            Safe (x, y) position or None if not found
        """
        robots = robots or {}
        margin = collision_radius * 2
        
        for _ in range(max_attempts):
            x = np.random.uniform(margin, self.map_width - margin)
            y = np.random.uniform(margin, self.map_height - margin)
            
            # Check if position is collision-free
            if self.is_position_collision_free(x, y, collision_radius):
                # Check against existing robots
                safe = True
                test_shape = get_robot_collision_shape(x, y, 0, collision_radius)
                
                for robot in robots.values():
                    robot_shape = get_robot_collision_shape(robot.x, robot.y, robot.theta, 
                                                          robot.collision_radius)
                    if circle_circle_collision(test_shape, robot_shape):
                        safe = False
                        break
                
                if safe:
                    return (x, y)
        
        return None
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get detailed collision statistics."""
        summary = self.get_collision_summary()
        
        stats = {
            "total_collisions": summary["total"],
            "collision_breakdown": summary,
            "active_collisions": len(self.active_collisions),
            "collision_rate": 0.0
        }
        
        if len(self.collision_history) > 0:
            # Calculate collision density
            recent_collisions = self.collision_history[-100:]  # Last 100 collisions
            stats["recent_collision_types"] = {}
            for collision in recent_collisions:
                collision_type = collision.collision_type
                stats["recent_collision_types"][collision_type] = \
                    stats["recent_collision_types"].get(collision_type, 0) + 1
        
        return stats


def create_collision_detector(map_width: float, map_height: float,
                            with_obstacles: bool = True) -> CollisionDetector:
    """
    Factory function to create a collision detector.
    
    Args:
        map_width: Width of the map
        map_height: Height of the map
        with_obstacles: Whether to add default obstacles
        
    Returns:
        Configured CollisionDetector
    """
    detector = CollisionDetector(map_width, map_height)
    
    if with_obstacles:
        from .obstacles import create_simple_obstacles
        obstacle_manager = create_simple_obstacles(map_width, map_height)
        detector.set_obstacle_manager(obstacle_manager)
    
    return detector
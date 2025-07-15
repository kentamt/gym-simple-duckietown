import numpy as np
import math
from typing import Tuple, List, Union, Optional
from dataclasses import dataclass


@dataclass
class Point:
    """2D point representation."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)


@dataclass
class Circle:
    """Circle representation for collision detection."""
    center: Point
    radius: float
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside circle."""
        return self.center.distance_to(point) <= self.radius
    
    def intersects_circle(self, other: 'Circle') -> bool:
        """Check if this circle intersects with another circle."""
        distance = self.center.distance_to(other.center)
        return distance <= (self.radius + other.radius)
    
    def intersects_rectangle(self, rect: 'Rectangle') -> bool:
        """Check if circle intersects with rectangle."""
        return circle_rectangle_collision(self, rect)


@dataclass
class Rectangle:
    """Rectangle representation for collision detection."""
    center: Point
    width: float
    height: float
    rotation: float = 0.0  # Rotation in radians
    
    def get_corners(self) -> List[Point]:
        """Get the four corner points of the rectangle."""
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Local corners (before rotation)
        local_corners = [
            Point(-half_width, -half_height),
            Point(half_width, -half_height),
            Point(half_width, half_height),
            Point(-half_width, half_height)
        ]
        
        # Apply rotation and translation
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        
        corners = []
        for corner in local_corners:
            # Rotate
            rotated_x = corner.x * cos_r - corner.y * sin_r
            rotated_y = corner.x * sin_r + corner.y * cos_r
            
            # Translate
            world_corner = Point(
                rotated_x + self.center.x,
                rotated_y + self.center.y
            )
            corners.append(world_corner)
        
        return corners
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside rectangle."""
        return point_in_rectangle(point, self)
    
    def intersects_rectangle(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another rectangle."""
        return rectangle_rectangle_collision(self, other)
    
    def intersects_circle(self, circle: Circle) -> bool:
        """Check if rectangle intersects with circle."""
        return circle_rectangle_collision(circle, self)


def distance_point_to_point(p1: Point, p2: Point) -> float:
    """Calculate distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def circle_circle_collision(c1: Circle, c2: Circle) -> bool:
    """
    Check collision between two circles.
    
    Args:
        c1: First circle
        c2: Second circle
        
    Returns:
        True if circles are colliding
    """
    distance = c1.center.distance_to(c2.center)
    return distance <= (c1.radius + c2.radius)


def point_in_rectangle(point: Point, rect: Rectangle) -> bool:
    """
    Check if a point is inside a rectangle.
    
    Args:
        point: Point to check
        rect: Rectangle to check against
        
    Returns:
        True if point is inside rectangle
    """
    # Transform point to rectangle's local coordinate system
    cos_r = math.cos(-rect.rotation)
    sin_r = math.sin(-rect.rotation)
    
    # Translate to rectangle center
    local_x = point.x - rect.center.x
    local_y = point.y - rect.center.y
    
    # Rotate to align with rectangle axes
    rotated_x = local_x * cos_r - local_y * sin_r
    rotated_y = local_x * sin_r + local_y * cos_r
    
    # Check bounds
    half_width = rect.width / 2
    half_height = rect.height / 2
    
    return (abs(rotated_x) <= half_width and abs(rotated_y) <= half_height)


def circle_rectangle_collision(circle: Circle, rect: Rectangle) -> bool:
    """
    Check collision between a circle and a rectangle.
    
    Args:
        circle: Circle to check
        rect: Rectangle to check
        
    Returns:
        True if circle and rectangle are colliding
    """
    # Transform circle center to rectangle's local coordinate system
    cos_r = math.cos(-rect.rotation)
    sin_r = math.sin(-rect.rotation)
    
    # Translate to rectangle center
    local_x = circle.center.x - rect.center.x
    local_y = circle.center.y - rect.center.y
    
    # Rotate to align with rectangle axes
    rotated_x = local_x * cos_r - local_y * sin_r
    rotated_y = local_x * sin_r + local_y * cos_r
    
    # Find closest point on rectangle to circle center
    half_width = rect.width / 2
    half_height = rect.height / 2
    
    closest_x = max(-half_width, min(half_width, rotated_x))
    closest_y = max(-half_height, min(half_height, rotated_y))
    
    # Calculate distance from circle center to closest point
    distance_sq = (rotated_x - closest_x)**2 + (rotated_y - closest_y)**2
    
    return distance_sq <= circle.radius**2


def rectangle_rectangle_collision(rect1: Rectangle, rect2: Rectangle) -> bool:
    """
    Check collision between two rectangles using Separating Axis Theorem.
    
    Args:
        rect1: First rectangle
        rect2: Second rectangle
        
    Returns:
        True if rectangles are colliding
    """
    # Get corners of both rectangles
    corners1 = rect1.get_corners()
    corners2 = rect2.get_corners()
    
    # Test all potential separating axes
    for rect in [rect1, rect2]:
        # Get two edges of the rectangle to form axes
        cos_r = math.cos(rect.rotation)
        sin_r = math.sin(rect.rotation)
        
        # Two perpendicular axes for this rectangle
        axes = [
            Point(cos_r, sin_r),      # Parallel to width
            Point(-sin_r, cos_r)      # Parallel to height
        ]
        
        for axis in axes:
            # Project all corners onto this axis
            proj1 = [project_point_onto_axis(corner, axis) for corner in corners1]
            proj2 = [project_point_onto_axis(corner, axis) for corner in corners2]
            
            # Find min/max projections
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no collision
    
    return True  # No separating axis found, collision exists


def project_point_onto_axis(point: Point, axis: Point) -> float:
    """Project a point onto an axis (dot product)."""
    return point.x * axis.x + point.y * axis.y


def line_circle_collision(line_start: Point, line_end: Point, circle: Circle) -> bool:
    """
    Check collision between a line segment and a circle.
    
    Args:
        line_start: Start point of line segment
        line_end: End point of line segment
        circle: Circle to check
        
    Returns:
        True if line segment intersects circle
    """
    # Vector from line start to end
    line_vec = Point(line_end.x - line_start.x, line_end.y - line_start.y)
    line_length_sq = line_vec.x**2 + line_vec.y**2
    
    if line_length_sq == 0:
        # Line is a point, check distance to circle center
        return line_start.distance_to(circle.center) <= circle.radius
    
    # Vector from line start to circle center
    to_circle = Point(circle.center.x - line_start.x, circle.center.y - line_start.y)
    
    # Project circle center onto line
    t = max(0, min(1, (to_circle.x * line_vec.x + to_circle.y * line_vec.y) / line_length_sq))
    
    # Find closest point on line segment
    closest = Point(
        line_start.x + t * line_vec.x,
        line_start.y + t * line_vec.y
    )
    
    # Check distance
    return closest.distance_to(circle.center) <= circle.radius


def get_robot_collision_shape(x: float, y: float, theta: float, 
                            collision_radius: float) -> Circle:
    """
    Get collision shape for a robot.
    
    Args:
        x: Robot x position
        y: Robot y position
        theta: Robot orientation (not used for circle)
        collision_radius: Robot collision radius
        
    Returns:
        Circle representing robot collision shape
    """
    return Circle(Point(x, y), collision_radius)


def get_rectangular_obstacle_shape(x: float, y: float, width: float, height: float,
                                 rotation: float = 0.0) -> Rectangle:
    """
    Get rectangular collision shape for an obstacle.
    
    Args:
        x: Obstacle center x position
        y: Obstacle center y position
        width: Obstacle width
        height: Obstacle height
        rotation: Obstacle rotation in radians
        
    Returns:
        Rectangle representing obstacle collision shape
    """
    return Rectangle(Point(x, y), width, height, rotation)


def get_circular_obstacle_shape(x: float, y: float, radius: float) -> Circle:
    """
    Get circular collision shape for an obstacle.
    
    Args:
        x: Obstacle center x position
        y: Obstacle center y position
        radius: Obstacle radius
        
    Returns:
        Circle representing obstacle collision shape
    """
    return Circle(Point(x, y), radius)


def check_collision_with_map_boundaries(x: float, y: float, collision_radius: float,
                                      map_width: float, map_height: float) -> bool:
    """
    Check if robot collides with map boundaries.
    
    Args:
        x: Robot x position
        y: Robot y position
        collision_radius: Robot collision radius
        map_width: Map width
        map_height: Map height
        
    Returns:
        True if robot collides with boundaries
    """
    return (x - collision_radius < 0 or 
            x + collision_radius > map_width or
            y - collision_radius < 0 or 
            y + collision_radius > map_height)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles."""
    diff = angle2 - angle1
    return normalize_angle(diff)
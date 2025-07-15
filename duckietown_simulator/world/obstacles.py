import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..utils.geometry import (
    Point, Circle, Rectangle,
    get_rectangular_obstacle_shape, get_circular_obstacle_shape
)


class ObstacleType(Enum):
    """Types of obstacles."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    WALL = "wall"


@dataclass
class ObstacleConfig:
    """Configuration for an obstacle."""
    x: float  # Center x position
    y: float  # Center y position
    obstacle_type: ObstacleType
    # For rectangular obstacles
    width: Optional[float] = None
    height: Optional[float] = None
    rotation: float = 0.0  # Rotation in radians
    # For circular obstacles
    radius: Optional[float] = None
    # General properties
    name: Optional[str] = None
    color: str = "#2c3e50"  # Default dark color
    
    def __post_init__(self):
        if self.obstacle_type == ObstacleType.RECTANGLE or self.obstacle_type == ObstacleType.WALL:
            if self.width is None or self.height is None:
                raise ValueError("Rectangle/Wall obstacles require width and height")
        elif self.obstacle_type == ObstacleType.CIRCLE:
            if self.radius is None:
                raise ValueError("Circle obstacles require radius")


class Obstacle:
    """
    Represents a static obstacle in the environment.
    """
    
    def __init__(self, config: ObstacleConfig):
        """
        Initialize obstacle.
        
        Args:
            config: Obstacle configuration
        """
        self.config = config
        self.x = config.x
        self.y = config.y
        self.obstacle_type = config.obstacle_type
        self.name = config.name or f"{config.obstacle_type.value}_{id(self)}"
        self.color = config.color
        
        # Create collision shape
        if config.obstacle_type == ObstacleType.RECTANGLE or config.obstacle_type == ObstacleType.WALL:
            self.collision_shape = get_rectangular_obstacle_shape(
                config.x, config.y, config.width, config.height, config.rotation
            )
            self.width = config.width
            self.height = config.height
            self.rotation = config.rotation
        elif config.obstacle_type == ObstacleType.CIRCLE:
            self.collision_shape = get_circular_obstacle_shape(
                config.x, config.y, config.radius
            )
            self.radius = config.radius
        else:
            raise ValueError(f"Unknown obstacle type: {config.obstacle_type}")
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of obstacle.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if isinstance(self.collision_shape, Circle):
            return (
                self.x - self.radius,
                self.y - self.radius,
                self.x + self.radius,
                self.y + self.radius
            )
        elif isinstance(self.collision_shape, Rectangle):
            corners = self.collision_shape.get_corners()
            x_coords = [corner.x for corner in corners]
            y_coords = [corner.y for corner in corners]
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside obstacle."""
        point = Point(x, y)
        return self.collision_shape.contains_point(point)
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization."""
        data = {
            'type': self.obstacle_type.value,
            'x': self.x,
            'y': self.y,
            'color': self.color,
            'name': self.name
        }
        
        if isinstance(self.collision_shape, Circle):
            data['radius'] = self.radius
        elif isinstance(self.collision_shape, Rectangle):
            data['width'] = self.width
            data['height'] = self.height
            data['rotation'] = self.rotation
            data['corners'] = [(c.x, c.y) for c in self.collision_shape.get_corners()]
        
        return data
    
    def __str__(self) -> str:
        if self.obstacle_type == ObstacleType.CIRCLE:
            return f"{self.name}: Circle at ({self.x:.2f}, {self.y:.2f}), r={self.radius:.2f}"
        else:
            return f"{self.name}: Rectangle at ({self.x:.2f}, {self.y:.2f}), {self.width:.2f}x{self.height:.2f}"


class ObstacleManager:
    """
    Manages static obstacles in the environment.
    """
    
    def __init__(self):
        """Initialize obstacle manager."""
        self.obstacles: List[Obstacle] = []
        self._obstacle_dict: Dict[str, Obstacle] = {}
    
    def add_obstacle(self, config: ObstacleConfig) -> Obstacle:
        """
        Add an obstacle to the environment.
        
        Args:
            config: Obstacle configuration
            
        Returns:
            Created obstacle
        """
        obstacle = Obstacle(config)
        self.obstacles.append(obstacle)
        self._obstacle_dict[obstacle.name] = obstacle
        return obstacle
    
    def remove_obstacle(self, name: str) -> bool:
        """
        Remove an obstacle by name.
        
        Args:
            name: Name of obstacle to remove
            
        Returns:
            True if obstacle was removed
        """
        if name in self._obstacle_dict:
            obstacle = self._obstacle_dict[name]
            self.obstacles.remove(obstacle)
            del self._obstacle_dict[name]
            return True
        return False
    
    def get_obstacle(self, name: str) -> Optional[Obstacle]:
        """Get obstacle by name."""
        return self._obstacle_dict.get(name)
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()
        self._obstacle_dict.clear()
    
    def get_obstacles_in_region(self, min_x: float, min_y: float, 
                               max_x: float, max_y: float) -> List[Obstacle]:
        """
        Get obstacles that might be in a given region.
        
        Args:
            min_x, min_y, max_x, max_y: Bounding box of region
            
        Returns:
            List of obstacles that could intersect the region
        """
        obstacles_in_region = []
        
        for obstacle in self.obstacles:
            obs_min_x, obs_min_y, obs_max_x, obs_max_y = obstacle.get_bounds()
            
            # Check if bounding boxes overlap
            if (obs_max_x >= min_x and obs_min_x <= max_x and
                obs_max_y >= min_y and obs_min_y <= max_y):
                obstacles_in_region.append(obstacle)
        
        return obstacles_in_region
    
    def get_all_obstacles(self) -> List[Obstacle]:
        """Get all obstacles."""
        return self.obstacles.copy()
    
    def count_obstacles(self) -> int:
        """Get number of obstacles."""
        return len(self.obstacles)
    
    def create_wall_boundary(self, map_width: float, map_height: float, 
                           wall_thickness: float = 0.1):
        """
        Create wall boundaries around the map.
        
        Args:
            map_width: Width of the map
            map_height: Height of the map
            wall_thickness: Thickness of boundary walls
        """
        # Bottom wall
        self.add_obstacle(ObstacleConfig(
            x=map_width/2, y=-wall_thickness/2,
            obstacle_type=ObstacleType.WALL,
            width=map_width, height=wall_thickness,
            name="bottom_wall"
        ))
        
        # Top wall
        self.add_obstacle(ObstacleConfig(
            x=map_width/2, y=map_height + wall_thickness/2,
            obstacle_type=ObstacleType.WALL,
            width=map_width, height=wall_thickness,
            name="top_wall"
        ))
        
        # Left wall
        self.add_obstacle(ObstacleConfig(
            x=-wall_thickness/2, y=map_height/2,
            obstacle_type=ObstacleType.WALL,
            width=wall_thickness, height=map_height,
            name="left_wall"
        ))
        
        # Right wall
        self.add_obstacle(ObstacleConfig(
            x=map_width + wall_thickness/2, y=map_height/2,
            obstacle_type=ObstacleType.WALL,
            width=wall_thickness, height=map_height,
            name="right_wall"
        ))
    
    def create_random_obstacles(self, map_width: float, map_height: float,
                              num_obstacles: int, min_size: float = 0.2,
                              max_size: float = 0.5, margin: float = 0.5) -> List[Obstacle]:
        """
        Create random obstacles within the map bounds.
        
        Args:
            map_width: Width of the map
            map_height: Height of the map
            num_obstacles: Number of obstacles to create
            min_size: Minimum obstacle size
            max_size: Maximum obstacle size
            margin: Margin from map edges
            
        Returns:
            List of created obstacles
        """
        created_obstacles = []
        
        for i in range(num_obstacles):
            # Random position within margins
            x = np.random.uniform(margin, map_width - margin)
            y = np.random.uniform(margin, map_height - margin)
            
            # Random obstacle type
            if np.random.random() < 0.5:
                # Circle obstacle
                radius = np.random.uniform(min_size, max_size)
                config = ObstacleConfig(
                    x=x, y=y,
                    obstacle_type=ObstacleType.CIRCLE,
                    radius=radius,
                    name=f"random_circle_{i}"
                )
            else:
                # Rectangle obstacle
                width = np.random.uniform(min_size, max_size)
                height = np.random.uniform(min_size, max_size)
                rotation = np.random.uniform(0, 2 * np.pi)
                config = ObstacleConfig(
                    x=x, y=y,
                    obstacle_type=ObstacleType.RECTANGLE,
                    width=width, height=height, rotation=rotation,
                    name=f"random_rect_{i}"
                )
            
            obstacle = self.add_obstacle(config)
            created_obstacles.append(obstacle)
        
        return created_obstacles
    
    def __str__(self) -> str:
        """String representation of obstacle manager."""
        return f"ObstacleManager with {len(self.obstacles)} obstacles"


def create_simple_obstacles(map_width: float, map_height: float) -> ObstacleManager:
    """
    Create a simple obstacle setup for testing.
    
    Args:
        map_width: Width of the map
        map_height: Height of the map
        
    Returns:
        ObstacleManager with simple obstacles
    """
    manager = ObstacleManager()
    
    # Central circular obstacle
    manager.add_obstacle(ObstacleConfig(
        x=map_width/2, y=map_height/2,
        obstacle_type=ObstacleType.CIRCLE,
        radius=0.3,
        name="central_circle"
    ))
    
    # Rectangular obstacles in corners
    margin = 0.8
    size = 0.4
    
    # Bottom-left
    manager.add_obstacle(ObstacleConfig(
        x=margin, y=margin,
        obstacle_type=ObstacleType.RECTANGLE,
        width=size, height=size,
        name="corner_bl"
    ))
    
    # Top-right
    manager.add_obstacle(ObstacleConfig(
        x=map_width-margin, y=map_height-margin,
        obstacle_type=ObstacleType.RECTANGLE,
        width=size, height=size, rotation=np.pi/4,
        name="corner_tr"
    ))
    
    return manager
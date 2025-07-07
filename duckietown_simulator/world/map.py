import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MapConfig:
    """Configuration for the map layout."""
    width: int  # Number of tiles in x direction
    height: int  # Number of tiles in y direction
    tile_size: float = 0.61  # Size of each tile in meters (61cm)
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map dimensions must be positive")
        if self.tile_size <= 0:
            raise ValueError("Tile size must be positive")


class Map:
    """
    Tile-based map representation for Duckietown simulator.
    
    Each tile is 61cm x 61cm square. The map consists of a grid of tiles
    that can be configured with different sizes (e.g., 5x5, 6x5, etc.).
    """
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.width_tiles = config.width
        self.height_tiles = config.height
        self.tile_size = config.tile_size
        
        # Calculate map dimensions in meters
        self.width_meters = self.width_tiles * self.tile_size
        self.height_meters = self.height_tiles * self.tile_size
        
        # Create tile grid (0 = empty, 1 = obstacle, 2 = road, etc.)
        self.tiles = np.zeros((self.height_tiles, self.width_tiles), dtype=int)
        
        # Store tile boundaries for collision detection
        self._compute_tile_boundaries()
    
    def _compute_tile_boundaries(self):
        """Compute the boundaries of each tile for collision detection."""
        self.tile_boundaries = []
        for row in range(self.height_tiles):
            tile_row = []
            for col in range(self.width_tiles):
                x_min = col * self.tile_size
                x_max = (col + 1) * self.tile_size
                y_min = row * self.tile_size
                y_max = (row + 1) * self.tile_size
                
                tile_row.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'center_x': (x_min + x_max) / 2,
                    'center_y': (y_min + y_max) / 2
                })
            self.tile_boundaries.append(tile_row)
    
    def get_tile_at_position(self, x: float, y: float) -> Tuple[int, int]:
        """
        Get the tile coordinates for a given world position.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            Tuple of (row, col) tile coordinates, or (-1, -1) if out of bounds
        """
        if x < 0 or x >= self.width_meters or y < 0 or y >= self.height_meters:
            return (-1, -1)
        
        col = int(x / self.tile_size)
        row = int(y / self.tile_size)
        
        # Ensure we don't go out of bounds due to floating point precision
        col = min(col, self.width_tiles - 1)
        row = min(row, self.height_tiles - 1)
        
        return (row, col)
    
    def get_tile_center(self, row: int, col: int) -> Tuple[float, float]:
        """
        Get the center coordinates of a tile.
        
        Args:
            row: Tile row index
            col: Tile column index
            
        Returns:
            Tuple of (x, y) center coordinates in meters
        """
        if not self.is_valid_tile(row, col):
            raise ValueError(f"Invalid tile coordinates: ({row}, {col})")
        
        return (self.tile_boundaries[row][col]['center_x'],
                self.tile_boundaries[row][col]['center_y'])
    
    def is_valid_tile(self, row: int, col: int) -> bool:
        """Check if tile coordinates are valid."""
        return 0 <= row < self.height_tiles and 0 <= col < self.width_tiles
    
    def set_tile_type(self, row: int, col: int, tile_type: int):
        """
        Set the type of a specific tile.
        
        Args:
            row: Tile row index
            col: Tile column index
            tile_type: Type of tile (0=empty, 1=obstacle, 2=road, etc.)
        """
        if not self.is_valid_tile(row, col):
            raise ValueError(f"Invalid tile coordinates: ({row}, {col})")
        
        self.tiles[row, col] = tile_type
    
    def get_tile_type(self, row: int, col: int) -> int:
        """Get the type of a specific tile."""
        if not self.is_valid_tile(row, col):
            return -1  # Out of bounds
        
        return self.tiles[row, col]
    
    def get_map_boundaries(self) -> Dict[str, float]:
        """Get the overall map boundaries."""
        return {
            'x_min': 0.0,
            'x_max': self.width_meters,
            'y_min': 0.0,
            'y_max': self.height_meters
        }
    
    def is_position_in_bounds(self, x: float, y: float) -> bool:
        """Check if a position is within map boundaries."""
        return 0 <= x < self.width_meters and 0 <= y < self.height_meters
    
    def get_tile_corners(self, row: int, col: int) -> List[Tuple[float, float]]:
        """
        Get the four corner coordinates of a tile.
        
        Returns:
            List of (x, y) tuples for the four corners
        """
        if not self.is_valid_tile(row, col):
            raise ValueError(f"Invalid tile coordinates: ({row}, {col})")
        
        bounds = self.tile_boundaries[row][col]
        return [
            (bounds['x_min'], bounds['y_min']),  # Bottom-left
            (bounds['x_max'], bounds['y_min']),  # Bottom-right
            (bounds['x_max'], bounds['y_max']),  # Top-right
            (bounds['x_min'], bounds['y_max'])   # Top-left
        ]
    
    def create_straight_track(self):
        """Create a simple straight track layout."""
        # Mark all tiles as road
        self.tiles.fill(2)
        
        # Add boundaries on the edges
        self.tiles[0, :] = 1  # Top boundary
        self.tiles[-1, :] = 1  # Bottom boundary
        self.tiles[:, 0] = 1  # Left boundary
        self.tiles[:, -1] = 1  # Right boundary
    
    def create_loop_track(self):
        """Create a simple loop track layout."""
        # Start with all obstacles
        self.tiles.fill(1)
        
        # Create a simple rectangular loop
        if self.width_tiles >= 3 and self.height_tiles >= 3:
            # Create road in the middle area
            for row in range(1, self.height_tiles - 1):
                for col in range(1, self.width_tiles - 1):
                    # Create a hollow rectangle
                    if (row == 1 or row == self.height_tiles - 2 or 
                        col == 1 or col == self.width_tiles - 2):
                        self.tiles[row, col] = 2  # Road
    
    def __str__(self) -> str:
        """String representation of the map."""
        lines = [f"Map: {self.width_tiles}x{self.height_tiles} tiles ({self.width_meters:.2f}m x {self.height_meters:.2f}m)"]
        lines.append("Tile layout:")
        for row in range(self.height_tiles):
            line = ""
            for col in range(self.width_tiles):
                tile_type = self.tiles[row, col]
                if tile_type == 0:
                    line += "."  # Empty
                elif tile_type == 1:
                    line += "#"  # Obstacle
                elif tile_type == 2:
                    line += "="  # Road
                else:
                    line += "?"  # Unknown
            lines.append(line)
        return "\n".join(lines)


def create_map_from_config(width: int, height: int, track_type: str = "straight") -> Map:
    """
    Factory function to create a map with specified dimensions and track type.
    
    Args:
        width: Number of tiles in x direction
        height: Number of tiles in y direction
        track_type: Type of track layout ("straight", "loop")
        
    Returns:
        Configured Map instance
    """
    config = MapConfig(width=width, height=height)
    map_instance = Map(config)
    
    if track_type == "straight":
        map_instance.create_straight_track()
    elif track_type == "loop":
        map_instance.create_loop_track()
    else:
        raise ValueError(f"Unknown track type: {track_type}")
    
    return map_instance
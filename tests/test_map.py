#!/usr/bin/env python3
"""
Test script to demonstrate the tile-based map system.
"""

import sys
sys.path.append('.')

from duckietown_simulator.world.map import Map, MapConfig, create_map_from_config
from duckietown_simulator.utils.config import ConfigManager


def test_basic_map_creation():
    """Test basic map creation with different sizes."""
    print("=== Testing Basic Map Creation ===")
    
    # Test different map sizes
    sizes = [(3, 3), (5, 5), (6, 5), (10, 5), (8, 8)]
    
    for width, height in sizes:
        print(f"\nCreating {width}x{height} map:")
        config = MapConfig(width=width, height=height)
        map_instance = Map(config)
        
        print(f"  Map dimensions: {map_instance.width_meters:.2f}m x {map_instance.height_meters:.2f}m")
        print(f"  Tile size: {map_instance.tile_size}m")
        print(f"  Total tiles: {map_instance.width_tiles * map_instance.height_tiles}")


def test_track_layouts():
    """Test different track layout creation."""
    print("\n=== Testing Track Layouts ===")
    
    # Test straight track
    print("\nStraight track (5x5):")
    straight_map = create_map_from_config(5, 5, "straight")
    print(straight_map)
    
    # Test loop track
    print("\nLoop track (6x5):")
    loop_map = create_map_from_config(6, 5, "loop")
    print(loop_map)


def test_tile_operations():
    """Test tile-based operations."""
    print("\n=== Testing Tile Operations ===")
    
    map_instance = create_map_from_config(5, 5, "straight")
    
    # Test position to tile conversion
    positions = [(0.0, 0.0), (0.3, 0.3), (1.5, 1.5), (2.8, 2.8)]
    
    print("\nPosition to tile conversion:")
    for x, y in positions:
        row, col = map_instance.get_tile_at_position(x, y)
        print(f"  Position ({x:.1f}, {y:.1f}) -> Tile ({row}, {col})")
        
        if row != -1 and col != -1:
            center_x, center_y = map_instance.get_tile_center(row, col)
            print(f"    Tile center: ({center_x:.3f}, {center_y:.3f})")
    
    # Test tile boundaries
    print("\nTile boundaries for (1, 1):")
    corners = map_instance.get_tile_corners(1, 1)
    for i, (x, y) in enumerate(corners):
        print(f"  Corner {i}: ({x:.3f}, {y:.3f})")


def test_config_manager():
    """Test configuration manager."""
    print("\n=== Testing Configuration Manager ===")
    
    config_manager = ConfigManager()
    
    # Create default configs
    config_manager.create_default_configs()
    
    # List available configs
    configs = config_manager.list_track_configs()
    print(f"\nAvailable track configs: {configs}")
    
    # Test loading a config
    if configs:
        config_name = configs[0]
        print(f"\nLoading config: {config_name}")
        config = config_manager.load_track_config(config_name)
        print(f"  Config: {config}")
        
        # Get map dimensions
        width, height = config_manager.get_map_dimensions(config_name)
        print(f"  Map dimensions: {width}x{height} tiles")
        
        width_m, height_m = config_manager.get_map_size_meters(config_name)
        print(f"  Map size: {width_m:.2f}m x {height_m:.2f}m")


def test_boundary_checking():
    """Test boundary checking functionality."""
    print("\n=== Testing Boundary Checking ===")
    
    map_instance = create_map_from_config(5, 5, "straight")
    
    test_positions = [
        (0.0, 0.0),      # Valid - corner
        (1.5, 1.5),      # Valid - center
        (3.04, 3.04),    # Valid - near edge
        (3.05, 3.05),    # Invalid - just outside
        (-0.1, 0.0),     # Invalid - negative x
        (0.0, -0.1),     # Invalid - negative y
        (10.0, 10.0)     # Invalid - way outside
    ]
    
    boundaries = map_instance.get_map_boundaries()
    print(f"Map boundaries: {boundaries}")
    
    print("\nBoundary checking:")
    for x, y in test_positions:
        in_bounds = map_instance.is_position_in_bounds(x, y)
        print(f"  Position ({x:.2f}, {y:.2f}): {'✓' if in_bounds else '✗'}")


if __name__ == "__main__":
    test_basic_map_creation()
    test_track_layouts()
    test_tile_operations()
    test_config_manager()
    test_boundary_checking()
    
    print("\n=== All tests completed successfully! ===")
#!/usr/bin/env python3
"""
Script to create rotated versions of road tiles for directional mapping.
"""

import pygame
import numpy as np
import os
from pathlib import Path


def create_rotated_tiles():
    """Create rotated versions of road tiles."""
    
    # Initialize pygame for image operations
    pygame.init()
    
    # Paths
    tiles_dir = Path("duckietown_simulator/assets/tiles")
    
    # Base road tiles to rotate
    base_tiles = {
        'road': 'road.png',
        'straight': 'straight.png',
        'turn_left': 'turn_left.png',
        'turn_right': 'turn_right.png'
    }
    
    print("Creating rotated road tiles...")
    
    for tile_name, tile_file in base_tiles.items():
        base_path = tiles_dir / tile_file
        
        if not base_path.exists():
            print(f"Warning: Base tile {base_path} not found, skipping...")
            continue
            
        try:
            # Load the base image
            base_image = pygame.image.load(str(base_path))
            print(f"\nProcessing {tile_name} ({tile_file}):")
            print(f"  Original size: {base_image.get_size()}")
            
            # Create rotated versions
            rotations = {
                '0': 0,      # North/Up
                '90': 90,    # East/Right  
                '180': 180,  # South/Down
                '270': 270   # West/Left
            }
            
            for angle_name, angle in rotations.items():
                if angle == 0:
                    # Use original image for 0 degrees
                    rotated_image = base_image
                else:
                    # Rotate the image
                    rotated_image = pygame.transform.rotate(base_image, angle)
                
                # Save rotated version
                output_name = f"{tile_name}_{angle_name}.png"
                output_path = tiles_dir / output_name
                
                pygame.image.save(rotated_image, str(output_path))
                print(f"  Created: {output_name} (rotated {angle}°)")
                
        except Exception as e:
            print(f"Error processing {tile_file}: {e}")
    
    print(f"\nRotated tiles created in {tiles_dir}/")
    pygame.quit()


def create_enhanced_road_tiles():
    """Create enhanced road tiles with clear directional indicators."""
    
    pygame.init()
    tile_size = 256
    tiles_dir = Path("duckietown_simulator/assets/tiles")
    
    print("Creating enhanced directional road tiles...")
    
    # Define road types with their characteristics
    road_types = {
        'road_straight': {
            'description': 'Straight road with lane markings',
            'lane_width': 20,
            'center_line': True,
            'side_lines': True
        },
        'road_curve_left': {
            'description': 'Left curve road',
            'is_curve': True,
            'curve_direction': 'left'
        },
        'road_curve_right': {
            'description': 'Right curve road', 
            'is_curve': True,
            'curve_direction': 'right'
        },
        'road_intersection': {
            'description': 'Four-way intersection',
            'is_intersection': True
        }
    }
    
    for road_type, config in road_types.items():
        print(f"\nCreating {road_type}:")
        
        # Create 4 rotational variants
        for rotation in [0, 90, 180, 270]:
            surface = pygame.Surface((tile_size, tile_size))
            surface.fill((50, 50, 50))  # Dark gray road base
            
            if config.get('is_curve'):
                # Create curved road
                create_curved_road(surface, tile_size, config['curve_direction'], rotation)
            elif config.get('is_intersection'):
                # Create intersection
                create_intersection_road(surface, tile_size, rotation)
            else:
                # Create straight road
                create_straight_road(surface, tile_size, rotation)
            
            # Save the tile
            filename = f"{road_type}_{rotation}.png"
            filepath = tiles_dir / filename
            pygame.image.save(surface, str(filepath))
            print(f"  Created: {filename}")
    
    pygame.quit()


def create_straight_road(surface, tile_size, rotation):
    """Create a straight road with lane markings."""
    
    # Road dimensions
    road_width = tile_size - 40  # Leave borders
    lane_width = road_width // 2
    
    if rotation in [0, 180]:  # Vertical road
        # Center the road horizontally
        road_x = (tile_size - road_width) // 2
        
        # Yellow center line
        center_x = tile_size // 2
        pygame.draw.line(surface, (255, 255, 0), 
                        (center_x, 0), (center_x, tile_size), 4)
        
        # White side lines
        left_x = road_x + 10
        right_x = road_x + road_width - 10
        pygame.draw.line(surface, (255, 255, 255),
                        (left_x, 0), (left_x, tile_size), 3)
        pygame.draw.line(surface, (255, 255, 255),
                        (right_x, 0), (right_x, tile_size), 3)
                        
    else:  # Horizontal road (90, 270 degrees)
        # Center the road vertically
        road_y = (tile_size - road_width) // 2
        
        # Yellow center line
        center_y = tile_size // 2
        pygame.draw.line(surface, (255, 255, 0),
                        (0, center_y), (tile_size, center_y), 4)
        
        # White side lines
        top_y = road_y + 10
        bottom_y = road_y + road_width - 10
        pygame.draw.line(surface, (255, 255, 255),
                        (0, top_y), (tile_size, top_y), 3)
        pygame.draw.line(surface, (255, 255, 255),
                        (0, bottom_y), (tile_size, bottom_y), 3)


def create_curved_road(surface, tile_size, direction, rotation):
    """Create a curved road section."""
    
    center = tile_size // 2
    radius_outer = tile_size // 2 - 10
    radius_inner = tile_size // 4
    
    # Adjust curve based on direction and rotation
    start_angle = 0
    end_angle = 90
    
    if direction == 'left':
        if rotation == 0:    # North to West
            start_angle, end_angle = 180, 270
        elif rotation == 90: # East to North  
            start_angle, end_angle = 90, 180
        elif rotation == 180: # South to East
            start_angle, end_angle = 0, 90
        elif rotation == 270: # West to South
            start_angle, end_angle = 270, 360
    else:  # right curve
        if rotation == 0:    # North to East
            start_angle, end_angle = 270, 360
        elif rotation == 90: # East to South
            start_angle, end_angle = 180, 270
        elif rotation == 180: # South to West
            start_angle, end_angle = 90, 180
        elif rotation == 270: # West to North
            start_angle, end_angle = 0, 90
    
    # Draw curved road sections (simplified for now)
    # This creates a basic curved appearance
    points = []
    for angle in range(start_angle, end_angle + 1, 5):
        x = center + radius_outer * np.cos(np.radians(angle))
        y = center + radius_outer * np.sin(np.radians(angle))
        points.append((x, y))
    
    if len(points) > 2:
        pygame.draw.lines(surface, (255, 255, 255), False, points, 3)


def create_intersection_road(surface, tile_size, rotation):
    """Create an intersection road."""
    
    center = tile_size // 2
    road_width = tile_size - 40
    
    # Vertical road
    road_x = (tile_size - road_width) // 2
    pygame.draw.rect(surface, (70, 70, 70), 
                    (road_x, 0, road_width, tile_size))
    
    # Horizontal road  
    road_y = (tile_size - road_width) // 2
    pygame.draw.rect(surface, (70, 70, 70),
                    (0, road_y, tile_size, road_width))
    
    # Center marking
    pygame.draw.circle(surface, (255, 255, 0), (center, center), 8)
    
    # Corner markings based on rotation
    corner_offset = 30
    corners = [
        (corner_offset, corner_offset),
        (tile_size - corner_offset, corner_offset),
        (tile_size - corner_offset, tile_size - corner_offset),
        (corner_offset, tile_size - corner_offset)
    ]
    
    for corner in corners:
        pygame.draw.circle(surface, (255, 255, 255), corner, 6)


def update_tile_mapping():
    """Update tile mapping configuration to include rotated tiles."""
    
    tiles_dir = Path("duckietown_simulator/assets/tiles")
    config_file = tiles_dir / "tile_mapping.ini"
    
    # Enhanced tile mapping content
    mapping_content = """# Tile Type to Image Mapping
# This file maps tile types (integers) to image filenames
# Numbers 0-9 are reserved for basic tiles
# Numbers 10+ are used for directional/rotated tiles

[tile_images]
0 = empty.png
1 = obstacle.png
2 = road.png
3 = grass.png
4 = intersection.png

# Directional road tiles (10-19: Straight roads)
[straight_roads]
10 = road_straight_0.png     # North-South (vertical)
11 = road_straight_90.png    # East-West (horizontal)
12 = road_straight_180.png   # North-South (vertical, same as 10)
13 = road_straight_270.png   # East-West (horizontal, same as 11)

# Curved road tiles (20-29: Curves)
[curved_roads]
20 = road_curve_left_0.png   # North to West
21 = road_curve_left_90.png  # East to North
22 = road_curve_left_180.png # South to East
23 = road_curve_left_270.png # West to South
24 = road_curve_right_0.png  # North to East
25 = road_curve_right_90.png # East to South
26 = road_curve_right_180.png # South to West
27 = road_curve_right_270.png # West to North

# Intersection tiles (30-39: Intersections)
[intersections]
30 = road_intersection_0.png
31 = road_intersection_90.png
32 = road_intersection_180.png
33 = road_intersection_270.png

# Legacy rotated tiles (for backward compatibility)
[legacy_rotated]
road_0 = road_0.png
road_90 = road_90.png
road_180 = road_180.png
road_270 = road_270.png
straight_0 = straight_0.png
straight_90 = straight_90.png
straight_180 = straight_180.png
straight_270 = straight_270.png

# Image settings
[settings]
tile_size = 256
enable_caching = true
smooth_scaling = true
support_rotation = true
"""
    
    with open(config_file, 'w') as f:
        f.write(mapping_content)
    
    print(f"Updated tile mapping configuration: {config_file}")


if __name__ == "__main__":
    print("Creating Rotated Road Tiles")
    print("=" * 40)
    
    # Create rotated versions of existing tiles
    create_rotated_tiles()
    
    # Create enhanced directional road tiles
    create_enhanced_road_tiles()
    
    # Update the tile mapping configuration
    update_tile_mapping()
    
    print("\n" + "=" * 40)
    print("Rotated tile creation completed!")
    print("\nCreated tile types:")
    print("Basic tiles: 0-4 (empty, obstacle, road, grass, intersection)")
    print("Straight roads: 10-13 (vertical/horizontal)")
    print("Left curves: 20-23 (four rotations)")
    print("Right curves: 24-27 (four rotations)")
    print("Intersections: 30-33 (four rotations)")
    print("\nYou can now use these numbers in your map layouts for directional roads!")
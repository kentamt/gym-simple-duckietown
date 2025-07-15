#!/usr/bin/env python3
"""
Script to create dummy tile images for the Duckietown simulator.
These are placeholder images that can be replaced with actual photos.
"""

import pygame
import numpy as np
import os


def create_dummy_tile_images():
    """Create dummy tile images for different tile types."""
    
    # Initialize pygame for image creation
    pygame.init()
    
    # Tile size in pixels (should be square)
    tile_size = 256  # High resolution for good scaling
    
    # Create assets directory
    assets_dir = "duckietown_simulator/assets/tiles"
    os.makedirs(assets_dir, exist_ok=True)
    
    print(f"Creating dummy tile images ({tile_size}x{tile_size} pixels)...")
    
    # Define colors for different tile types
    colors = {
        'empty': (240, 240, 240),      # Light gray
        'road': (50, 50, 50),          # Dark gray (asphalt)
        'obstacle': (139, 69, 19),     # Brown (wood/barrier)
        'grass': (34, 139, 34),        # Forest green
        'intersection': (100, 100, 100), # Medium gray
        'turn_left': (70, 70, 70),     # Darker gray with markings
        'turn_right': (70, 70, 70),    # Darker gray with markings
        'straight': (60, 60, 60),      # Road with lane markings
    }
    
    # Create each tile type
    for tile_type, base_color in colors.items():
        print(f"  Creating {tile_type}.png...")
        
        # Create surface
        surface = pygame.Surface((tile_size, tile_size))
        surface.fill(base_color)
        
        # Add specific details for each tile type
        if tile_type == 'empty':
            # Add subtle texture
            for i in range(0, tile_size, 20):
                pygame.draw.line(surface, (230, 230, 230), (i, 0), (i, tile_size), 1)
                pygame.draw.line(surface, (230, 230, 230), (0, i), (tile_size, i), 1)
        
        elif tile_type == 'road' or tile_type == 'straight':
            # Add lane markings
            center = tile_size // 2
            # Yellow center line
            pygame.draw.line(surface, (255, 255, 0), (center, 0), (center, tile_size), 4)
            # White side lines
            pygame.draw.line(surface, (255, 255, 255), (20, 0), (20, tile_size), 3)
            pygame.draw.line(surface, (255, 255, 255), (tile_size-20, 0), (tile_size-20, tile_size), 3)
        
        elif tile_type == 'obstacle':
            # Add wood grain pattern
            for i in range(5):
                y = i * tile_size // 5 + tile_size // 10
                pygame.draw.line(surface, (160, 82, 45), (0, y), (tile_size, y), 8)
                pygame.draw.line(surface, (101, 67, 33), (0, y+4), (tile_size, y+4), 2)
        
        elif tile_type == 'grass':
            # Add grass texture
            for i in range(100):
                x = np.random.randint(0, tile_size)
                y = np.random.randint(0, tile_size)
                grass_color = (np.random.randint(20, 50), np.random.randint(120, 160), np.random.randint(20, 50))
                pygame.draw.circle(surface, grass_color, (x, y), 2)
        
        elif tile_type == 'intersection':
            # Add intersection markings
            center = tile_size // 2
            # Cross pattern
            pygame.draw.line(surface, (255, 255, 255), (center, 0), (center, tile_size), 3)
            pygame.draw.line(surface, (255, 255, 255), (0, center), (tile_size, center), 3)
            # Corner markings
            for corner in [(30, 30), (tile_size-30, 30), (tile_size-30, tile_size-30), (30, tile_size-30)]:
                pygame.draw.circle(surface, (255, 255, 0), corner, 8)
        
        elif tile_type == 'turn_left':
            # Add left turn arrow
            center = tile_size // 2
            # Road base
            pygame.draw.line(surface, (255, 255, 255), (20, 0), (20, tile_size), 3)
            pygame.draw.line(surface, (255, 255, 255), (tile_size-20, 0), (tile_size-20, tile_size), 3)
            # Arrow pointing left
            arrow_points = [
                (center + 40, center - 20),
                (center - 40, center),
                (center + 40, center + 20),
                (center + 20, center + 20),
                (center + 20, center + 10),
                (center - 10, center + 10),
                (center - 10, center - 10),
                (center + 20, center - 10),
                (center + 20, center - 20)
            ]
            pygame.draw.polygon(surface, (255, 255, 0), arrow_points)
        
        elif tile_type == 'turn_right':
            # Add right turn arrow
            center = tile_size // 2
            # Road base
            pygame.draw.line(surface, (255, 255, 255), (20, 0), (20, tile_size), 3)
            pygame.draw.line(surface, (255, 255, 255), (tile_size-20, 0), (tile_size-20, tile_size), 3)
            # Arrow pointing right
            arrow_points = [
                (center - 40, center - 20),
                (center + 40, center),
                (center - 40, center + 20),
                (center - 20, center + 20),
                (center - 20, center + 10),
                (center + 10, center + 10),
                (center + 10, center - 10),
                (center - 20, center - 10),
                (center - 20, center - 20)
            ]
            pygame.draw.polygon(surface, (255, 255, 0), arrow_points)
        
        # Add border for all tiles
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, tile_size, tile_size), 2)
        
        # Save the image
        filename = os.path.join(assets_dir, f"{tile_type}.png")
        pygame.image.save(surface, filename)
    
    # Create a tile mapping file
    create_tile_mapping_file(assets_dir)
    
    print(f"Created {len(colors)} dummy tile images in {assets_dir}/")
    print("You can replace these with actual Duckietown photos!")
    
    pygame.quit()


def create_tile_mapping_file(assets_dir):
    """Create a mapping file that defines which image to use for each tile type."""
    
    mapping_content = """# Tile Type to Image Mapping
# This file maps tile types (integers) to image filenames
# You can modify this to use different images for different tile types

# Format: tile_type_id = image_filename
# Tile types:
# 0 = Empty/Floor
# 1 = Obstacle/Wall
# 2 = Road/Drivable

[tile_images]
0 = empty.png
1 = obstacle.png
2 = road.png

# Additional tile types for future use
[extended_tiles]
grass = grass.png
intersection = intersection.png
turn_left = turn_left.png
turn_right = turn_right.png
straight = straight.png

# Image settings
[settings]
tile_size = 256
enable_caching = true
smooth_scaling = true
"""
    
    config_file = os.path.join(assets_dir, "tile_mapping.ini")
    with open(config_file, 'w') as f:
        f.write(mapping_content)
    
    print(f"Created tile mapping configuration: {config_file}")


def create_robot_dummy_image():
    """Create a dummy robot image as well."""
    
    assets_dir = "duckietown_simulator/assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Robot image size
    robot_width = 64
    robot_height = 32
    
    print(f"Creating dummy robot image ({robot_width}x{robot_height} pixels)...")
    
    # Create robot surface
    surface = pygame.Surface((robot_width, robot_height), pygame.SRCALPHA)
    
    # Robot body (yellow like classic Duckiebot)
    pygame.draw.rect(surface, (255, 255, 0), (0, 0, robot_width, robot_height))
    pygame.draw.rect(surface, (0, 0, 0), (0, 0, robot_width, robot_height), 2)
    
    # Front indicator (red)
    pygame.draw.rect(surface, (255, 0, 0), (robot_width-8, 4, 6, robot_height-8))
    
    # Wheels (black circles)
    wheel_radius = 6
    pygame.draw.circle(surface, (0, 0, 0), (wheel_radius, wheel_radius), wheel_radius)
    pygame.draw.circle(surface, (0, 0, 0), (wheel_radius, robot_height-wheel_radius), wheel_radius)
    pygame.draw.circle(surface, (0, 0, 0), (robot_width-wheel_radius, wheel_radius), wheel_radius)
    pygame.draw.circle(surface, (0, 0, 0), (robot_width-wheel_radius, robot_height-wheel_radius), wheel_radius)
    
    # Center dot
    pygame.draw.circle(surface, (0, 0, 0), (robot_width//2, robot_height//2), 3)
    
    # Save robot image
    filename = os.path.join(assets_dir, "robot.png")
    pygame.image.save(surface, filename)
    
    print(f"Created dummy robot image: {filename}")


if __name__ == "__main__":
    print("Creating dummy images for Duckietown simulator...")
    create_dummy_tile_images()
    create_robot_dummy_image()
    print("\nDummy image creation completed!")
    print("Replace the images in duckietown_simulator/assets/ with your own photos!")
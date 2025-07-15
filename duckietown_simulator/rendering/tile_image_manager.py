import pygame
import os
import configparser
from typing import Dict, Optional, Tuple
from pathlib import Path


class TileImageManager:
    """
    Manages loading and caching of tile images for pygame rendering.
    """
    
    def __init__(self, assets_dir: str = None):
        """
        Initialize the tile image manager.
        
        Args:
            assets_dir: Path to assets directory. If None, uses default location.
        """
        if assets_dir is None:
            # Get default assets directory relative to this file
            current_dir = Path(__file__).parent
            self.assets_dir = current_dir.parent / "assets" / "tiles"
        else:
            self.assets_dir = Path(assets_dir)
        
        # Image cache
        self.image_cache: Dict[str, pygame.Surface] = {}
        self.scaled_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self.rotated_cache: Dict[Tuple[str, int, int, int], pygame.Surface] = {}  # (filename, width, height, rotation)
        
        # Configuration
        self.tile_mapping: Dict[int, str] = {}
        self.config = {
            'tile_size': 256,
            'enable_caching': True,
            'smooth_scaling': True
        }
        
        # Set up DT17 tile mappings
        self._setup_dt17_mappings()
    
    def _load_dt17_image(self, base_tile_key: str) -> Optional[pygame.Surface]:
        """Load a DT17 base tile image."""
        filename = self.dt17_base_tiles.get(base_tile_key)
        if not filename:
            return None
            
        if filename in self.image_cache:
            return self.image_cache[filename]
        
        image_path = self.assets_dir / filename
        
        if not image_path.exists():
            print(f"Warning: DT17 tile image not found: {image_path}")
            return None
        
        try:
            image = pygame.image.load(str(image_path))
            
            if self.config['enable_caching']:
                self.image_cache[filename] = image
            
            return image
            
        except pygame.error as e:
            print(f"Error loading DT17 tile image {filename}: {e}")
            return None
    
    def _setup_dt17_mappings(self):
        """Set up DT17 tile mappings using only DT17_tile*.png files with rotation."""
        # Base DT17 tiles - we'll use these as the foundation and rotate as needed
        self.dt17_base_tiles = {
            'empty': 'DT17_tile_empty-texture.png',
            'straight': 'DT17_tile_straight-texture.png',
            'curve_left': 'DT17_tile_curve_left-texture.png',
            'curve_right': 'DT17_tile_curve_right-texture.png',
            'three_way': 'DT17_tile_three_way_center-texture.png',
            'four_way': 'DT17_tile_four_way_center-texture.png'
        }
        
        # Map tile types to base tiles and rotations (in 90-degree increments)
        # Format: tile_type -> (base_tile_key, rotation_degrees)
        self.tile_mapping = {
            0: ('empty', 0),              # Empty tile
            1: ('empty', 0),              # Obstacle (will be handled separately)
            2: ('straight', 0),           # Road - straight horizontal
            3: ('straight', 90),          # Road - straight vertical
            4: ('curve_left', 0),         # Curve - left turn
            5: ('curve_left', 90),        # Curve - rotated left turn
            6: ('curve_left', 180),       # Curve - rotated left turn
            7: ('curve_left', 270),       # Curve - rotated left turn
            8: ('curve_right', 0),        # Curve - right turn
            9: ('curve_right', 90),       # Curve - rotated right turn
            10: ('curve_right', 180),     # Curve - rotated right turn
            11: ('curve_right', 270),     # Curve - rotated right turn
            12: ('three_way', 0),         # Three-way intersection
            13: ('three_way', 90),        # Three-way intersection rotated
            14: ('three_way', 180),       # Three-way intersection rotated
            15: ('three_way', 270),       # Three-way intersection rotated
            16: ('four_way', 0),          # Four-way intersection
        }
    
    
    def get_tile_image(self, tile_type, size: Tuple[int, int] = None) -> Optional[pygame.Surface]:
        """
        Get a tile image for the specified tile type using DT17 tiles with rotation.
        
        Args:
            tile_type: Tile type (int)
            size: (width, height) to scale image to. If None, uses original size.
            
        Returns:
            Pygame Surface with the tile image, or None if not found
        """
        # Get base tile and rotation for this tile type
        tile_config = self.tile_mapping.get(tile_type)
        if tile_config is None:
            print(f"Warning: No DT17 mapping for tile type {tile_type}")
            return self._create_fallback_tile(tile_type, size)
        
        base_tile_key, rotation = tile_config
        
        # Handle obstacle tiles specially
        if tile_type == 1:  # Obstacle
            return self._create_obstacle_tile(size)
        
        # Check rotated and scaled cache first
        if size is not None and self.config['enable_caching']:
            cache_key = (base_tile_key, size[0], size[1], rotation)
            if cache_key in self.rotated_cache:
                return self.rotated_cache[cache_key]
        
        # Load base DT17 image
        base_image = self._load_dt17_image(base_tile_key)
        if base_image is None:
            return self._create_fallback_tile(tile_type, size)
        
        # Scale first if needed
        if size is not None and size != base_image.get_size():
            if self.config['smooth_scaling']:
                scaled_image = pygame.transform.smoothscale(base_image, size)
            else:
                scaled_image = pygame.transform.scale(base_image, size)
        else:
            scaled_image = base_image
        
        # Rotate if needed
        if rotation != 0:
            # Use rotozoom with scale=1.0 for better quality rotation
            rotated_image = pygame.transform.rotozoom(scaled_image, rotation, 1.0)
            # Crop to original size to avoid scaling artifacts
            if rotated_image.get_size() != scaled_image.get_size():
                # Calculate crop area to maintain original size
                orig_w, orig_h = scaled_image.get_size()
                rot_w, rot_h = rotated_image.get_size()
                x_offset = (rot_w - orig_w) // 2
                y_offset = (rot_h - orig_h) // 2
                
                # Create a new surface with the original size
                cropped_image = pygame.Surface((orig_w, orig_h))
                cropped_image.blit(rotated_image, (0, 0), (x_offset, y_offset, orig_w, orig_h))
                rotated_image = cropped_image
        else:
            rotated_image = scaled_image
        
        # Cache the final result
        if size is not None and self.config['enable_caching']:
            cache_key = (base_tile_key, size[0], size[1], rotation)
            self.rotated_cache[cache_key] = rotated_image
        
        return rotated_image
    
    def _create_obstacle_tile(self, size: Tuple[int, int] = None) -> pygame.Surface:
        """
        Create an obstacle tile by darkening the empty tile.
        
        Args:
            size: Size of the obstacle tile
            
        Returns:
            Pygame Surface with obstacle appearance
        """
        if size is None:
            size = (self.config['tile_size'], self.config['tile_size'])
        
        # Start with empty tile
        base_image = self._load_dt17_image('empty')
        if base_image is None:
            return self._create_fallback_tile(1, size)
        
        # Scale if needed
        if size != base_image.get_size():
            if self.config['smooth_scaling']:
                scaled_image = pygame.transform.smoothscale(base_image, size)
            else:
                scaled_image = pygame.transform.scale(base_image, size)
        else:
            scaled_image = base_image.copy()
        
        # Create dark overlay for obstacle
        overlay = pygame.Surface(size)
        overlay.fill((60, 60, 60))  # Dark gray
        overlay.set_alpha(180)  # Semi-transparent
        
        # Blit overlay onto the empty tile
        scaled_image.blit(overlay, (0, 0))
        
        return scaled_image
    
    def _create_fallback_tile(self, tile_type: int, size: Tuple[int, int] = None) -> pygame.Surface:
        """
        Create a fallback tile image when the actual image is not available.
        
        Args:
            tile_type: Tile type to create fallback for
            size: Size of the fallback tile
            
        Returns:
            Pygame Surface with a simple colored rectangle
        """
        if size is None:
            size = (self.config['tile_size'], self.config['tile_size'])
        
        surface = pygame.Surface(size)
        
        # Choose color based on tile type
        if tile_type == 0:  # Empty
            color = (240, 240, 240)
        elif tile_type == 1:  # Obstacle
            color = (139, 69, 19)
        elif tile_type == 2:  # Road
            color = (50, 50, 50)
        else:
            color = (128, 128, 128)  # Unknown type
        
        surface.fill(color)
        
        # Add border
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, size[0], size[1]), 2)
        
        # Add tile type number in center
        if size[0] > 30:  # Only if tile is large enough
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                font = pygame.font.Font(None, min(36, size[0] // 4))
                text = font.render(str(tile_type), True, (255, 255, 255))
                text_rect = text.get_rect(center=(size[0] // 2, size[1] // 2))
                surface.blit(text, text_rect)
            except pygame.error:
                # If font fails, just skip the text
                pass
        
        return surface
    
    def preload_tiles(self, tile_types: list = None):
        """
        Preload DT17 tile images into cache.
        
        Args:
            tile_types: List of tile types to preload. If None, loads all mapped types.
        """
        if tile_types is None:
            tile_types = list(self.tile_mapping.keys())
        
        print(f"Preloading {len(tile_types)} DT17 tile images...")
        
        # First preload all base DT17 images
        for base_tile_key in self.dt17_base_tiles.keys():
            self._load_dt17_image(base_tile_key)
        
        print(f"Preloaded {len(self.image_cache)} DT17 base images")
    
    def clear_cache(self):
        """Clear all cached images."""
        self.image_cache.clear()
        self.scaled_cache.clear()
        self.rotated_cache.clear()
        print("Tile image cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cache usage."""
        return {
            'base_images': len(self.image_cache),
            'scaled_images': len(self.scaled_cache),
            'rotated_images': len(self.rotated_cache),
            'total_cached': len(self.image_cache) + len(self.scaled_cache) + len(self.rotated_cache)
        }
    
    def add_custom_mapping(self, tile_type: int, base_tile_key: str, rotation: int = 0):
        """
        Add a custom tile type to DT17 base tile mapping.
        
        Args:
            tile_type: Integer tile type
            base_tile_key: Base DT17 tile key (e.g., 'straight', 'curve_left')
            rotation: Rotation in degrees (0, 90, 180, 270)
        """
        if base_tile_key not in self.dt17_base_tiles:
            print(f"Warning: Unknown base tile key '{base_tile_key}'. Available: {list(self.dt17_base_tiles.keys())}")
            return
            
        if rotation not in [0, 90, 180, 270]:
            print(f"Warning: Invalid rotation {rotation}. Must be 0, 90, 180, or 270")
            return
            
        self.tile_mapping[tile_type] = (base_tile_key, rotation)
        print(f"Added custom mapping: tile type {tile_type} -> {base_tile_key} rotated {rotation}°")
    
    def list_available_images(self) -> list:
        """List all available tile image files."""
        if not self.assets_dir.exists():
            return []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        images = []
        
        for file_path in self.assets_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path.name)
        
        return sorted(images)
    
    def get_tile_mappings(self) -> Dict[int, tuple]:
        """Get current tile type to (base_tile_key, rotation) mappings."""
        return self.tile_mapping.copy()
    
    def get_available_base_tiles(self) -> Dict[str, str]:
        """Get available DT17 base tiles."""
        return self.dt17_base_tiles.copy()


def create_tile_manager(assets_dir: str = None) -> TileImageManager:
    """
    Factory function to create a tile image manager.
    
    Args:
        assets_dir: Path to assets directory
        
    Returns:
        TileImageManager instance
    """
    return TileImageManager(assets_dir)
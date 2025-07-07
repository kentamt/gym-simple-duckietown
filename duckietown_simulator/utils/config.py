import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """
    Configuration manager for loading and validating map and environment configs.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Get the directory containing this file
            current_dir = Path(__file__).parent
            # Navigate to the configs directory
            self.config_dir = current_dir.parent / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self.tracks_dir = self.config_dir / "tracks"
        self.robots_dir = self.config_dir / "robots"
        self.environments_dir = self.config_dir / "environments"
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create configuration directories if they don't exist."""
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.robots_dir.mkdir(parents=True, exist_ok=True)
        self.environments_dir.mkdir(parents=True, exist_ok=True)
    
    def load_track_config(self, track_name: str) -> Dict[str, Any]:
        """
        Load a track configuration file.
        
        Args:
            track_name: Name of the track configuration
            
        Returns:
            Dictionary containing track configuration
            
        Raises:
            ConfigError: If config file is not found or invalid
        """
        config_path = self.tracks_dir / f"{track_name}.json"
        
        if not config_path.exists():
            raise ConfigError(f"Track config not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            self._validate_track_config(config)
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in track config {track_name}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading track config {track_name}: {e}")
    
    def _validate_track_config(self, config: Dict[str, Any]):
        """Validate track configuration structure."""
        required_fields = ["name", "map_config", "track_type"]
        
        for field in required_fields:
            if field not in config:
                raise ConfigError(f"Missing required field in track config: {field}")
        
        # Validate map_config
        map_config = config["map_config"]
        required_map_fields = ["width", "height"]
        
        for field in required_map_fields:
            if field not in map_config:
                raise ConfigError(f"Missing required field in map_config: {field}")
        
        # Validate data types and ranges
        if not isinstance(map_config["width"], int) or map_config["width"] <= 0:
            raise ConfigError("map_config.width must be a positive integer")
        
        if not isinstance(map_config["height"], int) or map_config["height"] <= 0:
            raise ConfigError("map_config.height must be a positive integer")
        
        if "tile_size" in map_config:
            if not isinstance(map_config["tile_size"], (int, float)) or map_config["tile_size"] <= 0:
                raise ConfigError("map_config.tile_size must be a positive number")
    
    def save_track_config(self, track_name: str, config: Dict[str, Any]):
        """
        Save a track configuration file.
        
        Args:
            track_name: Name of the track configuration
            config: Configuration dictionary to save
        """
        # Validate before saving
        self._validate_track_config(config)
        
        config_path = self.tracks_dir / f"{track_name}.json"
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            raise ConfigError(f"Error saving track config {track_name}: {e}")
    
    def list_track_configs(self) -> list:
        """List all available track configurations."""
        if not self.tracks_dir.exists():
            return []
        
        configs = []
        for file_path in self.tracks_dir.glob("*.json"):
            configs.append(file_path.stem)
        
        return sorted(configs)
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist."""
        # Create default straight track configs
        straight_configs = [
            {
                "name": "straight_3x3",
                "description": "Minimal straight track on 3x3 tile grid",
                "map_config": {"width": 3, "height": 3, "tile_size": 0.61},
                "track_type": "straight"
            },
            {
                "name": "straight_5x5",
                "description": "Standard straight track on 5x5 tile grid",
                "map_config": {"width": 5, "height": 5, "tile_size": 0.61},
                "track_type": "straight"
            },
            {
                "name": "straight_10x5",
                "description": "Long straight track on 10x5 tile grid",
                "map_config": {"width": 10, "height": 5, "tile_size": 0.61},
                "track_type": "straight"
            }
        ]
        
        # Create default loop track configs
        loop_configs = [
            {
                "name": "loop_5x5",
                "description": "Small loop track on 5x5 tile grid",
                "map_config": {"width": 5, "height": 5, "tile_size": 0.61},
                "track_type": "loop"
            },
            {
                "name": "loop_6x5",
                "description": "Medium loop track on 6x5 tile grid",
                "map_config": {"width": 6, "height": 5, "tile_size": 0.61},
                "track_type": "loop"
            },
            {
                "name": "loop_8x8",
                "description": "Large loop track on 8x8 tile grid",
                "map_config": {"width": 8, "height": 8, "tile_size": 0.61},
                "track_type": "loop"
            }
        ]
        
        # Save configs if they don't exist
        all_configs = straight_configs + loop_configs
        for config in all_configs:
            config_path = self.tracks_dir / f"{config['name']}.json"
            if not config_path.exists():
                self.save_track_config(config["name"], config)
    
    def get_map_dimensions(self, track_name: str) -> tuple:
        """
        Get the map dimensions for a track configuration.
        
        Args:
            track_name: Name of the track configuration
            
        Returns:
            Tuple of (width, height) in tiles
        """
        config = self.load_track_config(track_name)
        map_config = config["map_config"]
        return (map_config["width"], map_config["height"])
    
    def get_map_size_meters(self, track_name: str) -> tuple:
        """
        Get the map size in meters for a track configuration.
        
        Args:
            track_name: Name of the track configuration
            
        Returns:
            Tuple of (width_meters, height_meters)
        """
        config = self.load_track_config(track_name)
        map_config = config["map_config"]
        
        tile_size = map_config.get("tile_size", 0.61)
        width_meters = map_config["width"] * tile_size
        height_meters = map_config["height"] * tile_size
        
        return (width_meters, height_meters)
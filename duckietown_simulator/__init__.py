"""
Duckietown Simulator

A Python-based simulator for Duckietown environments with OpenAI Gym interface.
Features differential drive robot dynamics, tile-based maps, collision detection,
and various reward functions for RL training.
"""

from .environment import DuckietownEnv, make_env
from .robot.duckiebot import Duckiebot, RobotConfig, create_duckiebot
from .world.map import Map, MapConfig, create_map_from_array, create_map_from_file

__version__ = "0.1.0"
__author__ = "Claude Code"

__all__ = [
    "DuckietownEnv",
    "make_env", 
    "Duckiebot",
    "RobotConfig",
    "create_duckiebot",
    "Map",
    "MapConfig", 
    "create_map_from_array",
    "create_map_from_file",
]
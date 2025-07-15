"""
Duckietown Simulator Environment Module

This module provides OpenAI Gym compatible environments for the Duckietown simulator.
"""

from .duckietown_env import DuckietownEnv, make_env
from .reward_functions import (
    lane_following_reward,
    exploration_reward,
    racing_reward,
    navigation_reward,
    sparse_reward,
    shaped_reward,
    get_reward_function,
    REWARD_FUNCTIONS,
)

__all__ = [
    "DuckietownEnv",
    "make_env",
    "lane_following_reward",
    "exploration_reward", 
    "racing_reward",
    "navigation_reward",
    "sparse_reward",
    "shaped_reward",
    "get_reward_function",
    "REWARD_FUNCTIONS",
]
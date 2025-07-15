"""
Gymnasium environment registration for Duckietown simulator.

This module registers the Duckietown environments with Gymnasium
so they can be created using gym.make().
"""

from gymnasium.envs.registration import register
from .environment.duckietown_env import DuckietownEnv
from .environment.reward_functions import get_reward_function


# Register basic environments
register(
    id='Duckietown-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 5, 'height': 5, 'track_type': 'straight'},
        'max_steps': 500,
    },
    max_episode_steps=500,
)

register(
    id='DuckietownLoop-v0', 
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 6, 'height': 5, 'track_type': 'loop'},
        'max_steps': 1000,
    },
    max_episode_steps=1000,
)

register(
    id='DuckietownSmall-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv', 
    kwargs={
        'map_config': {'width': 3, 'height': 3, 'track_type': 'straight'},
        'max_steps': 300,
    },
    max_episode_steps=300,
)

register(
    id='DuckietownLarge-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 8, 'height': 8, 'track_type': 'loop'},
        'max_steps': 2000,
    },
    max_episode_steps=2000,
)

# Register task-specific environments
register(
    id='DuckietownLaneFollowing-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 5, 'height': 5, 'track_type': 'straight'},
        'reward_function': get_reward_function('lane_following'),
        'max_steps': 500,
    },
    max_episode_steps=500,
)

register(
    id='DuckietownExploration-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 6, 'height': 6, 'track_type': 'loop'},
        'reward_function': get_reward_function('exploration'),
        'max_steps': 1000,
    },
    max_episode_steps=1000,
)

register(
    id='DuckietownRacing-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 8, 'height': 6, 'track_type': 'loop'},
        'reward_function': get_reward_function('racing'),
        'max_steps': 1500,
    },
    max_episode_steps=1500,
)

register(
    id='DuckietownNavigation-v0',
    entry_point='duckietown_simulator.environment:DuckietownEnv',
    kwargs={
        'map_config': {'width': 5, 'height': 5, 'track_type': 'straight'},
        'reward_function': get_reward_function('navigation', target_position=(2.5, 4.0)),
        'max_steps': 800,
    },
    max_episode_steps=800,
)


def list_environments():
    """List all available Duckietown environments."""
    envs = [
        'Duckietown-v0',
        'DuckietownLoop-v0',
        'DuckietownSmall-v0', 
        'DuckietownLarge-v0',
        'DuckietownLaneFollowing-v0',
        'DuckietownExploration-v0',
        'DuckietownRacing-v0',
        'DuckietownNavigation-v0',
    ]
    return envs


def create_custom_env(
    map_config=None,
    reward_function=None,
    max_steps=500,
    render_mode=None,
    **kwargs
):
    """
    Create a custom Duckietown environment.
    
    Args:
        map_config: Map configuration
        reward_function: Reward function or name
        max_steps: Maximum steps per episode
        render_mode: Rendering mode
        **kwargs: Additional environment arguments
        
    Returns:
        DuckietownEnv instance
    """
    if isinstance(reward_function, str):
        reward_function = get_reward_function(reward_function)
    
    return DuckietownEnv(
        map_config=map_config,
        reward_function=reward_function,
        max_steps=max_steps,
        render_mode=render_mode,
        **kwargs
    )
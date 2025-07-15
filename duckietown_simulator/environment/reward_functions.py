import numpy as np
from typing import Dict, Any


def lane_following_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    """
    Reward function for lane following task.
    
    Encourages the robot to:
    - Move forward along the road
    - Stay in the center of lanes
    - Avoid collisions
    - Maintain stable motion
    
    Args:
        state: Previous state dictionary
        action: Action taken [left_wheel_vel, right_wheel_vel]
        next_state: New state dictionary
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Forward movement reward
    distance_moved = next_state.get('distance_step', 0.0)
    linear_velocity = next_state.get('linear_velocity', 0.0)
    
    # Reward forward motion
    if linear_velocity > 0:
        reward += distance_moved * 100.0  # Strong reward for forward movement
    else:
        reward -= 10.0  # Penalty for going backwards
    
    # Collision penalty
    if next_state.get('collision', False):
        reward -= 1000.0  # Large penalty for collision
    
    # Stability rewards
    angular_velocity = abs(next_state.get('angular_velocity', 0.0))
    
    # Penalty for excessive turning (promotes stability)
    if angular_velocity > 1.0:  # rad/s
        reward -= angular_velocity * 10.0
    
    # Action smoothness reward (penalize jerky motions)
    left_wheel, right_wheel = action
    wheel_diff = abs(left_wheel - right_wheel)
    if wheel_diff > 2.0:  # Large difference indicates sharp turn
        reward -= wheel_diff * 5.0
    
    # Speed reward (encourage moderate speeds)
    target_speed = 0.3  # m/s
    speed_error = abs(linear_velocity - target_speed)
    if speed_error < 0.1:
        reward += 10.0  # Bonus for target speed
    else:
        reward -= speed_error * 20.0
    
    # Small time penalty to encourage efficiency
    reward -= 1.0
    
    return reward


def exploration_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    """
    Reward function for exploration task.
    
    Encourages the robot to:
    - Explore new areas
    - Avoid collisions
    - Maintain motion
    
    Args:
        state: Previous state dictionary
        action: Action taken
        next_state: New state dictionary
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Movement reward
    distance_moved = next_state.get('distance_step', 0.0)
    reward += distance_moved * 50.0
    
    # Collision penalty
    if next_state.get('collision', False):
        reward -= 500.0
    
    # Encourage any motion (exploration)
    linear_velocity = abs(next_state.get('linear_velocity', 0.0))
    angular_velocity = abs(next_state.get('angular_velocity', 0.0))
    
    if linear_velocity > 0.1 or angular_velocity > 0.1:
        reward += 5.0  # Bonus for movement
    
    # Small time penalty
    reward -= 0.5
    
    return reward


def racing_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    """
    Reward function for racing task.
    
    Encourages the robot to:
    - Move as fast as possible
    - Avoid collisions
    - Complete laps quickly
    
    Args:
        state: Previous state dictionary
        action: Action taken
        next_state: New state dictionary
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Speed reward (prioritize high speed)
    linear_velocity = next_state.get('linear_velocity', 0.0)
    if linear_velocity > 0:
        reward += linear_velocity * 200.0  # High reward for speed
    else:
        reward -= 50.0  # Strong penalty for not moving forward
    
    # Distance reward
    distance_moved = next_state.get('distance_step', 0.0)
    reward += distance_moved * 150.0
    
    # Collision penalty (race ending)
    if next_state.get('collision', False):
        reward -= 2000.0  # Very high penalty
    
    # Minimize turning for racing (encourage straight line speed)
    angular_velocity = abs(next_state.get('angular_velocity', 0.0))
    reward -= angular_velocity * 20.0
    
    # Efficiency penalty
    reward -= 2.0
    
    return reward


def navigation_reward(
    state: Dict, 
    action: np.ndarray, 
    next_state: Dict,
    target_position: tuple = (0.0, 0.0)
) -> float:
    """
    Reward function for navigation to target task.
    
    Encourages the robot to:
    - Navigate towards target position
    - Avoid collisions
    - Take efficient paths
    
    Args:
        state: Previous state dictionary
        action: Action taken
        next_state: New state dictionary
        target_position: Target (x, y) position
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Distance to target
    current_x = next_state.get('x', 0.0)
    current_y = next_state.get('y', 0.0)
    prev_x = state.get('x', 0.0)
    prev_y = state.get('y', 0.0)
    
    target_x, target_y = target_position
    
    # Previous distance to target
    prev_distance = np.sqrt((prev_x - target_x)**2 + (prev_y - target_y)**2)
    
    # Current distance to target
    current_distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
    
    # Reward for getting closer to target
    distance_improvement = prev_distance - current_distance
    reward += distance_improvement * 100.0
    
    # Bonus for reaching target
    if current_distance < 0.2:  # Within 20cm
        reward += 500.0
        if current_distance < 0.1:  # Within 10cm
            reward += 1000.0
    
    # Movement reward
    distance_moved = next_state.get('distance_step', 0.0)
    reward += distance_moved * 10.0
    
    # Collision penalty
    if next_state.get('collision', False):
        reward -= 800.0
    
    # Encourage forward motion
    linear_velocity = next_state.get('linear_velocity', 0.0)
    if linear_velocity < 0:
        reward -= 20.0
    
    # Time penalty
    reward -= 1.0
    
    return reward


def custom_reward_wrapper(base_reward_func: callable, **kwargs):
    """
    Wrapper to create custom reward functions with parameters.
    
    Args:
        base_reward_func: Base reward function
        **kwargs: Additional parameters for the reward function
        
    Returns:
        Wrapped reward function
    """
    def wrapped_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
        return base_reward_func(state, action, next_state, **kwargs)
    
    return wrapped_reward


def sparse_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    """
    Sparse reward function that only gives reward for task completion.
    
    Args:
        state: Previous state dictionary
        action: Action taken
        next_state: New state dictionary
        
    Returns:
        Reward value (0 or large positive/negative)
    """
    # Only collision penalty and success bonus
    if next_state.get('collision', False):
        return -100.0
    
    # Could add task-specific success conditions here
    # For now, small movement reward
    distance_moved = next_state.get('distance_step', 0.0)
    if distance_moved > 0.01:  # Minimum movement threshold
        return 1.0
    
    return 0.0


def shaped_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    """
    Dense shaped reward function for easier learning.
    
    Args:
        state: Previous state dictionary  
        action: Action taken
        next_state: New state dictionary
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Basic movement reward
    distance_moved = next_state.get('distance_step', 0.0)
    reward += distance_moved * 20.0
    
    # Velocity rewards
    linear_velocity = next_state.get('linear_velocity', 0.0)
    angular_velocity = abs(next_state.get('angular_velocity', 0.0))
    
    # Reward forward motion
    if linear_velocity > 0:
        reward += linear_velocity * 10.0
    
    # Penalty for excessive turning
    reward -= angular_velocity * 2.0
    
    # Collision penalty
    if next_state.get('collision', False):
        reward -= 200.0
    
    # Action penalties for efficiency
    left_wheel, right_wheel = action
    total_effort = abs(left_wheel) + abs(right_wheel)
    reward -= total_effort * 0.1
    
    # Small time step penalty
    reward -= 0.2
    
    return reward


# Dictionary of available reward functions
REWARD_FUNCTIONS = {
    "lane_following": lane_following_reward,
    "exploration": exploration_reward,
    "racing": racing_reward,
    "navigation": navigation_reward,
    "sparse": sparse_reward,
    "shaped": shaped_reward,
}


def get_reward_function(name: str, **kwargs):
    """
    Get a reward function by name.
    
    Args:
        name: Name of the reward function
        **kwargs: Additional parameters for the reward function
        
    Returns:
        Reward function
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(REWARD_FUNCTIONS.keys())}")
    
    base_func = REWARD_FUNCTIONS[name]
    
    if kwargs:
        return custom_reward_wrapper(base_func, **kwargs)
    else:
        return base_func
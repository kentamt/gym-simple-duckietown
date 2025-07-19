# Duckietown Gym Interface

This document explains how to use the OpenAI Gym interface for the Duckietown simulator.

## Overview

The Duckietown Gym interface provides a standard RL environment that can be used with any RL framework. The environment simulates a Duckiebot robot with differential drive kinematics on a tile-based map.

## Quick Start

```python
from duckietown_simulator import DuckietownEnv, make_env

# Create basic environment
env = DuckietownEnv()

# Or use factory function with predefined maps
env = make_env("default")  # 5x5 straight track
env = make_env("loop")     # 6x5 loop track
env = make_env("small")    # 3x3 small track
env = make_env("large")    # 8x8 large loop

# Standard gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Action and Observation Spaces

### Action Space
- **Type**: Box(2,) with range [-max_wheel_speed, max_wheel_speed]
- **Format**: `[left_wheel_velocity, right_wheel_velocity]` (rad/s)
- **Default range**: [-10.0, 10.0] rad/s

### Observation Space  
- **Type**: Box(7,) 
- **Format**: `[x, y, theta, linear_vel, angular_vel, left_wheel_vel, right_wheel_vel]`
- **Units**: 
  - x, y: position in meters
  - theta: orientation in radians [-π, π]
  - linear_vel: forward velocity in m/s
  - angular_vel: angular velocity in rad/s
  - wheel velocities: rad/s

## Rendering

The environment supports multiple rendering modes for visualization:

### Human Rendering (`render_mode="human"`)

```python
env = DuckietownEnv(render_mode="human")
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Display current state
    
    if terminated or truncated:
        break

env.close()
```

**Rendering Options:**
1. **Pygame (Recommended)**: Full interactive real-time rendering
   - Real-time visualization with smooth animations
   - Interactive controls (mouse pan/zoom, keyboard shortcuts)
   - Collision detection visualization
   - Robot trajectory tracking
   - **Installation**: `pip install pygame`

2. **Matplotlib**: Static visual plots
   - High-quality plots suitable for analysis
   - Can save images for documentation
   - **Installation**: `pip install matplotlib`

3. **Text Mode**: ASCII art fallback (always available)
   - Works in any terminal
   - Shows map layout and robot position as text
   - No additional dependencies required

### RGB Array Rendering (`render_mode="rgb_array"`)

```python
env = DuckietownEnv(render_mode="rgb_array")
obs, info = env.reset()

rgb_array = env.render()  # Returns numpy array
print(f"Image shape: {rgb_array.shape}")  # (height, width, 3)
```

**Note**: RGB array mode requires pygame. Returns `None` with fallback renderers.

### Rendering Examples

```python
# Demo with text rendering (always works)
env = DuckietownEnv(render_mode="human")
obs, info = env.reset()

for step in range(10):
    action = np.array([2.0, 2.0])  # Move forward
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)  # Pause to see changes

env.close()

# Save matplotlib visualization
from duckietown_simulator.rendering.simple_renderer import MatplotlibRenderer
renderer = MatplotlibRenderer(env.map)
renderer.set_robots({"robot": env.robot})
renderer.render(save_path="duckietown_visualization.png")
```

## Environment Configuration

### Map Configuration

```python
# Custom map from 2D array
custom_layout = [
    [1, 1, 1, 1, 1],  # 1 = wall/obstacle
    [1, 2, 2, 2, 1],  # 2 = road
    [1, 2, 0, 2, 1],  # 0 = empty space
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1],
]

env = DuckietownEnv(
    map_config={"layout": custom_layout, "tile_size": 0.61}
)

# From config dict
env = DuckietownEnv(
    map_config={"width": 5, "height": 5, "track_type": "straight"}
)

# From file
env = DuckietownEnv(map_config="path/to/map.json")
```

### Robot Configuration

```python
from duckietown_simulator.robot.duckiebot import RobotConfig

robot_config = RobotConfig(
    wheelbase=0.102,           # Distance between wheels (m)
    wheel_radius=0.0318,       # Wheel radius (m)  
    max_wheel_speed=15.0,      # Max wheel speed (rad/s)
    collision_radius=0.06,     # Robot collision radius (m)
    initial_x=1.0,             # Starting x position
    initial_y=1.0,             # Starting y position
    initial_theta=0.0          # Starting orientation
)

env = DuckietownEnv(robot_config=robot_config)
```

## Reward Functions

The simulator includes several predefined reward functions:

### Lane Following
```python
from duckietown_simulator.environment.reward_functions import get_reward_function

env = DuckietownEnv(
    reward_function=get_reward_function('lane_following')
)
```

Available reward functions:
- `lane_following`: Rewards forward motion and lane keeping
- `exploration`: Rewards exploration and movement
- `racing`: Rewards high speed and efficient racing
- `navigation`: Rewards moving towards target position
- `sparse`: Minimal sparse rewards
- `shaped`: Dense shaped rewards for easier learning

### Custom Reward Functions

```python
def custom_reward(state, action, next_state):
    reward = 0.0
    
    # Reward forward movement
    distance = next_state.get('distance_step', 0.0)
    reward += distance * 10.0
    
    # Penalty for collision
    if next_state.get('collision', False):
        reward -= 100.0
        
    return reward

env = DuckietownEnv(reward_function=custom_reward)
```

## Environment Information

The `info` dictionary returned by `step()` and `reset()` contains:

```python
info = {
    'step_count': 42,
    'total_distance': 1.23,
    'collision': False,
    'robot_state': {
        'pose': [x, y, theta],
        'x': x_position,
        'y': y_position, 
        'theta': orientation,
        'linear_velocity': forward_speed,
        'angular_velocity': turn_rate,
        'wheel_velocities': [left_wheel, right_wheel],
        'collision_radius': radius,
        'is_collided': collision_state,
        'total_distance': total_distance_traveled,
        'step_count': robot_step_count
    },
    'action': [left_wheel_cmd, right_wheel_cmd],
}
```

## Usage Examples

### Basic RL Training Loop

```python
import numpy as np
from duckietown_simulator import DuckietownEnv

env = DuckietownEnv(
    reward_function=get_reward_function('lane_following'),
    max_steps=500
)

for episode in range(100):
    obs, info = env.reset()
    episode_reward = 0
    
    while True:
        # Your policy here
        action = your_policy(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            print(f"Episode {episode}: reward = {episode_reward}")
            break

env.close()
```

### Navigation Task

```python
env = DuckietownEnv(
    reward_function=get_reward_function('navigation', target_position=(3.0, 3.0)),
    max_steps=200
)

obs, info = env.reset()
target_x, target_y = 3.0, 3.0

for step in range(200):
    # Simple navigation policy
    current_x = info['robot_state']['x']
    current_y = info['robot_state']['y']
    current_theta = info['robot_state']['theta']
    
    # Calculate direction to target
    dx = target_x - current_x
    dy = target_y - current_y
    target_angle = np.arctan2(dy, dx)
    
    # Simple control logic
    angle_diff = target_angle - current_theta
    if abs(angle_diff) > 0.2:
        # Turn towards target
        action = np.array([1.0, -1.0]) if angle_diff > 0 else np.array([-1.0, 1.0])
    else:
        # Move forward
        action = np.array([2.0, 2.0])
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

## Integration with RL Libraries

The environment is compatible with popular RL libraries:

### Stable-Baselines3 (when gymnasium is available)
```python
# Note: Requires gymnasium to be installed
from stable_baselines3 import PPO
from duckietown_simulator import DuckietownEnv

env = DuckietownEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Ray RLlib
```python
# Custom wrapper may be needed for RLlib integration
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure the simulator package is in your Python path
2. **Action Space Error**: Ensure actions are numpy arrays with shape (2,)
3. **Map Loading Error**: Check that map files exist and have correct format
4. **Collision Detection**: Robot may get stuck if starting position is invalid

### Performance Tips

1. Use smaller maps for faster training
2. Limit max_steps to prevent infinite episodes  
3. Use appropriate reward scaling
4. Consider using sparse rewards for initial training

## Files

- `test_gym_interface.py`: Comprehensive test suite
- `example_gym_usage.py`: Usage examples and demonstrations
- `duckietown_simulator/environment/`: Main environment implementation
- `duckietown_simulator/environment/reward_functions.py`: Reward function definitions
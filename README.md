# Duckietown RL Simulator
<img src="https://github.com/user-attachments/assets/604a2b0e-075c-4f9f-9b53-31a30e5fa219" width="500" />

A simplified 2D simulator for training reinforcement learning models on Duckietown navigation tasks. This simulator focuses on waypoint-based navigation and differential drive control, removing the complexity of 3D rendering and computer vision found in existing solutions.

## Features

- **Simplified Physics**: 2D top-down simulation with differential drive kinematics
- **Waypoint Navigation**: Robots detect and follow waypoints instead of lane detection
- **RL-Ready**: OpenAI Gym compatible environment for easy integration with RL frameworks
- **Configurable**: Flexible track layouts, robot parameters, and reward functions
- **Lightweight**: Fast simulation suitable for training at scale

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/duckietown-rl-simulator.git
cd duckietown-rl-simulator

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import gym
from duckietown_simulator import DuckietownEnv

# Create environment
env = DuckietownEnv(
    track_layout="straight",
    max_episode_steps=1000,
    render_mode="human"
)

# Reset environment
obs = env.reset()

# Run simulation
for step in range(1000):
    # Random action: [left_wheel_speed, right_wheel_speed]
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
        
env.close()
```

## Environment Specifications

### Observation Space
- **Waypoints**: Relative positions and orientations of detected waypoints
- **Robot State**: Current velocity, orientation, and position relative to track
- **Proximity**: Distance to boundaries and obstacles
- **Progress**: Metrics for goal-directed navigation

### Action Space
- **Continuous Control**: `[left_wheel_speed, right_wheel_speed]`
- **Range**: `[-max_speed, max_speed]` for each wheel
- **Units**: Radians per second

### Reward Function
- **Progress**: +1.0 for reaching waypoints in sequence
- **Lane Keeping**: -0.1 per step for deviation from centerline
- **Collision**: -10.0 for hitting boundaries or obstacles
- **Efficiency**: Small penalty for excessive speed changes

## Configuration

### Track Layouts
- `straight`: Simple straight track with waypoints
- `curve`: S-curve track for testing turning behavior
- `intersection`: T-junction with multiple path options
- `loop`: Closed circuit for continuous navigation
- `custom`: Load custom track from configuration file

### Robot Parameters
```python
robot_config = {
    "wheelbase": 0.1,           # Distance between wheels (meters)
    "max_wheel_speed": 10.0,    # Maximum wheel speed (rad/s)
    "waypoint_detection_range": 2.0,  # Detection radius (meters)
    "collision_radius": 0.05    # Robot collision radius (meters)
}
```

### Environment Parameters
```python
env_config = {
    "track_layout": "straight",
    "max_episode_steps": 1000,
    "render_mode": "human",     # "human", "rgb_array", or None
    "reward_function": "default",
    "random_start": True,
    "noise_level": 0.0          # Sensor noise (0.0 = perfect)
}
```

## Project Structure

```
duckietown_simulator/
├── environment/
│   ├── __init__.py
│   ├── duckietown_env.py      # Main gym environment
│   └── reward_functions.py    # Reward computation
├── world/
│   ├── __init__.py
│   ├── map.py                 # Track layout and boundaries
│   ├── waypoints.py           # Waypoint network management
│   └── obstacles.py           # Static obstacles
├── robot/
│   ├── __init__.py
│   ├── duckiebot.py           # Robot dynamics and state
│   ├── kinematics.py          # Differential drive model
│   └── sensors.py             # Waypoint detection logic
├── rendering/
│   ├── __init__.py
│   ├── visualizer.py          # 2D visualization
│   └── matplotlib_renderer.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py            # Collision detection, transformations
│   └── config.py              # Configuration management
└── configs/
    ├── tracks/                # Track layout definitions
    ├── robots/                # Robot parameter presets
    └── environments/          # Environment configurations
```

## Training Examples

### Basic PPO Training
```python
from stable_baselines3 import PPO
from duckietown_simulator import DuckietownEnv

env = DuckietownEnv(track_layout="straight")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("duckiebot_navigation")
```

### Custom Training Loop
```python
import numpy as np
from duckietown_simulator import DuckietownEnv

env = DuckietownEnv(track_layout="curve")
obs = env.reset()

for episode in range(1000):
    done = False
    total_reward = 0
    
    while not done:
        # Your RL algorithm here
        action = your_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode}: Reward = {total_reward}")
    obs = env.reset()
```

## Evaluation Metrics

- **Success Rate**: Percentage of episodes reaching the goal
- **Collision Rate**: Percentage of episodes ending in collision
- **Path Efficiency**: Ratio of optimal path length to actual path length
- **Lane Deviation**: Average distance from track centerline
- **Completion Time**: Steps required to reach goal

## Extending the Simulator

### Adding New Track Layouts
1. Create track definition in `configs/tracks/`
2. Define waypoint network and boundaries
3. Register in `world/map.py`

### Custom Reward Functions
1. Implement reward function in `environment/reward_functions.py`
2. Register in environment configuration
3. Use via `reward_function` parameter

### Advanced Robot Models
1. Extend `robot/duckiebot.py` with additional dynamics
2. Modify observation space in `environment/duckietown_env.py`
3. Update kinematics in `robot/kinematics.py`


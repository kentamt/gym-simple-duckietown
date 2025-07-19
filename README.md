# Duckietown Simulator with TSL (Trajectory Specification Language)

A comprehensive simulation environment for autonomous robots in Duckietown, featuring trajectory editing, PID control, collision detection, and both single-agent and multi-agent reinforcement learning environments.

## Features

- **🎯 Interactive Trajectory Editor**: GUI-based waypoint creation and editing using mouse input
- **🤖 PID Controller**: Robust trajectory following with collision detection and speed visualization
- **🎮 Gym Environments**: OpenAI Gym/Gymnasium compatible environments for RL training
- **👥 Multi-Agent Support**: Multi-robot environments with collision avoidance
- **🔧 Discrete Action Space**: Simple {STOP, GO} action space for easy learning
- **📊 Real-time Visualization**: Pygame-based rendering with comprehensive overlays

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gym-tsl-duckie

# Install dependencies
pip install numpy pygame gymnasium pillow
```

### Main Interface

Use the unified main entry point to access all functionality:

```bash
# Launch trajectory editor
python main.py trajectory-editor

# Run PID demo with collision detection
python main.py pid-demo

# Run single-agent gym environment
python main.py single-gym

# Run multi-agent gym environment
python main.py multi-gym

# Run all tests
python main.py test

# List available trajectory files
python main.py list-trajectories
```

## Repository Structure

```
gym-tsl-duckie/
├── main.py                 # Main entry point
├── examples/               # Gym environment examples
│   ├── gym_pid_road_network.py      # Single-agent gym env
│   ├── multi_agent_gym_env.py       # Multi-agent gym env
│   └── example_gym_usage.py         # Basic usage examples
├── demos/                  # Interactive demonstrations
│   ├── demo_pid_road_network.py     # PID controller demo
│   ├── demo_discrete_actions.py     # Discrete action demo
│   └── demo_*.py                    # Other demo scripts
├── tests/                  # Test suite
│   ├── test_multi_agent_env.py      # Multi-agent tests
│   ├── test_gym_env.py              # Single-agent tests
│   ├── test_duckiebot.py            # Robot tests
│   ├── test_collision_detection.py  # Collision tests
│   └── test_map.py                  # Map tests
├── tools/                  # Utility tools
│   ├── trajectory_editor.py         # GUI trajectory editor
│   └── *.py                         # Debug and utility scripts
├── data/                   # Data files
│   ├── trajectory_*.json            # Trajectory files
│   └── *.png                        # Visualization images
├── duckietown_simulator/   # Core simulator package
│   ├── environment/                 # Environment definitions
│   ├── robot/                       # Robot models and controllers
│   ├── rendering/                   # Visualization components
│   ├── world/                       # Map and collision systems
│   └── utils/                       # Utility functions
└── docs/                   # Documentation
```

## Core Components

### 1. Trajectory Editor
Interactive GUI for creating and editing robot trajectories:

```python
# Launch the trajectory editor
python tools/trajectory_editor.py

# Features:
# - Mouse-based waypoint creation
# - Three modes: View, Edit, Delete (TAB to switch)
# - Save/load trajectories to JSON
# - Real-time visualization
```

### 2. Single-Agent Gym Environment
OpenAI Gym environment for single robot training:

```python
from examples.gym_pid_road_network import make_env

env = make_env(trajectory_file="data/trajectory_1.json", render_mode="human")
obs, info = env.reset()

for step in range(1000):
    action = 1 if step % 2 == 0 else 0  # Alternate STOP/GO
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

**Action Space**: `Discrete(2)` - {0: STOP, 1: GO}
**Observation Space**: `[x, y, theta, v_linear, v_angular, dist_to_waypoint, collision, progress]`

### 3. Multi-Agent Gym Environment
Multi-robot environment with coordinated action spaces:

```python
from examples.multi_agent_gym_env import make_multi_agent_env

env = make_multi_agent_env(
    num_agents=3,
    trajectory_files=["data/trajectory_1.json", "data/trajectory_2.json"],
    render_mode="human"
)

obs, infos = env.reset()

for step in range(1000):
    # Actions for each agent
    actions = {
        "robot1": 1,  # GO
        "robot2": 0,  # STOP
        "robot3": 1   # GO
    }
    
    obs, rewards, terminated, truncated, infos = env.step(actions)
    env.render()
    
    if any(terminated.values()) or any(truncated.values()):
        break

env.close()
```

**Action Space**: `Dict` of `Discrete(2)` for each agent
**Observation Space**: Own state + other agents' positions and collision status

### 4. PID Controller Demo
Comprehensive demonstration with collision detection:

```python
# Run the demo
python demos/demo_pid_road_network.py

# Features:
# - Trajectory following with PID control
# - Real-time collision detection
# - Speed and progress visualization
# - Waypoint interpolation
# - Interactive controls
```

## Key Features

### Collision Detection
- **Robot-Robot Collisions**: Multi-agent collision detection
- **Robot-Obstacle Collisions**: Static obstacle avoidance
- **Boundary Collisions**: Map boundary detection
- **Real-time Visualization**: Visual collision feedback

### Trajectory System
- **Waypoint-based**: Define paths using waypoints
- **Interpolation**: Smooth trajectory interpolation
- **Multiple Formats**: Support for various trajectory types
- **Dynamic Loading**: Runtime trajectory switching

### Visualization
- **Pygame Rendering**: Real-time 2D visualization
- **Speed Overlays**: Robot speed and status information
- **Progress Tracking**: Trajectory completion progress
- **Collision Indicators**: Visual collision feedback

## Configuration

### Robot Configuration
```python
from duckietown_simulator.robot.duckiebot import RobotConfig

config = RobotConfig(
    wheelbase=0.102,           # Distance between wheels (m)
    wheel_radius=0.0318,       # Wheel radius (m)
    max_wheel_speed=10.0,      # Maximum wheel speed (rad/s)
    collision_radius=0.05      # Robot collision radius (m)
)
```

### PID Configuration
```python
from duckietown_simulator.robot.pid_controller import PIDConfig, PIDGains

pid_config = PIDConfig(
    distance_gains=PIDGains(kp=0.5, ki=0.0, kd=0.0),
    heading_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
    max_linear_velocity=2.5,
    max_angular_velocity=4.0,
    position_tolerance=0.10
)
```

## Examples

### Basic Single-Agent Training
```python
import numpy as np
from examples.gym_pid_road_network import make_env

env = make_env("data/trajectory_1.json", render_mode="human")
obs, info = env.reset()

# Simple policy: Random actions with bias toward GO
for step in range(500):
    action = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% GO, 20% STOP
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 50 == 0:
        print(f"Step {step}: Reward={reward:.2f}, Progress={info['waypoint_progress']['progress_ratio']*100:.1f}%")
    
    if terminated or truncated:
        print("Episode completed!")
        break

env.close()
```

### Multi-Agent Coordination
```python
from examples.multi_agent_gym_env import make_multi_agent_env

env = make_multi_agent_env(num_agents=2, render_mode="human")
obs, infos = env.reset()

# Coordinated policy: Robots take turns
for step in range(300):
    if step < 100:
        actions = {"robot1": 1, "robot2": 1}  # Both GO
    elif step < 200:
        actions = {"robot1": 0, "robot2": 1}  # robot1 STOP, robot2 GO
    else:
        actions = {"robot1": 1, "robot2": 0}  # robot1 GO, robot2 STOP
    
    obs, rewards, terminated, truncated, infos = env.step(actions)
    
    if step % 50 == 0:
        for agent_id in env.agent_ids:
            print(f"{agent_id}: Reward={rewards[agent_id]:.2f}, Collisions={infos[agent_id]['collisions']}")

env.close()
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python main.py test

# Run specific tests
python tests/test_multi_agent_env.py
python tests/test_gym_env.py
python tests/test_collision_detection.py
```

## Development

### Adding New Trajectories
1. Use the trajectory editor: `python main.py trajectory-editor`
2. Create waypoints by clicking on the map
3. Save the trajectory to `data/` directory
4. Use in environments by specifying the file path

### Custom Environments
Extend the base environments for custom scenarios:

```python
from examples.gym_pid_road_network import PIDRoadNetworkEnv

class CustomEnv(PIDRoadNetworkEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def _calculate_reward(self, action):
        # Custom reward function
        reward = super()._calculate_reward(action)
        # Add custom reward components
        return reward
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Pygame Window Issues**: Make sure display is available for rendering
3. **Trajectory File Not Found**: Check file paths in `data/` directory
4. **Performance Issues**: Reduce rendering FPS or disable visualization

### Debug Mode
Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Duckietown project for inspiration and assets
- OpenAI Gym for the environment interface
- Pygame community for visualization tools
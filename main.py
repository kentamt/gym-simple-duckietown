#!/usr/bin/env python3
"""
Main entry point for Duckietown Simulator with TSL (Trajectory Specification Language).

This script provides a unified interface to all the functionality in the repository:
- Trajectory editing with GUI
- PID controller demos with collision detection
- Single-agent gym environment
- Multi-agent gym environment
- Various testing and debugging tools
"""

import sys
import argparse
import os

def trajectory_editor():
    """Launch the trajectory editor GUI."""
    import subprocess
    subprocess.run([sys.executable, 'tools/trajectory_editor.py'])

def pid_demo():
    """Run PID controller demonstration."""
    import subprocess
    subprocess.run([sys.executable, 'demos/demo_pid_road_network.py'])

def single_agent_gym():
    """Run single-agent gym environment demo."""
    sys.path.append('.')
    import subprocess
    subprocess.run([sys.executable, 'examples/gym_pid_road_network.py'])

def multi_agent_gym():
    """Run multi-agent gym environment demo."""
    sys.path.append('.')
    import subprocess
    subprocess.run([sys.executable, 'examples/multi_agent_gym_env.py'])

def discrete_actions_demo():
    """Run discrete actions demonstration."""
    import subprocess
    subprocess.run([sys.executable, 'demos/demo_discrete_actions.py'])

def run_tests():
    """Run test suite."""
    import subprocess
    test_files = [
        'tests/test_duckiebot.py',
        'tests/test_collision_detection.py',
        'tests/test_gym_env.py',
        'tests/test_multi_agent_env.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*50}")
            print(f"Running {test_file}")
            print('='*50)
            try:
                subprocess.run([sys.executable, test_file], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Test {test_file} failed with exit code {e.returncode}")
            except FileNotFoundError:
                print(f"Test file {test_file} not found")

def list_trajectories():
    """List available trajectory files."""
    trajectory_dir = 'data'
    if os.path.exists(trajectory_dir):
        trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.json')]
        print("Available trajectory files:")
        for file in sorted(trajectory_files):
            print(f"  - {file}")
    else:
        print("No trajectory data directory found.")

def main():
    parser = argparse.ArgumentParser(
        description="Duckietown Simulator with Trajectory Specification Language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  trajectory-editor    Launch GUI trajectory editor for creating waypoints
  pid-demo            Run PID controller demonstration with collision detection
  single-gym          Run single-agent gym environment demo
  multi-gym           Run multi-agent gym environment demo
  discrete-demo       Run discrete actions demonstration
  test                Run all tests
  list-trajectories   List available trajectory files

Examples:
  python main.py trajectory-editor
  python main.py pid-demo
  python main.py multi-gym
  python main.py test
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'trajectory-editor', 'pid-demo', 'single-gym', 'multi-gym', 
            'discrete-demo', 'test', 'list-trajectories'
        ],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    # Execute the chosen command
    if args.command == 'trajectory-editor':
        trajectory_editor()
    elif args.command == 'pid-demo':
        pid_demo()
    elif args.command == 'single-gym':
        single_agent_gym()
    elif args.command == 'multi-gym':
        multi_agent_gym()
    elif args.command == 'discrete-demo':
        discrete_actions_demo()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'list-trajectories':
        list_trajectories()

if __name__ == "__main__":
    main()
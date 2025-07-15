#!/usr/bin/env python3
"""
Quick test for rendering functionality.
"""

import sys
sys.path.append('.')

from duckietown_simulator.environment import DuckietownEnv


def test_render_setup():
    """Test that rendering setup works."""
    print("Testing render setup...")
    
    try:
        # Create environment with human rendering
        env = DuckietownEnv(
            map_config={"width": 3, "height": 3, "track_type": "straight"},
            render_mode="human",
            max_steps=10
        )
        
        print(f"Environment created with renderer type: {getattr(env, 'renderer_type', 'unknown')}")
        
        # Reset and test one render call
        obs, info = env.reset()
        print(f"Environment reset. Robot at: ({info['robot_state']['x']:.3f}, {info['robot_state']['y']:.3f})")
        
        # Try one render
        print("Attempting to render...")
        result = env.render()
        print(f"Render result: {result}")
        
        print("Render test successful!")
        env.close()
        
    except Exception as e:
        print(f"Error during render test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_no_render():
    """Test environment without rendering."""
    print("\nTesting environment without rendering...")
    
    try:
        env = DuckietownEnv(
            map_config={"width": 3, "height": 3, "track_type": "straight"},
            render_mode=None,
            max_steps=10
        )
        
        obs, info = env.reset()
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("No-render test successful!")
        return True
        
    except Exception as e:
        print(f"Error during no-render test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Quick Render Test ===")
    
    success1 = test_no_render()
    success2 = test_render_setup()
    
    if success1 and success2:
        print("\n✅ All quick tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
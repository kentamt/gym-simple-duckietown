#!/usr/bin/env python3
"""
Demo of manual wheel speed control on a road network.
Control left and right wheel speeds independently using keyboard.
"""

import sys
import numpy as np
import pygame
sys.path.append('.')

from duckietown_simulator.world.map import create_map_from_array
from duckietown_simulator.robot.duckiebot import create_duckiebot
from duckietown_simulator.rendering.pygame_renderer import create_pygame_renderer, RenderConfig


def create_road_network_map():
    """Create the same road network as the PID demo."""
    
    road_network = [
        [9, 2, 2, 8],  # Top: left curve, horizontal roads, right curve
        [3, 0,  0, 3],  # Vertical roads with grass in middle
        [14, 2, 2, 12],  # Middle: intersections with horizontal road
        [3, 0,  0, 3],  # Vertical roads with grass in middle
        [7, 2, 2, 4]   # Bottom: right curve, horizontal roads, left curve
    ]
    
    print("Creating road network:")
    print("  Layout: 4x5 tiles with curves, straights, and intersections")
    print("  Road types: curves (20,24,26,27), verticals (10), horizontals (11), intersections (30)")
    
    # Create map from 2D array
    map_instance = create_map_from_array(road_network, tile_size=0.61)
    
    print(f"Loaded map layout: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    return map_instance


class ManualWheelController:
    """Manual wheel speed controller for Duckiebot."""
    
    def __init__(self):
        """Initialize with zero wheel speeds."""
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.max_speed = 1.0  # Maximum wheel speed in m/s
        self.speed_increment = 0.1  # Speed change per key press
        
    def update_speeds(self, keys_pressed):
        """Update wheel speeds based on pressed keys."""
        # Left wheel controls (Q/A keys)
        if keys_pressed[pygame.K_q]:
            self.left_wheel_speed = min(self.max_speed, self.left_wheel_speed + self.speed_increment)
        elif keys_pressed[pygame.K_a]:
            self.left_wheel_speed = max(-self.max_speed, self.left_wheel_speed - self.speed_increment)
        else:
            # Gradually reduce speed when no key is pressed
            if abs(self.left_wheel_speed) > 0.01:
                self.left_wheel_speed *= 0.95
            else:
                self.left_wheel_speed = 0.0
        
        # Right wheel controls (E/D keys)
        if keys_pressed[pygame.K_e]:
            self.right_wheel_speed = min(self.max_speed, self.right_wheel_speed + self.speed_increment)
        elif keys_pressed[pygame.K_d]:
            self.right_wheel_speed = max(-self.max_speed, self.right_wheel_speed - self.speed_increment)
        else:
            # Gradually reduce speed when no key is pressed
            if abs(self.right_wheel_speed) > 0.01:
                self.right_wheel_speed *= 0.95
            else:
                self.right_wheel_speed = 0.0
    
    def get_wheel_speeds(self):
        """Get current wheel speeds."""
        return self.left_wheel_speed, self.right_wheel_speed
    
    def reset_speeds(self):
        """Reset both wheel speeds to zero."""
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0


def main():
    """Main demo function."""
    print("Manual Wheel Control Demo")
    print("=" * 50)
    
    # Create map and robots
    map_instance = create_road_network_map()
    
    # Create a single robot at the starting position
    robot = create_duckiebot(
        x=0.30,  # Starting x position
        y=0.30,  # Starting y position
        theta=0.0  # Starting orientation
    )
    
    # Create manual controller
    controller = ManualWheelController()
    
    # Create renderer
    render_config = RenderConfig(
        width=1200,
        height=800,
        show_grid=False,
        show_collision_circles=True,
        show_robot_ids=True,
        use_tile_images=True
    )
    
    renderer = create_pygame_renderer(map_instance, render_config)
    renderer.set_robots({"manual_robot": robot})
    
    # Add custom speed display
    def draw_speed_display():
        """Draw wheel speed display on screen."""
        left_speed, right_speed = controller.get_wheel_speeds()
        
        # Create text surfaces
        font = renderer.font
        
        # Speed text
        left_text = f"Left Wheel:  {left_speed:+.2f} m/s"
        right_text = f"Right Wheel: {right_speed:+.2f} m/s"
        
        # Render text
        left_surface = font.render(left_text, True, (255, 255, 255))
        right_surface = font.render(right_text, True, (255, 255, 255))
        
        # Position on screen (bottom left)
        y_offset = renderer.config.height - 80
        renderer.screen.blit(left_surface, (10, y_offset))
        renderer.screen.blit(right_surface, (10, y_offset + 25))
        
        # Controls reminder
        controls_text = "Controls: Q/A = Left Wheel  |  E/D = Right Wheel  |  SPACE = Stop"
        controls_surface = renderer.small_font.render(controls_text, True, (200, 200, 200))
        renderer.screen.blit(controls_surface, (10, y_offset + 55))
    
    # Store the original render method
    original_render = renderer.render
    
    # Create new render method that includes speed display
    def enhanced_render():
        result = original_render()
        draw_speed_display()
        return result
    
    # Replace the render method
    renderer.render = enhanced_render
    
    print(f"Map: {map_instance.width_tiles}x{map_instance.height_tiles} tiles")
    print("Robot: 1 manual robot")
    print()
    print("Manual Wheel Controls:")
    print("  Q: Increase Left Wheel Speed")
    print("  A: Decrease Left Wheel Speed") 
    print("  E: Increase Right Wheel Speed")
    print("  D: Decrease Right Wheel Speed")
    print("  SPACE: Emergency Stop (reset all speeds)")
    print("  R: Reset Robot Position")
    print()
    print("Other Controls:")
    print("  =/+/-: Zoom In/Out")
    print("  Mouse: Pan")
    print("  ESC: Quit")
    print()
    print("Use Q/A for left wheel and E/D for right wheel!")
    print("Different wheel speeds will make the robot turn.")
    
    # Simulation loop
    clock = pygame.time.Clock()
    step = 0
    
    try:
        while True:
            # Handle pygame events first
            if not renderer.handle_events():
                break
            
            # Handle additional keyboard events for wheel control
            keys = pygame.key.get_pressed()
            
            # Check for emergency stop
            if keys[pygame.K_SPACE]:
                controller.reset_speeds()
            
            # Check for robot reset
            if keys[pygame.K_r]:
                robot.pose = np.array([0.30, 0.30, 0.0])
                robot.linear_velocity = 0.0
                robot.angular_velocity = 0.0
                robot.omega_l = 0.0
                robot.omega_r = 0.0
                controller.reset_speeds()
            
            # Update wheel speeds based on input
            controller.update_speeds(keys)
            left_speed, right_speed = controller.get_wheel_speeds()
            
            # Apply wheel speeds to robot (convert m/s to rad/s)
            # Wheel radius is typically 0.0318m for Duckiebot
            wheel_radius = 0.0318
            omega_l = left_speed / wheel_radius  # Convert linear speed to angular speed
            omega_r = right_speed / wheel_radius
            
            # Update robot physics
            dt = 1.0 / 60.0  # 60 FPS
            robot.step([omega_l, omega_r], dt)
            
            # Render everything
            renderer.render()
            
            # Print status every second
            if step % 60 == 0:
                print(f"Step: {step:4d} | Left: {left_speed:+.2f} | Right: {right_speed:+.2f} | "
                      f"Pos: ({robot.pose[0]:.2f}, {robot.pose[1]:.2f}) | Angle: {np.degrees(robot.pose[2]):.1f}°")
            
            step += 1
            clock.tick(60)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted!")
    
    finally:
        renderer.cleanup()
    
    print("=" * 50)
    print("Demo completed!")
    print()
    print("Key features demonstrated:")
    print("✓ Manual left/right wheel speed control")
    print("✓ Independent wheel control for turning")
    print("✓ Real-time speed and position feedback")
    print("✓ Emergency stop functionality") 
    print("✓ Robot reset capability")
    print("✓ Realistic differential drive physics")


if __name__ == "__main__":
    main()
"""
Simple text-based renderer for when pygame is not available.

This provides basic visualization using text output and optional matplotlib plots.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

from ..world.map import Map
from ..robot.duckiebot import Duckiebot


class SimpleRenderer:
    """
    Simple text-based renderer for Duckietown environment.
    
    This renderer provides basic visualization when pygame is not available,
    using ASCII art and text output.
    """
    
    def __init__(self, map_instance: Map, show_coordinates: bool = True):
        """
        Initialize simple renderer.
        
        Args:
            map_instance: Map to render
            show_coordinates: Whether to show coordinate information
        """
        self.map = map_instance
        self.show_coordinates = show_coordinates
        self.frame_count = 0
        self.start_time = time.time()
        self.robots: Dict[str, Duckiebot] = {}
        
        # Character mapping for tiles
        self.tile_chars = {
            0: '.',  # Empty
            1: '#',  # Obstacle/Wall
            2: '=',  # Road
            3: '~',  # Grass
            4: '+',  # Intersection
        }
    
    def set_robots(self, robots: Dict[str, Duckiebot]):
        """Set robots to render."""
        self.robots = robots.copy()
    
    def render(self) -> bool:
        """
        Render current frame to console.
        
        Returns:
            True (always continue for simple renderer)
        """
        self.frame_count += 1
        
        # Clear screen (works on most terminals)
        print('\033[2J\033[H', end='')
        
        # Title
        print("=" * 60)
        print("           Duckietown Simulator (Text Mode)")
        print("=" * 60)
        
        # Map visualization
        print("\nMap Layout:")
        self._render_map_with_robots()
        
        # Robot information
        if self.robots:
            print("\nRobot Status:")
            for robot_id, robot in self.robots.items():
                self._render_robot_info(robot_id, robot)
        
        # Frame info
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            print(f"\nFrame: {self.frame_count}, FPS: {fps:.1f}, Time: {elapsed:.1f}s")
        
        print("\nControls: Ctrl+C to quit")
        print("-" * 60)
        
        return True
    
    def _render_map_with_robots(self):
        """Render map with robots overlaid."""
        # Create a copy of the map for rendering
        map_display = []
        for row in range(self.map.height_tiles):
            map_row = []
            for col in range(self.map.width_tiles):
                tile_type = self.map.get_tile_type(row, col)
                char = self.tile_chars.get(tile_type, '?')
                map_row.append(char)
            map_display.append(map_row)
        
        # Add robots to the display
        robot_chars = ['@', '%', '&', '*', '$']  # Different chars for multiple robots
        for i, (robot_id, robot) in enumerate(self.robots.items()):
            # Convert robot position to tile coordinates
            robot_row, robot_col = self.map.get_tile_at_position(robot.x, robot.y)
            
            if robot_row >= 0 and robot_col >= 0:  # Valid position
                robot_char = robot_chars[i % len(robot_chars)]
                map_display[robot_row][robot_col] = robot_char
        
        # Print the map
        print("  ", end="")
        for col in range(self.map.width_tiles):
            print(f"{col}", end="")
        print()
        
        for row in range(self.map.height_tiles):
            print(f"{row} ", end="")
            for col in range(self.map.width_tiles):
                print(map_display[row][col], end="")
            print()
        
        # Legend
        print("\nLegend:")
        print("  . = Empty   # = Wall   = = Road   ~ = Grass   + = Intersection")
        print("  @ = Robot1  % = Robot2  & = Robot3  * = Robot4  $ = Robot5")
    
    def _render_robot_info(self, robot_id: str, robot: Duckiebot):
        """Render detailed robot information."""
        print(f"  {robot_id}:")
        print(f"    Position: ({robot.x:.3f}, {robot.y:.3f}) m")
        print(f"    Angle: {robot.theta:.3f} rad ({np.degrees(robot.theta):.1f}°)")
        print(f"    Velocity: {robot.linear_velocity:.3f} m/s")
        print(f"    Angular Vel: {robot.angular_velocity:.3f} rad/s")
        print(f"    Wheels: L={robot.omega_l:.2f}, R={robot.omega_r:.2f} rad/s")
        print(f"    Distance: {robot.total_distance:.3f} m")
        print(f"    Collision: {'YES' if robot.is_collided else 'No'}")
    
    def close(self):
        """Clean up resources (nothing needed for simple renderer)."""
        pass


class MatplotlibRenderer:
    """
    Matplotlib-based renderer for when pygame is not available.
    
    This provides visual rendering using matplotlib, suitable for
    Jupyter notebooks or saving images.
    """
    
    def __init__(self, map_instance: Map, figsize: tuple = (8, 6)):
        """
        Initialize matplotlib renderer.
        
        Args:
            map_instance: Map to render
            figsize: Figure size for matplotlib
        """
        self.map = map_instance
        self.figsize = figsize
        self.robots: Dict[str, Duckiebot] = {}
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            self.plt = plt
            self.patches = patches
            self.available = True
        except ImportError:
            print("Warning: matplotlib not available for visual rendering")
            self.available = False
    
    def set_robots(self, robots: Dict[str, Duckiebot]):
        """Set robots to render."""
        self.robots = robots.copy()
    
    def render(self, save_path: Optional[str] = None) -> bool:
        """
        Render current frame using matplotlib.
        
        Args:
            save_path: Optional path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        
        fig, ax = self.plt.subplots(figsize=self.figsize)
        
        # Draw map tiles
        self._draw_map(ax)
        
        # Draw robots
        for robot_id, robot in self.robots.items():
            self._draw_robot(ax, robot, robot_id)
        
        # Set up plot
        ax.set_xlim(0, self.map.width_meters)
        ax.set_ylim(0, self.map.height_meters)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Duckietown Simulator')
        
        # Show or save
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved render to {save_path}")
        else:
            self.plt.show(block=False)
            self.plt.pause(0.01)
        
        self.plt.close(fig)
        return True
    
    def _draw_map(self, ax):
        """Draw the map tiles."""
        colors = {
            0: 'lightgray',   # Empty
            1: 'darkgray',    # Obstacle/Wall
            2: 'white',       # Road
            3: 'lightgreen',  # Grass
            4: 'yellow',      # Intersection
        }
        
        for row in range(self.map.height_tiles):
            for col in range(self.map.width_tiles):
                tile_type = self.map.get_tile_type(row, col)
                color = colors.get(tile_type, 'gray')
                
                # Calculate tile position
                x = col * self.map.tile_size
                y = row * self.map.tile_size
                
                # Draw tile
                rect = self.patches.Rectangle(
                    (x, y), self.map.tile_size, self.map.tile_size,
                    linewidth=0.5, edgecolor='black', facecolor=color
                )
                ax.add_patch(rect)
    
    def _draw_robot(self, ax, robot: Duckiebot, robot_id: str):
        """Draw a robot."""
        # Robot colors
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        color_idx = hash(robot_id) % len(colors)
        color = colors[color_idx]
        
        # Draw robot as a circle
        circle = self.patches.Circle(
            (robot.x, robot.y), robot.collision_radius,
            color=color, alpha=0.7
        )
        ax.add_patch(circle)
        
        # Draw direction arrow
        arrow_length = robot.collision_radius * 2
        dx = arrow_length * np.cos(robot.theta)
        dy = arrow_length * np.sin(robot.theta)
        
        ax.arrow(robot.x, robot.y, dx, dy,
                head_width=robot.collision_radius * 0.5,
                head_length=robot.collision_radius * 0.3,
                fc=color, ec=color)
        
        # Add robot label
        ax.text(robot.x, robot.y + robot.collision_radius + 0.1,
               robot_id, ha='center', va='bottom',
               fontsize=8, color=color, weight='bold')
    
    def close(self):
        """Clean up resources."""
        if self.available:
            self.plt.close('all')


def create_fallback_renderer(map_instance: Map, prefer_matplotlib: bool = False):
    """
    Create a fallback renderer when pygame is not available.
    
    Args:
        map_instance: Map to render
        prefer_matplotlib: Whether to prefer matplotlib over text
        
    Returns:
        Renderer instance
    """
    if prefer_matplotlib:
        renderer = MatplotlibRenderer(map_instance)
        if renderer.available:
            return renderer
    
    # Fall back to simple text renderer
    return SimpleRenderer(map_instance)
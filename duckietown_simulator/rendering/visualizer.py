import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
from matplotlib.animation import FuncAnimation

from ..world.map import Map
from ..robot.duckiebot import Duckiebot


class DuckietownVisualizer:
    """
    2D visualization for Duckietown simulator using matplotlib.
    """
    
    def __init__(self, map_instance: Map, figsize: Tuple[int, int] = (10, 10)):
        """
        Initialize the visualizer.
        
        Args:
            map_instance: Map to visualize
            figsize: Figure size (width, height)
        """
        self.map = map_instance
        self.figsize = figsize
        
        # Color scheme
        self.colors = {
            'empty': '#f0f0f0',      # Light gray
            'obstacle': '#2c3e50',    # Dark blue-gray
            'road': '#ecf0f1',        # Very light gray
            'robot': '#e74c3c',       # Red
            'robot_direction': '#c0392b',  # Dark red
            'trajectory': '#3498db',   # Blue
            'collision_circle': '#f39c12',  # Orange
            'grid': '#bdc3c7'         # Light gray
        }
        
        # Initialize figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_plot()
        
        # Store trajectory
        self.trajectory = []
        
    def setup_plot(self):
        """Setup the plot with proper scaling and labels."""
        self.ax.set_xlim(0, self.map.width_meters)
        self.ax.set_ylim(0, self.map.height_meters)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Duckietown Simulator')
        self.ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
    def draw_map(self):
        """Draw the map tiles."""
        self.ax.clear()
        self.setup_plot()
        
        # Draw tiles
        for row in range(self.map.height_tiles):
            for col in range(self.map.width_tiles):
                tile_type = self.map.get_tile_type(row, col)
                
                # Get tile boundaries
                bounds = self.map.tile_boundaries[row][col]
                
                # Choose color based on tile type
                if tile_type == 0:  # Empty
                    color = self.colors['empty']
                elif tile_type == 1:  # Obstacle
                    color = self.colors['obstacle']
                elif tile_type == 2:  # Road
                    color = self.colors['road']
                else:
                    color = self.colors['empty']
                
                # Draw tile
                rect = patches.Rectangle(
                    (bounds['x_min'], bounds['y_min']),
                    self.map.tile_size,
                    self.map.tile_size,
                    facecolor=color,
                    edgecolor=self.colors['grid'],
                    linewidth=0.5
                )
                self.ax.add_patch(rect)
        
        # Draw tile grid lines
        for i in range(self.map.width_tiles + 1):
            x = i * self.map.tile_size
            self.ax.axvline(x, color=self.colors['grid'], linewidth=0.5, alpha=0.5)
        
        for i in range(self.map.height_tiles + 1):
            y = i * self.map.tile_size
            self.ax.axhline(y, color=self.colors['grid'], linewidth=0.5, alpha=0.5)
    
    def draw_robot(self, robot: Duckiebot, show_collision_circle: bool = True):
        """
        Draw the robot on the map.
        
        Args:
            robot: Robot to draw
            show_collision_circle: Whether to show collision circle
        """
        x, y, theta = robot.pose
        
        # Draw collision circle
        if show_collision_circle:
            circle = patches.Circle(
                (x, y),
                robot.collision_radius,
                facecolor=self.colors['collision_circle'],
                edgecolor=self.colors['collision_circle'],
                alpha=0.3,
                linewidth=1
            )
            self.ax.add_patch(circle)
        
        # Draw robot body as a small rectangle
        robot_length = 0.18  # cm
        robot_width = 0.15   # cm
        
        # Robot corners in local coordinates
        corners = np.array([
            [-robot_length/2, -robot_width/2],
            [robot_length/2, -robot_width/2],
            [robot_length/2, robot_width/2],
            [-robot_length/2, robot_width/2]
        ])
        
        # Rotate and translate to world coordinates
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        world_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
        
        # Draw robot body
        robot_patch = patches.Polygon(
            world_corners,
            facecolor=self.colors['robot'],
            edgecolor=self.colors['robot_direction'],
            linewidth=2
        )
        self.ax.add_patch(robot_patch)
        
        # Draw direction arrow
        arrow_length = 0.1
        arrow_x = x + arrow_length * np.cos(theta)
        arrow_y = y + arrow_length * np.sin(theta)
        
        self.ax.arrow(
            x, y, arrow_x - x, arrow_y - y,
            head_width=0.02,
            head_length=0.02,
            fc=self.colors['robot_direction'],
            ec=self.colors['robot_direction']
        )
        
        # Add robot position text
        self.ax.text(
            x, y - 0.15,
            f'({x:.2f}, {y:.2f})',
            ha='center',
            va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    def draw_trajectory(self, trajectory: List[Tuple[float, float, float]]):
        """
        Draw the robot's trajectory.
        
        Args:
            trajectory: List of (x, y, theta) poses
        """
        if len(trajectory) < 2:
            return
        
        # Extract x, y coordinates
        x_coords = [pose[0] for pose in trajectory]
        y_coords = [pose[1] for pose in trajectory]
        
        # Draw trajectory line
        self.ax.plot(
            x_coords, y_coords,
            color=self.colors['trajectory'],
            linewidth=2,
            alpha=0.7,
            label='Trajectory'
        )
        
        # Draw start and end points
        self.ax.plot(
            x_coords[0], y_coords[0],
            'go', markersize=8,
            label='Start'
        )
        
        self.ax.plot(
            x_coords[-1], y_coords[-1],
            'ro', markersize=8,
            label='End'
        )
        
        # Add trajectory statistics
        total_distance = sum(
            np.linalg.norm(np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]]))
            for i in range(len(x_coords) - 1)
        )
        
        self.ax.text(
            0.02, 0.98,
            f'Steps: {len(trajectory)}\nDistance: {total_distance:.3f}m',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
    
    def show(self, block: bool = True):
        """Show the visualization."""
        self.ax.legend()
        plt.tight_layout()
        plt.show(block=block)
    
    def save(self, filename: str, dpi: int = 300):
        """Save the visualization to file."""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    def clear_trajectory(self):
        """Clear the stored trajectory."""
        self.trajectory.clear()
    
    def create_animation(self, trajectories: List[List[Tuple[float, float, float]]], 
                        interval: int = 100) -> FuncAnimation:
        """
        Create an animation of the robot moving through trajectories.
        
        Args:
            trajectories: List of trajectory lists
            interval: Animation interval in milliseconds
            
        Returns:
            FuncAnimation object
        """
        def animate(frame):
            self.draw_map()
            
            # Draw trajectory up to current frame
            if frame < len(trajectories):
                current_trajectory = trajectories[frame]
                if current_trajectory:
                    # Draw partial trajectory
                    self.draw_trajectory(current_trajectory)
                    
                    # Draw robot at current position
                    if len(current_trajectory) > 0:
                        x, y, theta = current_trajectory[-1]
                        robot_pose = np.array([x, y, theta])
                        # Create a dummy robot for visualization
                        class DummyRobot:
                            def __init__(self, pose):
                                self.pose = pose
                                self.collision_radius = 0.05
                        
                        dummy_robot = DummyRobot(robot_pose)
                        self.draw_robot(dummy_robot)
        
        return FuncAnimation(
            self.fig, animate,
            frames=len(trajectories),
            interval=interval,
            repeat=True
        )


def create_visualizer(map_instance: Map, figsize: Tuple[int, int] = (10, 10)) -> DuckietownVisualizer:
    """
    Factory function to create a visualizer.
    
    Args:
        map_instance: Map to visualize
        figsize: Figure size
        
    Returns:
        DuckietownVisualizer instance
    """
    return DuckietownVisualizer(map_instance, figsize)
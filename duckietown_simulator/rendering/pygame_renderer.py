import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..world.map import Map
from ..robot.duckiebot import Duckiebot
from ..world.obstacles import ObstacleManager, Obstacle
from ..world.collision_detection import CollisionResult
from .tile_image_manager import TileImageManager


@dataclass
class RenderConfig:
    """Configuration for pygame renderer."""
    width: int = 1200
    height: int = 800
    fps: int = 60
    background_color: Tuple[int, int, int] = (240, 240, 240)
    grid_color: Tuple[int, int, int] = (200, 200, 200)
    show_grid: bool = True
    show_coordinates: bool = True
    show_collision_circles: bool = True
    show_robot_ids: bool = True
    show_fps: bool = True
    use_tile_images: bool = True
    assets_dir: str = None


class Colors:
    """Color constants for pygame rendering."""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)
    
    # Map colors
    EMPTY_TILE = (240, 240, 240)
    ROAD_TILE = (220, 220, 220)
    OBSTACLE_TILE = (60, 60, 60)
    
    # Robot colors
    ROBOT_COLORS = [
        (31, 119, 180),   # Blue
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
    ]
    
    # Collision colors
    COLLISION_CIRCLE = (255, 165, 0, 100)  # Orange with alpha
    COLLISION_POINT = (255, 0, 0)  # Red


class PygameRenderer:
    """
    Real-time pygame renderer for Duckietown simulator.
    
    Provides interactive visualization with real-time updates,
    zoom, pan, and collision detection visualization.
    """
    
    def __init__(self, map_instance: Map, config: RenderConfig = None):
        """
        Initialize pygame renderer.
        
        Args:
            map_instance: Map to render
            config: Renderer configuration
        """
        self.map = map_instance
        self.config = config or RenderConfig()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Duckietown Simulator - Real-time Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Camera/viewport settings
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom = 1.0
        self.target_zoom = 1.0
        
        # Calculate initial zoom to fit map
        margin = 50
        zoom_x = (self.config.width - 2 * margin) / self.map.width_meters
        zoom_y = (self.config.height - 2 * margin) / self.map.height_meters
        self.zoom = min(zoom_x, zoom_y) * 0.8  # Add some extra margin
        self.target_zoom = self.zoom
        
        # Center camera on map
        self.camera_x = -self.map.width_meters / 2
        self.camera_y = -self.map.height_meters / 2
        
        # Interaction state
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        self.running = True
        self.paused = False
        
        # Tile image manager
        self.tile_manager = None
        if self.config.use_tile_images:
            try:
                self.tile_manager = TileImageManager(self.config.assets_dir)
                self.tile_manager.preload_tiles()
                print(f"Loaded tile images: {self.tile_manager.get_cache_info()}")
            except Exception as e:
                print(f"Warning: Could not load tile images: {e}")
                print("Falling back to colored rectangles")
                self.config.use_tile_images = False
        
        # Rendering data
        self.robots: Dict[str, Duckiebot] = {}
        self.obstacle_manager: Optional[ObstacleManager] = None
        self.collision_results: List[CollisionResult] = []
        self.trajectories: Dict[str, List[Tuple[float, float]]] = {}
        self.planned_trajectories: Dict[str, List[Tuple[float, float]]] = {}
        self.show_trajectories = True
        self.show_planned_trajectories = True
        
        # Waypoint visualization
        self.current_waypoint: Optional[Tuple[float, float]] = None
        self.show_waypoint = True
        
        # Load robot image
        self.robot_image = None
        self.robot_image_scaled = {}  # Cache for scaled versions
        try:
            import os
            robot_image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'robot', 'duckie_top_view.png')
            self.robot_image = pygame.image.load(robot_image_path)
            print(f"Loaded robot image: {robot_image_path}")
        except Exception as e:
            print(f"Warning: Could not load robot image: {e}")
            print("Falling back to rectangle rendering")
        
        # Performance tracking
        self.frame_count = 0
        self.fps_display = 0
        self.last_fps_update = 0
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = (world_x + self.camera_x) * self.zoom + self.config.width // 2
        screen_y = (world_y + self.camera_y) * self.zoom + self.config.height // 2
        return (int(screen_x), int(screen_y))
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - self.config.width // 2) / self.zoom - self.camera_x
        world_y = (screen_y - self.config.height // 2) / self.zoom - self.camera_y
        return (world_x, world_y)
    
    def world_distance_to_screen(self, distance: float) -> int:
        """Convert world distance to screen pixels."""
        return int(distance * self.zoom)
    
    def set_robots(self, robots: Dict[str, Duckiebot]):
        """Set robots to render."""
        self.robots = robots
        
        # Initialize trajectories
        for robot_id in robots.keys():
            if robot_id not in self.trajectories:
                self.trajectories[robot_id] = []
    
    def set_obstacle_manager(self, obstacle_manager: ObstacleManager):
        """Set obstacle manager."""
        self.obstacle_manager = obstacle_manager
    
    def set_collision_results(self, collision_results: List[CollisionResult]):
        """Set collision results to visualize."""
        self.collision_results = collision_results
    
    def set_current_waypoint(self, waypoint: Tuple[float, float]):
        """Set the current waypoint target for visualization."""
        self.current_waypoint = waypoint
    
    def update_trajectories(self):
        """Update robot trajectories."""
        for robot_id, robot in self.robots.items():
            if robot_id in self.trajectories:
                self.trajectories[robot_id].append((robot.x, robot.y))
                # Limit trajectory length
                if len(self.trajectories[robot_id]) > 500:
                    self.trajectories[robot_id].pop(0)
    
    def handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            True if should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    # Reset camera
                    self.camera_x = -self.map.width_meters / 2
                    self.camera_y = -self.map.height_meters / 2
                    margin = 50
                    zoom_x = (self.config.width - 2 * margin) / self.map.width_meters
                    zoom_y = (self.config.height - 2 * margin) / self.map.height_meters
                    self.target_zoom = min(zoom_x, zoom_y) * 0.8
                elif event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                elif event.key == pygame.K_c:
                    # Clear trajectories
                    for robot_id in self.trajectories:
                        self.trajectories[robot_id].clear()
                elif event.key == pygame.K_g:
                    self.config.show_grid = not self.config.show_grid
                elif event.key == pygame.K_i:
                    # Toggle image/color rendering
                    if self.tile_manager:
                        self.config.use_tile_images = not self.config.use_tile_images
                        mode = "images" if self.config.use_tile_images else "colors"
                        print(f"Switched to {mode} rendering")
                elif event.key == pygame.K_p:
                    # Toggle planned trajectory display
                    self.show_planned_trajectories = not self.show_planned_trajectories
                    print(f"Planned trajectories: {'ON' if self.show_planned_trajectories else 'OFF'}")
                elif event.key == pygame.K_w:
                    # Toggle waypoint display
                    self.show_waypoint = not self.show_waypoint
                    print(f"Waypoint display: {'ON' if self.show_waypoint else 'OFF'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    # Zoom in with =, +, or keypad + key
                    zoom_factor = 1.5
                    self.target_zoom *= zoom_factor
                    self.target_zoom = max(0.1, min(10000.0, self.target_zoom))
                    print(f"Zoom in: {self.target_zoom:.2f}")
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    # Zoom out with - or keypad - key
                    zoom_factor = 1.0 / 1.5
                    self.target_zoom *= zoom_factor
                    self.target_zoom = max(0.1, min(1000.0, self.target_zoom))
                    print(f"Zoom out: {self.target_zoom:.2f}")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera_x += dx / self.zoom
                    self.camera_y += dy / self.zoom
                    self.last_mouse_pos = event.pos
        
        # Smooth zoom
        if abs(self.zoom - self.target_zoom) > 0.01:
            self.zoom += (self.target_zoom - self.zoom) * 0.3
        
        return True
    
    def draw_map(self):
        """Draw the map tiles."""
        for row in range(self.map.height_tiles):
            for col in range(self.map.width_tiles):
                tile_type = self.map.get_tile_type(row, col)
                
                # Get tile boundaries
                bounds = self.map.tile_boundaries[row][col]
                
                # Convert to screen coordinates
                top_left = self.world_to_screen(bounds['x_min'], bounds['y_min'])
                bottom_right = self.world_to_screen(bounds['x_max'], bounds['y_max'])
                
                # Use integer coordinates and ensure no gaps
                x = int(top_left[0])
                y = int(top_left[1])
                width = int(bottom_right[0]) - x
                height = int(bottom_right[1]) - y
                
                # Add overlap to prevent gaps - expand tiles by 1 pixel in each direction
                if col > 0:  # Not first column, extend left
                    x -= 1
                    width += 1
                if row > 0:  # Not first row, extend up
                    y -= 1
                    height += 1
                if col < self.map.width_tiles - 1:  # Not last column, extend right
                    width += 1
                if row < self.map.height_tiles - 1:  # Not last row, extend down
                    height += 1
                
                # Ensure minimum size
                width = max(width, 1)
                height = max(height, 1)
                
                # Skip tiles that are too small or off-screen
                if width <= 1 or height <= 1:
                    continue
                
                # Draw tile using image or color
                if self.config.use_tile_images and self.tile_manager:
                    self._draw_tile_image(tile_type, (x, y), (width, height))
                else:
                    self._draw_tile_color(tile_type, (x, y), (width, height))
                
                # Draw tile border if grid is enabled
                if self.config.show_grid and width > 2 and height > 2:
                    pygame.draw.rect(self.screen, self.config.grid_color,
                                   (x, y, width, height), 1)
    
    def _draw_tile_image(self, tile_type: int, position: Tuple[int, int], size: Tuple[int, int]):
        """Draw a tile using an image."""
        tile_image = self.tile_manager.get_tile_image(tile_type, size)
        
        if tile_image:
            self.screen.blit(tile_image, position)
        else:
            # Fallback to color if image fails
            self._draw_tile_color(tile_type, position, size)
    
    def _draw_tile_color(self, tile_type: int, position: Tuple[int, int], size: Tuple[int, int]):
        """Draw a tile using solid color (fallback method)."""
        # Choose color based on tile type
        if tile_type == 0:  # Empty
            color = Colors.EMPTY_TILE
        elif tile_type == 1:  # Obstacle
            color = Colors.OBSTACLE_TILE
        elif tile_type == 2:  # Road
            color = Colors.ROAD_TILE
        else:
            color = Colors.EMPTY_TILE
        
        pygame.draw.rect(self.screen, color, (position[0], position[1], size[0], size[1]))
    
    def draw_obstacles(self):
        """Draw static obstacles."""
        if not self.obstacle_manager:
            return
        
        for obstacle in self.obstacle_manager.get_all_obstacles():
            viz_data = obstacle.get_visualization_data()
            
            if viz_data['type'] == 'circle':
                center = self.world_to_screen(viz_data['x'], viz_data['y'])
                radius = self.world_distance_to_screen(viz_data['radius'])
                
                if radius > 1:  # Only draw if visible
                    pygame.draw.circle(self.screen, Colors.DARK_GRAY, center, radius)
                    pygame.draw.circle(self.screen, Colors.BLACK, center, radius, 2)
                    
                    # Draw obstacle label
                    if radius > 20:
                        label = viz_data['name'].replace('obstacle_', 'O')
                        text = self.small_font.render(label, True, Colors.WHITE)
                        text_rect = text.get_rect(center=center)
                        self.screen.blit(text, text_rect)
            
            elif viz_data['type'] == 'rectangle':
                corners = viz_data['corners']
                screen_corners = [self.world_to_screen(c[0], c[1]) for c in corners]
                
                # Only draw if at least one corner is on screen
                if any(0 <= x < self.config.width and 0 <= y < self.config.height 
                       for x, y in screen_corners):
                    pygame.draw.polygon(self.screen, Colors.DARK_GRAY, screen_corners)
                    pygame.draw.polygon(self.screen, Colors.BLACK, screen_corners, 2)
                    
                    # Draw obstacle label
                    center = self.world_to_screen(viz_data['x'], viz_data['y'])
                    label = viz_data['name'].replace('obstacle_', 'O')
                    text = self.small_font.render(label, True, Colors.WHITE)
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
    
    def draw_trajectories(self):
        """Draw robot trajectories."""
        if not self.show_trajectories:
            return
        
        for i, (robot_id, trajectory) in enumerate(self.trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            color = Colors.ROBOT_COLORS[i % len(Colors.ROBOT_COLORS)]
            
            # Convert trajectory to screen coordinates
            screen_points = [self.world_to_screen(x, y) for x, y in trajectory]
            
            # Filter points that are on screen or close to it
            visible_points = []
            for point in screen_points:
                if (-50 <= point[0] <= self.config.width + 50 and 
                    -50 <= point[1] <= self.config.height + 50):
                    visible_points.append(point)
            
            # Draw trajectory line
            if len(visible_points) >= 2:
                pygame.draw.lines(self.screen, color, False, visible_points, 2)
    
    def draw_planned_trajectories(self):
        """Draw planned trajectories for robots."""
        if not self.show_planned_trajectories or not self.planned_trajectories:
            return
        
        for i, (robot_id, trajectory) in enumerate(self.planned_trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            # Use robot color but make it semi-transparent by using a lighter shade
            base_color = Colors.ROBOT_COLORS[i % len(Colors.ROBOT_COLORS)]
            # Make it lighter for the planned trajectory
            planned_color = tuple(min(255, c + 60) for c in base_color)
            
            # Convert trajectory to screen coordinates
            screen_points = [self.world_to_screen(x, y) for x, y in trajectory]
            
            # Filter points that are on screen or close to it
            visible_points = []
            for point in screen_points:
                if (-50 <= point[0] <= self.config.width + 50 and 
                    -50 <= point[1] <= self.config.height + 50):
                    visible_points.append(point)
            
            # Draw planned trajectory line (dashed style)
            if len(visible_points) >= 2:
                # Draw dashed line by drawing segments
                # for j in range(0, len(visible_points) - 1):
                #     if j % 2 == 0:  # Only draw every other segment for dashed effect
                #         pygame.draw.line(self.screen, planned_color,
                #                        visible_points[j], visible_points[j + 1], 3)
                
                # Draw waypoint markers
                for point in visible_points:
                    if (0 <= point[0] <= self.config.width and 
                        0 <= point[1] <= self.config.height):
                        pygame.draw.circle(self.screen, planned_color, point, 2)
                        # pygame.draw.circle(self.screen, Colors.WHITE, point, 2, 2)
                
                # Draw start and end markers
                if visible_points:
                    # Start marker (green circle)
                    start_point = visible_points[0]
                    pygame.draw.circle(self.screen, Colors.GREEN, start_point, 8)
                    pygame.draw.circle(self.screen, Colors.WHITE, start_point, 8, 2)
                    
                    # End marker (red circle)
                    end_point = visible_points[-1]
                    pygame.draw.circle(self.screen, Colors.RED, end_point, 8)
                    pygame.draw.circle(self.screen, Colors.WHITE, end_point, 8, 2)
    
    def draw_current_waypoint(self):
        """Draw the current waypoint target."""
        if not self.show_waypoint or self.current_waypoint is None:
            return
        
        # Convert waypoint to screen coordinates
        screen_x, screen_y = self.world_to_screen(self.current_waypoint[0], self.current_waypoint[1])
        
        # Check if waypoint is on screen
        if not (0 <= screen_x <= self.config.width and 0 <= screen_y <= self.config.height):
            return
        
        # Draw waypoint as a large circle with cross
        waypoint_color = Colors.YELLOW
        center = (int(screen_x), int(screen_y))
        
        # Draw outer circle (target ring)
        # pygame.draw.circle(self.screen, waypoint_color, center, 15, 3)
        
        # Draw inner circle (filled)
        # pygame.draw.circle(self.screen, waypoint_color, center, 5)
        
        # Draw cross inside
        cross_size = 10
        pygame.draw.line(self.screen, Colors.RED,
                        (center[0] - cross_size, center[1]), 
                        (center[0] + cross_size, center[1]), 2)
        pygame.draw.line(self.screen, Colors.RED,
                        (center[0], center[1] - cross_size), 
                        (center[0], center[1] + cross_size), 2)
    
    def draw_robots(self):
        """Draw robots."""
        for i, (robot_id, robot) in enumerate(self.robots.items()):
            color = Colors.ROBOT_COLORS[i % len(Colors.ROBOT_COLORS)]
            
            # Robot position
            center = self.world_to_screen(robot.x, robot.y)
            
            # Skip if robot is off screen
            if (center[0] < -100 or center[0] > self.config.width + 100 or
                center[1] < -100 or center[1] > self.config.height + 100):
                continue
            
            # Draw collision circle
            if self.config.show_collision_circles:
                collision_radius = self.world_distance_to_screen(robot.collision_radius)
                if collision_radius > 1:
                    pygame.draw.circle(self.screen, Colors.COLLISION_CIRCLE[:3], 
                                     center, collision_radius, 1)
            
            # Draw robot body - use actual robot dimensions
            # Real Duckiebot is about 18cm long x 10cm wide
            robot_length = 0.18  # 18cm length
            robot_width = 0.10   # 10cm width

            # Convert to screen coordinates with minimum pixel size
            screen_length = max(12, self.world_distance_to_screen(robot_length))
            screen_width = max(8, self.world_distance_to_screen(robot_width))
            
            # Draw robot using image or fallback to rectangle
            if self.robot_image:
                self._draw_robot_image(robot, center, screen_length, screen_width)
            else:
                self._draw_robot_rectangle(robot, center, screen_length, screen_width, color)
            
            # Draw robot ID
            if self.config.show_robot_ids and screen_length > 15:
                text = self.small_font.render(robot_id, True, Colors.WHITE)
                text_rect = text.get_rect(center=(center[0], center[1] - screen_length//2 - 15))
                self.screen.blit(text, text_rect)
    
    def _draw_robot_image(self, robot, center, screen_length, screen_width):
        """Draw robot using the duckie_top_view.png image."""
        # Get or create scaled image
        size_key = (screen_length, screen_width)
        if size_key not in self.robot_image_scaled:
            # Scale the image to fit the robot dimensions
            scaled_image = pygame.transform.scale(self.robot_image, (int(screen_length), int(screen_width)))
            self.robot_image_scaled[size_key] = scaled_image
        
        scaled_image = self.robot_image_scaled[size_key]
        
        # Rotate the image based on robot orientation
        # Convert theta to degrees and adjust for pygame's coordinate system
        angle_degrees = -math.degrees(robot.theta)  # Negative because pygame rotates clockwise
        rotated_image = pygame.transform.rotate(scaled_image, angle_degrees)
        
        # Get the rect for proper positioning
        rotated_rect = rotated_image.get_rect(center=center)
        
        # Blit the rotated image
        self.screen.blit(rotated_image, rotated_rect)
    
    def _draw_robot_rectangle(self, robot, center, screen_length, screen_width, color):
        """Draw robot using rectangle (fallback method)."""
        # Calculate robot corners in screen pixels
        half_length = screen_length * 0.5
        half_width = screen_width * 0.5
        
        # Local corners in screen space
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Rotate corners in screen space
        cos_theta = math.cos(robot.theta)
        sin_theta = math.sin(robot.theta)
        
        screen_corners = []
        for lx, ly in corners:
            # Rotate in screen space
            rotated_x = lx * cos_theta - ly * sin_theta
            rotated_y = lx * sin_theta + ly * cos_theta
            screen_corners.append((
                center[0] + rotated_x,
                center[1] + rotated_y
            ))
        
        # Convert to integer coordinates
        screen_corners = [(int(x), int(y)) for x, y in screen_corners]
        
        # Draw robot body
        pygame.draw.polygon(self.screen, color, screen_corners)
        pygame.draw.polygon(self.screen, Colors.BLACK, screen_corners, 2)
        
        # Draw direction arrow
        arrow_length = screen_length * 0.6
        arrow_end = (
            int(center[0] + arrow_length * cos_theta),
            int(center[1] + arrow_length * sin_theta)
        )
        
        pygame.draw.line(self.screen, Colors.BLACK, center, arrow_end, 3)
    
    def draw_collisions(self):
        """Draw collision indicators."""
        for collision in self.collision_results:
            if collision.collision_point:
                center = self.world_to_screen(collision.collision_point[0], collision.collision_point[1])
                
                # Skip if off screen
                if (center[0] < 0 or center[0] > self.config.width or
                    center[1] < 0 or center[1] > self.config.height):
                    continue
                
                # Draw collision marker
                marker_size = 8
                pygame.draw.line(self.screen, Colors.COLLISION_POINT,
                               (center[0] - marker_size, center[1] - marker_size),
                               (center[0] + marker_size, center[1] + marker_size), 3)
                pygame.draw.line(self.screen, Colors.COLLISION_POINT,
                               (center[0] - marker_size, center[1] + marker_size),
                               (center[0] + marker_size, center[1] - marker_size), 3)
    
    def draw_ui(self):
        """Draw user interface elements."""
        # Background for UI
        ui_height = 120
        pygame.draw.rect(self.screen, (0, 0, 0, 180), (0, 0, self.config.width, ui_height))
        
        y_offset = 10
        
        # FPS
        if self.config.show_fps:
            fps_text = f"FPS: {self.fps_display}"
            text = self.font.render(fps_text, True, Colors.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        # Robot count
        robot_text = f"Robots: {len(self.robots)}"
        text = self.font.render(robot_text, True, Colors.WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Collision count
        collision_text = f"Collisions: {len(self.collision_results)}"
        text = self.font.render(collision_text, True, Colors.WHITE)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Camera",
            "T: Toggle Trajectories",
            "P: Toggle Planned Paths",
            "W: Toggle Waypoints",
            "C: Clear Trajectories",
            "G: Toggle Grid",
            "I: Toggle Images/Colors",
            "=/+/-: Zoom In/Out",
            "Mouse: Pan",
            "ESC: Quit"
        ]
        
        x_offset = 200
        y_offset = 10
        for i, control in enumerate(controls):
            if i % 4 == 0 and i > 0:
                x_offset += 200
                y_offset = 10
            
            text = self.small_font.render(control, True, Colors.WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 15
        
        # Pause indicator
        if self.paused:
            pause_text = "PAUSED"
            text = self.font.render(pause_text, True, Colors.RED)
            text_rect = text.get_rect(center=(self.config.width // 2, 50))
            self.screen.blit(text, text_rect)
    
    def render(self) -> bool:
        """
        Render one frame.
        
        Returns:
            True if should continue, False to quit
        """
        # Handle events
        if not self.handle_events():
            return False
        
        # Update FPS display
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fps_update > 500:  # Update every 0.5 seconds
            self.fps_display = int(self.clock.get_fps())
            self.last_fps_update = current_time
        
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Draw map
        self.draw_map()
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw planned trajectories (behind actual trajectories)
        self.draw_planned_trajectories()
        
        # Draw trajectories
        self.draw_trajectories()
        
        # Draw current waypoint target
        self.draw_current_waypoint()
        
        # Draw robots
        self.draw_robots()
        
        # Draw collisions
        self.draw_collisions()
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        
        self.frame_count += 1
        return True
    
    def run_simulation_step(self, update_trajectories: bool = True):
        """Run one simulation step (update trajectories, etc.)."""
        if not self.paused and update_trajectories:
            self.update_trajectories()
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
    
    def is_running(self) -> bool:
        """Check if renderer is still running."""
        return self.running
    
    def get_camera_info(self) -> Dict[str, float]:
        """Get current camera information."""
        return {
            'camera_x': self.camera_x,
            'camera_y': self.camera_y,
            'zoom': self.zoom,
            'target_zoom': self.target_zoom
        }


def create_pygame_renderer(map_instance: Map, config: RenderConfig = None) -> PygameRenderer:
    """
    Factory function to create a pygame renderer.
    
    Args:
        map_instance: Map to render
        config: Renderer configuration
        
    Returns:
        PygameRenderer instance
    """
    return PygameRenderer(map_instance, config)
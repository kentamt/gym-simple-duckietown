#!/usr/bin/env python3

import pygame
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from enum import Enum

from duckietown_simulator.environment.duckietown_env import DuckietownEnv
from duckietown_simulator.rendering.pygame_renderer import PygameRenderer


class EditorMode(Enum):
    VIEW = "view"
    EDIT = "edit"
    DELETE = "delete"


@dataclass
class Waypoint:
    x: float
    y: float
    speed: float = 1.0
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class TrajectoryEditor:
    def __init__(self, env: DuckietownEnv, renderer: PygameRenderer):
        self.env = env
        self.renderer = renderer
        self.mode = EditorMode.VIEW
        self.current_trajectory: List[Waypoint] = []
        self.selected_waypoint_idx: Optional[int] = None
        self.dragging_waypoint = False
        self.waypoint_radius = 10
        self.trajectory_name = "unnamed_trajectory"
        
        # Colors
        self.waypoint_color = (255, 255, 0)  # Yellow
        self.selected_waypoint_color = (255, 100, 100)  # Red
        self.trajectory_color = (0, 255, 0)  # Green
        self.preview_color = (150, 150, 255)  # Light blue
        
        # UI state
        self.show_help = True
        self.mouse_pos = (0, 0)
        
    def handle_events(self, events) -> bool:
        """Handle mouse and keyboard events for trajectory editing"""
        continue_running = True
        
        for event in events:
            if event.type == pygame.QUIT:
                continue_running = False
                
            elif event.type == pygame.KEYDOWN:
                # Let renderer handle some basic keys
                if event.key in [pygame.K_r, pygame.K_PLUS, pygame.K_MINUS, pygame.K_EQUALS]:
                    # These are handled by renderer for camera controls
                    pass
                else:
                    continue_running = self._handle_keyboard(event)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - trajectory editing
                    self._handle_mouse_down(event)
                # Right click and middle click can be handled by renderer for pan/zoom
                
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    self._handle_mouse_up(event)
                
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
        
        # Also let renderer handle events for pan/zoom
        # We need to process events again for the renderer but filter out trajectory editing events
        renderer_continue = True
        for event in events:
            if event.type == pygame.QUIT:
                renderer_continue = False
            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_r, pygame.K_PLUS, pygame.K_MINUS, pygame.K_EQUALS]:
                # Handle basic renderer keys
                if event.key == pygame.K_r:
                    self.renderer.reset_camera()
                elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    self.renderer.zoom *= 1.1
                elif event.key == pygame.K_MINUS:
                    self.renderer.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in [2, 3]:  # Middle, right click
                # Handle pan/zoom
                pass
        
        return continue_running and renderer_continue
    
    def _handle_keyboard(self, event) -> bool:
        """Handle keyboard input"""
        key = event.key
        
        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_TAB:
            # Cycle through modes
            modes = list(EditorMode)
            current_idx = modes.index(self.mode)
            self.mode = modes[(current_idx + 1) % len(modes)]
            self.selected_waypoint_idx = None
            print(f"Switched to {self.mode.value} mode")
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_c:
            self.current_trajectory.clear()
            self.selected_waypoint_idx = None
            print("Cleared trajectory")
        elif key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._save_trajectory()
        elif key == pygame.K_o and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._load_trajectory()
        elif key == pygame.K_u and self.current_trajectory:
            # Undo last waypoint
            self.current_trajectory.pop()
            self.selected_waypoint_idx = None
            print("Removed last waypoint")
        
        return True
    
    def _handle_mouse_down(self, event):
        """Handle mouse button press"""
        if event.button == 1:  # Left click
            mouse_screen = pygame.mouse.get_pos()
            mouse_world = self.renderer.screen_to_world(mouse_screen[0], mouse_screen[1])
            
            if self.mode == EditorMode.EDIT:
                # Check if clicking on existing waypoint
                clicked_waypoint = self._get_waypoint_at_position(mouse_world)
                if clicked_waypoint is not None:
                    self.selected_waypoint_idx = clicked_waypoint
                    self.dragging_waypoint = True
                else:
                    # Add new waypoint
                    new_waypoint = Waypoint(x=mouse_world[0], y=mouse_world[1])
                    self.current_trajectory.append(new_waypoint)
                    print(f"Added waypoint at ({mouse_world[0]:.2f}, {mouse_world[1]:.2f})")
                    
            elif self.mode == EditorMode.DELETE:
                # Delete waypoint
                clicked_waypoint = self._get_waypoint_at_position(mouse_world)
                if clicked_waypoint is not None:
                    self.current_trajectory.pop(clicked_waypoint)
                    self.selected_waypoint_idx = None
                    print(f"Deleted waypoint {clicked_waypoint}")
            
            elif self.mode == EditorMode.VIEW:
                # Just select waypoint for viewing
                clicked_waypoint = self._get_waypoint_at_position(mouse_world)
                self.selected_waypoint_idx = clicked_waypoint
    
    def _handle_mouse_up(self, event):
        """Handle mouse button release"""
        if event.button == 1:  # Left click
            self.dragging_waypoint = False
    
    def _handle_mouse_motion(self, event):
        """Handle mouse movement"""
        self.mouse_pos = pygame.mouse.get_pos()
        
        if self.dragging_waypoint and self.selected_waypoint_idx is not None:
            # Update waypoint position
            mouse_world = self.renderer.screen_to_world(self.mouse_pos[0], self.mouse_pos[1])
            waypoint = self.current_trajectory[self.selected_waypoint_idx]
            waypoint.x = mouse_world[0]
            waypoint.y = mouse_world[1]
    
    def _get_waypoint_at_position(self, world_pos: Tuple[float, float]) -> Optional[int]:
        """Find waypoint index at given world position"""
        for i, waypoint in enumerate(self.current_trajectory):
            screen_pos = self.renderer.world_to_screen(waypoint.x, waypoint.y)
            distance = np.sqrt((screen_pos[0] - self.renderer.world_to_screen(world_pos[0], world_pos[1])[0])**2 + 
                              (screen_pos[1] - self.renderer.world_to_screen(world_pos[0], world_pos[1])[1])**2)
            if distance <= self.waypoint_radius:
                return i
        return None
    
    def render_trajectory(self, screen):
        """Render the current trajectory and waypoints"""
        if not self.current_trajectory:
            return
        
        # Draw trajectory lines
        if len(self.current_trajectory) > 1:
            points = []
            for waypoint in self.current_trajectory:
                screen_pos = self.renderer.world_to_screen(waypoint.x, waypoint.y)
                points.append(screen_pos)
            
            if len(points) > 1:
                pygame.draw.lines(screen, self.trajectory_color, False, points, 2)
        
        # Draw waypoints
        for i, waypoint in enumerate(self.current_trajectory):
            screen_pos = self.renderer.world_to_screen(waypoint.x, waypoint.y)
            
            # Choose color based on selection
            color = self.selected_waypoint_color if i == self.selected_waypoint_idx else self.waypoint_color
            
            # Draw waypoint circle
            pygame.draw.circle(screen, color, screen_pos, self.waypoint_radius)
            pygame.draw.circle(screen, (0, 0, 0), screen_pos, self.waypoint_radius, 2)
            
            # Draw waypoint number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i), True, (0, 0, 0))
            text_rect = text.get_rect(center=screen_pos)
            screen.blit(text, text_rect)
        
        # Draw preview waypoint in edit mode
        if self.mode == EditorMode.EDIT:
            mouse_world = self.renderer.screen_to_world(self.mouse_pos[0], self.mouse_pos[1])
            if self._get_waypoint_at_position(mouse_world) is None:
                pygame.draw.circle(screen, self.preview_color, self.mouse_pos, self.waypoint_radius // 2)
    
    def render_ui(self, screen):
        """Render UI elements"""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Mode indicator
        mode_text = font.render(f"Mode: {self.mode.value.upper()}", True, (255, 255, 255))
        screen.blit(mode_text, (10, 10))
        
        # Waypoint count
        count_text = small_font.render(f"Waypoints: {len(self.current_trajectory)}", True, (255, 255, 255))
        screen.blit(count_text, (10, 50))
        
        # Selected waypoint info
        if self.selected_waypoint_idx is not None:
            waypoint = self.current_trajectory[self.selected_waypoint_idx]
            info_text = small_font.render(f"Selected: #{self.selected_waypoint_idx} ({waypoint.x:.2f}, {waypoint.y:.2f})", 
                                        True, (255, 255, 255))
            screen.blit(info_text, (10, 75))
        
        # Help text
        if self.show_help:
            help_y = screen.get_height() - 200
            help_texts = [
                "Controls:",
                "TAB - Switch mode (View/Edit/Delete)",
                "Left Click - Add waypoint (Edit) / Select (View) / Delete (Delete)",
                "Drag - Move waypoint (Edit mode)",
                "C - Clear trajectory",
                "U - Undo last waypoint",
                "Ctrl+S - Save trajectory",
                "Ctrl+O - Load trajectory",
                "H - Toggle help",
                "ESC - Exit"
            ]
            
            for i, text in enumerate(help_texts):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                help_surface = small_font.render(text, True, color)
                screen.blit(help_surface, (10, help_y + i * 20))
    
    def _save_trajectory(self):
        """Save current trajectory to JSON file"""
        if not self.current_trajectory:
            print("No trajectory to save")
            return
        
        filename = f"{self.trajectory_name}.json"
        trajectory_data = {
            "name": self.trajectory_name,
            "waypoints": [wp.to_dict() for wp in self.current_trajectory]
        }
        
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Saved trajectory to {filename}")
    
    def _load_trajectory(self):
        """Load trajectory from JSON file"""
        filename = f"{self.trajectory_name}.json"
        
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return
        
        try:
            with open(filename, 'r') as f:
                trajectory_data = json.load(f)
            
            self.current_trajectory = [Waypoint.from_dict(wp) for wp in trajectory_data["waypoints"]]
            self.trajectory_name = trajectory_data.get("name", "loaded_trajectory")
            self.selected_waypoint_idx = None
            
            print(f"Loaded trajectory from {filename} ({len(self.current_trajectory)} waypoints)")
            
        except Exception as e:
            print(f"Error loading trajectory: {e}")
    
    def get_trajectory_for_robot(self) -> List[Tuple[float, float]]:
        """Convert trajectory to format compatible with PID controller"""
        return [(wp.x, wp.y) for wp in self.current_trajectory]


def main():
    """Demo of the trajectory editor"""
    # Initialize environment
    env = DuckietownEnv(
        map_config="duckietown_simulator/assets/maps/simple_room.json",
        max_steps=1000,
        render_mode="human"
    )
    
    # Initialize renderer
    from duckietown_simulator.rendering.pygame_renderer import RenderConfig
    config = RenderConfig(width=1200, height=800)
    renderer = PygameRenderer(
        map_instance=env.map,
        config=config
    )
    
    # Initialize trajectory editor
    editor = TrajectoryEditor(env, renderer)
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    print("Trajectory Editor Started!")
    print("Press H for help, TAB to switch modes")
    
    while running:
        # Handle events
        events = pygame.event.get()
        
        # Let editor handle all events (including renderer coordination)
        running = editor.handle_events(events)
        
        # Render
        renderer.render()
        screen = renderer.screen
        
        # Render trajectory editor elements
        editor.render_trajectory(screen)
        editor.render_ui(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
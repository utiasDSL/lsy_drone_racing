"""
Matplotlib-based 2D Top-Down Trajectory Visualizer (Fixed Version)

Features:
- 2D overhead view (top-down perspective)
- Interactive pan and zoom controls
- Screenshot functionality
- Visual distinction for detected vs undetected gates
- Clean, minimalist UI
- FIXED: Proper real gate position updates when detected
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Button, CheckButtons

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


class TrajectoryVisualizer:
    """Enhanced 2D top-down trajectory visualizer using Matplotlib."""
    
    # Visualization constants
    DEFAULT_FIGSIZE = (16, 10)
    TRAJECTORY_SAMPLES = 200
    GATE_ARROW_LENGTH = 0.3
    GATE_RADIUS = 0.3
    OBSTACLE_RADIUS = 0.1
    
    # Color scheme
    COLOR_WAYPOINTS = "#ec80cc"           # Pink - waypoints
    COLOR_TRAJECTORY = '#ff7f0e'          # Orange - trajectory
    COLOR_GATES_DETECTED = '#2ca02c'      # Green - detected gates
    COLOR_GATES_REAL = '#00ff00'          # Bright green - real detected position
    COLOR_GATES_UNDETECTED = '#d3d3d3'    # Light gray - undetected gates
    COLOR_OBSTACLES = '#7f7f7f'           # Gray - obstacles
    COLOR_DRONE = '#1f77b4'               # Blue - drone
    COLOR_WAYPOINT_ARROW = '#d62728'      # Red - changes
    
    def __init__(
        self, 
        width: int = None,
        height: int = None,
        figsize: tuple = None,
        title: str = "Drone Racing - 2D Top-Down View",
        output_dir: str = None
    ):
        """Initialize the visualizer.
        
        Args:
            width: Figure width in pixels.
            height: Figure height in pixels.
            figsize: Figure size as tuple (width, height) in inches.
            title: Window title.
            output_dir: Directory for saving screenshots.
                       If None, uses ~/repos/lsy_drone_racing_ws26/lsy_drone_racing/pics.
        """
        if figsize is not None:
            self.figsize = figsize
        elif width is not None and height is not None:
            self.figsize = (width / 100, height / 100)
        else:
            self.figsize = self.DEFAULT_FIGSIZE
            
        self.title = title
        self.fig: Optional[Figure] = None
        self.ax = None
        self.trajectory_data = {}
        
        # Output directory
        if output_dir is None:
            output_dir = Path.home() / "repos" / "lsy_drone_racing_ws26" / "lsy_drone_racing" / "pics"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Info] Screenshot directory set to: {self.output_dir}")
        
        # Plot objects
        self._drone_plot = None
        self._trajectory_plot = None
        self._waypoint_plots = []
        self._gate_plots = []
        self._gate_detected_status = []  # Track which gates are detected
        self._gate_real_positions = None  # Track real gate positions when detected
        self._gate_real_plots = []  # Plot objects for real gate positions
        self._waypoint_change_arrows = []
        
        
        # Display toggles
        self._show_waypoints = False
        self._show_trajectory = True
        self._show_gates = True
        
        # View state for pan/zoom
        self._initial_xlim = None
        self._initial_ylim = None
        self._pan_start = None
        self._press_event = None
        
        # UI control references
        self.check = None
        self.btn_screenshot = None
        self.btn_reset = None
        
    
    def create_figure(self) -> tuple[Figure, plt.Axes]:
        """Create matplotlib figure with 2D axes and controls."""
        plt.ion()
        
        # Create figure without toolbar
        fig = plt.figure(figsize=self.figsize)
        fig.canvas.toolbar_visible = False
        
        # Main 2D plot - adjust for sidebar
        ax = fig.add_subplot(111, position=[0.05, 0.05, 0.75, 0.9])
        
        # Set equal aspect ratio for proper visualization
        ax.set_aspect('equal', adjustable='box')
        
        # Connect mouse events for pan and zoom
        self._connect_mouse_controls(fig, ax)
        
        # Styling
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('X (m)', fontsize=11, labelpad=8)
        ax.set_ylabel('Y (m)', fontsize=11, labelpad=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Mouse control hint
        fig.text(0.02, 0.97, 
                 'Mouse: Left/Right=Pan | Scroll=Zoom',
                 fontsize=9, style='italic', alpha=0.6,
                 verticalalignment='top')
        
        # Add sidebar controls
        self._add_sidebar_controls(fig)
        
        return fig, ax
    
    def _connect_mouse_controls(self, fig: Figure, ax: plt.Axes) -> None:
        """Connect mouse events for interactive pan and zoom."""
        
        # Scroll wheel zoom
        def on_scroll(event):
            if event.inaxes == ax:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                zoom_factor = 1.1 if event.button == 'down' else 0.9
                
                # Zoom relative to mouse position
                x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
                y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
                
                x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
                y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
                
                ax.set_xlim([x_center - x_range, x_center + x_range])
                ax.set_ylim([y_center - y_range, y_center + y_range])
                
                fig.canvas.draw_idle()
        
        # Pan with mouse drag (both left and right buttons)
        def on_press(event):
            if event.inaxes == ax and event.button in [1, 3]:  # Left or right button
                self._press_event = event
                self._pan_start = (event.xdata, event.ydata)
        
        def on_release(event):
            self._press_event = None
            self._pan_start = None
        
        def on_motion(event):
            if self._press_event is None or self._pan_start is None:
                return
            if event.inaxes == ax and event.xdata and event.ydata:
                dx = event.xdata - self._pan_start[0]
                dy = event.ydata - self._pan_start[1]
                
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
                ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
                
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    def _add_sidebar_controls(self, fig: Figure) -> None:
        """Add integrated control panel in sidebar."""
        sidebar_x = 0.82
        sidebar_width = 0.16
        
        # === Display Checkboxes ===
        ax_check = plt.axes([sidebar_x, 0.55, sidebar_width, 0.25])
        ax_check.set_title('Display', fontsize=11, fontweight='bold', loc='left', pad=8)
        
        labels = ['Trajectory', 'Waypoints', 'Gates']
        visibility = [self._show_trajectory, self._show_waypoints, self._show_gates]
        self.check = CheckButtons(ax_check, labels, visibility)
        
        def toggle_visibility(label):
            try:
                if label == 'Trajectory':
                    self._show_trajectory = not self._show_trajectory
                    if self._trajectory_plot is not None:
                        self._trajectory_plot.set_visible(self._show_trajectory)
                elif label == 'Waypoints':
                    self._show_waypoints = not self._show_waypoints
                    if self._waypoint_plots:
                        for plot in self._waypoint_plots:
                            if plot is not None:
                                plot.set_visible(self._show_waypoints)
                    if self._waypoint_change_arrows:
                        for arrow in self._waypoint_change_arrows:
                            if arrow is not None:
                                arrow.set_visible(self._show_waypoints)
                elif label == 'Gates':
                    self._show_gates = not self._show_gates
                    if self._gate_plots:
                        for gate_plot in self._gate_plots:
                            if isinstance(gate_plot, tuple) and len(gate_plot) == 2:
                                circle, arrow = gate_plot
                                if circle is not None:
                                    circle.set_visible(self._show_gates)
                                if arrow is not None:
                                    arrow.set_visible(self._show_gates)
                    if hasattr(self, '_gate_real_plots') and self._gate_real_plots:
                        for plot in self._gate_real_plots:
                            if plot is not None:
                                plot.set_visible(self._show_gates)
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"[Error] Toggle visibility failed for {label}: {e}")
        
        self.check.on_clicked(toggle_visibility)
        
        # === Screenshot Button ===
        ax_screenshot = plt.axes([sidebar_x + 0.01, 0.30, sidebar_width - 0.02, 0.05])
        self.btn_screenshot = Button(ax_screenshot, 'Screenshot', color='lightblue', hovercolor='blue')
        
        def save_screenshot(event):
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"screenshot_{timestamp}.png"
                self.save_image(filepath, dpi=150)
                print(f"✓ Screenshot saved: {filepath}")
            except Exception as e:
                print(f"[Error] Screenshot failed: {e}")
        
        self.btn_screenshot.on_clicked(save_screenshot)
        
        # === Reset View Button ===
        ax_reset = plt.axes([sidebar_x + 0.01, 0.23, sidebar_width - 0.02, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset View', color='lightcoral', hovercolor='red')
        
        def reset_view(event):
            try:
                if self._initial_xlim is not None and self._initial_ylim is not None:
                    self.ax.set_xlim(self._initial_xlim)
                    self.ax.set_ylim(self._initial_ylim)
                    fig.canvas.draw_idle()
                else:
                    print("[Warning] Initial view limits not set yet")
            except Exception as e:
                print(f"[Error] Reset view failed: {e}")
        
        self.btn_reset.on_clicked(reset_view)
        
        # === Instructions ===
        instructions = (
            "Controls:\n"
            "• Pan: Click & drag\n"
            "• Zoom: Scroll wheel\n"
            "• Reset: Button or 'r'\n"
            "\nDisplay:\n"
            "• Green: Detected\n"
            "• Gray: Undetected\n"
            "• Bright Green: Real pos"
        )
        
        ax_info = plt.axes([sidebar_x, 0.02, sidebar_width, 0.18])
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, instructions, fontsize=8,
                    verticalalignment='top', 
                    fontfamily='monospace',
                    color='gray',
                    transform=ax_info.transAxes)
    
    def add_obstacles(
        self,
        ax: plt.Axes,
        obstacle_positions: NDArray[np.floating],
        color: Optional[str] = None
    ) -> None:
        """Add obstacles to 2D plot."""
        if color is None:
            color = self.COLOR_OBSTACLES
        
        for pos in obstacle_positions:
            circle = Circle(
                (pos[0], pos[1]),
                self.OBSTACLE_RADIUS,
                color=color,
                fill=True,
                alpha=0.5,
                edgecolor='black',
                linewidth=1,
                zorder=2
            )
            ax.add_patch(circle)
        
        # Add to legend
        ax.plot([], [], 'o', color=color, markersize=8,
                label='Obstacles', alpha=0.5)
    
    def add_drone(
        self,
        ax: plt.Axes,
        drone_position: NDArray[np.floating],
        color: Optional[str] = None
    ) -> None:
        """Add drone marker to 2D plot."""
        if color is None:
            color = self.COLOR_DRONE
        
        self._drone_plot = ax.scatter(
            drone_position[0],
            drone_position[1],
            color=color,
            s=200,
            marker='^',
            edgecolors='black',
            linewidths=2,
            label='Drone',
            zorder=20
        )
    
    def add_trajectory(
        self,
        ax: plt.Axes,
        trajectory: CubicSpline,
        trajectory_duration: float,
        color: Optional[str] = None
    ) -> None:
        """Add trajectory line to 2D plot."""
        if color is None:
            color = self.COLOR_TRAJECTORY
        
        t_samples = np.linspace(0, trajectory_duration, self.TRAJECTORY_SAMPLES)
        traj_points = trajectory(t_samples)
        
        # Store trajectory data
        self.trajectory_data = {
            'time': t_samples.tolist(),
            'positions': traj_points.tolist()
        }
        
        # Plot XY coordinates only (top-down view)
        self._trajectory_plot, = ax.plot(
            traj_points[:, 0],
            traj_points[:, 1],
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label='Trajectory',
            zorder=5
        )
    
    def add_waypoints(
        self,
        ax: plt.Axes,
        waypoints: NDArray[np.floating],
        color: Optional[str] = None
    ) -> None:
        """Add waypoint markers to 2D plot."""
        if color is None:
            color = self.COLOR_WAYPOINTS
        
        # Plot XY coordinates
        plot = ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            color=color,
            s=100,
            marker='o',
            edgecolors='black',
            linewidths=1.5,
            label='Waypoints',
            alpha=0.7,
            zorder=10
        )
        
        self._waypoint_plots.append(plot)
        
        # Initially hide waypoints
        plot.set_visible(self._show_waypoints)
    
    def add_waypoint_changes(
        self,
        ax: plt.Axes,
        old_waypoints: NDArray[np.floating],
        new_waypoints: NDArray[np.floating],
        color: Optional[str] = None
    ) -> None:
        """Add arrows showing waypoint changes."""
        if color is None:
            color = self.COLOR_WAYPOINT_ARROW
        
        # Clear old arrows
        for arrow in self._waypoint_change_arrows:
            arrow.remove()
        self._waypoint_change_arrows.clear()
        
        # Add new arrows for changed waypoints
        min_len = min(len(old_waypoints), len(new_waypoints))
        for i in range(min_len):
            old_pos = old_waypoints[i][:2]  # XY only
            new_pos = new_waypoints[i][:2]
            
            distance = np.linalg.norm(new_pos - old_pos)
            if distance > 0.01:  # Significant change
                arrow = FancyArrowPatch(
                    old_pos, new_pos,
                    arrowstyle='->,head_width=0.3,head_length=0.3',
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    zorder=15
                )
                ax.add_patch(arrow)
                self._waypoint_change_arrows.append(arrow)
                arrow.set_visible(self._show_waypoints)
    
    def add_gates(
        self,
        ax: plt.Axes,
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        detected_status: Optional[NDArray[np.bool_]] = None
    ) -> None:
        """Add gates to 2D plot with detection status."""
        # If no detection status provided, assume all undetected
        if detected_status is None:
            detected_status = np.zeros(len(gate_positions), dtype=bool)
        
        self._gate_detected_status = list(detected_status)  # Convert to list for easier updates
        
        for i, (pos, normal) in enumerate(zip(gate_positions, gate_normals)):
            is_detected = detected_status[i]
            color = self.COLOR_GATES_DETECTED if is_detected else self.COLOR_GATES_UNDETECTED
            
            # Draw gate as circle (top-down view)
            circle = Circle(
                (pos[0], pos[1]),
                self.GATE_RADIUS,
                color=color,
                fill=True,
                alpha=0.3,
                edgecolor=color,
                linewidth=2,
                zorder=3
            )
            ax.add_patch(circle)
            
            # Draw arrow showing gate direction (normal vector)
            arrow_end = pos[:2] + normal[:2] * self.GATE_ARROW_LENGTH
            arrow = FancyArrowPatch(
                pos[:2], arrow_end,
                arrowstyle='->,head_width=0.15,head_length=0.15',
                color=color,
                linewidth=2.5,
                alpha=0.8,
                zorder=4
            )
            ax.add_patch(arrow)
            
            # Add gate number
            text = ax.text(
                pos[0], pos[1],
                str(i + 1),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                color='white' if is_detected else 'gray',
                zorder=5
            )
            
            self._gate_plots.append((circle, arrow, text))  # Also store text object
        
        # Add legend entries
        ax.plot([], [], 'o', color=self.COLOR_GATES_DETECTED, 
                markersize=10, label='Gates (Detected)', alpha=0.7)
        ax.plot([], [], 'o', color=self.COLOR_GATES_UNDETECTED, 
                markersize=10, label='Gates (Undetected)', alpha=0.5)
        ax.plot([], [], 'o', color=self.COLOR_GATES_REAL, 
                markersize=10, markerfacecolor='none', markeredgewidth=2,
                linestyle='--', label='Real Position', alpha=0.9)
    
    def update_gate_detection(
        self,
        gate_index: int,
        is_detected: bool,
        real_position: Optional[NDArray[np.floating]] = None
    ) -> None:
        """Update detection status of a specific gate and optionally its real position.
        
        Args:
            gate_index: Index of the gate to update.
            is_detected: Whether the gate has been detected.
            real_position: Real position of the gate (if different from initial estimate).
                          If provided, will display the real position when detected.
        """
        if gate_index >= len(self._gate_detected_status):
            return
        
        self._gate_detected_status[gate_index] = is_detected
        
        # Update gate color and text
        if gate_index < len(self._gate_plots):
            circle, arrow, text = self._gate_plots[gate_index]
            color = self.COLOR_GATES_DETECTED if is_detected else self.COLOR_GATES_UNDETECTED
            
            circle.set_color(color)
            circle.set_edgecolor(color)
            arrow.set_color(color)
            text.set_color('white' if is_detected else 'gray')
        
        # Update real position if provided and detected
        if is_detected and real_position is not None:
            self._update_gate_real_position(gate_index, real_position)
        
        self.fig.canvas.draw_idle()
    
    def _update_gate_real_position(
        self,
        gate_index: int,
        real_position: NDArray[np.floating]
    ) -> None:
        """Display the real position of a detected gate.
        
        Args:
            gate_index: Index of the gate.
            real_position: Real position of the gate [x, y, z].
        """
        if self.ax is None:
            return
        
        # Store real position
        if self._gate_real_positions is None:
            # Initialize array to store real positions
            num_gates = len(self._gate_detected_status)
            self._gate_real_positions = np.full((num_gates, 3), np.nan)
        
        self._gate_real_positions[gate_index] = real_position
        
        # Remove old real position marker if exists
        if gate_index < len(self._gate_real_plots) and self._gate_real_plots[gate_index] is not None:
            for obj in self._gate_real_plots[gate_index]:
                if obj is not None:
                    obj.remove()
        
        # Ensure list is long enough
        while len(self._gate_real_plots) <= gate_index:
            self._gate_real_plots.append(None)
        
        # Add marker for real position (bright green circle with cross)
        circle = Circle(
            (real_position[0], real_position[1]),
            self.GATE_RADIUS * 0.8,  # Slightly smaller
            color=self.COLOR_GATES_REAL,
            fill=False,
            edgecolor=self.COLOR_GATES_REAL,
            linewidth=3,
            linestyle='--',
            zorder=6
        )
        self.ax.add_patch(circle)
        
        # Add cross marker at center
        cross_size = self.GATE_RADIUS * 0.3
        cross_x = [real_position[0] - cross_size, real_position[0] + cross_size]
        cross_y = [real_position[1], real_position[1]]
        line1, = self.ax.plot(cross_x, cross_y, color=self.COLOR_GATES_REAL, 
                              linewidth=2, zorder=7)
        
        cross_x = [real_position[0], real_position[0]]
        cross_y = [real_position[1] - cross_size, real_position[1] + cross_size]
        line2, = self.ax.plot(cross_x, cross_y, color=self.COLOR_GATES_REAL, 
                              linewidth=2, zorder=7)
        
        # Store all objects
        self._gate_real_plots[gate_index] = (circle, line1, line2)
        
        print(f"[VISUALIZER] Gate {gate_index + 1} real position updated to: "
              f"[{real_position[0]:.3f}, {real_position[1]:.3f}, {real_position[2]:.3f}]")
    
    def visualize(
        self,
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        obstacle_positions: Optional[NDArray[np.floating]] = None,
        trajectory: Optional[CubicSpline] = None,
        trajectory_duration: float = 15.0,
        waypoints: Optional[NDArray[np.floating]] = None,
        drone_position: Optional[NDArray[np.floating]] = None,
        gate_detected_status: Optional[NDArray[np.bool_]] = None,
        gate_real_positions: Optional[NDArray[np.floating]] = None,
        show: bool = True
    ) -> tuple[Figure, plt.Axes]:
        """Create complete visualization.
        
        Args:
            gate_positions: Gate positions array.
            gate_normals: Gate normal vectors.
            obstacle_positions: Obstacle positions.
            trajectory: Trajectory spline.
            trajectory_duration: Duration of trajectory.
            waypoints: Waypoint positions.
            drone_position: Current drone position.
            gate_detected_status: Gate detection status.
            gate_real_positions: Real gate positions (for already detected gates).
            show: Whether to display immediately.
        
        Returns:
            Tuple of (Figure, Axes).
        """
        # Create figure if needed
        if self.fig is None:
            self.fig, self.ax = self.create_figure()
        
        # Clear existing plots
        self.ax.clear()
        self._waypoint_plots.clear()
        self._gate_plots.clear()
        self._gate_real_plots.clear()
        self._waypoint_change_arrows.clear()
        
        # Add trajectory
        if trajectory is not None:
            self.add_trajectory(self.ax, trajectory, trajectory_duration)
        
        # Add waypoints
        if waypoints is not None:
            self.add_waypoints(self.ax, waypoints)
        
        # Add gates with detection status
        self.add_gates(self.ax, gate_positions, gate_normals, gate_detected_status)
        
        # Add real gate positions if provided
        if gate_real_positions is not None and gate_detected_status is not None:
            for i, (is_detected, real_pos) in enumerate(zip(gate_detected_status, gate_real_positions)):
                if is_detected and not np.any(np.isnan(real_pos)):
                    self._update_gate_real_position(i, real_pos)
        
        # Add obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            self.add_obstacles(self.ax, obstacle_positions)
        
        # Add drone
        if drone_position is not None:
            self.add_drone(self.ax, drone_position)
        
        # Set initial view limits and save them
        self._set_equal_aspect(self.ax)
        self._initial_xlim = self.ax.get_xlim()
        self._initial_ylim = self.ax.get_ylim()
        
        # Legend
        self.ax.legend(loc='upper left', fontsize=9, framealpha=0.85, 
                      edgecolor='gray', fancybox=True)
        
        if show:
            plt.draw()
            plt.pause(0.001)
        
        return self.fig, self.ax
    
    def update(
        self,
        drone_position: Optional[NDArray[np.floating]] = None,
        trajectory: Optional[CubicSpline] = None,
        trajectory_duration: float = 15.0,
        waypoints: Optional[NDArray[np.floating]] = None,
        old_waypoints: Optional[NDArray[np.floating]] = None,
        gate_detected_status: Optional[NDArray[np.bool_]] = None,
        gate_real_positions: Optional[NDArray[np.floating]] = None
    ) -> None:
        """Update visualization.
        
        Args:
            drone_position: Current drone position [x, y, z].
            trajectory: Updated trajectory spline.
            trajectory_duration: Duration of trajectory.
            waypoints: Updated waypoints.
            old_waypoints: Previous waypoints (for showing changes).
            gate_detected_status: Array of gate detection status.
            gate_real_positions: Array of real gate positions [N, 3] where N is number of gates.
                                First dimension corresponds to gate index.
                                Use np.nan for gates where real position is unknown.
        """
        if self.fig is None or self.ax is None:
            return
        
        # Update drone position
        if drone_position is not None and self._drone_plot is not None:
            self._drone_plot.set_offsets([[drone_position[0], drone_position[1]]])
        
        # Update trajectory
        if trajectory is not None and self._trajectory_plot is not None:
            t_samples = np.linspace(0, trajectory_duration, self.TRAJECTORY_SAMPLES)
            traj_points = trajectory(t_samples)
            
            self._trajectory_plot.set_data(
                traj_points[:, 0],
                traj_points[:, 1]
            )
        
        # Update waypoints with arrows
        if waypoints is not None and old_waypoints is not None:
            self.add_waypoint_changes(self.ax, old_waypoints, waypoints)
        
        # Update gate detection status and real positions
        if gate_detected_status is not None:
            for i, is_detected in enumerate(gate_detected_status):
                # Check if detection status changed
                if i < len(self._gate_detected_status) and is_detected != self._gate_detected_status[i]:
                    # Get real position if provided
                    real_pos = None
                    if gate_real_positions is not None and i < len(gate_real_positions):
                        pos = gate_real_positions[i]
                        if not np.any(np.isnan(pos)):
                            real_pos = pos
                            print(f"[UPDATE] Gate {i+1} detected with real position: "
                                  f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    
                    self.update_gate_detection(i, is_detected, real_pos)
                    
                # Also check if real position is provided even if detection status hasn't changed
                elif is_detected and gate_real_positions is not None and i < len(gate_real_positions):
                    pos = gate_real_positions[i]
                    if not np.any(np.isnan(pos)):
                        # Check if we haven't already displayed this real position
                        if self._gate_real_positions is None or \
                           i >= len(self._gate_real_positions) or \
                           np.any(np.isnan(self._gate_real_positions[i])):
                            self._update_gate_real_position(i, pos)
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _set_equal_aspect(self, ax: plt.Axes) -> None:
        """Set equal aspect ratio and appropriate limits."""
        # Get all plotted data bounds
        x_data = []
        y_data = []
        
        for line in ax.get_lines():
            x_data.extend(line.get_xdata())
            y_data.extend(line.get_ydata())
        
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                x_data.extend(offsets[:, 0])
                y_data.extend(offsets[:, 1])
        
        if x_data and y_data:
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
            
            # Add margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            ax.set_xlim([x_min - x_margin, x_max + x_margin])
            ax.set_ylim([y_min - y_margin, y_max + y_margin])
    
    def save_image(self, filepath: str | Path, dpi: int = 150, transparent: bool = False) -> None:
        """Save current view as image."""
        if self.fig is None:
            raise RuntimeError("No figure to save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.savefig(str(filepath), dpi=dpi, bbox_inches='tight', transparent=transparent)
        print(f"✓ Saved: {filepath}")
    
    def save_html(self, filepath: str | Path, include_plotlyjs: bool = True, auto_open: bool = False) -> None:
        """Compatibility method - saves as PNG instead."""
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.html':
            filepath = filepath.with_suffix('.png')
        self.save_image(filepath)
    
    def save_trajectory_data(self, filepath: str | Path, format: str = 'json') -> None:
        """Save trajectory data to file."""
        if not self.trajectory_data:
            raise RuntimeError("No trajectory data available.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.trajectory_data, f, indent=2)
        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'z'])
                for t, pos in zip(self.trajectory_data['time'], self.trajectory_data['positions']):
                    writer.writerow([t, pos[0], pos[1], pos[2]])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✓ Saved: {filepath}")
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)


def quick_visualize(
    gate_positions: NDArray[np.floating],
    gate_normals: NDArray[np.floating],
    obstacle_positions: Optional[NDArray[np.floating]] = None,
    trajectory: Optional[CubicSpline] = None,
    trajectory_duration: float = 15.0,
    waypoints: Optional[NDArray[np.floating]] = None,
    drone_position: Optional[NDArray[np.floating]] = None,
    gate_detected_status: Optional[NDArray[np.bool_]] = None,
    save_path: Optional[str | Path] = None,
    output_dir: Optional[str] = None
) -> tuple[Figure, plt.Axes]:
    """Quick visualization function.
    
    Args:
        gate_positions: Gate positions array.
        gate_normals: Gate normal vectors.
        obstacle_positions: Obstacle positions.
        trajectory: Trajectory spline.
        trajectory_duration: Duration of trajectory.
        waypoints: Waypoint positions.
        drone_position: Current drone position.
        gate_detected_status: Gate detection status.
        save_path: Path to save visualization image.
        output_dir: Directory for saving outputs.
    """
    viz = TrajectoryVisualizer(output_dir=output_dir)
    fig, ax = viz.visualize(
        gate_positions=gate_positions,
        gate_normals=gate_normals,
        obstacle_positions=obstacle_positions,
        trajectory=trajectory,
        trajectory_duration=trajectory_duration,
        waypoints=waypoints,
        drone_position=drone_position,
        gate_detected_status=gate_detected_status
    )
    
    if save_path is not None:
        viz.save_image(save_path)
    
    return fig, ax
"""Controller that follows a pre-defined trajectory with minimal adaptive re-planning.

Uses a cubic spline through hard-coded waypoints. When the sensor reveals that a gate's
actual position deviates significantly from nominal, the closest waypoint is nudged by the
same offset and the spline is rebuilt. All other waypoints stay unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_REPLAN_THRESHOLD = 0.05

# Nominal gate centers (x, y, z) matching level-2 track layout.
_NOMINAL_GATE_POS = np.array([
    [0.5,  0.25, 0.7],
    [1.05, 0.75, 1.2],
    [-1.0, -0.25, 0.7],
    [0.0,  -0.75, 1.2],
], dtype=np.float64)


class StateController(Controller):
    """State controller following a pre-defined trajectory with gate-nudge re-planning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        self._base_waypoints = np.array([
            [-1.5,  0.75, 0.05],
            [-1.0,  0.55, 0.4],
            [0.3,   0.35, 0.7],   # near gate 0
            [1.3,  -0.15, 0.9],
            [0.85,  0.85, 1.2],   # near gate 1
            [-0.5, -0.05, 0.7],
            [-1.2, -0.2,  0.8],   # near gate 2
            [-1.2, -0.2,  1.2],
            [-0.0, -0.7,  1.2],   # near gate 3
            [0.5,  -0.75, 1.2],
        ], dtype=np.float64)

        # Which waypoint index corresponds to each gate (closest to nominal gate pos).
        self._gate_wp_idx = [
            int(np.argmin(np.linalg.norm(self._base_waypoints - g, axis=1)))
            for g in _NOMINAL_GATE_POS
        ]

        self._waypoints = self._base_waypoints.copy()
        self._planned_gates_pos = np.array(obs["gates_pos"], dtype=np.float64)
        self._replanned_for: set[int] = set()

        self._t_total = 15.0
        self._des_pos_spline: CubicSpline | None = None
        self._build_spline()
        self._tick = 0
        self._finished = False

    def _build_spline(self) -> None:
        t = np.linspace(0, self._t_total, len(self._waypoints))
        self._des_pos_spline = CubicSpline(t, self._waypoints)

    def _check_and_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        target_gate = int(obs["target_gate"])
        if target_gate < 0 or target_gate in self._replanned_for:
            return
        new_pos = np.asarray(obs["gates_pos"][target_gate], dtype=np.float64)
        delta = new_pos - self._planned_gates_pos[target_gate]
        if np.linalg.norm(delta) > _REPLAN_THRESHOLD:
            idx = self._gate_wp_idx[target_gate]
            self._waypoints[idx] += delta  # nudge only the matching waypoint
            self._planned_gates_pos[target_gate] = new_pos
            self._replanned_for.add(target_gate)
            self._build_spline()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True
        self._check_and_replan(obs)
        des_pos = self._des_pos_spline(t)
        return np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._finished = False
        self._waypoints = self._base_waypoints.copy()
        self._replanned_for = set()
        self._build_spline()

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

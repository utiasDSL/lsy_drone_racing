"""Gate-traversing state controller with verbose phase-2 debugging."""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

TAKEOFF_HEIGHT = 0.5
TAKEOFF_TIME   = 3.0

class MyController(Controller):

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq   = config.env.freq
        self._tick   = 0
        self._n_gates = len(obs["gates_pos"])
        self._gate_pos    = obs["gates_pos"].copy()
        self._gate_quat   = obs["gates_quat"].copy()
        self._gates_visited = obs["gates_visited"].copy()
        self._prev_target_gate = int(obs["target_gate"])
        self._spline = None
        self._spline_vel = None
        self._t_total = 0.0
        self._t_start_tick = 0

        print("\n========== INIT ==========")
        print(f"  n_gates      : {self._n_gates}")
        print(f"  start pos    : {obs['pos']}")
        print(f"  gate_pos     :\n{self._gate_pos}")
        print(f"  gates_visited: {self._gates_visited}")
        print(f"  target_gate  : {obs['target_gate']}")
        print(f"  freq         : {self._freq}")
        print("==========================\n")

    def _build_spline(self, start_pos: np.ndarray, label: str = "") -> None:
        waypoints = np.vstack([start_pos, self._gate_pos])
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        dists = np.maximum(dists, 0.01)
        t_knots = np.concatenate([[0.0], np.cumsum(dists * 4.0)])
        self._t_total = t_knots[-1]
        # Force zero velocity at the start — don't inherit drone's current velocity.
        # This prevents the spline from dipping below the starting height.
        self._spline = CubicSpline(
            t_knots, waypoints,
            bc_type=((1, np.zeros(3)), (1, np.zeros(3)))
        )
        self._spline_vel = self._spline.derivative()
        self._t_start_tick = self._tick

        print(f"\n--- _build_spline [{label}] at tick={self._tick} ---")
        print(f"  start_pos : {np.round(start_pos, 3)}")
        print(f"  waypoints :\n{np.round(waypoints, 3)}")
        print(f"  dists     : {np.round(dists, 3)}")
        print(f"  t_knots   : {np.round(t_knots, 3)}")
        print(f"  t_total   : {self._t_total:.3f}s")
        # Sanity check: sample first few spline points
        for ts in np.linspace(0, min(1.0, self._t_total), 5):
            p = self._spline(ts)
            v = self._spline_vel(ts)
            print(f"    t={ts:.2f} → pos={np.round(p,3)}  vel={np.round(v,3)}")
        print()

    def _update_gates(self, obs: dict) -> bool:
        new_visited = obs["gates_visited"]
        rebuild = False
        for i in range(self._n_gates):
            if new_visited[i] and not self._gates_visited[i]:
                print(f"  [gate update] gate {i} now visited: "
                      f"{self._gate_pos[i]} → {obs['gates_pos'][i]}")
                self._gate_pos[i] = obs["gates_pos"][i]
                self._gate_quat[i] = obs["gates_quat"][i]
                self._gates_visited[i] = True
                rebuild = True
        return rebuild

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        pos = obs["pos"]
        vel = obs["vel"]
        t   = self._tick / self._freq
        current_gate = int(obs["target_gate"])

        # ── Phase 1: takeoff ──────────────────────────────────────────
        if t < TAKEOFF_TIME:
            frac      = t / TAKEOFF_TIME
            target_z  = TAKEOFF_HEIGHT * frac
            des_vel_z = TAKEOFF_HEIGHT / TAKEOFF_TIME
            acc_z     = 3.0 * (target_z - pos[2]) + 1.5 * (des_vel_z - vel[2])
            return np.array([
                pos[0], pos[1], target_z,
                0.0, 0.0, des_vel_z,
                0.0, 0.0, float(np.clip(acc_z, -10, 10)),
                0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)

        # ── Phase 2: first tick — build spline once ───────────────────
        if self._spline is None:
            print(f"\n>>> Entering phase 2 at tick={self._tick}, t={t:.3f}")
            print(f"    pos={np.round(pos,3)}  vel={np.round(vel,3)}")
            self._build_spline(pos, label="phase2_init")

        # ── Phase 2: gate/sensor updates ─────────────────────────────
        if self._update_gates(obs):
            self._build_spline(pos, label="gate_visited")

        if current_gate != self._prev_target_gate:
            print(f"\n>>> Gate passed! {self._prev_target_gate} → {current_gate}")
            self._build_spline(pos, label="gate_passed")
            self._prev_target_gate = current_gate

        # ── Evaluate spline ───────────────────────────────────────────
        t_elapsed = (self._tick - self._t_start_tick) / self._freq
        t_sp = min(t_elapsed, self._t_total)
        des_pos = self._spline(t_sp)
        des_vel = self._spline_vel(t_sp)

        kp  = np.array([1.0, 1.0, 1.5])
        kd  = np.array([0.5, 0.5, 0.8])
        acc = kp * (des_pos - pos) + kd * (des_vel - vel)
        acc = np.clip(acc, -15.0, 15.0)

        target_pos = self._gate_pos[min(current_gate, self._n_gates - 1)]
        delta      = target_pos - pos
        des_yaw    = float(np.arctan2(delta[1], delta[0]))

        # ── Per-tick debug (every 20 ticks = 0.2s at 100Hz) ──────────
        if self._tick % 20 == 0:
            print(f"  [t={t:.2f} sp={t_sp:.2f}] gate={current_gate} "
                  f"pos={np.round(pos,3)} → des={np.round(des_pos,3)} "
                  f"vel={np.round(vel,3)} des_vel={np.round(des_vel,3)} "
                  f"acc={np.round(acc,3)}")

        return np.array([
            des_pos[0], des_pos[1], des_pos[2],
            des_vel[0], des_vel[1], des_vel[2],
            acc[0],     acc[1],     acc[2],
            des_yaw,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return terminated or truncated

    def episode_callback(self):
        self._tick = 0
        self._spline = None
        self._spline_vel = None
        self._t_total = 0.0
        self._t_start_tick = 0
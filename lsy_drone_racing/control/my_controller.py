"""Gate-traversing state controller with verbose phase-2 debugging."""
from __future__ import annotations
from typing import TYPE_CHECKING

import heapq
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

        self._pos_integral = np.zeros(3)
        self._last_tick = 0

        self._subgoal_stage = 0  # 0=approach, 1=center, 2=exit
        self._current_goal = None
        print(obs.keys())

        print("\n========== INIT ==========")
        print(f"  n_gates      : {self._n_gates}")
        print(f"  start pos    : {obs['pos']}")
        print(f"  gate_pos     :\n{self._gate_pos}")
        print(f"  gates_visited: {self._gates_visited}")
        print(f"  target_gate  : {obs['target_gate']}")
        print(f"  freq         : {self._freq}")
        print("==========================\n")

    def _build_occupancy_grid(self, obs, resolution=0.2, margin=4.0):
        obstacles = obs["obstacles_pos"]

        # Bounds
        all_points = np.vstack([self._gate_pos, obstacles])
        min_bounds = np.min(all_points, axis=0) - 1.0
        max_bounds = np.max(all_points, axis=0) + 1.0

        self._grid_origin = min_bounds
        self._grid_res = resolution

        grid_size = np.ceil((max_bounds - min_bounds) / resolution).astype(int)
        grid = np.zeros(grid_size, dtype=bool)

        inflate = int(margin / resolution)

        for obstacle in obstacles:

            idx = ((obstacle - min_bounds) / resolution).astype(int).flatten()

            # safety clamp
            idx = np.clip(idx, 0, np.array(grid.shape) - 1)

            for dx in range(-inflate, inflate+1):
                for dy in range(-inflate, inflate+1):

                    # circular mask
                    if dx*dx + dy*dy > inflate*inflate:
                        continue

                    # block full vertical column (cylinder)
                    for dz in range(grid.shape[2]):
                        i = idx + np.array([dx, dy, dz])

                        # print(f"Marking obstacle at grid {i} (world {self._grid_origin + i * self._grid_res})")

                        if np.all(i >= 0) and np.all(i < grid.shape):
                            grid[tuple(i)] = True

        # for i in range(self._n_gates):
        #     if i == int(self._prev_target_gate):
        #         continue
            
        #     gate = self._gate_pos[i]
        #     idx = ((gate - min_bounds) / resolution).astype(int)

        #     for dx in range(-inflate, inflate+1):
        #         for dy in range(-inflate, inflate+1):
        #             for dz in range(-inflate, inflate+1):
        #                 i3 = idx + np.array([dx, dy, dz])
        #                 if np.all(i3 >= 0) and np.all(i3 < grid.shape):
        #                     grid[tuple(i3)] = True
        
        return grid

    def _astar(self, start, goal, grid):
        def to_grid(p):
            return tuple(((p - self._grid_origin) / self._grid_res).astype(int))

        def to_world(idx):
            return self._grid_origin + self._grid_res * np.array(idx)

        start_idx = to_grid(start)
        goal_idx  = to_grid(goal)

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        # neighbors = [
        #     (dx, dy, dz)
        #     for dx in [-1, 0, 1]
        #     for dy in [-1, 0, 1]
        #     for dz in [-1, 0, 1]
        #     if not (dx == 0 and dy == 0 and dz == 0)
        # ]

        neighbors = [
            (1,0,0), (-1,0,0),
            (0,1,0), (0,-1,0),
            (0,0,1), (0,0,-1)
        ]

        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                # reconstruct path
                path = []
                while current in came_from:
                    path.append(to_world(current))
                    current = came_from[current]
                path.append(start)
                print("\n--- A* PATH ---")
                for p in path:
                    print(p)
                return path[::-1]

            for d in neighbors:
                neighbor = tuple(np.array(current) + np.array(d))

                if not all(0 <= neighbor[i] < grid.shape[i] for i in range(3)):
                    continue

                if grid[neighbor]:  # obstacle
                    continue

                tentative = g_score[current] + 1

                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # no path

    def _build_spline(self, start_pos: np.ndarray, gate_id: int, obs: dict, label: str = "") -> None:
        
        gate_center = self._gate_pos[gate_id]
        rot = R.from_quat(self._gate_quat[gate_id])
        gate_normal = rot.apply([1, 0, 0])

        approach = gate_center - 0.3 * gate_normal
        exit_pt  = gate_center + 0.6 * gate_normal

        if self._subgoal_stage == 0:
            goal = approach
        elif self._subgoal_stage == 1:
            goal = gate_center
            print(f"  [subgoal] stage {self._subgoal_stage} → goal set to {goal}")
        else:
            goal = exit_pt
            print(f"  [subgoal] stage {self._subgoal_stage} → goal set to {goal}")

        self._current_goal = goal

        grid = self._build_occupancy_grid(obs)

        path = self._astar(start_pos, goal, grid)

        if path is None or len(path) < 3:
            print("No path found — fallback to straight line")
            waypoints = np.vstack([start_pos, approach])
        else:
            waypoints = np.array(path)

        # waypoints = waypoints[::3]  # downsample for smoother spline (tune this!)

        # for i in range(gate_id, self._n_gates):
        #     if self._gates_visited[i]:
        #         continue
            
        #     gate_center = self._gate_pos[i]
        #     rot = R.from_quat(self._gate_quat[i])
        #     gate_normal = rot.apply([1, 0, 0])

        #     approach_point = gate_center - 0.5 * gate_normal

        #     waypoints.append(approach_point)
        #     waypoints.append(gate_center)

        # waypoints = np.vstack(waypoints)

        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        dists = np.maximum(dists, 0.01)
        weights = np.ones(len(dists))

        # Slow down near the END (approach to goal)
        slow_radius = 5  # last N segments
        weights[-slow_radius:] *= 3.0   # increase time → slower motion

        t_knots = np.concatenate([[0.0], np.cumsum(dists * weights * 4.0)])        
        self._t_total = t_knots[-1]
        # Force zero velocity at the start — don't inherit drone's current velocity.
        # This prevents the spline from dipping below the starting height.
        # self._spline = CubicSpline(
        #     t_knots, waypoints,
        #     bc_type=((1, np.zeros(3)), (1, np.zeros(3)))
        # )
        self._spline = CubicSpline(
            t_knots, waypoints,
            bc_type="natural"
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

        dt = 1.0 / self._freq

        if self._current_goal is not None:
            pos_err = np.linalg.norm(pos - self._current_goal)
        vel_err = np.linalg.norm(vel)

        # print(f"Stage {self._subgoal_stage} → goal currently at {self._current_goal}")
        
        # if self._tick % 50 == 0:
        #     print("\n--- Obstacles ---")
        #     print("positions:\n", np.round(obs["obstacles_pos"], 3))
        #     print("visited:\n", obs["obstacles_visited"])

        # Goal evaluation logic
        if self._current_goal is not None:
            if pos_err < 0.05 and vel_err < 0.05:
            
                print(f"Reached subgoal stage {self._subgoal_stage}")

                self._subgoal_stage += 1

                if self._subgoal_stage > 2:
                    print("Gate fully passed!")
                    self._subgoal_stage = 0

                # force replan to next stage
                self._build_spline(pos, current_gate, obs, label="subgoal_switch")

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
            self._build_spline(pos, gate_id=current_gate, obs=obs, label="phase2_init")

        # ── Phase 2: gate/sensor updates ─────────────────────────────
        # if self._update_gates(obs):
        #     self._build_spline(pos, gate_id=current_gate, obs=obs, label="gate_visited")

        # if current_gate != self._prev_target_gate:
        #     print(f"\n>>> Gate passed! {self._prev_target_gate} → {current_gate}")
        #     self._build_spline(pos, gate_id=current_gate, obs=obs, label="gate_passed")
        #     self._prev_target_gate = current_gate            

        # ── Evaluate spline ───────────────────────────────────────────
        t_elapsed = (self._tick - self._t_start_tick) / self._freq
        t_sp = min(t_elapsed, self._t_total)
        des_pos = self._spline(t_sp)
        des_vel = self._spline_vel(t_sp)

        kp  = np.array([2.0, 2.0, 3.0])
        kd  = np.array([0.5, 0.5, 1.0])
        ki = np.array([0.3, 0.3, 0.8])

        pos_error = des_pos - pos
        vel_error = des_vel - vel

        self._pos_integral += pos_error * dt
        self._pos_integral = np.clip(self._pos_integral, -0.5, 0.5)

        acc = kp * (des_pos - pos) + kd * (des_vel - vel) + ki * self._pos_integral
        acc = np.clip(acc, -5.0, 5.0)

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
        self._pos_integral = np.zeros(3)
        self._last_tick = 0
"""AIGP reward system (ported from DronePrix).

This module provides a modular reward definition with:
- a dataclass config (`RewardConfig`)
- named presets (`REWARD_PRESETS`)
- a vectorized JAX implementation (`RewardCalculator.compute`)

The implementation is intentionally close to the DronePrix version, but adapted to lsy/crazyflow
state tensors and vectorized environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jp

if TYPE_CHECKING:
    from jax import Array


@dataclass
class RewardConfig:
    """Configuration for reward components."""

    # Gate passage rewards
    gate_passage_reward: float = 100.0
    gate_passage_enabled: bool = True

    # Progress reward (distance to next gate)
    progress_weight: float = 2.0
    progress_enabled: bool = True

    # Velocity toward gate reward
    progress_velocity_weight: float = 0.5
    progress_velocity_enabled: bool = True
    progress_velocity_max: float = 3.0  # Cap velocity component (m/s) to prevent dominance

    # Raw speed bonus
    speed_bonus_weight: float = 0.1
    speed_bonus_enabled: bool = True
    speed_bonus_max: float = 5.0  # Cap speed bonus contribution

    # Speed efficiency (velocity in optimal direction)
    speed_efficiency_weight: float = 0.3
    speed_efficiency_enabled: bool = True

    # Orientation alignment (pointing toward gate)
    orientation_weight: float = 0.2
    orientation_enabled: bool = True

    # Action smoothness penalty
    smoothness_weight: float = 0.1
    smoothness_enabled: bool = True

    # Altitude maintenance
    altitude_weight: float = 0.5
    altitude_enabled: bool = True
    altitude_target: float = 1.0  # Ideal flying altitude
    altitude_tolerance: float = 0.5  # Acceptable deviation
    altitude_penalty_scale: float = 2.0  # How quickly penalty grows

    # Boundary proximity penalty
    boundary_weight: float = 1.0
    boundary_enabled: bool = True
    boundary_danger_distance: float = 0.5  # Start penalising within this distance

    # Crash penalty
    crash_penalty: float = -100.0
    crash_enabled: bool = True

    # Timeout penalty
    timeout_penalty: float = -50.0
    timeout_enabled: bool = True

    # Time penalty (per step)
    time_penalty: float = -0.01
    time_penalty_enabled: bool = True

    # Completion time bonus
    completion_bonus_base: float = 200.0
    completion_bonus_enabled: bool = True
    completion_time_target: float = 10.0  # Target time in seconds for max bonus

    # Gate approach angle reward
    approach_angle_weight: float = 0.3
    approach_angle_enabled: bool = True

    # Hover penalty (penalise staying still)
    hover_penalty_weight: float = 0.5
    hover_penalty_enabled: bool = True
    hover_speed_threshold: float = 0.5  # Below this speed is "hovering"

    # Upward velocity penalty (prevent flying straight up)
    upward_penalty_weight: float = 1.0
    upward_penalty_enabled: bool = True
    upward_velocity_threshold: float = 1.0  # Penalise upward velocity above this

    # Perception awareness (reward keeping gate in camera view)
    perception_awareness_weight: float = 0.3
    perception_awareness_enabled: bool = True

    # Lookahead alignment (reward velocity aligned with NEXT gate for turning)
    lookahead_alignment_weight: float = 0.5
    lookahead_alignment_enabled: bool = True

    # Speed attenuation near gates (reduce speed reward when approaching gate)
    speed_attenuation_enabled: bool = True
    speed_attenuation_distance: float = 2.0

    # Racing line rewards (trajectory following)
    racing_line_enabled: bool = False
    racing_line_deviation_weight: float = 0.5  # Penalty for lateral deviation
    racing_line_alignment_weight: float = 0.3  # Reward for velocity along line
    racing_line_max_deviation: float = 1.0  # Clip deviation penalty at this distance

    # Gate-approach curiosity bonus (exploration aid when struggling)
    gate_approach_curiosity_enabled: bool = False
    gate_approach_curiosity_weight: float = 5.0  # Weight for curiosity bonus
    gate_approach_curiosity_success_threshold: float = 0.2  # Only active below this success rate

    @classmethod
    def from_dict(cls, config: dict) -> RewardConfig:
        """Create a `RewardConfig` from a dictionary, ignoring unknown keys."""
        known_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in config.items() if k in known_keys})

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Preset configurations for different training stages (ported from DronePrix).
REWARD_PRESETS: dict[str, RewardConfig] = {
    "swift": RewardConfig(
        # Swift-style reward based on UZH/Intel Nature 2023 paper
        gate_passage_reward=10.0,
        gate_passage_enabled=True,
        progress_weight=2.0,
        progress_enabled=True,
        progress_velocity_enabled=False,
        speed_bonus_enabled=False,
        speed_efficiency_enabled=False,
        orientation_enabled=False,
        smoothness_weight=0.1,
        smoothness_enabled=True,
        altitude_enabled=False,
        boundary_enabled=True,
        boundary_weight=1.0,
        boundary_danger_distance=1.5,
        crash_penalty=-30.0,
        crash_enabled=True,
        timeout_enabled=False,
        time_penalty=-0.01,
        time_penalty_enabled=True,
        completion_bonus_base=10.0,
        completion_bonus_enabled=True,
        approach_angle_enabled=False,
        hover_penalty_enabled=False,
        upward_penalty_enabled=True,
        upward_penalty_weight=3.0,
        upward_velocity_threshold=0.5,
        perception_awareness_weight=0.3,
        perception_awareness_enabled=True,
        lookahead_alignment_enabled=False,
        speed_attenuation_enabled=False,
        racing_line_enabled=False,
        racing_line_deviation_weight=0.0,
        racing_line_alignment_weight=0.0,
        gate_approach_curiosity_enabled=False,
    ),
    "grandprix": RewardConfig(
        # Qualifier preset: Swift-inspired with gate-dominant rewards.
        gate_passage_reward=50.0,
        gate_passage_enabled=True,
        progress_weight=2.0,
        progress_enabled=True,
        progress_velocity_weight=0.1,
        progress_velocity_enabled=True,
        speed_bonus_enabled=False,
        speed_efficiency_enabled=False,
        orientation_enabled=False,
        smoothness_weight=0.1,
        smoothness_enabled=True,
        altitude_enabled=True,
        altitude_target=1.0,
        altitude_tolerance=0.3,
        altitude_penalty_scale=1.0,
        boundary_enabled=True,
        boundary_weight=1.0,
        boundary_danger_distance=0.8,
        crash_penalty=-150.0,
        crash_enabled=True,
        timeout_enabled=False,
        time_penalty=-0.01,
        time_penalty_enabled=True,
        completion_bonus_base=50.0,
        completion_bonus_enabled=True,
        approach_angle_enabled=False,
        hover_penalty_enabled=False,
        upward_penalty_enabled=True,
        upward_penalty_weight=1.0,
        upward_velocity_threshold=0.5,
        perception_awareness_weight=0.3,
        perception_awareness_enabled=True,
        lookahead_alignment_weight=0.3,
        lookahead_alignment_enabled=True,
        speed_attenuation_enabled=True,
        speed_attenuation_distance=2.0,
        racing_line_enabled=False,
        gate_approach_curiosity_enabled=False,
    ),
    "grandprix_lite": RewardConfig(
        # Bridge preset between grandprix and minimal.
        gate_passage_reward=100.0,
        gate_passage_enabled=True,
        progress_weight=1.0,
        progress_enabled=True,
        progress_velocity_enabled=False,
        speed_bonus_enabled=False,
        speed_efficiency_enabled=False,
        orientation_enabled=False,
        smoothness_enabled=False,
        altitude_enabled=False,
        boundary_enabled=True,
        boundary_weight=0.5,
        boundary_danger_distance=0.5,
        crash_penalty=-300.0,
        crash_enabled=True,
        timeout_enabled=False,
        time_penalty=-0.01,
        time_penalty_enabled=True,
        completion_bonus_enabled=False,
        approach_angle_enabled=False,
        hover_penalty_enabled=False,
        upward_penalty_enabled=True,
        upward_penalty_weight=2.0,
        perception_awareness_enabled=False,
        lookahead_alignment_enabled=False,
        speed_attenuation_enabled=False,
        racing_line_enabled=False,
        gate_approach_curiosity_enabled=False,
    ),
    "minimal": RewardConfig(
        # Absolute minimum reward function (3 core components only).
        gate_passage_reward=100.0,
        gate_passage_enabled=True,
        progress_enabled=False,
        progress_velocity_enabled=False,
        speed_bonus_enabled=False,
        speed_efficiency_enabled=False,
        orientation_enabled=False,
        smoothness_enabled=False,
        altitude_enabled=False,
        boundary_enabled=False,
        crash_penalty=-300.0,
        crash_enabled=True,
        timeout_enabled=False,
        time_penalty=-0.01,
        time_penalty_enabled=True,
        completion_bonus_enabled=False,
        approach_angle_enabled=False,
        hover_penalty_enabled=False,
        upward_penalty_enabled=False,
        perception_awareness_enabled=False,
        lookahead_alignment_enabled=False,
        speed_attenuation_enabled=False,
        racing_line_enabled=False,
        racing_line_deviation_weight=0.0,
        racing_line_alignment_weight=0.0,
        gate_approach_curiosity_enabled=False,
    ),
    "minimal_curiosity": RewardConfig(
        # Minimal reward + curiosity bonus for exploration when struggling.
        gate_passage_reward=100.0,
        gate_passage_enabled=True,
        progress_enabled=False,
        progress_velocity_enabled=False,
        speed_bonus_enabled=False,
        speed_efficiency_enabled=False,
        orientation_enabled=False,
        smoothness_enabled=False,
        altitude_enabled=False,
        boundary_enabled=False,
        crash_penalty=-300.0,
        crash_enabled=True,
        timeout_enabled=False,
        time_penalty=-0.01,
        time_penalty_enabled=True,
        completion_bonus_enabled=False,
        approach_angle_enabled=False,
        hover_penalty_enabled=False,
        upward_penalty_enabled=False,
        perception_awareness_enabled=False,
        lookahead_alignment_enabled=False,
        speed_attenuation_enabled=False,
        racing_line_enabled=False,
        gate_approach_curiosity_enabled=True,
        gate_approach_curiosity_weight=5.0,
    ),
}


def get_preset(name: str) -> RewardConfig:
    """Get a preset reward configuration by name."""
    if name not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset: {name}. Available: {sorted(REWARD_PRESETS)}")
    return REWARD_PRESETS[name]


def _quat_to_forward(quat_xyzw: Array) -> Array:
    """Compute the body X-axis (forward) in world coordinates from a xyzw quaternion.

    Args:
        quat_xyzw: Quaternion in xyzw order, shape (..., 4).

    Returns:
        Forward unit vector in world coordinates, shape (..., 3).
    """
    x, y, z, w = (
        quat_xyzw[..., 0],
        quat_xyzw[..., 1],
        quat_xyzw[..., 2],
        quat_xyzw[..., 3],
    )
    # First column of rotation matrix R(q): R[:, 0]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    return jp.stack([r00, r10, r20], axis=-1)


class RewardCalculator:
    """Modular reward calculator for AIGP environments."""

    def __init__(self, config: RewardConfig | dict | str | None = "swift"):
        """Initialize the reward calculator.

        Args:
            config: Reward preset name, RewardConfig, dict of RewardConfig fields, or None.
        """
        if config is None:
            self.config = RewardConfig()
        elif isinstance(config, str):
            self.config = get_preset(config)
        elif isinstance(config, dict):
            self.config = RewardConfig.from_dict(config)
        else:
            self.config = config

        # External signal for curiosity gating (updated by curriculum/callbacks).
        self.current_success_rate: float = 0.0

    def compute(
        self,
        *,
        pos: Array,
        vel: Array,
        quat: Array,
        target_gate: Array,
        active_gate_count: Array,
        gates_pos: Array,
        passed_gate: Array,
        progress: Array,
        disabled_drones: Array,
        completed: Array,
        truncated: Array,
        steps: Array,
        max_episode_steps: Array,
        freq: int,
        pos_limit_low: Array,
        pos_limit_high: Array,
        action_diff: Array | None = None,
        has_prev_action: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute reward and individual components (vectorized).

        Args:
            pos: Drone positions, shape (n_worlds, n_drones, 3).
            vel: Drone linear velocities, shape (n_worlds, n_drones, 3).
            quat: Drone orientations (xyzw), shape (n_worlds, n_drones, 4).
            target_gate: Current target gate indices, shape (n_worlds, n_drones).
            active_gate_count: Active (non-padding) gate count, shape (n_worlds, n_drones).
            gates_pos: Gate positions, shape (n_worlds, n_gates, 3).
            passed_gate: Gate passage flags for this step, shape (n_worlds, n_drones).
            progress: Signed progress to current target gate, shape (n_worlds, n_drones).
            disabled_drones: Disabled flags, shape (n_worlds, n_drones).
            completed: Completion flags, shape (n_worlds, n_drones).
            truncated: Timeout flags, shape (n_worlds, n_drones).
            steps: Step counters per world, shape (n_worlds,).
            max_episode_steps: Maximum steps per episode, shape (1,) or scalar array.
            freq: Control frequency (Hz).
            pos_limit_low: Lower bounds (x, y, z), shape (3,).
            pos_limit_high: Upper bounds (x, y, z), shape (3,).
            action_diff: Norm of action delta, shape (n_worlds, n_drones). Required for smoothness.
            has_prev_action: Mask indicating valid `action_diff`, shape (n_worlds, n_drones).

        Returns:
            Total reward and a dict of named component tensors, each shape (n_worlds, n_drones).
        """
        cfg = self.config

        passed_f = passed_gate.astype(jp.float32)
        completed_f = completed.astype(jp.float32)
        crashed_f = (disabled_drones & ~completed).astype(jp.float32)
        truncated_f = truncated.astype(jp.float32)

        # Gate indexing helpers.
        n_worlds, n_drones = pos.shape[0], pos.shape[1]
        n_gates = gates_pos.shape[1]
        has_gate = (target_gate >= 0) & (target_gate < active_gate_count)
        safe_gate_idx = jp.clip(target_gate, 0, n_gates - 1)
        safe_gate_idx = jp.minimum(safe_gate_idx, active_gate_count - 1)
        env_idx = jp.arange(n_worlds)[:, None]
        gate_pos = gates_pos[env_idx, safe_gate_idx]

        to_gate = gate_pos - pos
        dist_to_gate = jp.linalg.norm(to_gate, axis=-1)
        dir_to_gate = to_gate / (dist_to_gate[..., None] + 1e-8)

        speed = jp.linalg.norm(vel, axis=-1)
        vel_toward_gate = jp.sum(vel * dir_to_gate, axis=-1)

        # 1. Gate passage reward
        gate_passage = cfg.gate_passage_reward * passed_f if cfg.gate_passage_enabled else 0.0

        # 2. Progress reward
        if cfg.progress_enabled:
            progress_r = cfg.progress_weight * progress
            progress_r = jp.where(has_gate, progress_r, 0.0)
        else:
            progress_r = 0.0

        # 3. Progress velocity
        if cfg.progress_velocity_enabled:
            capped_vel = jp.clip(vel_toward_gate, 0.0, cfg.progress_velocity_max)
            progress_velocity = cfg.progress_velocity_weight * capped_vel
            progress_velocity = jp.where(has_gate, progress_velocity, 0.0)
        else:
            progress_velocity = 0.0

        # 4. Raw speed bonus (+ optional attenuation near gates)
        if cfg.speed_bonus_enabled:
            speed_reward = jp.minimum(speed, cfg.speed_bonus_max)
            if cfg.speed_attenuation_enabled:
                attenuation = jp.minimum(
                    1.0, dist_to_gate / (cfg.speed_attenuation_distance + 1e-8)
                )
                speed_reward = jp.where(
                    has_gate & (dist_to_gate < cfg.speed_attenuation_distance),
                    speed_reward * attenuation,
                    speed_reward,
                )
            speed_bonus = cfg.speed_bonus_weight * speed_reward
        else:
            speed_bonus = 0.0

        # 5. Speed efficiency
        if cfg.speed_efficiency_enabled:
            speed_efficiency = cfg.speed_efficiency_weight * vel_toward_gate
            speed_efficiency = jp.where(has_gate, speed_efficiency, 0.0)
        else:
            speed_efficiency = 0.0

        # 6. Orientation alignment (horizontal)
        if cfg.orientation_enabled:
            forward = _quat_to_forward(quat)
            forward_h = forward[..., :2]
            dir_h = dir_to_gate[..., :2]
            denom = jp.linalg.norm(forward_h, axis=-1) * jp.linalg.norm(dir_h, axis=-1) + 1e-8
            alignment = jp.sum(forward_h * dir_h, axis=-1) / denom
            alignment = jp.clip(alignment, -1.0, 1.0)
            alignment_reward = (alignment + 1.0) / 2.0
            orientation = cfg.orientation_weight * alignment_reward
            orientation = jp.where(has_gate, orientation, 0.0)
        else:
            orientation = 0.0

        # 7. Action smoothness penalty
        if cfg.smoothness_enabled and action_diff is not None:
            valid = (
                jp.ones_like(action_diff, dtype=bool)
                if has_prev_action is None
                else has_prev_action
            )
            smoothness = -cfg.smoothness_weight * jp.where(valid, action_diff, 0.0)
        else:
            smoothness = 0.0

        # 8. Altitude maintenance
        if cfg.altitude_enabled:
            target_z = jp.where(has_gate, gate_pos[..., 2], cfg.altitude_target)
            altitude_error = jp.abs(pos[..., 2] - target_z)
            excess = jp.maximum(0.0, altitude_error - cfg.altitude_tolerance)
            altitude = -cfg.altitude_weight * (excess * cfg.altitude_penalty_scale)
        else:
            altitude = 0.0

        # 9. Boundary proximity penalty
        if cfg.boundary_enabled:
            dist_to_min = pos - pos_limit_low
            dist_to_max = pos_limit_high - pos
            min_dist = jp.minimum(dist_to_min, dist_to_max)
            normalised = jp.clip(1.0 - (min_dist / (cfg.boundary_danger_distance + 1e-8)), 0.0, 1.0)
            boundary_penalty = jp.sum(normalised**2, axis=-1)
            boundary = -cfg.boundary_weight * boundary_penalty
        else:
            boundary = 0.0

        # 10. Crash penalty
        crash = cfg.crash_penalty * crashed_f if cfg.crash_enabled else 0.0

        # 11. Timeout penalty
        timeout = cfg.timeout_penalty * truncated_f if cfg.timeout_enabled else 0.0

        # 12. Time penalty (per step)
        if cfg.time_penalty_enabled:
            time_penalty = cfg.time_penalty * jp.ones_like(passed_f)
        else:
            time_penalty = 0.0

        # 13. Completion time bonus
        if cfg.completion_bonus_enabled:
            steps_d = jp.tile(steps[:, None], (1, n_drones))
            completion_time = steps_d / float(freq)
            time_ratio = jp.maximum(
                0.0,
                1.0
                - (completion_time - cfg.completion_time_target)
                / (cfg.completion_time_target + 1e-8),
            )
            multiplier = jp.maximum(0.1, time_ratio)
            completion_bonus = cfg.completion_bonus_base * multiplier * completed_f
        else:
            completion_bonus = 0.0

        # 14. Gate approach angle reward
        if cfg.approach_angle_enabled:
            vel_dir = vel / (speed[..., None] + 1e-8)
            approach_alignment = jp.sum(vel_dir * dir_to_gate, axis=-1)
            approach_reward = (jp.clip(approach_alignment, -1.0, 1.0) + 1.0) / 2.0
            approach_angle = cfg.approach_angle_weight * approach_reward
            approach_angle = jp.where(has_gate & (dist_to_gate > 0.5), approach_angle, 0.0)
        else:
            approach_angle = 0.0

        # 15. Hover penalty
        if cfg.hover_penalty_enabled:
            hover_factor = jp.clip(1.0 - speed / (cfg.hover_speed_threshold + 1e-8), 0.0, 1.0)
            hover_penalty = -cfg.hover_penalty_weight * hover_factor
        else:
            hover_penalty = 0.0

        # 16. Upward velocity penalty
        if cfg.upward_penalty_enabled:
            upward_vel = vel[..., 2]
            upward_penalty = -cfg.upward_penalty_weight * jp.maximum(
                0.0, upward_vel - cfg.upward_velocity_threshold
            )
        else:
            upward_penalty = 0.0

        # 17. Perception awareness (gate in front of camera)
        if cfg.perception_awareness_enabled:
            forward = _quat_to_forward(quat)
            camera_alignment = jp.sum(forward * dir_to_gate, axis=-1)
            perception_awareness = cfg.perception_awareness_weight * jp.maximum(
                0.0, camera_alignment
            )
            # Avoid a degenerate "hover and stare at the gate" policy: only reward alignment when
            # making forward progress toward the current target.
            perception_awareness = jp.where(
                has_gate & (vel_toward_gate > 0.1), perception_awareness, 0.0
            )
        else:
            perception_awareness = 0.0

        # 18. Lookahead alignment (velocity aligned with NEXT gate)
        if cfg.lookahead_alignment_enabled:
            has_next = (target_gate >= 0) & ((target_gate + 1) < active_gate_count)
            next_gate_idx = jp.clip(target_gate + 1, 0, n_gates - 1)
            next_gate_pos = gates_pos[env_idx, next_gate_idx]
            to_next = next_gate_pos - pos
            dist_next = jp.linalg.norm(to_next, axis=-1)
            dir_to_next = to_next / (dist_next[..., None] + 1e-8)
            vel_dir = vel / (speed[..., None] + 1e-8)
            lookahead = jp.sum(vel_dir * dir_to_next, axis=-1)
            lookahead_alignment = cfg.lookahead_alignment_weight * jp.maximum(0.0, lookahead)
            lookahead_alignment = jp.where(has_next & (speed > 0.5), lookahead_alignment, 0.0)
        else:
            lookahead_alignment = 0.0

        # 19-20. Racing line (not implemented here; keep keys for logging parity)
        racing_line_deviation = 0.0
        racing_line_alignment = 0.0

        # 21. Gate-approach curiosity bonus (exploration aid)
        if (
            cfg.gate_approach_curiosity_enabled
            and self.current_success_rate < cfg.gate_approach_curiosity_success_threshold
        ):
            progress_pos = jp.maximum(0.0, progress)
            gate_approach_curiosity = cfg.gate_approach_curiosity_weight * progress_pos
            gate_approach_curiosity = jp.where(has_gate, gate_approach_curiosity, 0.0)
        else:
            gate_approach_curiosity = 0.0

        components: dict[str, Array] = {
            "gate_passage": gate_passage,
            "progress": progress_r,
            "progress_velocity": progress_velocity,
            "speed_bonus": speed_bonus,
            "speed_efficiency": speed_efficiency,
            "orientation": orientation,
            "smoothness": smoothness,
            "altitude": altitude,
            "boundary": boundary,
            "crash": crash,
            "timeout": timeout,
            "time_penalty": time_penalty,
            "completion_bonus": completion_bonus,
            "approach_angle": approach_angle,
            "hover_penalty": hover_penalty,
            "upward_penalty": upward_penalty,
            "perception_awareness": perception_awareness,
            "lookahead_alignment": lookahead_alignment,
            "racing_line_deviation": racing_line_deviation,
            "racing_line_alignment": racing_line_alignment,
            "gate_approach_curiosity": gate_approach_curiosity,
        }

        # NaN/Inf guard: replace non-finite components with 0.0.
        for k, v in list(components.items()):
            if isinstance(v, (int, float)):
                components[k] = jp.zeros((n_worlds, n_drones), dtype=jp.float32) + float(v)
            else:
                components[k] = jp.where(jp.isfinite(v), v, 0.0)

        total_reward = jp.zeros((n_worlds, n_drones), dtype=jp.float32)
        for v in components.values():
            total_reward = total_reward + v
        total_reward = jp.where(jp.isfinite(total_reward), total_reward, 0.0)

        return total_reward, components

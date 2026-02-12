"""AIGP curriculum manager (ported and adapted from DronePrix).

This module is RL-library agnostic: it doesn't depend on Stable-Baselines3 or Torch.
It focuses on:
- stage definitions and config loading (TOML)
- advancement gating (binary success + stability)
- panic mode / adaptive difficulty / recovery mode (geometry easing + DR scaling)
- forgetting detection hooks
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from pathlib import Path

    from ml_collections import ConfigDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurriculumStage:
    """Definition of a single curriculum stage."""

    name: str
    description: str = ""

    tracks: list[str] = field(default_factory=list)
    track_weights: list[float] = field(default_factory=list)

    active_gate_count: int | None = None

    gate_width: float | None = None
    gate_height: float | None = None
    gate_tolerance: float | None = None

    reward_preset: str = "swift"
    dr_tier: str = "none"

    success_rate_threshold: float = 0.8
    min_episodes: int = 0
    max_episode_steps: int | None = None
    eval_episodes: int | None = None

    bossfight_enable: bool = True
    bossfight_success_rate_min_threshold: float | None = None
    bossfight_bottomk_fraction: float = 0.2
    bossfight_bottomk_mean_threshold: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumStage":
        """Create a stage from a config dictionary."""
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass(frozen=True)
class PanicModeConfig:
    """Configuration for panic mode difficulty/assist control."""

    enable: bool = False
    target_success: float = 0.25
    recover_success: float = 0.35
    patience_evals: int = 3
    step: float = 0.05
    dr_min_mult: float = 0.30
    assist_max_mult: float = 1.40
    hardlock_zero_evals: int = 8
    stuck_action: str = "rollback_stage"

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "PanicModeConfig":
        """Create a PanicModeConfig from a config dictionary."""
        if not data:
            return cls()
        return cls(
            enable=bool(data.get("enable", False)),
            target_success=float(data.get("target_success", 0.25)),
            recover_success=float(data.get("recover_success", 0.35)),
            patience_evals=int(data.get("patience_evals", 3)),
            step=float(data.get("step", 0.05)),
            dr_min_mult=float(data.get("dr_min_mult", 0.30)),
            assist_max_mult=float(data.get("assist_max_mult", 1.40)),
            hardlock_zero_evals=int(data.get("hardlock_zero_evals", 8)),
            stuck_action=str(data.get("stuck_action", "rollback_stage")),
        )


@dataclass
class PanicStageState:
    """Per-stage panic mode state."""

    dr_mult: float = 1.0
    assist_mult: float = 1.0
    below_target_count: int = 0
    zero_success_count: int = 0
    hardlock: bool = False
    stuck_action_taken: str | None = None


class PanicModeController:
    """Stage-aware panic mode controller with DR and assist scaling."""

    def __init__(self, config: PanicModeConfig, num_stages: int) -> None:
        """Initialize the controller with per-stage state."""
        self.config = config
        self._states: dict[int, PanicStageState] = {
            i: PanicStageState() for i in range(num_stages)
        }

    def get_state(self, stage_idx: int) -> PanicStageState:
        """Get the current panic state for a stage."""
        return self._states.get(stage_idx, PanicStageState())

    def reset_stage(self, stage_idx: int) -> None:
        """Reset panic state for a stage."""
        self._states[stage_idx] = PanicStageState()

    def update(
        self, stage_idx: int, *, eval_success_rate: float | None
    ) -> tuple[float, float, bool, str | None]:
        """Update panic mode after an evaluation cycle.

        Returns:
            (dr_mult, assist_mult, hardlock, stuck_action_taken)
        """
        state = self.get_state(stage_idx)
        state.stuck_action_taken = None

        if not self.config.enable or eval_success_rate is None:
            return state.dr_mult, state.assist_mult, state.hardlock, None

        sr = float(eval_success_rate)

        if sr <= 0.0:
            state.zero_success_count += 1
        else:
            state.zero_success_count = 0

        if sr < self.config.target_success:
            state.below_target_count += 1
        else:
            state.below_target_count = 0

        if state.below_target_count >= max(self.config.patience_evals, 1):
            if state.dr_mult > self.config.dr_min_mult + 1e-9:
                state.dr_mult = float(
                    max(self.config.dr_min_mult, state.dr_mult - self.config.step)
                )
            else:
                state.assist_mult = float(
                    min(self.config.assist_max_mult, state.assist_mult + self.config.step)
                )
            state.below_target_count = 0

        if sr >= self.config.recover_success:
            if state.assist_mult > 1.0 + 1e-9:
                state.assist_mult = float(max(1.0, state.assist_mult - self.config.step * 2.0))
            if state.dr_mult < 1.0 - 1e-9:
                state.dr_mult = float(min(1.0, state.dr_mult + self.config.step))

        state.hardlock = bool(
            state.zero_success_count >= self.config.hardlock_zero_evals
            and state.dr_mult <= self.config.dr_min_mult + 1e-9
            and state.assist_mult >= self.config.assist_max_mult - 1e-9
        )

        if state.hardlock:
            state.stuck_action_taken = self.config.stuck_action

        self._states[stage_idx] = state
        return state.dr_mult, state.assist_mult, state.hardlock, state.stuck_action_taken


@dataclass(frozen=True)
class AdaptiveDifficultyConfig:
    """Configuration for adaptive difficulty adjustment."""

    enable: bool = False
    target_success: float = 0.25
    tolerance: float = 0.05
    step: float = 0.05
    min_mult: float = 0.30
    max_mult: float = 1.0
    eval_window: int = 5
    patience: int = 3

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "AdaptiveDifficultyConfig":
        """Create an AdaptiveDifficultyConfig from a config dictionary."""
        if not data:
            return cls()
        return cls(
            enable=bool(data.get("enable", False)),
            target_success=float(data.get("target_success", 0.25)),
            tolerance=float(data.get("tolerance", 0.05)),
            step=float(data.get("step", 0.05)),
            min_mult=float(data.get("min_mult", 0.30)),
            max_mult=float(data.get("max_mult", 1.0)),
            eval_window=int(data.get("eval_window", 5)),
            patience=int(data.get("patience", 3)),
        )


class AdaptiveDifficultyController:
    """Stage-aware adaptive difficulty controller based on eval success."""

    def __init__(self, config: AdaptiveDifficultyConfig, num_stages: int) -> None:
        """Initialize the controller with per-stage multipliers and histories."""
        self.config = config
        self._mult: dict[int, float] = {i: 1.0 for i in range(num_stages)}
        self._history: dict[int, list[float]] = {i: [] for i in range(num_stages)}
        self._below_band: dict[int, int] = {i: 0 for i in range(num_stages)}

    def get_multiplier(self, stage_idx: int) -> float:
        """Get the current multiplier for a stage."""
        return float(self._mult.get(stage_idx, 1.0))

    def reset_stage(self, stage_idx: int) -> None:
        """Reset a stage's adaptive difficulty state."""
        self._mult[stage_idx] = 1.0
        self._history[stage_idx] = []
        self._below_band[stage_idx] = 0

    def update(
        self, stage_idx: int, *, eval_success_rate: float | None
    ) -> tuple[float, float | None]:
        """Update the multiplier from an eval success rate.

        Returns:
            (multiplier, smoothed_success_rate)
        """
        if not self.config.enable or eval_success_rate is None:
            return self.get_multiplier(stage_idx), None

        history = self._history.get(stage_idx, [])
        history.append(float(eval_success_rate))
        window = max(self.config.eval_window, 1)
        if len(history) > window:
            history = history[-window:]
        self._history[stage_idx] = history

        smoothed = float(np.mean(history)) if history else None
        if smoothed is None:
            return self.get_multiplier(stage_idx), None

        target = float(self.config.target_success)
        current = self.get_multiplier(stage_idx)

        if smoothed < target - self.config.tolerance:
            self._below_band[stage_idx] = self._below_band.get(stage_idx, 0) + 1
            if self._below_band[stage_idx] >= max(self.config.patience, 1):
                self._mult[stage_idx] = float(max(self.config.min_mult, current - self.config.step))
                self._below_band[stage_idx] = 0
        else:
            self._below_band[stage_idx] = 0
            if smoothed > target + self.config.tolerance:
                self._mult[stage_idx] = float(min(self.config.max_mult, current + self.config.step))

        return self.get_multiplier(stage_idx), smoothed


@dataclass(frozen=True)
class RecoveryModeConfig:
    """Configuration for hard-lock recovery mode (geometry easing)."""

    enable: bool = True
    patience: int = 5
    min_sr: float = 0.05
    exit_sr: float = 0.20
    tolerance_mult: float = 1.2
    gate_scale: float = 1.1
    require_min_mult: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "RecoveryModeConfig":
        """Create a RecoveryModeConfig from a config dictionary."""
        if not data:
            return cls()
        return cls(
            enable=bool(data.get("enable", True)),
            patience=int(data.get("patience", 5)),
            min_sr=float(data.get("min_sr", 0.05)),
            exit_sr=float(data.get("exit_sr", 0.20)),
            tolerance_mult=float(data.get("tolerance_mult", 1.2)),
            gate_scale=float(data.get("gate_scale", 1.1)),
            require_min_mult=bool(data.get("require_min_mult", True)),
        )


@dataclass
class RecoveryStageState:
    """Per-stage recovery mode state."""

    active: bool = False
    stuck_count: int = 0


class RecoveryModeController:
    """Stage-aware recovery controller that temporarily eases gate geometry."""

    def __init__(self, config: RecoveryModeConfig, num_stages: int) -> None:
        """Initialize the controller with per-stage state."""
        self.config = config
        self._states: dict[int, RecoveryStageState] = {
            i: RecoveryStageState() for i in range(num_stages)
        }

    def get_state(self, stage_idx: int) -> RecoveryStageState:
        """Get recovery state for a stage."""
        return self._states.get(stage_idx, RecoveryStageState())

    def is_active(self, stage_idx: int) -> bool:
        """Check if recovery mode is active for a stage."""
        return bool(self.get_state(stage_idx).active)

    def reset_stage(self, stage_idx: int) -> None:
        """Reset recovery state for a stage."""
        self._states[stage_idx] = RecoveryStageState()

    def update(
        self,
        stage_idx: int,
        *,
        eval_success_rate: float | None,
        adaptive_at_min: bool,
    ) -> bool:
        """Update recovery state after an evaluation."""
        if not self.config.enable or eval_success_rate is None:
            return self.is_active(stage_idx)

        state = self.get_state(stage_idx)
        sr = float(eval_success_rate)

        if state.active:
            if sr >= self.config.exit_sr:
                state.active = False
                state.stuck_count = 0
                logger.info(
                    "EXIT_RECOVERY_MODE stage=%s eval_sr=%.3f", stage_idx, sr
                )
            self._states[stage_idx] = state
            return state.active

        at_min = adaptive_at_min if self.config.require_min_mult else True
        if sr <= self.config.min_sr and at_min:
            state.stuck_count += 1
        else:
            state.stuck_count = 0

        if state.stuck_count >= self.config.patience:
            state.active = True
            logger.info(
                "ENTER_RECOVERY_MODE stage=%s eval_sr=%.3f tol_mult=%.2f gate_scale=%.2f",
                stage_idx,
                sr,
                self.config.tolerance_mult,
                self.config.gate_scale,
            )

        self._states[stage_idx] = state
        return state.active


@dataclass(frozen=True)
class CurriculumConfig:
    """Full curriculum specification loaded from TOML."""

    name: str
    max_gates: int = 11
    stability_window: int = 20
    stability_threshold: float = 0.15
    eval_episodes: int = 30

    panic: PanicModeConfig = field(default_factory=PanicModeConfig)
    adaptive: AdaptiveDifficultyConfig = field(default_factory=AdaptiveDifficultyConfig)
    recovery: RecoveryModeConfig = field(default_factory=RecoveryModeConfig)

    stages: list[CurriculumStage] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CurriculumConfig":
        """Load a curriculum TOML file into a typed spec."""
        cfg = load_config(path)
        root = cfg.get("curriculum", {})
        name = str(root.get("name", path.stem))
        max_gates = int(root.get("max_gates", 11))
        stability_window = int(root.get("stability_window", 20))
        stability_threshold = float(root.get("stability_threshold", 0.15))
        eval_episodes = int(root.get("eval_episodes", 30))

        panic = PanicModeConfig.from_dict(root.get("panic"))
        adaptive = AdaptiveDifficultyConfig.from_dict(root.get("adaptive"))
        recovery = RecoveryModeConfig.from_dict(root.get("recovery"))

        stages = [CurriculumStage.from_dict(s) for s in list(root.get("stages", []))]
        for s in stages:
            if not s.tracks:
                raise ValueError(f"Stage {s.name} must specify at least one track")
            if s.track_weights and len(s.track_weights) != len(s.tracks):
                raise ValueError(f"Stage {s.name} track_weights length mismatch")

        return cls(
            name=name,
            max_gates=max_gates,
            stability_window=stability_window,
            stability_threshold=stability_threshold,
            eval_episodes=eval_episodes,
            panic=panic,
            adaptive=adaptive,
            recovery=recovery,
            stages=stages,
        )


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate metrics from an evaluation sweep."""

    n_episodes: int
    success_rate: float
    completion_mean: float
    completion_std: float
    lap_time_s_median: float | None = None


@dataclass
class ForgettingState:
    """Track baseline performance for forgetting detection."""

    baseline_success_rate: float | None = None
    baseline_stage_idx: int | None = None


class CurriculumManager:
    """Manage stage progression and difficulty multipliers."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        """Initialize the manager and its controllers."""
        self.cfg = cfg
        self.stage_idx = 0

        self.panic = PanicModeController(cfg.panic, num_stages=len(cfg.stages))
        self.adaptive = AdaptiveDifficultyController(cfg.adaptive, num_stages=len(cfg.stages))
        self.recovery = RecoveryModeController(cfg.recovery, num_stages=len(cfg.stages))

        self.forgetting = ForgettingState()
        self._recent_completion: list[float] = []

    def current_stage(self) -> CurriculumStage:
        """Return the current stage spec."""
        return self.cfg.stages[self.stage_idx]

    def stage_by_idx(self, stage_idx: int) -> CurriculumStage:
        """Return the stage spec for an explicit index."""
        if stage_idx < 0 or stage_idx >= len(self.cfg.stages):
            raise IndexError("stage_idx out of range")
        return self.cfg.stages[int(stage_idx)]

    def reset_stage_state(self, stage_idx: int) -> None:
        """Reset controller state for a stage."""
        self.panic.reset_stage(stage_idx)
        self.adaptive.reset_stage(stage_idx)
        self.recovery.reset_stage(stage_idx)
        self._recent_completion = []

    def record_eval(self, summary: EvalSummary) -> None:
        """Record completion history for stability checks."""
        self._recent_completion.append(float(summary.completion_mean))
        if len(self._recent_completion) > max(self.cfg.stability_window, 1):
            self._recent_completion = self._recent_completion[-self.cfg.stability_window :]

    def is_stable(self) -> bool:
        """Return True if recent completion metrics are stable enough to allow advancement."""
        if len(self._recent_completion) < max(self.cfg.stability_window, 1):
            return False
        return float(np.std(self._recent_completion)) <= float(self.cfg.stability_threshold)

    def update_after_eval(
        self,
        *,
        summary: EvalSummary,
        stage_episodes: int,
        eval_tracks: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update controllers after an eval and decide whether to advance/rollback.

        Returns:
            Dict of decisions and current multipliers.
        """
        stage = self.current_stage()
        self.record_eval(summary)

        adaptive_mult, adaptive_smoothed = self.adaptive.update(
            self.stage_idx, eval_success_rate=summary.success_rate
        )
        dr_mult, assist_mult, hardlock, stuck_action = self.panic.update(
            self.stage_idx, eval_success_rate=summary.success_rate
        )
        recovery_active = self.recovery.update(
            self.stage_idx,
            eval_success_rate=(
                adaptive_smoothed if adaptive_smoothed is not None else summary.success_rate
            ),
            adaptive_at_min=adaptive_mult <= self.cfg.adaptive.min_mult + 1e-9,
        )

        gate_success_threshold = float(stage.success_rate_threshold)
        gate_success_rate = float(summary.success_rate)
        gate_success_ok = gate_success_rate >= gate_success_threshold

        gate_min_episodes_required = int(stage.min_episodes)
        gate_stage_episodes = int(stage_episodes)
        gate_min_episodes_ok = gate_stage_episodes >= gate_min_episodes_required

        gate_stability_ok = bool(self.is_stable())
        gate_recovery_clear = not bool(recovery_active)

        bossfight_ok = True
        bossfight_block_reason = "none"
        bossfight_tracks_total: int | None = None
        bossfight_tracks_covered: int | None = None
        bossfight_success_rate_min: float | None = None
        bossfight_success_rate_bottom20_mean: float | None = None
        if len(stage.tracks) > 1 and bool(stage.bossfight_enable):
            aggregate: dict[str, Any] = {}
            if isinstance(eval_tracks, dict):
                aggregate_raw = eval_tracks.get("_aggregate")
                if isinstance(aggregate_raw, dict):
                    aggregate = aggregate_raw

            bossfight_tracks_total = int(aggregate.get("tracks_total", len(stage.tracks)))
            covered_raw = aggregate.get("tracks_covered")
            bossfight_tracks_covered = int(covered_raw) if covered_raw is not None else 0

            min_raw = aggregate.get("success_rate_min")
            bossfight_success_rate_min = float(min_raw) if min_raw is not None else None

            bottom20_raw = aggregate.get("success_rate_bottom20_mean")
            bossfight_success_rate_bottom20_mean = (
                float(bottom20_raw) if bottom20_raw is not None else None
            )

            min_track_threshold = (
                float(stage.bossfight_success_rate_min_threshold)
                if stage.bossfight_success_rate_min_threshold is not None
                else max(0.10, gate_success_threshold - 0.10)
            )
            bottomk_mean_threshold = (
                float(stage.bossfight_bottomk_mean_threshold)
                if stage.bossfight_bottomk_mean_threshold is not None
                else None
            )

            if bossfight_tracks_covered != bossfight_tracks_total:
                bossfight_ok = False
                bossfight_block_reason = "bossfight_tracks_not_covered"
            elif (
                bossfight_success_rate_min is None
                or bossfight_success_rate_min < min_track_threshold
            ):
                bossfight_ok = False
                bossfight_block_reason = "bossfight_min_track_fail"
            elif (
                bottomk_mean_threshold is not None
                and (
                    bossfight_success_rate_bottom20_mean is None
                    or bossfight_success_rate_bottom20_mean < bottomk_mean_threshold
                )
            ):
                bossfight_ok = False
                bossfight_block_reason = "bossfight_bottomk_fail"

        can_advance = (
            gate_success_ok
            and gate_min_episodes_ok
            and gate_stability_ok
            and gate_recovery_clear
            and bossfight_ok
        )

        if can_advance:
            block_reason = "none"
        elif not gate_success_ok:
            block_reason = "success_rate"
        elif not gate_min_episodes_ok:
            block_reason = "min_episodes"
        elif not gate_stability_ok:
            block_reason = "stability"
        elif not gate_recovery_clear:
            block_reason = "recovery"
        elif not bossfight_ok:
            block_reason = (
                bossfight_block_reason
                if bossfight_block_reason != "none"
                else "bossfight_other"
            )
        else:
            block_reason = "other"

        decision: dict[str, Any] = {
            "stage_idx": int(self.stage_idx),
            "stage_name": stage.name,
            "advance": bool(can_advance),
            "rollback": False,
            "hardlock": bool(hardlock),
            "stuck_action": stuck_action,
            "adaptive_mult": float(adaptive_mult),
            "adaptive_smoothed": adaptive_smoothed,
            "panic_dr_mult": float(dr_mult),
            "panic_assist_mult": float(assist_mult),
            "recovery_active": bool(recovery_active),
            "gate_success_ok": bool(gate_success_ok),
            "gate_success_rate": gate_success_rate,
            "gate_success_threshold": gate_success_threshold,
            "gate_min_episodes_ok": bool(gate_min_episodes_ok),
            "gate_stage_episodes": gate_stage_episodes,
            "gate_min_episodes_required": gate_min_episodes_required,
            "gate_stability_ok": bool(gate_stability_ok),
            "gate_recovery_clear": bool(gate_recovery_clear),
            "block_reason": str(block_reason),
            "bossfight_ok": bool(bossfight_ok),
            "bossfight_tracks_total": bossfight_tracks_total,
            "bossfight_tracks_covered": bossfight_tracks_covered,
            "bossfight_success_rate_min": bossfight_success_rate_min,
            "bossfight_success_rate_bottom20_mean": bossfight_success_rate_bottom20_mean,
        }

        if hardlock and stuck_action == "rollback_stage" and self.stage_idx > 0:
            decision["rollback"] = True

        return decision

    def advance(self) -> None:
        """Advance to the next stage."""
        if self.stage_idx >= len(self.cfg.stages) - 1:
            return
        prev = self.stage_idx
        self.stage_idx += 1
        self.reset_stage_state(self.stage_idx)
        self.forgetting = ForgettingState(baseline_success_rate=None, baseline_stage_idx=prev)

    def rollback(self) -> None:
        """Rollback one stage (used for hardlocks/forgetting)."""
        if self.stage_idx <= 0:
            return
        self.stage_idx -= 1
        self.reset_stage_state(self.stage_idx)

    def set_forgetting_baseline(self, *, success_rate: float) -> None:
        """Store baseline success rate for the stage we just advanced from."""
        self.forgetting.baseline_success_rate = float(success_rate)

    @staticmethod
    def _load_track(path: Path) -> ConfigDict:
        """Load a track from a TOML env config file (returns `env.track`)."""
        cfg = load_config(path)
        if not hasattr(cfg, "env") or not hasattr(cfg.env, "track"):
            raise ValueError(f"Track file does not look like an env config: {path}")
        return copy.deepcopy(cfg.env.track)

    def build_tracks_for_stage(
        self, *, stage_idx: int, config_dir: Path
    ) -> tuple[list[ConfigDict], list[float] | None]:
        """Load and stage-adjust tracks (gate size, active gate count) for a given stage."""
        stage = self.stage_by_idx(stage_idx)
        tracks = [self._load_track(config_dir / t) for t in stage.tracks]
        weights = stage.track_weights if stage.track_weights else None

        for i, t in enumerate(tracks):
            t["_track_id"] = int(i)
            t["_track_name"] = str(stage.tracks[i])
            if stage.active_gate_count is not None:
                t["active_gate_count"] = int(stage.active_gate_count)
            if "gate_size" not in t:
                t["gate_size"] = {}
            gate_size = dict(t.gate_size)
            if stage.gate_width is not None:
                gate_size["width"] = float(stage.gate_width)
            if stage.gate_height is not None:
                gate_size["height"] = float(stage.gate_height)
            if stage.gate_tolerance is not None:
                gate_size["tolerance"] = float(stage.gate_tolerance)
            t["gate_size"] = gate_size

        return tracks, weights

    def build_stage_tracks(
        self, *, config_dir: Path
    ) -> tuple[list[ConfigDict], list[float] | None]:
        """Load and stage-adjust tracks (gate size, active gate count) for the current stage."""
        return self.build_tracks_for_stage(stage_idx=int(self.stage_idx), config_dir=config_dir)

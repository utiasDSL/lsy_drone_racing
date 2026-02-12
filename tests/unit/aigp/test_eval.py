import numpy as np
import pytest

from lsy_drone_racing.aigp.curriculum import EvalSummary
from lsy_drone_racing.aigp.eval import aggregate_track_eval_summaries, evaluate_sb3_vec_env


class _FakeSB3VecEnv:
    def __init__(self):
        self.num_envs = 2
        self._step_idx = 0
        self._obs = {"feat": np.zeros((2, 1), dtype=np.float32)}
        self._episodes = [
            [
                {"success": True, "completion_fraction": 1.0, "lap_time_s": 1.0},
                {"success": False, "completion_fraction": 0.5, "lap_time_s": 2.0},
            ],
            [
                {"success": True, "completion_fraction": 1.0, "lap_time_s": 1.5},
                {"success": True, "completion_fraction": 1.0, "lap_time_s": 1.2},
            ],
        ]

    def reset(self) -> dict[str, np.ndarray]:
        self._step_idx = 0
        return self._obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float | bool]]]:  # noqa: ARG002
        infos = self._episodes[min(self._step_idx, len(self._episodes) - 1)]
        self._step_idx += 1
        obs = {"feat": np.full((2, 1), self._step_idx, dtype=np.float32)}
        reward = np.zeros((2,), dtype=np.float32)
        done = np.array([True, True], dtype=bool)
        return obs, reward, done, infos


@pytest.mark.unit
def test_evaluate_sb3_vec_env_aggregates_episode_metrics():
    env = _FakeSB3VecEnv()
    summary = evaluate_sb3_vec_env(
        env,
        lambda obs: np.zeros((obs["feat"].shape[0], 1), dtype=np.float32),
        n_episodes=4,
    )

    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(0.75)
    assert summary.completion_mean == pytest.approx(0.875)
    assert summary.lap_time_s_median == pytest.approx(1.2)


@pytest.mark.unit
def test_aggregate_track_eval_summaries_computes_bottomk_metrics():
    rows = [
        (0, "track_a", EvalSummary(10, 0.20, 0.30, 0.01, None)),
        (1, "track_b", EvalSummary(10, 0.80, 0.70, 0.02, 2.5)),
        (2, "track_c", EvalSummary(10, 1.00, 0.95, 0.03, 2.0)),
    ]
    payload = aggregate_track_eval_summaries(rows, bottomk_fraction=0.2)
    aggregate = payload["_aggregate"]

    assert len(payload["tracks"]) == 3
    assert aggregate["tracks_total"] == 3
    assert aggregate["tracks_covered"] == 3
    assert aggregate["success_rate_min"] == pytest.approx(0.20)
    assert aggregate["success_rate_bottom20_mean"] == pytest.approx(0.20)

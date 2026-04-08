"""Unit tests for the RaceCoreEnv class.

These tests focus on the race-core logic (obs, reward, terminated, truncated, close, gate-pass
detection) rather than physics or rendering. They exercise both the public properties and a few
"hidden" helpers (``_step_env``, ``_disabled_drones``) by crafting the env data directly, following
the same pattern used in the render integration tests.
"""

from pathlib import Path
from typing import Any

import gymnasium
import jax.numpy as jp
import numpy as np
import pytest

from lsy_drone_racing.envs.race_core import RaceCoreEnv
from lsy_drone_racing.utils import load_config

# Note: ``scipy`` must NOT be imported at module load time. ``lsy_drone_racing`` depends on
# ``crazyflow``, which sets ``SCIPY_ARRAY_API=1`` on import and raises if scipy was imported first.
# The only test that needs scipy (``test_gate_pass_increments_target_gate``) imports it locally.

CONFIG_PATH = Path(__file__).parents[3] / "config"


def make_env(config_name: str = "level0.toml", **overrides: Any) -> gymnasium.Env:
    """Build a single-drone env with sensible defaults that individual tests can override."""
    config = load_config(CONFIG_PATH / config_name)
    kwargs = dict(
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    kwargs.update(overrides)
    return gymnasium.make("DroneRacing-v0", **kwargs)


# region close


@pytest.mark.unit
def test_close_after_reset():
    """close() after a normal reset must not raise."""
    env = make_env()
    env.reset()
    env.close()


@pytest.mark.unit
def test_close_without_reset():
    """close() without ever calling reset() must not raise."""
    env = make_env()
    env.close()


# region obs


@pytest.mark.unit
def test_obs_structure_and_initial_values():
    """obs() returns the expected keys, lives in observation_space, and starts at gate 0."""
    env = make_env()
    obs, _ = env.reset()
    expected_keys = {
        "pos",
        "quat",
        "vel",
        "ang_vel",
        "target_gate",
        "gates_pos",
        "gates_quat",
        "gates_visited",
        "obstacles_pos",
        "obstacles_visited",
    }
    assert set(obs.keys()) == expected_keys
    # Single-drone make() squeezes leading (world, drone) dims: pos is (3,), gates_pos is
    # (n_gates, 3), target_gate is a 0-d scalar.
    assert np.asarray(obs["pos"]).shape == (3,)
    assert np.asarray(obs["gates_pos"]).ndim == 2
    assert np.asarray(obs["gates_pos"]).shape[1] == 3
    assert int(np.asarray(obs["target_gate"]).item()) == 0
    env.close()


@pytest.mark.unit
def test_obs_returns_nominal_when_out_of_sensor_range():
    """With sensor_range=0, gates/obstacles are not visited and obs returns nominal poses."""
    env = make_env(sensor_range=0.0)
    obs, _ = env.reset()
    assert not bool(jp.any(obs["gates_visited"])), "no gate should be visited"
    assert not bool(jp.any(obs["obstacles_visited"])), "no obstacle should be visited"

    nominal_gate_pos = np.asarray(env.unwrapped.gates["nominal_pos"])
    nominal_obstacle_pos = np.asarray(env.unwrapped.obstacles["nominal_pos"])
    np.testing.assert_allclose(np.asarray(obs["gates_pos"]), nominal_gate_pos, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(obs["obstacles_pos"]), nominal_obstacle_pos, rtol=1e-5)
    env.close()


@pytest.mark.unit
def test_obs_returns_real_pose_when_in_sensor_range():
    """With a huge sensor range, obs returns the actual mocap pose for each gate."""
    env = make_env(sensor_range=100.0)
    obs, _ = env.reset()
    assert bool(jp.all(obs["gates_visited"])), "all gates should be visited"
    assert bool(jp.all(obs["obstacles_visited"])), "all obstacles should be visited"

    gate_ids = env.unwrapped.data.gate_mj_ids
    obstacle_ids = env.unwrapped.data.obstacle_mj_ids
    real_gates = np.asarray(env.unwrapped.sim.mjx_data.mocap_pos[0, gate_ids])
    real_obstacles = np.asarray(env.unwrapped.sim.mjx_data.mocap_pos[0, obstacle_ids])
    np.testing.assert_allclose(np.asarray(obs["gates_pos"]), real_gates, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(obs["obstacles_pos"]), real_obstacles, rtol=1e-5)
    env.close()


# region reward


@pytest.mark.unit
def test_reward_zero_during_race():
    """While target_gate >= 0, the sparse reward is 0."""
    env = make_env()
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert float(np.asarray(reward).sum()) == 0.0
    env.close()


@pytest.mark.unit
def test_reward_minus_one_when_course_finished():
    """When target_gate == -1 (course finished), reward is -1 per drone."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(target_gate=data.target_gate.at[...].set(-1))
    reward = np.asarray(env.unwrapped.reward())
    assert reward.sum() == -1.0 * env.unwrapped.sim.n_drones
    env.close()


# region terminated


@pytest.mark.unit
def test_terminated_false_after_reset():
    """Fresh reset: no drone is disabled, so terminated is False."""
    env = make_env()
    env.reset()
    assert not bool(jp.any(env.unwrapped.terminated()))
    env.close()


@pytest.mark.unit
def test_terminated_true_when_target_gate_negative():
    """Setting target_gate = -1 marks the drone disabled on the next step."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(target_gate=data.target_gate.at[...].set(-1))
    env.step(env.action_space.sample())
    assert bool(jp.all(env.unwrapped.terminated()))
    env.close()


@pytest.mark.unit
def test_disabled_drones_out_of_bounds():
    """``_disabled_drones`` flags a drone positioned above pos_limit_high."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    # Place the drone well above the z upper limit (2.5). Shape: (n_worlds, n_drones, 3).
    pos = jp.asarray(env.unwrapped.sim.data.states.pos).at[..., 2].set(5.0)
    contacts = jp.zeros_like(env.unwrapped.sim.contacts())
    disabled = RaceCoreEnv._disabled_drones(pos, contacts, data)
    assert bool(jp.all(disabled)), "drone above pos_limit_high should be disabled"
    env.close()


@pytest.mark.unit
def test_disabled_drones_nominal_not_disabled():
    """``_disabled_drones`` does not flag a drone at its nominal starting pose without contacts."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    pos = env.unwrapped.sim.data.states.pos
    contacts = jp.zeros_like(env.unwrapped.sim.contacts())
    disabled = RaceCoreEnv._disabled_drones(pos, contacts, data)
    assert not bool(jp.any(disabled)), "nominal drone with no contacts should not be disabled"
    env.close()


@pytest.mark.unit
def test_disabled_drones_on_contact():
    """A masked contact (e.g. hitting an obstacle) disables the drone."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    pos = env.unwrapped.sim.data.states.pos
    # Set every contact to True; contact_masks selects the relevant ones per drone.
    contacts = jp.ones_like(env.unwrapped.sim.contacts())
    disabled = RaceCoreEnv._disabled_drones(pos, contacts, data)
    assert bool(jp.all(disabled)), "drone with active masked contacts should be disabled"
    env.close()


# region truncated


@pytest.mark.unit
def test_truncated_false_after_reset():
    """Fresh reset: steps=0, so truncated is False."""
    env = make_env()
    env.reset()
    assert not bool(jp.any(env.unwrapped.truncated()))
    env.close()


@pytest.mark.unit
def test_truncated_on_timeout_does_not_terminate():
    """Bumping steps to max_episode_steps flips truncated True while terminated stays False.

    This pins down two things at once: (1) ``truncated()`` fires on timeout, and (2) timeout is
    a separate signal from termination — despite the intuition that "for a single drone
    truncated == terminated", the two branches in race_core are genuinely independent.
    """
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(steps=data.steps.at[...].set(data.max_episode_steps))
    assert bool(jp.all(env.unwrapped.truncated()))
    assert not bool(jp.any(env.unwrapped.terminated()))
    env.close()


# region gate-pass (hidden function in _step_env)


@pytest.mark.unit
def test_gate_pass_increments_target_gate():
    """Straddling the current target gate's plane makes ``_step_env`` increment target_gate."""
    # Local import: scipy must not be imported at module load time (see note at top of file).
    from scipy.spatial.transform import Rotation as R

    env = make_env()
    env.reset()
    data = env.unwrapped.data

    # Current target gate (index 0 after reset). Shapes: mocap is (n_worlds, n_mocap, 3/4).
    gate_mj_id = int(np.asarray(data.gate_mj_ids[0]))
    gate_pos = np.asarray(env.unwrapped.sim.mjx_data.mocap_pos[0, gate_mj_id])
    # MuJoCo quat is wxyz; gate_passed (via _step_env) expects scipy xyzw order.
    gate_quat_mj = np.asarray(env.unwrapped.sim.mjx_data.mocap_quat[0, gate_mj_id])
    gate_quat_xyzw = gate_quat_mj[[1, 2, 3, 0]]
    # Gates are crossed from -x to +x in the local gate frame (see gate_passed docstring).
    forward = R.from_quat(gate_quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

    behind = gate_pos - 0.05 * forward  # last drone position: just before the gate plane
    front = gate_pos + 0.05 * forward  # current drone position: just past it

    # Craft data so that last_drone_pos is "behind" and current sim pos is "front".
    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(behind))
    env.unwrapped.data = data.replace(last_drone_pos=new_last)

    sim_data = env.unwrapped.sim.data
    new_pos = sim_data.states.pos.at[0, 0, :].set(jp.asarray(front))
    env.unwrapped.sim.data = sim_data.replace(states=sim_data.states.replace(pos=new_pos))

    # Call _step_env directly so physics doesn't overwrite our crafted positions.
    contacts = env.unwrapped.sim.contacts()
    new_data = RaceCoreEnv._step_env(
        env.unwrapped.data,
        env.unwrapped.sim.data.states.pos,
        env.unwrapped.sim.mjx_data.mocap_pos,
        env.unwrapped.sim.mjx_data.mocap_quat,
        contacts,
    )
    assert int(np.asarray(new_data.target_gate[0, 0])) == 1
    env.close()


@pytest.mark.unit
def test_gate_not_passed_without_crossing():
    """Moving around without crossing the gate plane leaves target_gate unchanged."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    assert int(np.asarray(data.target_gate[0, 0])) == 0
    # Nominal step without any crafted crossing: drone still on the takeoff pad.
    contacts = env.unwrapped.sim.contacts()
    new_data = RaceCoreEnv._step_env(
        data,
        env.unwrapped.sim.data.states.pos,
        env.unwrapped.sim.mjx_data.mocap_pos,
        env.unwrapped.sim.mjx_data.mocap_quat,
        contacts,
    )
    assert int(np.asarray(new_data.target_gate[0, 0])) == 0
    env.close()

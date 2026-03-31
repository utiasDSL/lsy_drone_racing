"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

After the controller finishes (or the run is terminated/truncated), the drone
will automatically hover at a safe position and wait for a keypress before
landing. This ensures a safe end-of-run protocol regardless of which
controller is used.
"""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
import rclpy
from cflib.crazyflie import Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

from ece484_fly.utils import load_config, load_controller

if TYPE_CHECKING:
    from ece484_fly.envs.real_race_env import RealDroneRaceEnv

logger = logging.getLogger(__name__)

HOVER_HEIGHT = 0.5  # m, safe hover height after run
HOVER_OFFSET = 0.3  # m, small offset ahead of last position along velocity
HOVER_TRANSITION_DURATION = 3.0  # s, time to reach hover position
LAND_HEIGHT = 0.0  # m, target landing height
LAND_DURATION = 2.0  # s, landing descent duration


def _safe_hover_and_land(env: RealDroneRaceEnv):
    """Command the drone to hover at a safe position, wait for keypress, then land.

    This replaces the default env.close() return-to-start behavior with a more
    controlled sequence: hover in place -> user confirms -> land.
    """
    real_env = env.unwrapped
    if not real_env.data.drone_connected or not real_env.data.taken_off:
        return

    # Switch from low-level to high-level commander
    real_env.drone.commander.send_stop_setpoint()
    real_env.drone.commander.send_notify_setpoint_stop()
    real_env.drone.param.set_value("commander.enHighLevel", "1")
    real_env.drone.platform.send_arming_request(True)

    # Compute hover position: current pos + small offset along velocity, at safe height
    pos = real_env._ros_connector.pos[real_env.drone_name].copy()
    vel = real_env._ros_connector.vel[real_env.drone_name]
    speed = np.linalg.norm(vel)
    if speed > 0.1:
        hover_pos = pos + vel / speed * HOVER_OFFSET
    else:
        hover_pos = pos.copy()
    hover_pos[2] = max(HOVER_HEIGHT, pos[2])  # don't descend, only go up or hold

    # Fly to hover position
    logger.info(f"Hovering at {hover_pos} ...")
    real_env.drone.high_level_commander.go_to(*hover_pos, 0, HOVER_TRANSITION_DURATION)
    _wait_and_update_pose(real_env, HOVER_TRANSITION_DURATION)

    # Hold hover and wait for user keypress
    logger.info("Drone is hovering. Press Enter to land...")
    try:
        input()
    except EOFError:
        pass

    # Land
    logger.info("Landing...")
    land_pos = hover_pos.copy()
    land_pos[2] = LAND_HEIGHT
    real_env.drone.high_level_commander.go_to(*land_pos, 0, LAND_DURATION)
    _wait_and_update_pose(real_env, LAND_DURATION)
    logger.info("Landed. Turning off drone...")

    # Stop motors and close radio link before env.close()
    try:
        pk = CRTPPacket()
        pk.port = CRTPPort.LOCALIZATION
        pk.channel = Localization.GENERIC_CH
        pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
        real_env.drone.send_packet(pk)
        time.sleep(0.1)
        real_env.drone.close_link()
    except Exception:
        pass
    logger.info("Drone turned off.")

    # Mark as handled so env.close() skips return-to-start and radio close
    real_env.data.taken_off = False
    real_env.data.drone_connected = False


def _wait_and_update_pose(real_env, duration: float):
    """Wait for a high-level command to complete while sending pose updates."""
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        if not rclpy.ok():
            break
        obs = real_env.obs()
        real_env.drone.extpos.send_extpose(*obs["pos"][real_env.rank], *obs["quat"][real_env.rank])
        time.sleep(0.05)


def main(config: str = "level1.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `ece484_fly/control/` or None. If None,
         the controller specified in the config file is used.
    """
    rclpy.init()
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if controller is not None:
        config.controller.file = controller

    env: RealDroneRaceEnv = gymnasium.make(
        "RealDroneRacing-v0",
        drones=config.deploy.drones,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    try:
        obs, info = env.reset(options=config.deploy)
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters

        control_path = Path(__file__).parents[1] / "ece484_fly/control"
        controller_path = control_path / config.controller.file
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = next_obs["target_gate"] == -1
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")

        # Safe end-of-run: hover and wait for user to confirm landing
        _safe_hover_and_land(env)
    finally:
        env.close()


if __name__ == "__main__":
    import os
    import warnings

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lark")

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("ece484_fly").setLevel(logging.INFO)
    fire.Fire(main)

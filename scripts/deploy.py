#!/usr/bin/env python
"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.yaml>

"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np

from lsy_drone_racing.command import Command
from lsy_drone_racing.constants import (
    CTRL_FREQ,
    CTRL_TIMESTEP,
    QUADROTOR_KF,
    QUADROTOR_KM,
    SENSOR_RANGE,
    Z_HIGH,
    Z_LOW,
    HighGateDesc,
    LowGateDesc,
    ObstacleDesc,
    QuadrotorPhysicParams,
)
from lsy_drone_racing.import_utils import get_ros_package_path, pycrazyswarm
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.ros_utils import check_drone_start_pos, check_race_track
from lsy_drone_racing.vicon import Vicon
from lsy_drone_racing.wrapper import DroneRacingWrapper

if TYPE_CHECKING:
    from munch import Munch

# rospy.init_node changes the default logging configuration of Python, which is bad practice at
# best. As a workaround, we can create loggers under the ROS root logger `rosout`.
# Also see https://github.com/ros/ros_comm/issues/1384
logger = logging.getLogger("rosout." + __name__)


def get_init_info(
    env: Quadrotor, env_info: dict, vicon: Vicon, constraint_values: list, config: Munch
) -> dict:
    """Get the initial information dictionary for the controller.

    Args:
        env: Quadrotor environment.
        env_info: The environment information dictionary.
        vicon: Vicon interface.
        constraint_values: The values of the environment constraints evaluated at the start state.
        config: Competition configuration.
    """
    x_reference = config.quadrotor_config.task_info.stabilization_goal
    info = {
        "symbolic_model": env_info["symbolic_model"],
        "nominal_physical_parameters": asdict(QuadrotorPhysicParams()),
        "x_reference": [x_reference[0], 0.0, x_reference[1], 0.0, x_reference[2], 0.0] + [0.0] * 6,
        "u_reference": [0.084623, 0.084623, 0.084623, 0.084623],
        "symbolic_constraints": env_info["symbolic_constraints"],
        "ctrl_timestep": CTRL_TIMESTEP,
        "ctrl_freq": CTRL_FREQ,
        "episode_len_sec": 33,
        "quadrotor_kf": QUADROTOR_KF,
        "quadrotor_km": QUADROTOR_KM,
        "gate_dimensions": {"low": asdict(LowGateDesc()), "tall": asdict(HighGateDesc())},
        "obstacle_dimensions": asdict(ObstacleDesc()),
        "nominal_gates_pos_and_type": np.array(config.quadrotor_config.gates),
        "nominal_obstacles_pos": np.array(config.quadrotor_config.obstacles),
        "initial_state_randomization": env_info["initial_state_randomization"],
        "inertial_prop_randomization": env_info["inertial_prop_randomization"],
        "gates_and_obs_randomization": env_info["gates_and_obs_randomization"],
        "disturbances": env_info["disturbances"],
        "pyb_client": None,
        "urdf_dir": None,
        "constraint_values": constraint_values,
    }
    info.update(get_info(env, vicon, 0, config))
    return info


def sync_env(env: Quadrotor, vicon: Vicon):
    """Synchronize the internal env state with the real observations from vicon.

    Args:
        env: The Quadrotor environment.
        vicon: Vicon interface.
    """
    p, r = vicon.pos[vicon.drone_name], vicon.rpy[vicon.drone_name]
    env.state = [p[0], 0, p[1], 0, p[2], 0, r[0], r[1], r[2], 0, 0, 0]
    constraint_values = env.constraints.get_values(env, only_state=True)
    # IROS 2022 - Constrain violation flag for reward.
    env.cnstr_violation = env.constraints.is_violated(env, c_value=constraint_values)


def get_observation(info: dict, vicon: Vicon) -> np.ndarray:
    """Get the current observation from Vicon data.

    Args:
        info: The info dictionary corresponding to the current observation.
        vicon: Vicon interface.

    Returns:
        An observation from real data that matches the simulated observation space.
    """
    pos, rpy = vicon.pos[vicon.drone_name], vicon.rpy[vicon.drone_name]
    vel = vicon.vel[vicon.drone_name]
    ang_vel = vicon.ang_vel[vicon.drone_name]
    obs = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], *rpy, *ang_vel])
    return DroneRacingWrapper.observation_transform(obs, info)


def get_info(env: Quadrotor, vicon: Vicon, gate_id: int, config: Munch) -> dict:
    """Get the current environment info from Vicon data.

    Args:
        env: Quadrotor environment.
        vicon: Vicon interface.
        gate_id: Target gate ID.
        config: Competition configuration.

    Returns:
        An observation from real data that matches the simulated observation space.
    """
    gates = np.array(config.quadrotor_config.gates)
    # Infer gate z position based on type
    gates[:, 2] = np.where(gates[:, 6] == 1, Z_LOW, Z_HIGH)
    obstacles_pose = np.array(config.quadrotor_config.obstacles)
    obstacles_pose[:, 2] = ObstacleDesc.height
    drone_pos = vicon.pos[vicon.drone_name][:2]
    gates_in_range = np.linalg.norm(gates[:, :2] - drone_pos, axis=1) < SENSOR_RANGE
    obstacles_in_range = np.linalg.norm(obstacles_pose[:, :2] - drone_pos, axis=1) < SENSOR_RANGE

    # Update gates and obstacles that are closer than SENSOR_RANGE
    for i, in_range in enumerate(gates_in_range):
        if in_range:  # Overwrite both position and orientation
            gates[i, :2] = vicon.pos[f"gate{i+1}"][:2]
            gates[i, 3] = vicon.rpy[f"gate{i+1}"][2]
    for i, in_range in enumerate(obstacles_in_range):
        if in_range:  # Obstacles are symmetric, don't use orientation
            obstacles_pose[i, :2] = vicon.pos[f"obstacle{i+1}"][:2]

    info = {
        "mse": np.sum(((env.state - env.X_GOAL) * env.info_mse_metric_state_weight) ** 2),
        "collision": (None, False),  # Leave always False for deploy
        "gates_pose": gates[:, :6],
        "obstacles_pose": obstacles_pose,
        "gates_in_range": gates_in_range,
        "obstacles_in_range": obstacles_in_range,
        "gates_type": gates[:, 6],
        "current_gate_id": gate_id,
        "at_goal_position": False,  # Leave always False for deployment
        "task_completed": False,  # Leave always False for deployment
        "drone_pose": np.concatenate(vicon.pose(vicon.drone_name)),
        "constraint_values": env.constraints.get_values(env, only_state=True),
        "constraint_violation": 1 if env.cnstr_violation else 0,
    }
    return info


def main(config: str = "config/getting_started.yaml", controller: str = "examples/controller.py"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration.
        controller: Path to the controller implementation.
    """
    config = load_config(Path(config))

    # Initialize the crazyfly interface
    # pycrazyswarm expects strings, not Path objects, so we need to convert it first
    crazyswarm_config_path = get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml"
    swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
    time_helper = swarm.timeHelper

    # Check if the gates, obstacles and drone are positioned within tolerances. This needs to be
    # called after initializing crazyswarm. Vicon and crazyswarm attempt to initialize a ROS node.
    # Vicon handles the case that a node is already running, but crazyswarm does not
    check_race_track(config)
    check_drone_start_pos(config)

    # Initialize the environment with ROS as backend instead of the simulator
    kwargs = {"drone": swarm.allcfs.crazyflies[0]}
    env = gymnasium.make("DroneRacing-v0", config=config, backend="ros", backend_kwargs=kwargs)
    obs, info = env.reset()

    Controller = load_controller(Path(controller))
    ctrl = Controller(obs, info)

    # Run the main control loop
    try:
        start_time = time.time()
        while not time_helper.isShutdown():
            t_loop = time.perf_counter()
            # Use the latest observation of the environment, i.e. the latest state from Vicon
            obs, info = env.obs, env.info
            if last_obs.get("gate_id", 1) != obs["gate_id"]:
                logger.info(
                    f"Gate {last_obs.get("gate_id", 1)} passed in {time.time() - start_time:.4}s"
                )
            action = ctrl.compute_control(env.obs, env.info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ctrl.step_learn(action, next_obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            if dt := (time.perf_counter() - t_loop) < CTRL_TIMESTEP:
                time.sleep(CTRL_TIMESTEP - dt)  # Maintain the control loop frequency
            next_obs = env.obs  # Update the observation after the control loop
        total_time = time.time() - start_time
        ctrl.episode_learn()
        logger.info(f"Total time: {total_time:.3f}s" if obs["gate"] == -1 else "Task not completed")
    finally:
        env.apply(Command.NOTIFYSETPOINTSTOP, [])
        env.apply(Command.LAND, [0.0, 2.0])  # Args are height and duration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

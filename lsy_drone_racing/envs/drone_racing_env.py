"""Collection of environments for drone racing simulations.

This module is a core component of the lsy_drone_racing package, providing the primary interface
between the drone racing simulation and the user's control algorithms.

It serves as a bridge between the high-level race control and the low-level drone physics
simulation. The environments defined here
(:class:`~.DroneRacingEnv` and :class:`~.DroneRacingThrustEnv`) expose a common interface for all
controller types, allowing for easy integration and testing of different control algorithms,
comparison of control strategies, and deployment on our hardware.

Key roles in the project:

* Abstraction Layer: Provides a standardized Gymnasium interface for interacting with the
  drone racing simulation, abstracting away the underlying physics engine.
* State Management: Handles the tracking of race progress, gate passages, and termination
  conditions.
* Observation Processing: Manages the transformation of raw simulation data into structured
  observations suitable for control algorithms.
* Action Interpretation: Translates high-level control commands into appropriate inputs for the
  underlying simulation.
* Configuration Interface: Allows for easy customization of race scenarios, environmental
  conditions, and simulation parameters.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium import spaces
# import minsnap_trajectories as ms
from scipy.interpolate import CubicSpline, CubicHermiteSpline
# from roboticstoolbox import quintic
# from .quintic import QuinticSpline
import pybullet as p
import copy as copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import check_gate_pass
from lsy_drone_racing.utils.quintic_spline import QuinticSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingEnv(gymnasium.Env):
    """A Gymnasium environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It uses the Sim class for physics simulation and supports
    various configuration options for randomization, disturbances, and physics models.

    The environment provides:
    - A customizable track with gates and obstacles
    - Configurable simulation and control frequencies
    - Support for different physics models (e.g., PyBullet, mathematical dynamics)
    - Randomization of drone properties and initial conditions
    - Disturbance modeling for realistic flight conditions
    - Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:
    - "pos": Drone position
    - "rpy": Drone orientation (roll, pitch, yaw)
    - "vel": Drone linear velocity
    - "ang_vel": Drone angular velocity
    - "gates.pos": Positions of the gates
    - "gates.rpy": Orientations of the gates
    - "gates.in_range": Flags indicating if the drone is in the sensor range of the gates
    - "obstacles.pos": Positions of the obstacles
    - "obstacles.in_range": Flags indicating if the drone is in the sensor range of the obstacles
    - "target_gate": The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller.
    """

    CONTROLLER = "mellinger"  # specifies controller type

    def __init__(self, config: dict):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.config = config
        self.sim = Sim(
            track=config.env.track,
            sim_freq=config.sim.sim_freq,
            ctrl_freq=config.sim.ctrl_freq,
            disturbances=getattr(config.sim, "disturbances", {}),
            randomization=getattr(config.env, "randomization", {}),
            gui=config.sim.gui,
            n_drones=1,
            physics=config.sim.physics,
        )
        self.sim.seed(config.env.seed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(13,))
        n_gates, n_obstacles = len(self.sim.gates), len(self.sim.obstacles)
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "acc": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "target_gate": spaces.Discrete(n_gates, start=-1),
                "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
                "gates_rpy": spaces.Box(low=-np.pi, high=np.pi, shape=(n_gates, 3)),
                "gates_in_range": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=np.bool_),
                "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
                "obstacles_in_range": spaces.Box(
                    low=0, high=1, shape=(n_obstacles,), dtype=np.bool_
                ),
            }
        )
        self.target_gate = 0
        self.symbolic = self.sim.symbolic() if config.env.symbolic else None
        self._steps = 0
        self._last_drone_pos = np.zeros(3)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Observation and info.
        """
        # The system identification model is based on the attitude control interface. We cannot
        # support its use with the full state control interface
        if self.config.env.reseed:
            self.sim.seed(self.config.env.seed)
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        self.target_gate = 0
        self._steps = 0
        self.sim.drone.reset(self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel)
        self._last_drone_pos[:] = self.sim.drone.pos
        if self.sim.n_drones > 1:
            raise NotImplementedError("Firmware wrapper does not support multiple drones.")
        info = self.info
        info["sim_freq"] = self.config.sim.sim_freq
        info["low_level_ctrl_freq"] = self.config.sim.ctrl_freq
        info["drone_mass"] = self.sim.drone.params.mass
        info["env_freq"] = self.config.env.freq
        return self.obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        assert (
            self.config.sim.physics != PhysicsMode.SYS_ID
        ), "sys_id model not supported for full state control interface"
        action = action.astype(np.float64)  # Drone firmware expects float64
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.sim.drone.full_state_cmd(pos, vel, acc, yaw, rpy_rate)
        collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info

    def _inner_step_loop(self) -> bool:
        """Run the desired action for multiple simulation sub-steps.

        The outer controller is called at a lower frequency than the firmware loop. Each environment
        step therefore consists of multiple simulation steps. At each simulation step, the
        lower-level controller is called to compute the thrust and attitude commands.

        Returns:
            True if a collision occured at any point during the simulation steps, else False.
        """
        thrust = self.sim.drone.desired_thrust
        collision = False
        while (
            self.sim.drone.tick / self.sim.drone.firmware_freq
            < (self._steps + 1) / self.config.env.freq
        ):
            self.sim.step(thrust)
            self.target_gate += self.gate_passed()
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1
            collision |= bool(self.sim.collisions)
            pos, rpy, vel = self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel
            thrust = self.sim.drone.step_controller(pos, rpy, vel)[::-1]
        self.sim.drone.desired_thrust[:] = thrust
        self._last_drone_pos[:] = self.sim.drone.pos
        self._steps += 1
        return collision

    @property
    def obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the observation of the environment."""
        obs = {
            "pos": self.sim.drone.pos.astype(np.float32),
            "rpy": self.sim.drone.rpy.astype(np.float32),
            "vel": self.sim.drone.vel.astype(np.float32),
            "ang_vel": self.sim.drone.ang_vel.astype(np.float32),
        }
        self.last_vel = copy.deepcopy(obs["vel"])
        obs["ang_vel"][:] = R.from_euler("xyz", obs["rpy"]).apply(obs["ang_vel"], inverse=True)

        gates = self.sim.gates
        obs["target_gate"] = self.target_gate if self.target_gate < len(gates) else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        in_range = self.sim.in_range(gates, self.sim.drone, self.config.env.sensor_range)
        gates_pos = np.stack([g["nominal.pos"] for g in gates.values()])
        gates_pos[in_range] = np.stack([g["pos"] for g in gates.values()])[in_range]
        gates_rpy = np.stack([g["nominal.rpy"] for g in gates.values()])
        gates_rpy[in_range] = np.stack([g["rpy"] for g in gates.values()])[in_range]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_in_range"] = in_range

        obstacles = self.sim.obstacles
        in_range = self.sim.in_range(obstacles, self.sim.drone, self.config.env.sensor_range)
        obstacles_pos = np.stack([o["nominal.pos"] for o in obstacles.values()])
        obstacles_pos[in_range] = np.stack([o["pos"] for o in obstacles.values()])[in_range]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_in_range"] = in_range

        if "observation" in self.sim.disturbances:
            obs = self.sim.disturbances["observation"].apply(obs)
        return obs

    @property
    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 if self.target_gate != -1 else 0.0

    @property
    def terminated(self) -> bool:
        """Check if the episode is terminated.

        Returns:
            True if the drone is out of bounds, colliding with an obstacle, or has passed all gates,
            else False.
        """
        state = {k: getattr(self.sim.drone, k).copy() for k in self.sim.state_space}
        state["ang_vel"] = R.from_euler("xyz", state["rpy"]).apply(state["ang_vel"], inverse=True)
        if state not in self.sim.state_space:
            return True  # Drone is out of bounds
        if self.sim.collisions:
            return True
        if self.target_gate == -1:  # Drone has passed all gates
            return True
        return False

    @property
    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": self.sim.collisions, "symbolic_model": self.symbolic}

    def gate_passed(self) -> bool:
        """Check if the drone has passed a gate.

        Returns:
            True if the drone has passed a gate, else False.
        """
        if self.sim.n_gates > 0 and self.target_gate < self.sim.n_gates and self.target_gate != -1:
            gate_pos = self.sim.gates[self.target_gate]["pos"]
            gate_rot = R.from_euler("xyz", self.sim.gates[self.target_gate]["rpy"])
            drone_pos = self.sim.drone.pos
            last_drone_pos = self._last_drone_pos
            gate_size = (0.45, 0.45)
            return check_gate_pass(gate_pos, gate_rot, gate_size, drone_pos, last_drone_pos)
        return False

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""

        debug = True # print statements + makes the tracks plot in sim
        decelerate = False # makes the drone simulate the stopping motion
        return_home = False # makes the drone simulate the return to home after stopping
        if return_home: decelerate = True

        x_end = 2.0 # distance the drone is supposed to stop behind the last gate
        t_RTH = 9.0 # time it takes to get back to start (return to home)
        t_step_ctrl = 1/self.config.env.freq # control step
        t_hover = 2.0 # time the drone hovers before RTH and before landing
        height_homing = 2.0 # height of the return path
        
        ##### First, simply stop the drone with constant acceleration 
        if decelerate:
            # do one step to get the current acceleration
            start_pos = self.obs["pos"]
            start_vel = self.obs["vel"]
            self.step(np.array([*start_pos,
                                *start_vel,
                                0,0,0,
                                0,0,0,0]))
            time.sleep(t_step_ctrl)

            start_acc = -(start_vel - self.obs["vel"])/t_step_ctrl
            
            if debug:
                obs_x = []
                obs_v = []
                cmd_x = []
                cmd_v = []
                cmd_a = []
                start_pos = self.obs["pos"]
                start_vel = self.obs["vel"]
                obs_x.append(self.obs["pos"])
                obs_v.append(self.obs["vel"])
                cmd_x.append(start_pos)
                cmd_v.append(start_vel)
                cmd_a.append(start_acc)

            direction = start_vel/np.linalg.norm(start_vel) # unit vector in the direction of travel
            direction_angle = np.arccos( (np.dot(direction, [0,0,1])) / (1*1) ) 
            direction_angle = -(direction_angle-np.pi/2) # angle to the floor => negative means v_z < 0
            
            # drone can actually go further to no reach the x_end limit if angle!=0
            x_end = x_end/np.cos(direction_angle)
            # check if drone would crash into floor or ceiling
            x_end_z = start_pos[2] + x_end*np.sin(direction_angle)
            if x_end_z < 0.2: 
                if debug: print("x_end_z<0.2")
                x_end = (0.2 - start_pos[2])/np.sin(direction_angle)
            elif x_end_z > 2.5:
                if debug: print("x_end_z>2.5")
                x_end = (2.5 - start_pos[2])/np.sin(direction_angle)

            if debug: 
                print(f"start_pos_z={start_pos[2]}, x_end_z={start_pos[2] + x_end*np.sin(direction_angle)}, x_end={x_end}")
                print(f"direction_angle={direction_angle*180/np.pi}°")

            const_acc = np.linalg.norm(start_vel)**2/(x_end) # this is just an estimate of what constant deceleration is necessary
            t_brake_max = np.sqrt(4*x_end/const_acc) # the time it takes to brake completely
            t_brake = np.arange(0, t_brake_max, t_step_ctrl)

            if debug:
                print(f"t_brake={t_brake_max}")
                print(f"v_gate = {start_vel}")

            quintic_spline = QuinticSpline(0, t_brake_max, start_pos, start_vel, start_acc, 
                                           start_pos+direction*x_end, start_vel*0, start_acc*0)
            ref_pos_stop = quintic_spline(t_brake, order=0)
            ref_vel_stop = quintic_spline(t_brake, order=1)
            ref_acc_stop = quintic_spline(t_brake, order=2)

            if debug:
                try:
                    step = 5
                    for i in np.arange(0, len(ref_pos_stop[:,0]) - step, step):
                        p.addUserDebugLine(
                            ref_pos_stop[i,:],
                            ref_pos_stop[i + step,:],
                            lineColorRGB=[0, 1, 0],
                            lineWidth=2,
                            lifeTime=0,  # 0 means the line persists indefinitely
                            physicsClientId=0,
                        )
                except p.error:
                    print("PyBullet not available") # Ignore errors if PyBullet is not available

            # apply controls to slow down
            for i in np.arange(len(ref_pos_stop)):
                pos = self.obs["pos"]
                vel = self.obs["vel"]
                direction = vel/np.linalg.norm(vel[:2])
                if debug:
                    obs_x.append(self.obs["pos"])
                    obs_v.append(self.obs["vel"])
                    cmd_x.append(ref_pos_stop[i])
                    cmd_v.append(ref_vel_stop[i])
                    cmd_a.append(ref_acc_stop[i])
                if np.linalg.norm(vel) < 0.05:
                    print(f"leaving stopping loop at v={vel}")
                    break

                self.step(np.array([ref_pos_stop[i,0],ref_pos_stop[i,1],ref_pos_stop[i,2],
                                    ref_vel_stop[i,0],ref_vel_stop[i,1],ref_vel_stop[i,2],
                                    ref_acc_stop[i,0],ref_acc_stop[i,1],ref_acc_stop[i,2],
                                    0,0,0,0]))
                time.sleep(t_step_ctrl)

            # apply controls to hover for some more time
            for i in np.arange(int(t_hover/t_step_ctrl)): 
                if debug:
                    obs_x.append(self.obs["pos"])
                    obs_v.append(self.obs["vel"])
                    cmd_x.append(np.array(pos))
                    cmd_v.append(np.array([0,0,0]))
                    cmd_a.append(np.array([0,0,0]))
                self.step(np.array([pos[0],pos[1],pos[2],
                                    0,0,0,0,0,0,
                                    0,0,0,0]))
                time.sleep(t_step_ctrl)

        ##### Second, return to home along a spline
        if return_home:
            t_returnhome = np.arange(0, t_RTH, t_step_ctrl)
            start_pos = self.obs["pos"]
            start_vel = self.obs["vel"]

            landing_pos = np.array([*self.config.env.track.drone.pos[:2], 0.25]) # 0.25m above actual landing pos

            intermed_delta = landing_pos - start_pos
            intermed_pos1 = [start_pos[0] + intermed_delta[0]/4, start_pos[1] + intermed_delta[1]/4, height_homing]
            intermed_pos2 = [intermed_pos1[0] + intermed_delta[0]/2, intermed_pos1[1] + intermed_delta[1]/2, intermed_pos1[2]]
            intermed_pos3 = [0,0,0]
            intermed_pos3[0] = (5*landing_pos[0]+intermed_pos2[0])/6
            intermed_pos3[1] = (5*landing_pos[1]+intermed_pos2[1])/6
            intermed_pos3[2] = (landing_pos[2]+intermed_pos2[2])/2

            waypoints = np.array([
                    start_pos,
                    intermed_pos1,
                    intermed_pos2,
                    intermed_pos3,
                    landing_pos,
                ])
            spline = CubicSpline(np.linspace(0, t_RTH, len(waypoints)), waypoints, bc_type=((1, [0,0,0]), (1, [0,0,0]))) # bc type set boundary conditions for the derivative (here 1)
            ref_pos_return = spline(t_returnhome)
            spline_v = spline.derivative()
            ref_vel_return = spline_v(t_returnhome)

            if debug:
                try:
                    step = 5
                    for i in np.arange(0, len(ref_pos_return) - step, step):
                        p.addUserDebugLine(
                            ref_pos_return[i],
                            ref_pos_return[i + step],
                            lineColorRGB=[0, 0, 1],
                            lineWidth=2,
                            lifeTime=0,  # 0 means the line persists indefinitely
                            physicsClientId=0,
                        )
                    p.addUserDebugText("x", start_pos, textColorRGB=[0,1,0])
                    p.addUserDebugText("x", intermed_pos1, textColorRGB=[0,0,1])
                    p.addUserDebugText("x", intermed_pos2, textColorRGB=[0,0,1])
                    p.addUserDebugText("x", intermed_pos3, textColorRGB=[0,0,1])
                    p.addUserDebugText("x", landing_pos, textColorRGB=[0,0,1])
                except p.error:
                    print("PyBullet not available") # Ignore errors if PyBullet is not available
                

            for i in np.arange(len(t_returnhome)):
                self.step(np.array([ref_pos_return[i,0],ref_pos_return[i,1],ref_pos_return[i,2],
                                    ref_vel_return[i,0],ref_vel_return[i,1],ref_vel_return[i,2],
                                    0,0,0,0,0,0,0]))
                time.sleep(t_step_ctrl)

            
            for i in np.arange(int(t_hover/t_step_ctrl)): # hover for some more time
                self.step(np.array([ref_pos_return[-1,0],ref_pos_return[-1,1],ref_pos_return[-1,2],0,0,0,0,0,0,0,0,0,0]))
                time.sleep(t_step_ctrl)

        if debug:
            plt.rcParams.update(plt.rcParamsDefault)
            fig, ax = plt.subplots(1,3)
            obs_x = np.array(obs_x)
            cmd_x = np.array(cmd_x)
            ax[0].plot(np.arange(len(obs_x[:,0])), -obs_x[:,0], label="obs")
            ax[0].plot(np.arange(len(cmd_x[:,0])), -cmd_x[:,0], label="cmd")
            ax[0].set_title("Position")

            obs_v = np.array(obs_v)
            cmd_v = np.array(cmd_v)
            ax[1].plot(np.arange(len(obs_v[:,0])), -obs_v[:,0], label="obs")
            ax[1].plot(np.arange(len(cmd_v[:,0])), -cmd_v[:,0], label="cmd")
            ax[1].set_title("Velocity")

            obs_a = np.array(obs_v)
            cmd_a = np.array(cmd_a)
            obs_a[1:-1] = (obs_a[2:]-obs_a[:-2])/(2*t_step_ctrl)
            ax[2].plot(np.arange(len(obs_a[:,0])), -obs_a[:,0], label="obs")
            ax[2].plot(np.arange(len(cmd_a[:,0])), -cmd_a[:,0], label="cmd")
            ax[2].set_title("Acceleration")
            plt.legend()
            plt.show()    

        self.sim.close()


class DroneRacingThrustEnv(DroneRacingEnv):
    """Drone racing environment with a collective thrust attitude command interface.

    The action space consists of the collective thrust and body-fixed attitude commands
    [collective_thrust, roll, pitch, yaw].
    """

    def __init__(self, config: dict):
        """Initialize the DroneRacingThrustEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__(config)
        bounds = np.array([1, np.pi, np.pi, np.pi], dtype=np.float32)
        self.action_space = spaces.Box(low=-bounds, high=bounds)

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Step the drone racing environment with a thrust and attitude command.

        Args:
            action: Thrust command [thrust, roll, pitch, yaw].
        """
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        action = action.astype(np.float64)
        collision = False
        # We currently need to differentiate between the sys_id backend and all others because the
        # simulation step size is different for the sys_id backend (we do not substep in the
        # identified model). In future iterations, the sim API should be flexible to handle both
        # cases without an explicit step_sys_id function.
        if self.config.sim.physics == "sys_id":
            cmd_thrust, cmd_rpy = action[0], action[1:]
            self.sim.step_sys_id(cmd_thrust, cmd_rpy, 1 / self.config.env.freq)
            self.target_gate += self.gate_passed()
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1
            self._last_drone_pos[:] = self.sim.drone.pos
        else:
            # Crazyflie firmware expects negated pitch command. TODO: Check why this is the case and
            # fix this on the firmware side if possible.
            cmd_thrust, cmd_rpy = action[0], action[1:] * np.array([1, -1, 1])
            self.sim.drone.collective_thrust_cmd(cmd_thrust, cmd_rpy)
            collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info

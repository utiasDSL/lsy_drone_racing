"""Deployment environments for real-world drone racing.

This module provides environments for deploying drone racing algorithms on physical hardware,
mirroring the functionality of the simulation environments in the
:mod:`~lsy_drone_racing.envs.drone_racing_env` module.

Key components:

* :class:`~.DroneRacingDeployEnv`: A Gymnasium environment for controlling a real Crazyflie drone in
  a physical race track, using Vicon motion capture for positioning.
* :class:`~.DroneRacingThrustDeployEnv`: A variant of :class:`~.DroneRacingDeployEnv` that uses
  collective thrust and attitude commands for control.

These environments maintain consistent interfaces with their simulation counterparts
(:class:`~.DroneRacingEnv` and :class:`~.DroneRacingThrustEnv`), allowing for seamless transition
from simulation to real-world deployment. They handle the complexities of interfacing with physical
hardware while providing the same observation and action spaces as the simulation environments.

The module integrates with ROS, Crazyswarm, and Vicon systems to enable real-world drone control and
tracking in a racing scenario.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
import minsnap_trajectories as ms
from scipy.interpolate import CubicSpline
from lsy_drone_racing.utils.quintic_spline import QuinticSpline
import rospy
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.drone import Drone
from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import check_gate_pass
from lsy_drone_racing.utils.import_utils import get_ros_package_path, pycrazyswarm
from lsy_drone_racing.utils.ros_utils import check_drone_start_pos, check_race_track
from lsy_drone_racing.vicon import Vicon

if TYPE_CHECKING:
    from munch import Munch
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingDeployEnv(gymnasium.Env):
    """A Gymnasium environment for deploying drone racing algorithms on real hardware.

    This environment mirrors the functionality of the
    class:~lsy_drone_racing.envs.drone_racing_env.DroneRacingEnv, but interfaces with real-world
    hardware (Crazyflie drone and Vicon motion capture system) instead of a simulation.

    Key features:
    - Interfaces with a Crazyflie drone for physical control
    - Uses Vicon motion capture for precise position tracking
    - Maintains the same observation and action spaces as the simulation environment
    - Tracks progress through gates in a real-world racing scenario
    - Provides safety checks for drone positioning and track setup

    The observation space and action space are the same as the
    class:~lsy_drone_racing.envs.drone_racing_env.DroneRacingEnv.

    This environment allows for a transition from simulation to real-world deployment, maintaining
    consistent interfaces and functionalities to avoid any code changes when moving from simulation
    to physical hardware.
    """

    CONTROLLER = "mellinger"

    def __init__(self, config: dict | Munch):
        """Initialize the crazyflie drone and the motion capture system.

        Args:
            config: The configuration of the environment.
        """
        super().__init__()
        self.config = config
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(13,))
        n_gates, n_obstacles = (
            len(config.env.track.get("gates")),
            len(config.env.track.get("obstacles")),
        )
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
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
        crazyswarm_config_path = (
            get_ros_package_path("crazyswarm", heuristic_search=True) / "launch/crazyflies.yaml"
        )
        # pycrazyswarm expects strings, not Path objects, so we need to convert it first
        swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
        self.cf = swarm.allcfs.crazyflies[0]
        names = [f"gate{g}" for g in range(1, len(config.env.track.gates) + 1)]
        names += [f"obstacle{g}" for g in range(1, len(config.env.track.obstacles) + 1)]
        self.vicon = Vicon(track_names=names, timeout=5)
        self.symbolic = None
        if config.env.symbolic:
            sim = Sim(
                track=config.env.track,
                sim_freq=config.sim.sim_freq,
                ctrl_freq=config.sim.ctrl_freq,
                disturbances=getattr(config.sim, "disturbances", {}),
                randomization=getattr(config.env, "randomization", {}),
                physics=config.sim.physics,
            )
            self.symbolic = sim.symbolic()
        self._last_pos = np.zeros(3)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        """Reset the environment.

        We cannot reset the track in the real world. Instead, we check if the gates, obstacles and
        drone are positioned within tolerances.
        """
        check_race_track(self.config)
        check_drone_start_pos(self.config)
        self._last_pos[:] = self.vicon.pos[self.vicon.drone_name]
        self.target_gate = 0
        info = self.info
        info["sim_freq"] = self.config.sim.sim_freq
        info["low_level_ctrl_freq"] = self.config.sim.ctrl_freq
        info["env_freq"] = self.config.env.freq
        info["drone_mass"] = 0.033  # Crazyflie 2.1 mass in kg
        return self.obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Take a step in the environment.

        Warning:
            Step does *not* wait for the remaining time if the step took less than the control
            period. This ensures that controllers with longer action compute times can still hit
            their target frequency. Furthermore, it implies that loops using step have to manage the
            frequency on their own, and need to update the current observation and info before
            computing the next action.
        """
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rate)
        current_pos = self.vicon.pos[self.vicon.drone_name]
        self.target_gate += self.gate_passed(current_pos, self._last_pos)
        self._last_pos[:] = current_pos
        if self.target_gate >= len(self.config.env.track.gates):
            self.target_gate = -1
        terminated = self.target_gate == -1
        return self.obs, -1.0, terminated, False, self.info

    def close(self):
        """Close the environment by stopping the drone and landing."""
        
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

            # if debug:
            #     try:
            #         step = 5
            #         for i in np.arange(0, len(ref_pos_stop[:,0]) - step, step):
            #             p.addUserDebugLine(
            #                 ref_pos_stop[i,:],
            #                 ref_pos_stop[i + step,:],
            #                 lineColorRGB=[0, 1, 0],
            #                 lineWidth=2,
            #                 lifeTime=0,  # 0 means the line persists indefinitely
            #                 physicsClientId=0,
            #             )
            #     except p.error:
            #         print("PyBullet not available") # Ignore errors if PyBullet is not available

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

            # if debug:
            #     try:
            #         step = 5
            #         for i in np.arange(0, len(ref_pos_return) - step, step):
            #             p.addUserDebugLine(
            #                 ref_pos_return[i],
            #                 ref_pos_return[i + step],
            #                 lineColorRGB=[0, 0, 1],
            #                 lineWidth=2,
            #                 lifeTime=0,  # 0 means the line persists indefinitely
            #                 physicsClientId=0,
            #             )
            #         p.addUserDebugText("x", start_pos, textColorRGB=[0,1,0])
            #         p.addUserDebugText("x", intermed_pos1, textColorRGB=[0,0,1])
            #         p.addUserDebugText("x", intermed_pos2, textColorRGB=[0,0,1])
            #         p.addUserDebugText("x", intermed_pos3, textColorRGB=[0,0,1])
            #         p.addUserDebugText("x", landing_pos, textColorRGB=[0,0,1])
            #     except p.error:
            #         print("PyBullet not available") # Ignore errors if PyBullet is not available
                

            for i in np.arange(len(t_returnhome)):
                self.step(np.array([ref_pos_return[i,0],ref_pos_return[i,1],ref_pos_return[i,2],
                                    ref_vel_return[i,0],ref_vel_return[i,1],ref_vel_return[i,2],
                                    0,0,0,0,0,0,0]))
                time.sleep(t_step_ctrl)

            
            for i in np.arange(int(t_hover/t_step_ctrl)): # hover for some more time
                self.step(np.array([ref_pos_return[-1,0],ref_pos_return[-1,1],ref_pos_return[-1,2],0,0,0,0,0,0,0,0,0,0]))
                time.sleep(t_step_ctrl)

        # if debug:
        #     plt.rcParams.update(plt.rcParamsDefault)
        #     fig, ax = plt.subplots(1,3)
        #     obs_x = np.array(obs_x)
        #     cmd_x = np.array(cmd_x)
        #     ax[0].plot(np.arange(len(obs_x[:,0])), -obs_x[:,0], label="obs")
        #     ax[0].plot(np.arange(len(cmd_x[:,0])), -cmd_x[:,0], label="cmd")
        #     ax[0].set_title("Position")

        #     obs_v = np.array(obs_v)
        #     cmd_v = np.array(cmd_v)
        #     ax[1].plot(np.arange(len(obs_v[:,0])), -obs_v[:,0], label="obs")
        #     ax[1].plot(np.arange(len(cmd_v[:,0])), -cmd_v[:,0], label="cmd")
        #     ax[1].set_title("Velocity")

        #     obs_a = np.array(obs_v)
        #     cmd_a = np.array(cmd_a)
        #     obs_a[1:-1] = (obs_a[2:]-obs_a[:-2])/(2*t_step_ctrl)
        #     ax[2].plot(np.arange(len(obs_a[:,0])), -obs_a[:,0], label="obs")
        #     ax[2].plot(np.arange(len(cmd_a[:,0])), -cmd_a[:,0], label="cmd")
        #     ax[2].set_title("Acceleration")
        #     plt.legend()
        #     plt.show() 

        time.sleep(t_step_ctrl)
        self.cf.notifySetpointsStop()
        self.cf.land(0.05, 2.0)


    @property
    def obs(self) -> dict:
        """Return the observation of the environment."""
        drone = self.vicon.drone_name
        rpy = self.vicon.rpy[drone]
        ang_vel = R.from_euler("xyz", rpy).inv().apply(self.vicon.ang_vel[drone])
        obs = {
            "pos": self.vicon.pos[drone].astype(np.float32),
            "rpy": rpy.astype(np.float32),
            "vel": self.vicon.vel[drone].astype(np.float32),
            "ang_vel": ang_vel.astype(np.float32),
        }

        sensor_range = self.config.env.sensor_range
        n_gates = len(self.config.env.track.gates)

        obs["target_gate"] = self.target_gate if self.target_gate < n_gates else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        drone_pos = self.vicon.pos[self.vicon.drone_name]
        gates_pos = np.array([g.pos for g in self.config.env.track.gates])
        gate_names = [f"gate{g}" for g in range(1, len(gates_pos) + 1)]
        real_gates_pos = np.array([self.vicon.pos[g] for g in gate_names])
        in_range = np.linalg.norm(real_gates_pos - drone_pos, axis=1) < sensor_range
        gates_pos[in_range] = real_gates_pos[in_range]
        gates_rpy = np.array([g.rpy for g in self.config.env.track.gates])
        real_gates_rpy = np.array([self.vicon.rpy[g] for g in gate_names])
        gates_rpy[in_range] = real_gates_rpy[in_range]
        obs["gates_pos"] = gates_pos.astype(np.float32)
        obs["gates_rpy"] = gates_rpy.astype(np.float32)
        obs["gates_in_range"] = in_range

        obstacles_pos = np.array([o.pos for o in self.config.env.track.obstacles])
        obstacle_names = [f"obstacle{g}" for g in range(1, len(obstacles_pos) + 1)]
        real_obstacles_pos = np.array([self.vicon.pos[o] for o in obstacle_names])
        in_range = np.linalg.norm(real_obstacles_pos - drone_pos, axis=1) < sensor_range
        obstacles_pos[in_range] = real_obstacles_pos[in_range]
        obs["obstacles_pos"] = obstacles_pos.astype(np.float32)
        obs["obstacles_in_range"] = in_range
        return obs

    @property
    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {"collisions": [], "symbolic_model": self.symbolic}

    def gate_passed(self, pos: NDArray[np.floating], prev_pos: NDArray[np.floating]) -> bool:
        """Check if the drone has passed the current gate.

        Args:
            pos: Current drone position.
            prev_pos: Previous drone position.
        """
        n_gates = len(self.config.env.track.gates)
        if self.target_gate < n_gates and self.target_gate != -1:
            # Gate IDs go from 0 to N-1, but names go from 1 to N
            gate_id = "gate" + str(self.target_gate + 1)
            # Real gates measure 0.4m x 0.4m, we account for meas. error
            gate_size = (0.56, 0.56)
            gate_pos = self.vicon.pos[gate_id]
            gate_rot = R.from_euler("xyz", self.vicon.rpy[gate_id])
            return check_gate_pass(gate_pos, gate_rot, gate_size, pos, prev_pos)
        return False


class DroneRacingThrustDeployEnv(DroneRacingDeployEnv):
    """A Gymnasium environment for deploying drone racing algorithms on real hardware.

    This environment mirrors the functionality of the
    class:~lsy_drone_racing.envs.drone_racing_thrust_env.DroneRacingThrustEnv, but interfaces with
    real-world hardware (Crazyflie drone and Vicon motion capture system) instead of a simulation.
    """

    def __init__(self, config: dict | Munch):
        """Initialize the crazyflie drone and the motion capture system.

        Args:
            config: The configuration of the environment.
        """
        super().__init__(config)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        self.drone = Drone("mellinger")

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        """Take a step in the environment.

        Warning:
            Step does *not* wait for the remaining time if the step took less than the control
            period. This ensures that controllers with longer action compute times can still hit
            their target frequency. Furthermore, it implies that loops using step have to manage the
            frequency on their own, and need to update the current observation and info before
            computing the next action.
        """
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        collective_thrust, rpy = action[0], action[1:]
        rpy_deg = np.rad2deg(rpy)
        collective_thrust = self.drone._thrust_to_pwms(collective_thrust)
        self.cf.cmdVel(*rpy_deg, collective_thrust)
        current_pos = self.vicon.pos[self.vicon.drone_name]
        self.target_gate += self.gate_passed(current_pos, self._last_pos)
        self._last_pos[:] = current_pos
        if self.target_gate >= len(self.config.env.track.gates):
            self.target_gate = -1
        terminated = self.target_gate == -1
        return self.obs, -1.0, terminated, False, self.info

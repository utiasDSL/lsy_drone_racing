"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate


import subprocess as subway

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################
        self.OFFSET_CORRECTION_FACTOR = 5
        self.initial_info = initial_info

        self.start = initial_obs[0:4]
        self.gates =list([initial_obs[12:12+4], initial_obs[16:16+4], initial_obs[20:20+4], initial_obs[24:24+4]])
        self.obstacles = list([initial_obs[32:32+3], initial_obs[35:35+3], initial_obs[38:38+3], initial_obs[41:41+3]])

        self.gates_in_range = initial_obs[28:28+4]
        self.gate_id = initial_obs[-1]
        self.obstacles_in_range = initial_obs[44:44+4]
        self.updated_gates = [0, 0, 0, 0]
        self.updated_obstacles = [0, 0, 0, 0]
        self.passed_gates = [0, 0, 0, 0]
        self.trajectory_planner_active = False
        self.step = 0

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = np.loadtxt("working_waypoints.txt").transpose()
        #waypoints = np.unique(waypoints, axis=0)
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints
        duration = 15
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            #Draw the trajectory on PyBullet's GUI.
            #draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)
            pass

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        #########################
        # REPLACE THIS (END) ####
        #########################

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        pos = obs[0:3]
        self._check_if_gate_passed(pos)

        if self._update_obstacle_parameter(obs) or self._update_gate_parameter(obs):
            self.trajectory_planner_active = True
            gates_array, obstacles_array = self._convert_to_routing_format(np.array(self.gates), np.array(self.obstacles))
            print(self.gates)
            print(self.obstacles)
            np.savez('env_info.npz', gate_list=gates_array, obstacles=obstacles_array, start=pos)
            self.hover_position = pos
            ### start trajectory planner here ###
            self.trajectory_planning_process = subway.Popen(['python', 'examples/gridLessRouting.py', 'env_info.npz'])

        if self.trajectory_planner_active == True:
            result = self.trajectory_planning_process.poll()
            if result is not None:
                self.trajectory_planner_active = False
                # calculate new waypoints
                waypoints = np.loadtxt("optimal_waypoints.txt").transpose()
                #tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]],  s=0.1)
                #self.waypoints = waypoints
                #duration = 10
                #t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
                # Separate the x, y, z coordinates
                x = waypoints[:, 0]
                y = waypoints[:, 1]
                z = waypoints[:, 2]

                # Create an array of normalized path lengths for each waypoint
                path_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
                path_lengths = np.insert(np.cumsum(path_lengths), 0, 0)  # Add a 0 at the beginning

                # Normalize the path lengths to range from 0 to 1
                u = path_lengths / path_lengths[-1]
                print(u)
                i = 0
                while i < len(u)-1:
                    if u[i+1] <= u[i]:
                        u = np.delete(u, i+1)
                        x = np.delete(x, i+1)
                        y = np.delete(y, i+1)
                        z = np.delete(z, i+1)
                        i -= 3
                    i += 1
                    if i >= len(u)-2:
                        break

                # Create PCHIP interpolators for each coordinate
                interp_x = interpolate.Akima1DInterpolator(u, x)
                interp_y = interpolate.Akima1DInterpolator(u, y)
                interp_z = interpolate.Akima1DInterpolator(u, z)

                # Store the waypoints
                self.waypoints = waypoints

                # Generate the time vector
                duration = 10
                t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))

                # Interpolate to get the interpolated path
                x_interpolated = interp_x(t)
                y_interpolated = interp_y(t)
                z_interpolated = interp_z(t)
                self.ref_x, self.ref_y, self.ref_z = x_interpolated, y_interpolated, z_interpolated
                self.step = iteration
                draw_trajectory(self.initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        if self.trajectory_planner_active == True:
            #hover as long as new solution is calculated
            target_pos = self.hover_position
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command.FULLSTATE
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
        else:
            step = iteration - self.step
            print(step)
            if ep_time > 0 and step < len(self.ref_x):
                #last_set_pos = [self.ref_x[step-1], self.ref_y[step-1], self.ref_z[step-1]]
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                #offset = self._offset_calculation(pos, last_set_pos, target_pos)
                #target_pos = target_pos +  offset
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif step >= len(self.ref_x) and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif step >= len(self.ref_x) and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []


        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################

    def _offset_calculation(self, obs, set_last, set_now):
        offset_vec = np.array([set_last[0]- obs[0], set_last[1] - obs[1], set_last[2]- obs[2]])
        total_offset = np.dot(offset_vec, offset_vec)
        return offset_vec * total_offset * self.OFFSET_CORRECTION_FACTOR

    def _update_gate_parameter(self, obs):
        list_index = 0
        update = False
        for i in range(12, 25, 4):
            if not np.array_equal(obs[i:i+4], self.gates[list_index]) and self.updated_gates[list_index] == 0:
                self.gates[list_index] = obs[i:i + 4]
                self.updated_gates[list_index] = 1
                update = True
            list_index += 1
        return update #updated gate parameter route recalculation necessary

    def _update_obstacle_parameter(self, obs):
        list_index = 0
        update = False
        for i in range(32, 42, 3):
            if not np.array_equal(obs[i:i + 3], self.obstacles[list_index]) and self.updated_obstacles[list_index] == 0:
                self.obstacles[list_index] = obs[i:i + 3]
                self.updated_obstacles[list_index] = 1
                update = True
            list_index += 1
        return update #updated gate parameter route recalculation necessary

    def _check_if_gate_passed(self, pos):
        for gate in range(0, len(self.gates[0])):
            if np.allclose(pos, self.gates[gate][0:3], atol=0.15):
                self.passed_gates[gate] = 1

    def _convert_to_routing_format(self, gates, obstacles):
        for i in range(len(self.passed_gates)):
            if self.passed_gates[i] == 1:
                gates = np.delete(gates, i, axis=0)
        obstacles = np.append(obstacles, np.zeros((obstacles.shape[0], 3)), axis=1)
        gates = np.append(gates, np.zeros((gates.shape[0], 1)), axis=1)
        gates = np.insert(gates,[3], np.zeros((gates.shape[0], 2)), axis=1)
        return gates, obstacles

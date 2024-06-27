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
        self.initial_info = initial_info

        self.start = initial_obs[0:4]
        self.goal = [0, -1.5, 0.5]
        self.gates = list(
            [initial_obs[12:12 + 4], initial_obs[16:16 + 4], initial_obs[20:20 + 4], initial_obs[24:24 + 4]])
        self.obstacles = list(
            [initial_obs[32:32 + 3], initial_obs[35:35 + 3], initial_obs[38:38 + 3], initial_obs[41:41 + 3]])
        self.updated_gates = [0, 0, 0, 0]
        self.updated_obstacles = [0, 0, 0, 0]
        self.passed_gates = [0, 0, 0, 0]
        self.goal_duration = 12

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = []
        waypoints.append([self.initial_obs[0], self.initial_obs[1], 0.1])
        waypoints.append([self.initial_obs[0], self.initial_obs[1], 0.3])
        gates = self.NOMINAL_GATES
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]
        waypoints.append([1, 0, z_low])
        waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
        waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
        waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
        waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints.append(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )
        waypoints.append(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2] - 0.2,
                initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)

        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints
        duration = self.goal_duration
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)

        index, obstacle = self._check_collision(self.obstacles)
        if index is not None:
            print("Colliding", self.waypoints[index], obstacle)

        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        self.step = 0
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

        gate_updated = self._update_gate_parameter(obs)
        obstacle_update = self._update_obstacle_parameter(obs)

        if gate_updated:
            waypoints = self._regen_waypoints(self.gates, self.obstacles, pos, self.goal)
            self._recalc_trajectory(waypoints, iteration)

        step = iteration - self.step

        if step < len(self.ref_x):
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
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


    def _find_next_gate(self):
        for i in range(len(self.passed_gates)):
            if self.passed_gates[i] == 0:
                return i
        print("all gates passed")
        return None

    def _check_if_gate_passed(self, pos):
        for gate in range(0, len(self.gates[0])):
            if np.allclose(pos, self.gates[gate][0:3], atol=0.1):
                self.passed_gates[gate] = 1

    def _update_gate_parameter(self, obs):
        list_index = 0
        update = False
        for i in range(12, 25, 4):
            if not np.array_equal(obs[i:i+4], self.gates[list_index]) and self.updated_gates[list_index] == 0:
                self.gates[list_index] = obs[i : i + 4]
                self.updated_gates[list_index] = 1
                update = True
            list_index += 1
        return update #updated gate parameter route recalculation necessary

    def _update_obstacle_parameter(self, obs):
        list_index = 0
        update = False
        for i in range(32, 42, 3):
            if not np.array_equal(obs[i:i + 2], self.obstacles[list_index][0:2]) and self.updated_obstacles[list_index] == 0:
                self.obstacles[list_index] = obs[i : i + 3]
                self.obstacles[list_index][2] = 0.5
                self.updated_obstacles[list_index] = 1
                #if self._check_if_on_path(self.obstacles[list_index]):
                    #update = True
                update = True
            list_index += 1
        return update     #updated obstacle parameter route recalculation necessary

    def _resolve_collision(self, obstacles, gate_pos, direction):
        allowed_rot = [0, np.pi / 5, -np.pi / 5, np.pi / 4, -np.pi / 4, np.pi / 8, -np.pi / 8]
        lengths = [0.3, 0.2, 0.1]
        rot = gate_pos[3]

        for length in lengths:
            for offset in allowed_rot:
                blocked = [False, False]

                delta_x_ = np.cos(rot + np.pi / 2 + offset)
                delta_y_ = np.sin(rot + np.pi / 2 + offset)

                goal_pos = [gate_pos[0] + direction * delta_x_ * length, gate_pos[1] + direction * delta_y_ * length,
                            gate_pos[2]]
                for obstacle in obstacles:
                    for dim in range(0, 2):
                        if obstacle[dim] - 0.2 < goal_pos[dim] < obstacle[dim] + 0.2:
                            blocked[dim] = True
                if blocked != [True, True]:
                    return goal_pos
        print("Error no valid position found")
        return None

    def _recalc_trajectory(self, waypoints, iteration):
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints
        self.step = iteration - 1
        duration = self.goal_duration/(self._find_next_gate() + 1)
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        index, obstacle = self._check_collision(self.obstacles)
        if index is not None:
            print("Colliding", self.waypoints[index], obstacle)

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(self.initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

    def _check_collision(self, obstacles):
        for obstacle in obstacles:
            for i in range(len(self.ref_x)):
                point = [self.ref_x[i], self.ref_y[i], self.ref_z[i]]
                if np.linalg.norm(np.array(point[0:2]) - np.array(obstacle[0:2])) < 0.15:
                    if point[2] < 1.05:
                        collision_index = self._get_collision_waypoint(obstacle, point)
                        return collision_index, obstacle
        return None, None

    def _get_collision_waypoint(self, obstacle, point):
            distances = np.linalg.norm(self.waypoints - obstacle, axis=1)
            return np.argmin(distances)


    def _regen_waypoints(self, gates, obstacles, pos, goal):

        next_gate_index = self._find_next_gate()
        z_low = self.initial_info["gate_dimensions"]["low"]["height"]
        z_high = self.initial_info["gate_dimensions"]["tall"]["height"]


        waypoints = []
        waypoints.append(pos)
        #waypoints.append([1, 0, z_low])

        if next_gate_index< 1:
            waypoints.append(self._resolve_collision(obstacles,gates[0], -1))
            waypoints.append([gates[0][0], gates[0][1], gates[0][2]])
            waypoints.append(self._resolve_collision(obstacles, gates[0], 1))
            waypoints.append(
                [
                    (gates[0][0] + gates[1][0]) / 2 - 0.7,
                    (gates[0][1] + gates[1][1]) / 2 - 0.3,
                    (z_low + z_high) / 2,
                ]
            )
            waypoints.append(
                [
                    (gates[0][0] + gates[1][0]) / 2 - 0.5,
                    (gates[0][1] + gates[1][1]) / 2 - 0.6,
                    (z_low + z_high) / 2,
                ]
            )
        if next_gate_index < 2:
            waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
            waypoints.append([gates[1][0], gates[1][1], gates[1][2]])
            waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])

        if next_gate_index < 3:
            waypoints.append(self._resolve_collision(obstacles, gates[2], -1))
            #waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
            waypoints.append([gates[2][0], gates[2][1], gates[2][2]])
            #waypoints.append([gates[2][0], gates[2][1] + 0.1, z_low])
            waypoints.append(self._resolve_collision(obstacles, gates[2], 1))
            point = self._resolve_collision(obstacles, gates[2], 1)
            point[2] = z_high + 0.2
            waypoints.append(point)
            #waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])

        if next_gate_index < 4:
            waypoints.append([gates[3][0], gates[3][1] + 0.3, z_high+ 0.2])
            waypoints.append([gates[3][0], gates[3][1], gates[3][2]])
            waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.05])


        waypoints.append(
            [
                self.initial_info["x_reference"][0],
                self.initial_info["x_reference"][2],
                self.initial_info["x_reference"][4],
            ]
        )
        waypoints.append(
            [
                self.initial_info["x_reference"][0],
                self.initial_info["x_reference"][2] - 0.2,
                self.initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)
        return waypoints
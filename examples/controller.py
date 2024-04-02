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
            initial_obs: The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
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

        # Example: hardcode waypoints through the gates.
        waypoints = [
            (
                self.initial_obs[0],
                self.initial_obs[2],
                initial_info["gate_dimensions"]["tall"]["height"],
            )
        ]  # Height is hardcoded scenario knowledge.
        for idx, g in enumerate(self.NOMINAL_GATES):
            height = (
                initial_info["gate_dimensions"]["tall"]["height"]
                if g[6] == 0
                else initial_info["gate_dimensions"]["low"]["height"]
            )
            if g[5] > 0.75 or g[5] < 0:
                if idx == 2:  # Hardcoded scenario knowledge (direction in which to take gate 2).
                    waypoints.append((g[0] + 0.3, g[1] - 0.3, height))
                    waypoints.append((g[0] - 0.3, g[1] - 0.3, height))
                else:
                    waypoints.append((g[0] - 0.3, g[1], height))
                    waypoints.append((g[0] + 0.3, g[1], height))
            else:
                if idx == 3:  # Hardcoded scenario knowledge (correct how to take gate 3).
                    waypoints.append((g[0] + 0.1, g[1] - 0.3, height))
                    waypoints.append((g[0] + 0.1, g[1] + 0.3, height))
                else:
                    waypoints.append((g[0], g[1] - 0.3, height))
                    waypoints.append((g[0], g[1] + 0.3, height))
        waypoints.append(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )

        # Polynomial fit.
        self.waypoints = np.array(waypoints)
        deg = 6
        t = np.arange(self.waypoints.shape[0])
        fx = np.poly1d(np.polyfit(t, self.waypoints[:, 0], deg))
        fy = np.poly1d(np.polyfit(t, self.waypoints[:, 1], deg))
        fz = np.poly1d(np.polyfit(t, self.waypoints[:, 2], deg))
        duration = 15
        t_scaled = np.linspace(t[0], t[-1], int(duration * self.CTRL_FREQ))
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def compute_control(
        self,
        time: float,
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
            time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's getting_stated scenario.

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        elif iteration >= 3 * self.CTRL_FREQ and iteration < 20 * self.CTRL_FREQ:
            step = min(iteration - 3 * self.CTRL_FREQ, len(self.ref_x) - 1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, time]

        elif iteration == 20 * self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration == 20 * self.CTRL_FREQ + 1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5
            yaw = 0.0
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 23 * self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
            z = 1.5
            yaw = 0.0
            duration = 6

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 30 * self.CTRL_FREQ:
            height = 0.0
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == 33 * self.CTRL_FREQ - 1:
            command_type = Command(
                -1
            )  # Terminate command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
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

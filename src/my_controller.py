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
from src.map.map import Map
from src.path.rtt_star import RRTStar
from src.traj_gen.min_snap.traj_gen import TrajGenerator
from src.utils.calc_gate_center import calc_gate_center_and_normal


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

        # Print the initial information.
        if True or verbose:
            print("Initial information:")
            for key, value in initial_info.items():
                print(f"  {key}: {value}")
            

        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self.GATE_TYPES = initial_info["gate_dimensions"]

        # Generate map
        map = Map(-4, 4, -4, 4, 1.5)
        map.parse_gates(self.NOMINAL_GATES)
        map.parse_obstacles(self.NOMINAL_OBSTACLES)

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled

       
        gates_centers = []
        gates_normals = []
        for gate in self.NOMINAL_GATES:
            center, normal = calc_gate_center_and_normal(gate, self.GATE_TYPES)
            gates_centers.append(center)
            gates_normals.append(normal)
        

        # add the start and end points
        checkpoints = []
        initial_x = initial_obs[0]
        initial_y = initial_obs[2]
        initial_z = initial_obs[4]
        checkpoints.append(np.array([initial_x, initial_y, initial_z]))
        for gate_center, gate_normal in zip(gates_centers, gates_normals):
            gate_normal_normalized = gate_normal / np.linalg.norm(gate_normal)
            # add checkpoint before and after the gate center, 10 cm
            early_checkpoint = gate_center - 0.5 * gate_normal_normalized
            late_checkpoint = gate_center + 0.5 * gate_normal_normalized
            #checkpoints.append(early_checkpoint)
            checkpoints.append(gate_center)
            #checkpoints.append(late_checkpoint)

        checkpoints.append(np.array([0,-2,0.5])) # Hardcoded from the config file
        checkpoints = np.array(checkpoints)

        # Generate path using r_star
        path = []
        for i, (start_pos, end_pos) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
            print(f"Generating section {i} from {start_pos} to {end_pos}")
            rrt = RRTStar(start_pos, end_pos, map)
            waypoints, _ = rrt.plan()
            if waypoints is None:
                print("Failed to find path")
                exit(1)

            path.append(waypoints)
        
        path = np.concatenate(path)
        
        # visualize the path
        map.draw_scene(path)


        self.traj = TrajGenerator(checkpoints, max_vel=0.5)
        duration = self.traj.TS[-1]
        t = np.linspace(0, duration, int(duration * self.CTRL_FREQ))

        # gen ref traj for plotting
        if self.VERBOSE:
            ref_x = np.zeros_like(t)
            ref_y = np.zeros_like(t)
            ref_z = np.zeros_like(t)
            for i, time in enumerate(t[:-1]):
                state = self.traj.get_des_state(time)
                ref_x[i] = state.pos[0]
                ref_y[i] = state.pos[1]
                ref_z[i] = state.pos[2]
            
            draw_trajectory(initial_info, checkpoints, ref_x, ref_y, ref_z)
        

        

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
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
       
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        # print info
        # if self.VERBOSE:
        #     for key, value in info.items():
        #         print(f"  {key}: {value}")

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            
            if ep_time - 2 > 0 and ep_time - 2 < self.traj.TS[-1]:
                desired_state = self.traj.get_des_state(ep_time - 2)
                command_type = Command.FULLSTATE
                target_rpy_rates = np.zeros(3)
                args = [desired_state.pos, desired_state.vel, desired_state.acc, desired_state.yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif ep_time -2 >= self.traj.TS[-1] and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif self._setpoint_land and not self._land:
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

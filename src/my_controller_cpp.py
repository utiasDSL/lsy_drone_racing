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

from __future__ import annotations

import numpy as np

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory, draw_traj_without_ref, remove_trajectory
from online_traj_planner import OnlineTrajGenerator
from src.utils.config_reader import ConfigReader
from src.state_estimator import StateEstimator
import json


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
            
        # load config
        config_path = "./config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Save environment and control parameters.
        self.initial_info = initial_info
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        
        # Configuration
        self.takeoff_height = 0.2
        self.takeoff_time = 2

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        print("Nominal gates")
        print(type(self.NOMINAL_GATES))
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self.GATE_TYPES = initial_info["gate_dimensions"]
        start_point = np.array([initial_obs[0], initial_obs[2], self.takeoff_height])
        goal_point = np.array([0, -2, 0.5]) # Hardcoded from the config file

        self.traj_generator_cpp = OnlineTrajGenerator(start_point, goal_point, self.NOMINAL_GATES, self.NOMINAL_OBSTACLES, config_path)
        self.traj_generator_cpp.pre_compute_traj(self.takeoff_time)

        # Append extra info to scenario
        additional_static_obstacles = self.config["additional_statics_obstacles"]
        self.NOMINAL_OBSTACLES.extend(additional_static_obstacles)

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        self.current_gate_id = 0
        self.last_gate_id = 0
        self.next_potential_switching_time = 0
        self.state_estimator = StateEstimator(4)
        self.state_estimator.reset()


        #########################
        # REPLACE THIS (START) ##
        #########################

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled

        # gen ref traj for plotting
        if self.VERBOSE:
            traj = self.traj_generator_cpp.get_planned_traj()
            traj_positions = traj[:, [0, 3, 6]]
            draw_traj_without_ref(initial_info, traj_positions)
        

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        self.last_traj_recalc_time = None
        self.last_gate_change_time = 0
        self.last_gate_id = 0

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
        ########################
        
        current_drone_pos = np.array([obs[0], obs[2], obs[4]])
        self.state_estimator.add_measurement(current_drone_pos, ep_time)
        current_drone_vel, current_drone_acc = self.state_estimator.estimate_state()

        current_target_gate_pos = info.get("current_target_gate_pos", None)
        
        current_target_gate_id = info.get("current_target_gate_id", None)
        current_target_gate_in_range = info.get("current_target_gate_in_range", None)
        #print(f"Current target gate id {current_target_gate_id} In range {current_target_gate_in_range} pos {current_target_gate_pos}")
        if not(current_target_gate_pos != None and current_target_gate_id != None and current_target_gate_in_range != None):
            pass
        else:
            if current_target_gate_id != self.last_gate_id:
                print(f"update last gate change time at {ep_time}")
                self.last_gate_id = current_target_gate_id
                self.last_gate_change_time = ep_time
            
            # update_time = 0.5
            # if ep_time - self.last_gate_change_time > update_time:
            #     pos_updated = self.traj_generator_cpp.update_gate_pos(current_target_gate_id, current_target_gate_pos, current_drone_pos, current_target_gate_in_range, ep_time)
            #     if pos_updated: 
            #         self.last_traj_recalc_time = ep_time

            current_target_gate_pos[2] = 0
            pos_updated = self.traj_generator_cpp.update_gate_pos(current_target_gate_id, current_target_gate_pos, current_drone_pos, current_target_gate_in_range, ep_time)
            if pos_updated: 
                self.last_traj_recalc_time = ep_time
        
        traj_calc_duration = 0.2
        if self.VERBOSE and self.last_traj_recalc_time and ep_time - self.last_traj_recalc_time > traj_calc_duration:
            remove_trajectory()
            traj = self.traj_generator_cpp.get_planned_traj()
            traj_positions = traj[:, [0, 3, 6]]
            draw_traj_without_ref(self.initial_info, traj_positions)
            self.last_traj_recalc_time = None


        traj_end_time = self.traj_generator_cpp.get_traj_end_time()
        traj_has_ended = ep_time > traj_end_time

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [self.takeoff_height, self.takeoff_time]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            if ep_time < self.takeoff_time:
                command_type = Command.NONE
                args = []
            
            elif not traj_has_ended:
                cur_segment_id = current_target_gate_id
                #traj_sample = self.traj_generator_cpp.sample_traj_with_recompute(current_drone_pos, current_drone_vel, current_drone_acc, ep_time, cur_segment_id)
                traj_sample = self.traj_generator_cpp.sample_traj(ep_time)
                desired_pos = np.array([traj_sample[0], traj_sample[3], traj_sample[6]])
                desired_vel = np.array([traj_sample[1], traj_sample[4], traj_sample[7]])
                desired_acc = np.array([traj_sample[2], traj_sample[5], traj_sample[8]])
                command_type = Command.FULLSTATE
                target_rpy_rates = np.zeros(3)
                args = [desired_pos, desired_vel, desired_acc, 0, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif traj_has_ended and not self._setpoint_land:
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

"""Automatic tuning of the FxTDO observer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
import pybullet as p
from vizier import service
from vizier.service import clients, servers
from vizier.service import pyvizier as vz

from lsy_drone_racing.sim.noise import ExternalForceGrid
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.disturbance_observer import FxTDO

if TYPE_CHECKING:
    from munch import Munch

    from lsy_drone_racing.control.controller import BaseController
    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv


class AutoTuner():
    """Automatically Tunes the FxTDO."""
    def __init__(self, 
        config: str = "level3.toml",
        controller: str | None = None,
        n_runs: int = 1,
        gui: bool | None = False,
        env_id: str | None = None,
        opti_id: str = "default", 
        opti_algo: vz.Algorithm = vz.Algorithm.RANDOM_SEARCH,
        opti_runs: int = 10
    ) -> list[float]:
        
        self.opti_id = opti_id
        self.opti_algo = opti_algo
        self.opti_runs = opti_runs

        # Load configuration and check if firmare should be used.
        self.config = load_config(Path(__file__).parents[1] / "config" / config)
        if gui is None:
            gui = self.config.sim.gui
        else:
            self.config.sim.gui = gui

        # Load the controller module
        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        controller_path = control_path / (controller or self.config.controller.file)
        controller_cls = load_controller(controller_path)  # This returns a class, not an instance
        # Create the racing environment
        self.env: DroneRacingEnv = gymnasium.make(env_id or self.config.env.id, config=self.config)
        self.dist = ExternalForceGrid(3, [True, True, False], max_force=0.01, grid_size=0.5)
        self.env.sim.disturbances["dynamics"].append(self.dist) #WARN: env.sim to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.sim` for environment variables or `env.get_wrapper_attr('sim')` that will search the reminding wrappers.

        obs, info = self.env.reset()
        self.controller: BaseController = controller_cls(obs, info)
        self.observer = FxTDO(1 / self.config.env.freq)

    # Define the simulation setup
    def evaluate_observer(self, parameters: vz.Trial.parameters) -> float:
        """Evaluate the FxTDO observer with parameters from a Vizier trial.
        
        Args:
            parameters: parameters to test.
        
        Returns:
            The mean squared estimation error as the performance metric.
        """
        # Extract trial parameters
        L1 = parameters["L1"]
        L2 = parameters["L2"]
        k1 = np.array([
            parameters["k1_1"],
            parameters["k1_2"],
            parameters["k1_3"]
        ])
        k2 = np.array([
            parameters["k2_1"],
            parameters["k2_2"],
            parameters["k2_3"]
        ])
        d_inf = parameters["d_inf"]

        # Initialize FxTDO with trial parameters
        self.observer.set_parameters(
            f_d_max=0.1,
            f_d_dot_max=0.05,
            L1=L1,
            L2=L2,
            k1=k1,
            k2=k2,
            d_inf=d_inf
        )
        self.observer.reset()

        # Simulate the system
        metrics = self.simulate()

        # Return the mean squared error as the objective metric
        return metrics

    # Create a function to simulate the system
    def simulate(self) -> float:
        """Simulates the system and returns a performance metric.
        
        Args:
            observer: An instance of FxTDO to evaluate.
        
        Returns:
            Mean squared estimation error during simulation.
        """
        obs, info = self.env.reset()
        self.f_mass_z = (p.getDynamicsInfo(1, -1)[0] - 0.03454) * np.array([0, 0, -9.81])
        
        i = 0
        done = False
        predictions = []

        while not done:
            t_start = time.time()
            curr_time = i / self.config.env.freq
            # if gui:
            #     gui_timer = update_gui_timer(curr_time, env.unwrapped.sim.pyb_client, gui_timer) # this is slow!
            # p_info = p.getDynamicsInfo(1, -1)[0] #, _, _, _, _ 
            # print(f"mass={p_info}")

            action = self.controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.dist.update_pos(obs["pos"])
            done = terminated or truncated
            # Update the controller and observer internal state and models.
            self.controller.step_callback(action, obs, reward, terminated, truncated, info)

            # Desired thrust (=actual drone thrust) calculated like in physics DYN
            forces = np.array(self.env.sim.drone.rpm**2) * self.env.sim.drone.params.kf
            des_thrust = np.sum(forces)

            estimate, f_t = self.observer.step(obs, des_thrust)
            # Store observer predictions and real value                                         
            predictions.append([np.array(obs["pos"]), np.array(obs["vel"]), estimate[:3]*1.0, self.dist.force + self.f_mass_z, estimate[3:]*1.0, np.array(f_t)]) # the factor is to make a copy
            # Add up reward, collisions


            # Synchronize the GUI.
            if self.config.sim.gui:
                if (elapsed := time.time() - t_start) < 1 / self.config.env.freq:
                    time.sleep(1 / self.config.env.freq - elapsed)
            i += 1

        self.controller.episode_callback()  # Update the controller internal state and models.
        # log_episode_stats(obs, info, config, curr_time)
        self.controller.episode_reset()
        # ep_times.append(curr_time if obs["target_gate"] == -1 else None)


        # TODO calculate error
        predictions = np.array(predictions)
        v_ref = predictions[:,1,:]
        v_hat = predictions[:,2,:]
        f_ref = predictions[:,3,:]
        f_hat = predictions[:,4,:]
        # axis 0 is time, axis 1 is xyz
        e_v = np.linalg.norm(v_ref-v_hat, axis=1)
        e_f = np.linalg.norm(f_ref-f_hat, axis=1)*1000
        # print(f"e_f={e_f}")
        # TODO might wanna return infinite error if episode failed or normalize (0,1)
        mean_error = np.mean(e_v) + np.mean(e_f)
        variance_error = np.var(e_v) + np.var(e_f)

        # Return the metrics
        return {
            "error": mean_error,
            "variance": variance_error*1000
        }
        # return {
        #     "error": np.tanh(mean_error),
        #     "variance": np.tanh(variance_error)
        # }

    # Vizier Tuning
    def tune_observer(self):
        """Tunes the FxTDO using Google Vizier, minimizing both error and variance."""
        # Define the problem statement
        problem = vz.ProblemStatement()
        problem.metric_information.append(
            vz.MetricInformation(name="error", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        )
        problem.metric_information.append(
            vz.MetricInformation(name="variance", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        )
        problem.search_space.root.add_float_param("L1", 0.01, 100.0)
        problem.search_space.root.add_float_param("L2", 0.01, 100.0)
        problem.search_space.root.add_float_param("k1_1", 0.01, 100.0)
        problem.search_space.root.add_float_param("k1_2", 0.01, 100.0)
        problem.search_space.root.add_float_param("k1_3", 0.01, 100.0)
        problem.search_space.root.add_float_param("k2_1", 0.01, 100.0)
        problem.search_space.root.add_float_param("k2_2", 0.01, 100.0)
        problem.search_space.root.add_float_param("k2_3", 0.01, 100.0)
        problem.search_space.root.add_float_param("d_inf", 0.1, 0.99)

        # Initialize a random search policy
        study_config = vz.StudyConfig.from_problem(problem)

        study_config.algorithm = self.opti_algo
        study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.opti_id)
        print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

        # Optimization loop
        for i in range(self.opti_runs):
            suggestions = study_client.suggest(count=1)#, client_id=self.opti_id)
            # print(suggestions)
            for suggestion in suggestions:
                # print(f"Trying suggestion: {suggestion}")
                objective = self.evaluate_observer(suggestion.parameters)
                print(f'Iteration {i}, led to objective value {objective}.')
                final_measurement = vz.Measurement(objective)
                suggestion.complete(final_measurement)

        self.env.close()

        # Print the best parameters
        for optimal_trial in study_client.optimal_trials():
            optimal_trial = optimal_trial.materialize()
            print("Optimal Trial Suggestion and Objective: \n", optimal_trial.parameters, "\n", 
                    optimal_trial.final_measurement)


# Run the tuning
if __name__ == "__main__":
    A = AutoTuner(opti_id = 'fxtdo_study_full_wide_1000', opti_algo = vz.Algorithm.GAUSSIAN_PROCESS_BANDIT, opti_runs = 50)
    # https://oss-vizier.readthedocs.io/en/latest/guides/user/supported_algorithms.html
    # vz.Algorithm.GP_UCB_PE # one metric only
    # vz.Algorithm.RANDOM_SEARCH
    # vz.Algorithm.GAUSSIAN_PROCESS_BANDIT
    A.tune_observer()

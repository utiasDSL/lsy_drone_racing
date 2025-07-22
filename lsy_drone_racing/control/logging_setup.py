"""Logging setup for the MPC controller.

This module provides logging configuration and utilities for the drone racing
MPC controller with file and console output.
"""

import logging
import os
import time


def setup_logging() -> logging.Logger:
    """Setup logging configuration for the MPC controller with NO console output."""
    # Create logs directory if it doesn't exist
    os.makedirs("flight_logs", exist_ok=True)

    # Disable all existing loggers
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)

    # Create logger with unique name
    logger_name = f"MPController_{int(time.time())}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Prevent propagation to root logger
    logger.propagate = False

    # Create file handler with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"flight_logs/mpc_controller_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Tick:%(tick)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Add ONLY file handler
    logger.addHandler(file_handler)

    # Disable root logger completely
    root_logger = logging.getLogger()
    root_logger.disabled = True

    return logger


class FlightLogger:
    """Enhanced logging utilities for flight data and performance metrics."""

    def __init__(self, logger_name: str = "MPController"):
        """Initialize the flight logger.

        Args:
            logger_name: Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
        self.logger = setup_logging()

    def log_initialization(self, controller_params: dict, tick: int = 0):
        """Log controller initialization parameters."""
        self.logger.info("=== MPC CONTROLLER INITIALIZATION ===", extra={"tick": tick})

        # MPC Solver Parameters
        if any(key.startswith("MPC_") for key in controller_params.keys()):
            self.logger.info("MPC Solver Parameters:", extra={"tick": tick})
            for key in ["MPC_N", "MPC_T_HORIZON", "MPC_dt"]:
                if key in controller_params:
                    self.logger.info(f"  {key}: {controller_params[key]}", extra={"tick": tick})

        # MPC Weights
        if "MPC_weights" in controller_params:
            self.logger.info("MPC Cost Weights:", extra={"tick": tick})
            for weight_name, weight_value in controller_params["MPC_weights"].items():
                self.logger.info(f"  {weight_name}: {weight_value}", extra={"tick": tick})

        # Flight Parameters
        if "approach_distances" in controller_params:
            self.logger.info(
                f"Approach distances: {controller_params['approach_distances']}",
                extra={"tick": tick},
            )
        if "exit_distances" in controller_params:
            self.logger.info(
                f"Exit distances: {controller_params['exit_distances']}", extra={"tick": tick}
            )
        if "approach_height_offsets" in controller_params:
            self.logger.info(
                f"Approach height offsets: {controller_params['approach_height_offsets']}",
                extra={"tick": tick},
            )
        if "exit_height_offsets" in controller_params:
            self.logger.info(
                f"Exit height offsets: {controller_params['exit_height_offsets']}",
                extra={"tick": tick},
            )

        # Environment Parameters
        if "start_position" in controller_params:
            self.logger.info(
                f"Start position: {controller_params['start_position']}", extra={"tick": tick}
            )
        if "total_gates" in controller_params:
            self.logger.info(
                f"Total gates in track: {controller_params['total_gates']}", extra={"tick": tick}
            )
        if "freq" in controller_params:
            self.logger.info(
                f"Simulation frequency: {controller_params['freq']} Hz", extra={"tick": tick}
            )

        # Replanning Configuration
        if "replanning_frequency" in controller_params:
            self.logger.info(
                f"Replanning frequency: {controller_params['replanning_frequency']}",
                extra={"tick": tick},
            )

        # Default Parameters
        for key in [
            "default_approach_dist",
            "default_exit_dist",
            "default_approach_height_offset",
            "default_exit_height_offset",
        ]:
            if key in controller_params:
                self.logger.info(f"{key}: {controller_params[key]}", extra={"tick": tick})

        self.logger.info("=== END MPC CONTROLLER INITIALIZATION ===", extra={"tick": tick})

    def log_trajectory_update(self, trajectory_info: dict, tick: int):
        """Log trajectory update information."""
        self.logger.info(
            f"Trajectory updated: {trajectory_info.get('mode', 'standard')} mode, "
            f"waypoints: {trajectory_info.get('waypoints', 0)}, "
            f"points: {trajectory_info.get('points', 0)}, "
            f"target_gate: {trajectory_info.get('target_gate', 'N/A')}",
            extra={"tick": tick},
        )

    def log_gate_progress(self, gates_passed: int, total_gates: int, target_gate: int, tick: int):
        """Log gate passing progress."""
        if target_gate == -1:
            self.logger.info(
                f"FLIGHT COMPLETED SUCCESSFULLY! All {total_gates} gates passed",
                extra={"tick": tick},
            )
        else:
            self.logger.info(
                f"Gate {gates_passed} passed! Progress: {gates_passed}/{total_gates}",
                extra={"tick": tick},
            )

    def log_replanning_event(self, replan_info: dict, tick: int):
        """Log trajectory replanning events."""
        self.logger.info(
            f"Gate {replan_info.get('gate_idx', 'N/A')} position updated, replanning trajectory. "
            f"Horizontal diff: {replan_info.get('horizontal_diff', 0):.3f}m, "
            f"Config pos: {replan_info.get('config_pos', 'N/A')}, "
            f"Observed pos: {replan_info.get('observed_pos', 'N/A')}",
            extra={"tick": tick},
        )

    def log_mpc_status(self, mpc_status: dict, tick: int):
        """Log MPC solver status and performance."""
        self.logger.info(
            f"MPC Status - Gates: {mpc_status.get('gates_passed', 0)}/{mpc_status.get('total_gates', 0)}, "
            f"target_gate: {mpc_status.get('target_gate', 'N/A')}, "
            f"trajectory_idx: {mpc_status.get('trajectory_idx', 0)}, "
            f"finished: {mpc_status.get('finished', False)}, "
            f"solver_status: {mpc_status.get('solver_status', 'N/A')}",
            extra={"tick": tick},
        )

    def log_tracking_performance(self, tracking_info: dict, tick: int):
        """Log tracking performance metrics."""
        self.logger.info(
            f"MPC Tracking - Error: {tracking_info.get('error', 0):.3f}m, "
            f"Gates: {tracking_info.get('gates_passed', 0)}/{tracking_info.get('total_gates', 0)}, "
            f"Just replanned: {tracking_info.get('just_replanned', False)}, "
            f"Solver status: {tracking_info.get('solver_status', 'N/A')}",
            extra={"tick": tick},
        )

    def log_step_info(self, step_info: dict, tick: int):
        """Log step information."""
        self.logger.info(
            f"Step {tick}: reward={step_info.get('reward', 0):.3f}, "
            f"gates={step_info.get('gates_passed', 0)}/{step_info.get('total_gates', 0)}, "
            f"target_gate={step_info.get('target_gate', 'N/A')}, "
            f"terminated={step_info.get('terminated', False)}, "
            f"truncated={step_info.get('truncated', False)}",
            extra={"tick": tick},
        )

    def log_episode_summary(self, summary: dict, tick: int):
        """Log comprehensive episode summary."""
        self.logger.info("=== EPISODE SUMMARY ===", extra={"tick": tick})

        # Flight success
        self.logger.info(
            f"FLIGHT SUCCESS: {'YES' if summary.get('flight_successful', False) else 'NO'}",
            extra={"tick": tick},
        )

        # Gate completion
        gates_passed = summary.get("gates_passed", 0)
        total_gates = summary.get("total_gates", 1)
        completion_rate = (gates_passed / total_gates * 100) if total_gates > 0 else 0

        self.logger.info(f"Gates passed: {gates_passed}/{total_gates}", extra={"tick": tick})
        self.logger.info(f"Completion rate: {completion_rate:.1f}%", extra={"tick": tick})

        # Flight metrics
        freq = summary.get("freq", 1)
        self.logger.info(f"Total simulation ticks: {tick}", extra={"tick": tick})
        self.logger.info(f"Flight time: {tick / freq:.2f} seconds", extra={"tick": tick})

        # Episode status
        self.logger.info(
            f"Episode Status: {'COMPLETED' if summary.get('finished', False) else 'INCOMPLETE'}",
            extra={"tick": tick},
        )

        # Controller parameters
        if "controller_params" in summary:
            params = summary["controller_params"]
            for key, value in params.items():
                self.logger.info(f"{key}: {value}", extra={"tick": tick})

        self.logger.info("=== END EPISODE SUMMARY ===", extra={"tick": tick})

    def log_warning(self, message: str, tick: int = 0):
        """Log warning message."""
        self.logger.warning(message, extra={"tick": tick})

    def log_error(self, message: str, tick: int = 0):
        """Log error message."""
        self.logger.error(message, extra={"tick": tick})

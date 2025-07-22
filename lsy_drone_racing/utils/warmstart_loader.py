"""Warmstart data loader for MPC controller.

This module provides functions to load warmstart data from JSON files
for initializing the MPC solver.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from lsy_drone_racing.utils.controller_config import get_parameter


def load_warmstart_data() -> Tuple[
    List[np.ndarray], List[List[float]], List[np.ndarray], List[np.ndarray]
]:
    """Load warmstart data from JSON file specified in config.

    Returns:
        Tuple containing:
        - x_warmstart: List of state arrays (N+1 entries)
        - u_warmstart: List of control arrays (N entries)
        - pi_warmstart: List of pi arrays (N entries)

    Raises:
        FileNotFoundError: If warmstart file doesn't exist
        ValueError: If warmstart data is invalid or corrupted
    """
    # Get the directory of the current file
    current_dir = Path(__file__).parent

    # Default relative path to mpc_warmstart.json (2 levels up, then into data/)
    file_path = current_dir.parent / Path(
        get_parameter("mpc.warmstart.file_path", "control/data/mpc_warmstart.json")
    )

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Validate data structure
        required_keys = ["x_warmstart", "u_warmstart", "pi_warmstart"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in warmstart data")

        # Convert to numpy arrays
        x_warmstart = [np.array(entry) for entry in data["x_warmstart"]]
        u_warmstart = [np.array(entry) for entry in data["u_warmstart"]]
        pi_warmstart = [np.array(entry) for entry in data["pi_warmstart"]]

        return x_warmstart, u_warmstart, pi_warmstart

    except FileNotFoundError:
        raise FileNotFoundError(f"Warmstart file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in warmstart file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading warmstart data: {e}")

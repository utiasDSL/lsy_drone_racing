import numpy as np
from src.utils.config_reader import ConfigReader
from src.utils.types import Gate

def calc_gate_center_and_normal(gate: Gate):
    config = ConfigReader.get()
    
    # parse information
    gate_pos_xyz = gate.pos
    gate_rotations = gate.rot
    gate_type_id = gate.gate_type
    gate_properties = config.get_gate_properties_by_type(gate_type_id)

    # For now only simple calculation supported, where gate is only rotated around z-axis
    assert np.allclose(gate_rotations[0], 0) and np.allclose(gate_rotations[1], 0), "Only z-axis rotation supported"

    # Step 1, calculate the center, rotation not important as we only rotate around z-axis and z-axis aligned with gate
    center_pos_x = gate_pos_xyz[0]
    center_pos_y = gate_pos_xyz[1]
    center_pos_z = gate_pos_xyz[2] + gate_properties["height"]
    center_pos = np.array([center_pos_x, center_pos_y, center_pos_z])

    # Step 2, calculate the normal based on the rotation
    normal_x = -np.sin(gate_rotations[2])
    normal_y = np.cos(gate_rotations[2])
    normal_z = 0
    normal = np.array([normal_x, normal_y, normal_z])

    return center_pos, normal
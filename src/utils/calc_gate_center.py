import numpy as np

def calc_gate_center_and_normal(gate_pos_and_type, gate_types, type_id_to_type_mapping=None):
    if type_id_to_type_mapping is None:
        type_id_to_type_mapping = {0: "tall", 1: "low"}
    
    # parse information
    gate_pos_xyz = gate_pos_and_type[0:3]
    gate_rotations = gate_pos_and_type[3:6]
    gate_type_id = gate_pos_and_type[6]
    gate_type = gate_types[type_id_to_type_mapping[gate_type_id]]

    # For now only simple calculation supported, where gate is only rotated around z-axis
    assert np.allclose(gate_rotations[0], 0) and np.allclose(gate_rotations[1], 0), "Only z-axis rotation supported"

    # Step 1, calculate the center, rotation not important as we only rotate around z-axis and z-axis aligned with gate
    center_pos_x = gate_pos_xyz[0]
    center_pos_y = gate_pos_xyz[1]
    center_pos_z = gate_pos_xyz[2] + gate_type["height"]
    center_pos = np.array([center_pos_x, center_pos_y, center_pos_z])

    # Step 2, calculate the normal based on the rotation
    normal_y = -np.cos(gate_rotations[2])
    normal_x = np.sin(gate_rotations[2])
    normal_z = 0
    normal = np.array([normal_x, normal_y, normal_z])

    return center_pos, normal
# Fix, so we can import form the src module
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from src.map.map_utils import Object
from src.utils.config_reader import ConfigReader


if __name__ == "__main__":
    config_path = "./config.json"
    config_reader = ConfigReader.create(config_path=config_path)
    
    gate_type = 1
    component = config_reader.get_gate_geometry_by_type(gate_type)
    object = Object.transform_urdf_component_into_object(component)
    center = np.array([0,0,0])
    rot = 3.14/2
    rotation = np.array([0,0,rot])

    object.translate(center)
    object.rotate_z(rotation[2])
    for i, obb in enumerate(object.obbs):
        print(f"Obb {i}. Center: {obb.center}")
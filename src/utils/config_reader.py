import json
from typing import Dict


class ConfigReader:
    _instance = None

    def __init__(self, config_path: str):
        # Initialize only if this is the first time instance creation
        self.config_path = config_path
        self.config = self._read_config()

    @classmethod
    def create(cls, config_path: str):
        """Create and initialize the singleton instance."""
        if cls._instance is None:
            cls._instance = ConfigReader(config_path)
        else:
            raise Exception("ConfigReader instance already created.")
        return cls._instance

    @classmethod
    def get(cls):
        """Retrieve the existing singleton instance or raise an exception."""
        if cls._instance is None:
            raise Exception("ConfigReader instance has not been created yet.")
        return cls._instance

    def _read_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_gate_geometry_by_type(self, type_id: int):
        # convert int to string
        type_id = str(type_id)
        type_name = self.config['gate_id_to_name_mapping'][type_id]
        return self.config['component_geometry'][type_name]

    def get_gate_properties_by_type(self, type_id: int):
        # convert int to string
        type_id = str(type_id)
        type_name = self.config['gate_id_to_name_mapping'][type_id]
        return self.config['component_properties'][type_name]

    def get_obstacle_geometry(self):
        return self.config['component_geometry']['obstacle']

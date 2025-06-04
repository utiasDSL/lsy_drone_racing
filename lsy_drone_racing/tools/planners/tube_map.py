from __future__ import annotations
from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D

from scipy.interpolate import CubicSpline

from typing import Optional, Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

import pickle

class PathSegment:
    OUTER_ZONE = 0
    ENTRANCE_ZONE = 1
    EXIT_ZONE = 2
    PASSED = 3

class TubeMap():

    num_gates : int
    tubes : Dict[Tuple[int, int], OccupancyMap3D]

    occ_map_xlim : List[np.floating]
    occ_map_ylim : List[np.floating]
    occ_map_zlim : List[np.floating]
    occ_map_res : np.floating
    tube_radius : np.floating = 0.4

    
    def __init__(self):
        pass

    def generate_tube_map(self,
                          num_gates : int,
                           paths :List[str],
                           occ_map_xlim : List[np.floating],
                            occ_map_ylim : List[np.floating],
                             occ_map_zlim : List[np.floating],
                              occ_map_res : np.floating,
                              tube_radius : np.floating = 0.4,
                              save_to : Optional[str] = None):
        self.tubes = {}
        self.num_gates = num_gates
        self.occ_map_xlim = occ_map_xlim
        self.occ_map_ylim = occ_map_ylim
        self.occ_map_zlim = occ_map_zlim
        self.occ_map_res = occ_map_res
        self.tube_radius = tube_radius

        trajectories = [TubeMap.read_segmented_trajectory(path) for path in paths]
        
        for gate_idx in range(num_gates):
            self.tubes[(gate_idx, PathSegment.OUTER_ZONE)] = OccupancyMap3D(xlim = occ_map_xlim,
                                    ylim = occ_map_ylim,
                                    zlim = occ_map_zlim ,
                                    resolution = occ_map_res,
                                    init_val = 1)
            self.tubes[(gate_idx, PathSegment.ENTRANCE_ZONE)] = OccupancyMap3D(xlim = occ_map_xlim,
                                    ylim = occ_map_ylim,
                                    zlim = occ_map_zlim ,
                                    resolution=occ_map_res,
                                    init_val = 1)
            self.tubes[(gate_idx, PathSegment.EXIT_ZONE)] = OccupancyMap3D(xlim = occ_map_xlim,
                                    ylim = occ_map_ylim,
                                    zlim = occ_map_zlim ,
                                    resolution=occ_map_res,
                                    init_val = 1)
            
        for t_axis, pos, vel, gate_idx, zone in trajectories:
            current_gate_idx = gate_idx[0]
            current_zone = zone[0]
            pos_segment = []

            t_segment = []

            for idx,_ in enumerate(t_axis):
                if gate_idx[idx] != current_gate_idx or zone[idx] != current_zone:
                    if len(pos_segment) > 2:
                        spline = CubicSpline(t_segment, pos_segment)
                        self.tubes[(current_gate_idx, current_zone)].add_trajectory_tube(spline = spline, radius = tube_radius)
                    else:
                        for idx_2,_ in enumerate(t_segment):
                            self.tubes[(current_gate_idx, current_zone)].add_sphere(pos_segment[idx_2], radius = tube_radius)
                    pos_segment.clear()
                    t_segment.clear()
                current_gate_idx = gate_idx[idx]
                current_zone = zone[idx]
                pos_segment.append(pos[idx])
                t_segment.append(t_axis[idx])

        if save_to is not None:
            self.save_to_file(path = save_to)

        

    def read_segmented_trajectory(path : str) -> Tuple[List[np.floating], List[NDArray], List[NDArray], List[int], List[int]]:
        with open(path, 'r') as f:
            header = f.readline().strip().split(',')

        data = np.loadtxt(path, delimiter=',', skiprows=1)

        t_idx = header.index('t')
        x_idx = header.index('x')
        y_idx = header.index('y')
        z_idx = header.index('z')
        gate_idx = header.index('gate')
        zone_idx = header.index('zone')

        has_velocity = all(col in header for col in ['vx', 'vy', 'vz'])
        if has_velocity:
            vx_idx = header.index('vx')
            vy_idx = header.index('vy')
            vz_idx = header.index('vz')

        t_list = data[:, t_idx].tolist()
        pos_list = [data[i, [x_idx, y_idx, z_idx]] for i in range(data.shape[0])]
        vel_list = [data[i, [vx_idx, vy_idx, vz_idx]] for i in range(data.shape[0])] if has_velocity else []
        next_gate_list = data[:, gate_idx].tolist()
        zone_list = data[:, zone_idx].tolist()
        return t_list, pos_list, vel_list, next_gate_list, zone_list
    
    def save_to_file(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved TubeMap to {path}")

    @staticmethod
    def read_from_file(path: str) -> TubeMap:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Read TubeMap to {path}")

        return obj
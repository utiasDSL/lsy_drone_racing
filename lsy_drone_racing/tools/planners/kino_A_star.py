from __future__ import annotations
LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    import matplotlib.collections
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False

import itertools


import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Set, Union, Callable

import time
import heapq


from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D
from lsy_drone_racing.tools.ext_tools import PolynomialTool as poly
from lsy_drone_racing.tools.race_objects import Gate

class KinoAStarPlannerConfig:
    w_time : np.floating
    lambda_heu: np.floating

    max_vel : np.floating
    max_acc : np.floating
    tie_breaker : np.floating
    acc_resolution : np.floating
    time_resolution : np.floating
    max_duration: np.floating
    safety_check_res: np.floating

    def __init__(
        self,
        w_time : np.floating,
        lambda_heu: np.floating,
        max_vel : np.floating,
        max_acc : np.floating,
        tie_breaker : np.floating,
        acc_resolution : np.floating,
        time_resolution : np.floating,
        max_duration: np.floating,
        safety_check_res: np.floating,
    ) -> None:
        self.w_time = w_time 
        self.lambda_heu= lambda_heu
        self.max_vel = max_vel 
        self.max_acc = max_acc 
        self.tie_breaker = tie_breaker 
        self.acc_resolution = acc_resolution 
        self.time_resolution = time_resolution 
        self.max_duration = max_duration
        self.safety_check_res = safety_check_res

class NodeInfo():
    g_cost : np.floating
    f_cost : np.floating

    pos : NDArray[np.floating]
    vel : NDArray[np.floating]
    input : NDArray[np.floating]
    
    duration: np.floating
    time : np.floating

    parent : NodeInfo

    visited : int

    grid_idx : NDArray[np.integer]

    def __init__(self,
                g_cost : np.floating = 0.0,
                f_cost : np.floating = 0.0,
                pos : NDArray[np.floating] = np.array([0,0,0], dtype = np.floating),
                vel : NDArray[np.floating] = np.array([0,0,0], dtype = np.floating),
                duration : np.floating = 0.0,
                input : NDArray[np.floating] = np.array([0,0,0], dtype = np.floating),
                time : np.floating = 0.0,
                parent : NodeInfo = None,
                visited : int = 0,
                grid_idx : NDArray[np.integer] = np.array([0,0,0], dtype = np.integer)):
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pos = pos
        self.vel = vel
        self.duration = duration
        self.input = input
        self.time = time
        self.parent = parent
        self.visited = visited
        self.grid_idx = grid_idx
    
    def generate_copy(self) -> NodeInfo:
        return NodeInfo(g_cost = self.g_cost,
                        f_cost = self.f_cost,
                        pos = self.pos.copy(),
                        vel =self.vel.copy(),
                        duration = self.duration,
                        input = self.input.copy(),
                        time = self.time,
                        parent = self.parent,
                        visited = self.visited,
                         grid_idx = self.grid_idx.copy())

class SearchNode():
    grid_idx : NDArray[np.integer]
    info : NodeInfo
    g_cost : np.floating
    f_cost : np.floating

    class SearchNodeComparator():
        def __call__(self, node_1 : SearchNode, node_2 : SearchNode) -> bool:
            return node_1.f_cost > node_2.f_cost
    
    def __init__(self,
                info : NodeInfo,
                g_cost : np.floating,
                f_cost : np.floating,
                pos :  NDArray[np.floating],
                vel : NDArray[np.floating],
                input : NDArray[np.floating],
                duration : np.floating,
                time : np.floating ,
                parent : NodeInfo,
                grid_idx : NDArray[np.integer]):
        self.g_cost = g_cost
        self.f_cost = f_cost

        # self.info = info.generate_copy()
        # I modified here: cancel generate copy
        self.info = info
        self.info.g_cost = g_cost
        self.info.f_cost = f_cost
        self.info.pos = pos
        self.info.vel = vel
        self.info.input = input
        self.info.duration = duration
        self.info.time = time
        self.info.parent = parent

        self.grid_idx = grid_idx

    
class VisitState:
        UNVISITED = 0
        OPEN = 1
        CLOSED = -1

class KinoDynamicAStarPlanner():
    _grid_map : OccupancyMap3D

    _origin : NDArray[np.floating]
    _map_size : NDArray[np.floating]
    _resolution : NDArray[np.floating]

    _map_grid_size : NDArray[np.integer]

    _w_time : np.floating
    _max_vel : np.floating
    _max_acc : np.floating
    _tie_breaker : np.floating

    _acc_resolution : np.floating
    _time_resolution : np.floating
    _max_duration : np.floating
    _safety_check_resolution : np.floating

    _lambda_heu : np.floating

    _coef_shot : NDArray[np.floating]
    _t_shot: np.floating
    _is_shot_success : bool
    _path : List[NodeInfo]

    _node_counter : itertools.count

    _open_set_plot : matplotlib.collections

    def __init__(self, map : OccupancyMap3D, verbose : bool = True):
        self.verbose = verbose
        self._grid_map = map
        self._origin = np.array([0.,0.,0.])
        self._resolution = np.array([0.,0.,0.]) * map.resolution
        self._map_grid_size = map.size
        self._map_size = np.reshape(map.limit[:, 1] - map.limit[:,0], shape = (3,))
        self._open_set_plot : matplotlib.collections = None
        self._node_counter :itertools.count = itertools.count()

    def set_param(self,
                  w_time : np.floating,
                  max_vel : np.floating,
                  max_acc : np.floating,
                    tie_breaker: np.floating,
                  acc_resolution : np.floating,
                  time_resolution: np.floating,
                  max_duration: np.floating,
                  safety_check_res: np.floating,
                  lambda_heu: np.floating,  
    )->None:
        self._w_time = w_time
        self._max_vel = max_vel
        self._max_acc = max_acc
        self._tie_breaker = tie_breaker
        self._acc_resolution = acc_resolution
        self._time_resolution = time_resolution
        self._max_duration = max_duration
        self._safety_check_resolution = safety_check_res
        self._lambda_heu = lambda_heu
        return
    
    def setup_param(self,
                  params : KinoAStarPlannerConfig  
    )->None:
        self._w_time = params.w_time
        self._max_vel = params.max_vel
        self._max_acc = params.max_acc
        self._tie_breaker = params.tie_breaker
        self._acc_resolution = params.acc_resolution
        self._time_resolution = params.time_resolution
        self._max_duration = params.max_duration
        self._safety_check_resolution = params.safety_check_res
        self._lambda_heu = params.lambda_heu
        return
    

    def search(self,
    start_pt : NDArray[np.floating],
    start_v : NDArray[np.floating],
    end_pt : Union[NDArray[np.floating],List[NDArray[np.floating]]],
    end_v : Union[NDArray[np.floating], List[List[NDArray[np.floating]]]],
    time_out : np.floating = 300.0,
    max_search_size : int = 500000,
    fig : figure.Figure = None,
    ax : axes.Axes = None,
    close_enough : Callable[[NDArray[np.floating], NDArray[np.floating]], bool] = None,
    soft_collision_constraint : bool = False
    ) -> Tuple[bool, float]:
        self.print_message(f'Current posisition: {start_pt}')
        self.print_message(f'Current target posisition: {end_pt}')
        node_infos : List[List[List[NodeInfo]]] = []
        for i in range(self._map_grid_size[0]):
            temp_j = []
            for j in range(self._map_grid_size[1]):
                temp_k = []
                for k in range(self._map_grid_size[2]):
                    temp_k.append(NodeInfo(grid_idx = np.array([i,j,k]),
                                               g_cost = 0,
                                               f_cost = 0,
                                               time = 0,
                                               parent = None,
                                               visited = VisitState.UNVISITED))
                temp_j.append(temp_k)
            node_infos.append(temp_j)

        start_time_stamp = time.time()
        
        open_set : List[Tuple[float, float, SearchNode]] = []
        node_set : Set[NDArray[np.integer]]= set()

        start_idx = self._grid_map.world_to_map(pos = start_pt)
        end_idx = self._grid_map.world_to_map(pos = end_pt)
        
        
        
        node_infos[int(start_idx[0])][int(start_idx[1])][int(start_idx[2])].visited = VisitState.OPEN
        node = SearchNode(grid_idx = start_idx,
                           info = node_infos[int(start_idx[0])][int(start_idx[1])][int(start_idx[2])],
                           g_cost = 0,
                           f_cost = 0,
                           input = np.array([0.,0.,0.]),
                           pos = start_pt,
                           vel = start_v,
                           time = 0,
                           duration = 0,
                           parent = None)
        
        node_set.add(tuple(node.grid_idx))
        heapq.heappush(open_set, (node.f_cost, next(self._node_counter), node))

        end_node : SearchNode = None

        # Action space
        inputs = []
        for a_x in np.arange(-self._max_acc, self._max_acc + 1e-7, self._acc_resolution):
            for a_y in np.arange(-self._max_acc, self._max_acc + 1e-7, self._acc_resolution):
                for a_z in np.arange(-self._max_acc, self._max_acc + 1e-7, self._acc_resolution):
                    inputs.append(np.array([a_x, a_y, a_z]))

        durations = list(np.arange(self._time_resolution,
                                    self._max_duration + 1e-6,
                                    self._time_resolution))

        count = 0
        stop_expanding = False
        stop_expanding_threshold = np.inf
        while len(open_set) > 0:
            # self.print_message(f"Open set size: {len(open_set)}")
            if LOCAL_MODE:
                if fig is not None and ax is not None:
                    self.plot_open_set(open_set = open_set, fig = fig, ax = ax)
            _, _, node = heapq.heappop(open_set)

            if node.info.visited != VisitState.OPEN:
                continue

            node.info.visited = VisitState.CLOSED

            time_curr = node.info.time
            delta_pos = end_pt - node.info.pos
            delta_v = end_v - node.info.vel

            def default_comparison_func(current_pos: NDArray[np.floating],
                            current_vel : Union[NDArray[np.floating],List[NDArray[np.floating]]]) -> bool:
                    eps = 0.3
                    end_pos = end_pt
                    if isinstance(end_pt, list):
                        end_pos = np.stack(end_pt)  # shape = (N, D)
                    if end_pos.ndim == 1:
                        end_pos = end_pos[np.newaxis, :]  # shape = (1, D)
                    dists = np.linalg.norm(end_pos - current_pos, axis=1)
                    return np.any(dists < eps)

            if close_enough is None:
                close_enough = default_comparison_func
            if close_enough(current_pos = node.info.pos,
                            current_vel= node.info.vel):
                end_node = node
                self.print_message(f"The planner found a path in {time.time() - start_time_stamp} second, cost: {node.f_cost}")
                break
            if time.time() - start_time_stamp > time_out:
                self.print_message(f'The planner has reached max planning time {time_out} sec.')
                return False, time.time() - start_time_stamp
            if len(open_set) > stop_expanding_threshold:
                stop_expanding = True
                self.print_message(f'The planner has reached the max open set size {stop_expanding_threshold}. Stop expanding')
            if stop_expanding:
                continue
            expanded_node_infos : List[NodeInfo] = []
            for u in inputs:
                for tau in durations:
                    new_pos, new_vel = self._state_transit(node.info.pos, node.info.vel, u, tau)
                    new_time = time_curr + tau
                    new_idx = self._grid_map.world_to_map(new_pos)

                    if self._grid_map.out_of_range(new_idx):
                        # self.print_message(f"Arrived node out of range {new_idx}, prune node.")
                        continue

                    if np.array_equal(new_idx, node.grid_idx):
                        # self.print_message(f"Arrived node with same coordinate{new_idx}, prune node.")
                        continue

                    if node_infos[int(new_idx[0])][int(new_idx[1])][int(new_idx[2])].visited == VisitState.CLOSED:
                        # self.print_message("Arrived closed node, prune node.")
                        continue

                    if np.any(np.abs(new_vel) > self._max_vel):
                        # self.print_message("Exceeding velocity limit, prune node.")
                        continue

                    no_collision = True
                    collision_cost = 0.0
                    for t_sample in np.arange(tau * self._safety_check_resolution, tau + 1e-6, tau * self._safety_check_resolution):
                        check_pos, _ = self._state_transit(node.info.pos, node.info.vel, u, t_sample)
                        if not self._get_grid_occ(check_pos):
                            if not soft_collision_constraint:
                                no_collision = False
                                break
                            else:
                                collision_cost += 5000.0
                    if not no_collision:
                        continue
                    
                    g_cost = node.g_cost + (np.dot(u, u) + self._w_time) * tau + collision_cost
                    f_cost, optimal_time = self._estimate_heuristic(new_pos, new_vel, end_pt, end_v)
                    f_cost += self._lambda_heu * f_cost

                    prune = False
                    for info in expanded_node_infos:
                        if np.array_equal(info.grid_idx, new_idx):
                            prune = True
                            if f_cost < info.f_cost:
                                info.f_cost = f_cost
                                info.g_cost = g_cost
                                info.input = u
                                info.pos = new_pos
                                info.vel = new_vel
                                info.duration = tau
                                info.time = new_time
                            break
                    if prune:
                        continue

                    info = node_infos[int(new_idx[0])][int(new_idx[1])][int(new_idx[2])]
                    if info.visited == VisitState.UNVISITED:
                        info.visited = VisitState.OPEN
                        new_node = SearchNode(
                            info=info,
                            g_cost=g_cost,
                            f_cost=f_cost,
                            pos=new_pos,
                            vel=new_vel,
                            input=u,
                            duration=tau,
                            time=new_time,
                            parent=node.info,
                            grid_idx=new_idx
                        )
                        
                        heapq.heappush(open_set, (f_cost, next(self._node_counter),  new_node))
                        node_set.add(tuple(new_idx))
                        expanded_node_infos.append(info)
                    elif info.visited == VisitState.OPEN:
                        if g_cost < info.g_cost:
                            new_node = SearchNode(info=info,
                                                  g_cost=g_cost,
                                                  f_cost=f_cost,
                                                  pos=new_pos,
                                                  vel=new_vel,
                                                  input=u,
                                                  duration=tau,
                                                  time=new_time,
                                                  parent=node.info,
                                                  grid_idx=new_idx)
                            heapq.heappush(open_set, (f_cost, next(self._node_counter), new_node))
            # self.print_message("# Expanded nodes in this loop:" + str(len(expanded_node_infos)))
            if count > max_search_size:
                self.print_message(f'The planner has reached the max search size {max_search_size}.')
                return False, time.time() - start_time_stamp
            count += 1
        if end_node is None:
            self.print_message(f'The planner failed to find a path.')
            return False, time.time() - start_time_stamp
        
        # Backtrack the path
        self._path = []
        current = end_node.info
        while current is not None:
            self._path.append(current)
            current = current.parent
        self._path.reverse()
        return True, time.time() - start_time_stamp




    def print_message(self, message : str) -> None:
        if self.verbose:
            print(message)

    # def close_enough(current_pos: NDArray[np.floating],
    #                   current_vel : NDArray[np.floating],
    #                     end_pos : Union[NDArray[np.floating],List[NDArray[np.floating]]],
    #                       end_vel : Union[NDArray[np.floating],List[List[NDArray[np.floating]]]],
    #                       eps : np.floating = 0.3) -> bool:
        
    #     if isinstance(end_pos, list):
    #         end_pos = np.stack(end_pos)  # shape = (N, D)
    #     if end_pos.ndim == 1:
    #         end_pos = end_pos[np.newaxis, :]  # shape = (1, D)
    #     dists = np.linalg.norm(end_pos - current_pos, axis=1)

    #     return np.any(dists < eps)

    

    def get_sample_path(self,
                        dt : np.floating
                        ) -> Tuple[List[NDArray[np.floating]], List[NDArray[np.floating]]] :
        sample_path: List[NDArray[np.floating]] = []
        sample_vel: List[NDArray[np.floating]] = []

        
        if len(self._path) < 2:
            return sample_path, sample_vel
        
        # Get the actual total time for the planned path!
        total_time = self._path[-1].time

        # Recalculate the number of steps required and the time step dt
        num_steps = int(np.ceil(total_time / dt))
        dt = total_time / num_steps

        t = 0.0
        idx = 0

        while t < total_time + 1e-6:
            curr_node = self._path[idx]
            next_node = self._path[idx + 1]

            p_0 = self._path[idx].pos
            v_0 = self._path[idx].vel
            u = next_node.input

            tau = t - curr_node.time
            p_1, v_1 = self._state_transit(p_0 = p_0, v_0 = v_0, um = u, tau = tau)

            sample_path.append(p_1)
            sample_vel.append(v_1)
            t += dt

            while idx + 1 < len(self._path) - 1 and t >= self._path[idx + 1].time:
                idx += 1

        # if self._is_shot_success:
        #     t = 0.0
        #     while t < self._t_shot + 1e-6:
        #         time_vec = np.array([t**0, t**1, t**2, t**3])
        #         time_deriv = np.array([0.0, 1.0, 2*t, 3*t**2])

        #         pos = self._coef_shot @ time_vec
        #         vel = self._coef_shot @ time_deriv

        #         sample_path.append(pos)
        #         sample_vel.append(vel)

        #         t += dt

        return sample_path, sample_vel


    def _estimate_heuristic(self,
                            p_0 : NDArray[np.floating],
                            v_0 : NDArray[np.floating],
                            p_1 : NDArray[np.floating],
                            v_1 : NDArray[np.floating],
                            ) -> Tuple[np.floating, np.floating]:
        
        dp = p_1 - p_0
        c_1 = -36 * np.dot(dp, dp)
        c_2 = 24 * np.dot((v_0 + v_1), dp)
        c_3 = -4 * (np.dot(v_0, v_0) + np.dot(v_0, v_1) + np.dot(v_1, v_1))
        c_4 = 0.0
        c_5 = self._w_time

        ts = poly.quartic_solve_real(a = c_5, b = c_4, c = c_3, d = c_2, e = c_1)

        v_max = self._max_vel * 0.5
        t_bar = np.linalg.norm(dp, ord = np.inf) / v_max

        ts.append(t_bar)

        cost = np.inf
        t_d = t_bar
        
        for t in ts:
            if t < t_bar:
                continue
            c = -c_1 / (3 * t * t * t) - c_2 / (2 * t * t) - c_3 / t + self._w_time * t
            if c < cost:
                cost = c
                t_d = t
        
        optimal_time = t_d
        return 1.0 * (1 + self._tie_breaker) * cost, optimal_time



    def _compute_shot_traj(self,
                       state_1: NDArray[np.floating],
                       state_2: NDArray[np.floating],
                       time_to_goal: float) -> bool:

        p_0 = state_1[0:3]
        v_0 = state_1[3:6]
        p_1 = state_2[0:3]
        v_1 = state_2[3:6]
        d_p = p_1 - p_0
        d_v = v_1 - v_0
        T = time_to_goal

        a = ( -12.0 / T**3 * (d_p - v_0 * T) + 6.0 / T**2 * d_v ) / 6.0
        b = (  6.0 / T**2 * (d_p - v_0 * T) - 2.0 / T * d_v ) / 2.0
        c = v_0
        d = p_0


        coef = np.zeros((3, 4))
        coef[:, 0] = d
        coef[:, 1] = c
        coef[:, 2] = b
        coef[:, 3] = a

        Tm = np.array([
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0]
        ])

        t_delta = T / 20
        t = t_delta
        while t <= T + 1e-6:
            t_vec = np.array([1.0, t, t**2, t**3])
            for dim in range(3):
                poly = coef[dim, :]
                pos = poly @ t_vec
                if (pos < self._origin).any() or (pos >= self._origin + self._map_size).any():
                    return False
                if not self._get_grid_occ(pos):
                    return False
            t += t_delta

        self._coef_shot = coef
        self._t_shot = T
        self._is_shot_success = True
        return True


    def _get_grid_occ(self,
                      pos : NDArray[np.floating]) -> bool:
        return self._grid_map.is_free(self._grid_map.world_to_map(pos = pos), range_check = True)
    
    def _state_transit(self,
                        p_0 : NDArray[np.floating],
                        v_0 : NDArray[np.floating],
                        um : NDArray[np.floating],
                        tau : np.floating,
                        ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        return p_0 + 0.5 * tau * tau * um + v_0 * tau, v_0 + tau * um
    
    def remove_plot(self) -> bool:
        if hasattr(self,'_open_set_plot') and self._open_set_plot is not None:
            try:
                self._open_set_plot.remove()
                self._open_set_plot = None
                return True
            except:
                return False

    def plot_open_set(self, open_set: List[Tuple[float,float, SearchNode]],
                    fig : figure.Figure,
                      ax : axes.Axes,
                        size : float= 10.0,
                          color: str = 'purple',
                          label : str = 'open set') -> Tuple[figure.Figure, axes.Axes]:
        
        if self._open_set_plot is not None:
            self._open_set_plot.remove()
            self._open_set_plot = None

        if len(open_set) == 0:
            return
        xs = []
        ys = []
        zs = []
        for _, _, node in open_set:
            if node.info.visited == VisitState.OPEN:
                xs.append(node.info.pos[0])
                ys.append(node.info.pos[1])
                zs.append(node.info.pos[2])

        self._open_set_plot = ax.scatter(xs, ys, zs, c=color, s=size, alpha=0.6, label=label)
        fig.canvas.draw()
        fig.canvas.flush_events()

        return fig, ax


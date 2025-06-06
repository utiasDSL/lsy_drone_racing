"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from statistics import pvariance
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else
from scipy.fft import prev_fast_len
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from casadi import interpolant
from sympy import true
from traitlets import TraitError

from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.control.easy_controller import EasyController
from lsy_drone_racing.control import Controller
from lsy_drone_racing.tools.ext_tools import TrajectoryTool
from lsy_drone_racing.utils.utils import draw_line
LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    import matplotlib.collections
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False
if TYPE_CHECKING:
    from numpy.typing import NDArray



class MPCCPrescriptedController(EasyController):
    """Implementation of MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env=None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config, ros_tx_freq = 50)
        self.freq = config.env.freq
        self._tick = 0

        self.env = env
        self.init_gates(obs = obs,
                         gate_inner_size = [0.2,0.2,0.2,0.3],
                         gate_outer_size = [1.0,1.0,1.0,0.8],
                         gate_safe_radius = [0.4,0.4,0.4,0.4],
                         entry_offset = [0.3,0.3,0.1,0.05],
                         exit_offset = [0.4,0.2,0.3,0.3],
                        #  entry_offset = [0.3,0.7,0.3,0.2],
                        #  exit_offset = [0.5,0.1,0.1,0.3],
                         thickness = [0.2, 0.2, 0.2, 0.05],
                         vel_limit = [1.0, 1.0, 0.2, 1.0])

        # # pre-planned trajectory
        # t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv")
        t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_c_6_sec_bigger_pillar.csv")     
     
        trajectory = CubicSpline(t, pos)

        # global params
        self.N = 50
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05 # the segment interval for trajectory to be input to the model
        self.model_traj_length = 12 # maximum trajectory length the param can take

        # trajectory reparameterization
        self.traj_tool = TrajectoryTool()
        trajectory = self.traj_tool.extend_trajectory(trajectory)
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory)
        
        # build model & create solver
        self.acados_ocp_solver, self.ocp = self.create_ocp_solver(self.T_HORIZON, self.N, self.arc_trajectory)

        # initialize
        self.last_theta = 0.0
        self.last_v_theta = 0.0 # TODO: replan?
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

    def export_quadrotor_ode_model(self) -> AcadosModel:
        """Symbolic Quadrotor Model."""
        # Define name of solver to be used in script
        model_name = "lsy_example_mpc"

        # Define Gravitational Acceleration
        GRAVITY = 9.806

        # Sys ID Params
        params_pitch_rate = [-6.003842038081178, 6.213752925707588]
        params_roll_rate = [-3.960889336015948, 4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]
        params_acc = [20.907574256269616, 3.653687545690674]

        """Model setting"""
        # define basic variables in state and input vector
        self.px = MX.sym("px")  # 0
        self.py = MX.sym("py")  # 1
        self.pz = MX.sym("pz")  # 2
        self.vx = MX.sym("vx")  # 3
        self.vy = MX.sym("vy")  # 4
        self.vz = MX.sym("vz")  # 5
        self.roll = MX.sym("r")  # 6
        self.pitch = MX.sym("p")  # 7
        self.yaw = MX.sym("y")  # 8
        self.f_collective = MX.sym("f_collective")

        self.f_collective_cmd = MX.sym("f_collective_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")

        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")

        # expanded observation state
        self.theta = MX.sym("theta")
        # self.v_theta = MX.sym("v_theta")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        # define state and input vector
        states = vertcat(
            self.px,
            self.py,
            self.pz,
            self.vx,
            self.vy,
            self.vz,
            self.roll,
            self.pitch,
            self.yaw,
            self.f_collective,
            self.r_cmd,
            self.p_cmd,
            self.y_cmd,
            self.theta
        )
        inputs = vertcat(
            self.f_collective_cmd, 
            self.dr_cmd, 
            self.dp_cmd, 
            self.dy_cmd, 
            self.v_theta_cmd
        )

        # Define nonlinear system dynamics
        f = vertcat(
            self.vx,
            self.vy,
            self.vz,
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * cos(self.yaw) + sin(self.roll) * sin(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * sin(self.yaw) - sin(self.roll) * cos(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1]) * cos(self.roll) * cos(self.pitch) - GRAVITY,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_collective_cmd - self.f_collective),
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        # define dynamic trajectory input
        self.pd_list = MX.sym("pd_list", 3*int(self.model_traj_length/self.model_arc_length))
        self.tp_list = MX.sym("tp_list", 3*int(self.model_traj_length/self.model_arc_length))
        self.qc_dyn = MX.sym("qc_dyn", 1*int(self.model_traj_length/self.model_arc_length))
        self.qc_curv_dyn = MX.sym("qc_curv_dyn", 1*int(self.model_traj_length/self.model_arc_length))

        params = vertcat(
            self.pd_list, 
            self.tp_list,
            self.qc_dyn,
            self.qc_curv_dyn
        )

        # Initialize the nonlinear model for NMPC formulation
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f
        model.f_impl_expr = None
        model.x = states
        model.u = inputs
        model.p = params

        return model
    
    
    def casadi_linear_interp(self, theta, theta_list, p_flat, dim=3):
        """Manually interpolate a 3D path using CasADi symbolic expressions.
        
        Args:
            theta: CasADi symbol, scalar progress variable (0 ~ model_traj_length)
            theta_list: list or array, thetas of path points [0, 0.1, 0.2, ...]
            p_flat: CasADi symbol, 1D flattened path points [x0,y0,z0, x1,y1,z1, ...]
            dim: int, dimension of a single point (default=3)
        Returns:
            p_interp: CasADi 3x1 vector, interpolated path point at given theta
        """
        M = len(theta_list)
        
        # Find index interval
        # Normalize theta to index scale
        idx_float = (theta - theta_list[0]) / (theta_list[-1] - theta_list[0]) * (M - 1)

        idx_lower = floor(idx_float)
        idx_upper = idx_lower + 1
        alpha = idx_float - idx_lower

        # Handle boundary cases (clamping)
        idx_lower = if_else(idx_lower < 0, 0, idx_lower)
        idx_upper = if_else(idx_upper >= M, M-1, idx_upper)

        # Gather points
        p_lower = vertcat(*[p_flat[dim * idx_lower + d] for d in range(dim)])
        p_upper = vertcat(*[p_flat[dim * idx_upper + d] for d in range(dim)])

        # Interpolated point
        p_interp = (1 - alpha) * p_lower + alpha * p_upper

        return p_interp
    
    def get_updated_traj_param(self, trajectory: CubicSpline):
        """get updated trajectory parameters upon replaning"""
        # construct pd/tp lists from current trajectory

        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_list = trajectory(theta_list)
        tp_list = trajectory.derivative(1)(theta_list)
        qc_dyn_list = np.zeros_like(theta_list)
        
        radius_list =TrajectoryTool.compute_3d_turning_radius_from_vector_spline(trajectory, theta_list)
        radius_list[:40] = 10.0
        radius_list = np.clip(radius_list, 0.1, 10)
        radius_list_filtered = savgol_filter(radius_list, window_length=100, polyorder=5)
        radius_list_gaussian = 8 * np.exp(-50 * radius_list_filtered ** 2) # gaussian
        qc_curv_dyn_list = savgol_filter(radius_list_gaussian, window_length=100, polyorder=5)
        self.radius_filtered = CubicSpline(theta_list,qc_curv_dyn_list)
         
        
        for gate in self.gates:
            distances = np.linalg.norm(pd_list - gate.pos, axis=-1)
            qc_dyn_gate = np.exp(-5 * distances ** 2) # gaussian
            qc_dyn_list = np.maximum(qc_dyn_gate, qc_dyn_list)
        p_vals = np.concatenate([pd_list.flatten(), tp_list.flatten(), qc_dyn_list, qc_curv_dyn_list])
        return p_vals

    def mpcc_cost(self):
        """calculate mpcc cost function"""
        pos = vertcat(self.px, self.py, self.pz)
        ang = vertcat(self.roll, self.pitch, self.yaw)
        control_input = vertcat(self.f_collective_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        # interpolate spline dynamically
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_theta = self.casadi_linear_interp(self.theta, theta_list, self.pd_list)
        tp_theta = self.casadi_linear_interp(self.theta, theta_list, self.tp_list)
        qc_dyn_theta = self.casadi_linear_interp(self.theta, theta_list, self.qc_dyn, dim=1)
        qc_curv_theta = self.casadi_linear_interp(self.theta, theta_list, self.qc_curv_dyn, dim=1)
        tp_theta_norm = tp_theta / norm_2(tp_theta)
        e_theta = pos - pd_theta
        e_l = dot(tp_theta_norm, e_theta) * tp_theta_norm
        e_c = e_theta - e_l

        mpcc_cost = (self.q_l + self.q_l_peak * qc_dyn_theta) * dot(e_l, e_l) + \
                    (self.q_c  + self.q_c_peak * qc_dyn_theta) * dot(e_c, e_c) + \
                    (ang.T @ self.Q_w @ ang) + \
                    (control_input.T @ self.R_df @ control_input) + \
                    (-self.miu) * self.v_theta_cmd
        return mpcc_cost

    def create_ocp_solver(
        self, Tf: float, N: int, trajectory: CubicSpline,  verbose: bool = False
    ) -> tuple[AcadosOcpSolver, AcadosOcp]:
        """Creates an acados Optimal Control Problem and Solver."""
        ocp = AcadosOcp()

        # set model
        model = self.export_quadrotor_ode_model()
        ocp.model = model

        # Get Dimensions
        self.nx = model.x.rows()
        self.nu = model.u.rows()

        # Set dimensions
        ocp.solver_options.N_horizon = N

        ## Set Cost
        # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

        # Cost Type
        ocp.cost.cost_type = "EXTERNAL"

        # Weights
        self.q_l = 160
        self.q_l_peak = 640
        self.q_l_curv_peak = 0

        self.q_c =  80
        self.q_c_peak = 800
        self.q_c_curv_peak = 0
        
        self.Q_w = 1 * DM(np.eye(3))
        self.r_dv = 1
        self.R_df = DM(np.diag([0,1,1,1])) # cannot punish collective thrust
        self.miu = 2
        # param A: works and works well

        # Weights for easy planner
        # self.q_l = 120
        # self.q_l_peak = 100
        # self.q_l_curv_peak = 0
        # self.q_c = 50
        # self.q_c_peak = 100
        # self.q_c_curv_peak = 0

        
        # self.Q_w = DM(np.eye(3))
        # self.r_dv = 1
        # self.R_df = DM(np.eye(4))
        # self.miu = 1

        
        ocp.model.cost_expr_ext_cost = self.mpcc_cost()

        # Set State Constraints
        ocp.constraints.lbx = np.array([0.1, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([0.55, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12])

        # Set Input Constraints
        ocp.constraints.lbu = np.array([0.1, -10.0, -10.0, -10.0, 0]) # last term is v_theta should be positive
        ocp.constraints.ubu = np.array([0.55, 10.0, 10.0, 10.0, 3.0]) # contraint f_collective_thrust
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        # We have to set x0 even though we will overwrite it later on.
        ocp.constraints.x0 = np.zeros((self.nx))
        # Set initial reference trajectory
        p_vals = self.get_updated_traj_param(self.arc_trajectory)
        ocp.parameter_values = p_vals

        # Solver Options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        ocp.solver_options.tol = 1e-5

        ocp.solver_options.qp_solver_cond_N = N
        ocp.solver_options.qp_solver_warm_start = 1

        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50

        # set prediction horizon
        ocp.solver_options.tf = Tf

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted.json", verbose=verbose)

        return acados_ocp_solver, ocp

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # i = min(self._tick, len(self.x_des) - 1)
        # if self._tick > i:
        #     self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective]),
                self.last_rpy_cmd,
                np.array([self.last_theta])
            )
        )
        # xcurrent[-2], _ = self.traj_tool.find_nearest_waypoint(self.arc_trajectory, obs["pos"], self.last_theta+1) # correct theta
        ## warm-start - provide initial guess to guarantee stable convergence
        if not hasattr(self, "x_guess"):
            self.x_guess = [xcurrent for _ in range(self.N + 1)]
            self.u_guess = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self.x_guess = self.x_guess[1:] + [self.x_guess[-1]]
            self.u_guess = self.u_guess[1:] + [self.u_guess[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self.x_guess[i])
            self.acados_ocp_solver.set(i, "u", self.u_guess[i])
        self.acados_ocp_solver.set(self.N, "x", self.x_guess[self.N])
        
        ## replan trajectory: not needed now!
        # TODO: Gate changes when already getting close to it, but this causes a large jump of trajectory, e_c suddenly goes up, and it fails to track
        # TODO: Must we plan from current position?
        # if self.pos_change_detect(obs):
        #     gates_rotates = R.from_quat(obs['gates_quat'])
        #     rot_matrices = np.array(gates_rotates.as_matrix())
        #     self.gates_norm = np.array(rot_matrices[:,:,1])
        #     self.gates_pos = obs['gates_pos']
        #     # replan trajectory
        #     waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
        #     t, waypoints = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
        #     t, waypoints = self.add_drone_to_waypoints(waypoints, obs['pos'], 0.3, curr_theta=self.last_theta+1)
        #     trajectory = self.trajectory_generate(self.t_total, waypoints)
        #     trajectory = self.traj_tool.extend_trajectory(trajectory)
        #     self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory, epsilon=1e-3)
        #     # write trajectory as parameter to solver
        #     p_vals = self.get_updated_traj_param(self.arc_trajectory)
        #     # xcurrent[-2], _ = self.traj_tool.find_nearest_waypoint(self.arc_trajectory, obs["pos"]) # correct theta
        #     for i in range(self.N): 
        #         self.acados_ocp_solver.set(i, "p", p_vals)
        #         # TODO: maybe it needs to be initialized differently? uniform guess: solver failure; original warmup: unable to track new traj
        #         # self.acados_ocp_solver.set(i, "x", xcurrent) 
        #         # self.acados_ocp_solver.set(i, "u", np.zeros(self.nu))

        # set initial state
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        if self.acados_ocp_solver.solve() == 4:
            pass

        ## update initial guess
        self.x_guess = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self.u_guess = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]

        x1 = self.acados_ocp_solver.get(1, "x")
        u1 = self.acados_ocp_solver.get(1, "u")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_theta = self.last_theta * (1 - w) + x1[13] * w
        self.last_f_cmd = self.last_f_cmd * (1-w) + u1[0] * w
        self.last_rpy_cmd = self.last_rpy_cmd * (1-w) + x1[10:13] * w


        cmd = np.concatenate(
            (
                np.array([self.last_f_cmd]),
                self.last_rpy_cmd
            )
        )


        # guess_theta = self.last_theta
        # true_theta, _ = self.traj_tool.find_nearest_waypoint(self.arc_trajectory, obs["pos"], guess_theta + 1.5)
        # draw_line(self.env, np.stack([self.arc_trajectory(guess_theta), self.arc_trajectory(true_theta)]), rgba=np.array([8*max(true_theta-guess_theta, 0), 8*max(guess_theta-true_theta, 0), 0.0, 1.0]))


        ## visualization
        # test true theta and guess theta
        # print(self.radius_filtered(self.last_theta))
        # plt.plot(self.arc_trajectory.x, self.radius_filtered(self.arc_trajectory.x))
        # plt.show()
        # input()
        try:

            draw_line(self.env, self.arc_trajectory(self.arc_trajectory.x), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
            draw_line(self.env, np.stack([self.arc_trajectory(self.last_theta), obs["pos"]]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
            pos_traj = np.array([self.acados_ocp_solver.get(i, "x")[:3] for i in range(self.N+1)])
            draw_line(self.env, pos_traj[0:-1:5],rgba=np.array([1.0, 1.0, 0.0, 0.2]) )
        except:
            pass

        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self.step_update(obs = obs)
        self.update_next_gate()

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0

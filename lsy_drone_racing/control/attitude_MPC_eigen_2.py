"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc_ext"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")
    #
    # Obstacles as symbolic parameters (4 obstacles in 2D)
    p_obs1 = MX.sym("p_obs1", 2)
    p_obs2 = MX.sym("p_obs2", 2)
    p_obs3 = MX.sym("p_obs3", 2)
    p_obs4 = MX.sym("p_obs4", 2)

    p_ref = MX.sym("p_ref", 3)
    #



    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )


    #Define params necessary for external cost function
    params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.p = params

  
   
    

    # Penalize aggressive commands (smoother control)
    Q_control = 0.01
    control_penalty = df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2

    # Penalize large angles (prevents flips)
    Q_angle = 0.01
    angle_penalty = roll**2 + pitch**2  # Yaw penalty optional

    sharpness=8
    #Penalising proximity to obstacles
    d1 = (px - p_obs1[0])**sharpness + (py - p_obs1[1])**sharpness
    d2 = (px - p_obs2[0])**sharpness + (py - p_obs2[1])**sharpness
    d3 = (px - p_obs3[0])**sharpness + (py - p_obs3[1])**sharpness
    d4 = (px - p_obs4[0])**sharpness + (py - p_obs4[1])**sharpness
    safety_margin = 0.000002 # Min allowed distance squared
    Q_obs=2 
    obs_cost = (0.25*np.exp(-d1/(safety_margin)) + np.exp(-d2/safety_margin) + 
           np.exp(-d3/safety_margin) + 0.5*np.exp(-d4/safety_margin))

    #Penalising deviation from Reference trajectory #1
    Q_pos = 10.0  
    pos_error = (px - p_ref[0])**2 + (py - p_ref[1])**2 + (pz - p_ref[2])**2


    total_cost = (
        Q_pos * pos_error +
        Q_control * control_penalty +
        Q_angle * angle_penalty
        +Q_obs*obs_cost)

    model.cost_expr_ext_cost = total_cost
    model.cost_expr_ext_cost_e = Q_pos * pos_error 


    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model
    

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ocp.dims.np = model.p.rows()

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    default_params = np.zeros(ocp.dims.np)
    ocp.parameter_values = default_params  # Add this line
   

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    nx = model.x.rows()
    ocp.constraints.x0 = np.zeros((nx))





    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc_ext.json", verbose=verbose)

    return acados_ocp_solver, ocp


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self.y=[]
        self.y_mpc=[]
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self.waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.1, -1.5, 0.65],
                [1.1, -0.85, 1.15],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )
        self.prev_obstacle = [
            [1, 0, 1.4],
            [0.5, -1, 1.4],
            [0, 1.5, 1.4],
            [-0.5, 0.5, 1.4],
        ]
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])
        #visualising traj. Needed for visualisiing draw line##
        tvisual = np.linspace(0, 1, 50)
        x = cs_x(tvisual)
        y = cs_y(tvisual)
        z = cs_z(tvisual)
        self.traj_vis=np.array([x,y,z])
        #
        des_completion_time = 9
        ts = np.linspace(0, 1, int(self.freq * des_completion_time))

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        self.N = 20
        self.T_HORIZON = 1
        self.dt = self.T_HORIZON / self.N
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))
    
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

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


        update = self.check_for_update(obs)
        if update:
            print('Es hat sich was geändert')
        
                









        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        self.y=[]
        for j in range(self.N):
            
            yref = np.array(
                [self.prev_obstacle[0,0],self.prev_obstacle[0,1],
                self.prev_obstacle[1,0],self.prev_obstacle[1,1],
                self.prev_obstacle[2,0],self.prev_obstacle[2,1],
                self.prev_obstacle[3,0],self.prev_obstacle[3,1],
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
                   
                ]
            )
            self.acados_ocp_solver.set(j, "p", yref)
            self.y.append(yref)
        yref_N = np.hstack([
            self.prev_obstacle[:, :2].flatten(),
            self.x_des[i + self.N],
            self.y_des[i + self.N],
            self.z_des[i + self.N],
        ])
        self.acados_ocp_solver.set(self.N, "p", yref_N)
        #### Obs avoidance
        
        self.acados_ocp_solver.solve()
        self.y_mpc = []
        for j in range(self.N + 1):  # Include terminal state
            x_pred = self.acados_ocp_solver.get(j, "x")
            # Extract relevant states to match your y_ref format if needed
            # This depends on how you want to compare them
            y_mpc = x_pred[:len(yref_N)]  # Adjust this based on your state vector
            self.y_mpc.append(y_mpc)





        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]

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
        self._tick += 1

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0

    def check_for_update(self,obs):
        
        if not np.array_equal(self.prev_obstacle,obs["obstacles_pos"]):
            # print('Obstacle has changed:')  
             
            print(obs["obstacles_pos"])
            self.prev_obstacle=obs["obstacles_pos"]
            return True
        return False
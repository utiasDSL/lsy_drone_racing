"""Acados model and OCP solver for quadrotor MPC control.

This module contains the symbolic model definition and OCP solver
for the quadrotor attitude control interface.
"""

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    model_name = "lsy_example_mpc"
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    # Define state variables (14 total)
    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    roll = MX.sym("r")
    pitch = MX.sym("p")
    yaw = MX.sym("y")
    f_collective = MX.sym("f_collective")
    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    # Control inputs: 4 real controls only
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # State and input vectors
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

    # System dynamics (only for the 14 states)
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

    # Initialize the model
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.x = states
    model.u = inputs

    return model


def setup_ocp(
    model: AcadosModel, Tf: float, N: int, mpc_weights: dict = None
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    ocp.model = model

    # Get Dimensions - no slack variables
    nx = model.x.rows()  # 14 states
    nu = model.u.rows()  # 4 inputs (real controls only)
    ny = nx + nu  # 18 total for cost (14 states + 4 inputs)
    ny_e = nx  # 14 for terminal

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Define weight matrices for the cost function
    # Use provided weights or defaults
    if mpc_weights is not None:
        pos = mpc_weights.get("Q_pos", 7.0)
        vel = mpc_weights.get("Q_vel", 0.5)
        rpy = mpc_weights.get("Q_rpy", 0.01)
        f_col = mpc_weights.get("Q_thrust", 0.01)
        rpy_cmd = mpc_weights.get("Q_cmd", 0.01)
        r_param = mpc_weights.get("R", 0.007)
    else:
        # Default values
        pos = 7.0
        vel = 0.5
        rpy = 0.01
        f_col = 0.01
        rpy_cmd = 0.01
        r_param = 0.007

    Q = np.diag(
        [
            pos,
            pos,
            pos,  # Position
            vel,
            vel,
            vel,  # Velocity
            rpy,
            rpy,
            rpy,  # rpy - roll, pitch, yaw
            f_col,
            f_col,  # f_collective, f_collective_cmd
            rpy_cmd,
            rpy_cmd,
            rpy_cmd,  # rpy_cmd
        ]
    )

    # Control input regularization
    R = np.diag([r_param, r_param, r_param, r_param])  # Real control cost

    Q_e = Q.copy()

    # Stage cost matrix (18x18) - states + real controls only
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Vx matrix (18x14) - maps states to cost
    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vx = Vx

    # Vu matrix (18x4) - maps inputs to cost
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + 4, :4] = np.eye(4)  # Real controls
    ocp.cost.Vu = Vu

    # Terminal Vx matrix (14x14)
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # Set initial references (18 dimensions)
    ocp.cost.yref = np.array(
        [
            1.0,
            1.0,
            0.4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.35,
            0.35,
            0.0,
            0.0,
            0.0,  # states
            0.0,
            0.0,
            0.0,
            0.0,  # real controls
        ]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf
    return ocp

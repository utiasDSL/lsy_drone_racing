"""Acados model and OCP solver for quadrotor MPC control.

This module contains the symbolic model definition and OCP solver
for the quadrotor attitude control interface.
"""

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, transpose


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

    # Params
    # Parameters: reference position (3) + tangent vector (3)
    p_ref = MX.sym("p_ref", 6)

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
    model.p = p_ref
    model.x = states
    model.u = inputs

    return model


def export_quadrotor_ode_model_mpcc() -> AcadosModel:
    model = export_quadrotor_ode_model()
    model.name = "quadrotor_mpcc"

    # Add MPCC variables to the state
    theta = MX.sym("theta")  # Arc-length position along path
    v_theta = MX.sym("v_theta")  # Virtual progress velocity

    # Add to state vector
    model.x = vertcat(model.x, theta, v_theta)

    # Add virtual input ∆v_theta
    d_v_theta = MX.sym("d_v_theta")
    model.u = vertcat(model.u, d_v_theta)

    # Append progress dynamics
    delta_t = 0.02  # sample time, change as needed
    v_theta_next = v_theta + d_v_theta * delta_t
    theta_next = theta + v_theta * delta_t

    # Extend f_expl_expr to include progress dynamics
    model.f_expl_expr = vertcat(model.f_expl_expr, theta_next - theta, v_theta_next - v_theta)

    return model


def define_mpcc_cost(model):
    """Construct the external MPCC cost expression using parameterized reference."""

    # Get Dimensions - no slack variables
    nx = model.x.rows()  # 14 states
    nu = model.u.rows()  # 4 inputs (real controls only)
    ny = nx + nu  # 18 total for cost (14 states + 4 inputs)
    ny_e = nx  # 14 for terminal
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + 4, :4] = np.eye(4)  # Real controls

    # Parameters (must match ocp.parameter_values layout)
    p_ref = model.p[:3]  # desired position on reference path
    t_ref = model.p[3:6]  # tangent vector at that point

    # Extract state vars
    x = model.x
    pos = x[0:3]  # current position
    theta = x[-2]
    v_theta = x[-1]

    # Error from current to desired path
    e = pos - p_ref

    # Unit tangent vector
    t_hat = t_ref / (dot(t_ref, t_ref) + 1e-6) ** 0.5

    # Lag and contour errors
    lag_error = dot(t_hat, e)
    contour_error = e - lag_error * t_hat

    # Weights (tune as needed)
    q_c = 300.0
    q_l = 50.0
    mu = 0.001

    # u_ref
    u_ref = np.array([0.35, 0, 0, 0, 0])

    # Cost expression
    stage_cost = (
        q_c * dot(contour_error, contour_error)
        + q_l * lag_error**2
        - mu * v_theta
        + 0.1 * transpose(model.u - u_ref) @ (model.u - u_ref)
        + 200 * transpose(model.x[6:9]) @ (model.x[6:9])
    )
    terminal_cost = 0  # stage_cost

    return stage_cost, terminal_cost


def setup_ocp(
    model: AcadosModel, Tf: float, N: int, mpc_weights: dict = None
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    ocp.model = model

    # Get Dimensions - no slack variables
    nx = model.x.rows()  # 14 states

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost, ocp.model.cost_expr_ext_cost_e = define_mpcc_cost(model)

    ocp.parameter_values = np.array([0.0] * ocp.model.p.rows())

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57, 0, -0.5])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57, 10, 0.5])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 14, 15])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf
    return ocp

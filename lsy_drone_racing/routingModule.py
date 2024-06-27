
import pyomo.environ as pyo
import concurrent.futures
import numpy as np

########################################################################################################################
# MIP Model
########################################################################################################################
def initialize_model_variables(start, goal, gates, obstacles, steps):
    #initialize model
    model = pyo.ConcreteModel()

    model.start = start
    model.goal = goal
    model.gates = gates
    model.obstacles = obstacles
    model.steps = steps

    # initialize constant model parameters
    model.n_obstacles = len(model.obstacles)
    model.n_gates = len(model.gates)
    model.n_steps = steps[len(steps)-1] + 40
    model.M = 10000
    model.dim = 3
    model.gate_width = [0.21, 0.21, 0.3]
    model.obstacle_width = [0.35, 0.35, 0.65]
    model.obstacle_top = [0.25, 0.25, 0.2]
    model.upper_constraints = [3, 3, 2]
    model.lower_constraints = [-3, -3, 0]

    # initialize range variables
    model.obstacle_range = pyo.RangeSet(0,model.n_obstacles-1)
    model.gate_range = pyo.RangeSet(0, model.n_gates-1)
    model.step_range = pyo.RangeSet(0, model.n_steps-1)
    model.dim_range = pyo.RangeSet(0, model.dim-1)

    # initialize pyomo Variables for optimization
    model.x = pyo.Var(model.dim_range, model.step_range, within=pyo.Reals, bounds=(model.lower_constraints[0], model.upper_constraints[0]))
    model.bx = pyo.Var(model.dim_range, model.step_range, within=pyo.Binary)
    model.bobject = pyo.Var(model.dim_range, model.obstacle_range, pyo.RangeSet(0,4), model.step_range, within=pyo.Binary)
    model.length = pyo.Var( model.step_range, within=pyo.PositiveReals)

    return model

########################################################################################################################
# Model Constraints
########################################################################################################################
def travel_only_one_direction_1(model, step, dim):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    return model.x[dim, step] <= model.x[dim, step + 1] + model.bx[dim, step] * model.M
def travel_only_one_direction_2(model, step, dim):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    return model.x[dim, step] >= model.x[dim, step + 1] - model.bx[dim, step] * model.M
def travel_only_one_direction_3(model, step):
    return model.bx[0, step] + model.bx[1, step] + model.bx[2, step] == 1


def avoid_objects_1(model, object, step, dim):
    if model.obstacles[object][-1] <= 0.5:
        return model.x[dim, step] <= model.obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 0, step] * model.M
    elif model.obstacles[object][-1] <= 1.5:
        return model.x[dim, step] <= model.obstacles[object][dim] - model.obstacle_top[dim] + model.bobject[
            dim, object, 0, step] * model.M
    else:
        return model.x[dim, step] <= model.obstacles[object][dim] - model.gate_width[dim] + model.bobject[dim, object, 0, step] * model.M
def avoid_objects_2(model, object, step, dim):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    if model.obstacles[object][-1] <= 0.5:
        return model.x[dim, step + 1] <= model.obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 0, step] * model.M
    elif model.obstacles[object][-1] <= 1.5:
        return model.x[dim, step + 1] <= model.obstacles[object][dim] - model.obstacle_top[dim] + model.bobject[
            dim, object, 0, step] * model.M
    else:
        return model.x[dim, step + 1] <= model.obstacles[object][dim] - model.gate_width[dim] + model.bobject[dim, object, 0, step] * model.M
def avoid_objects_3(model, object, step, dim):
    if model.obstacles[object][-1] <= 0.5:
        return model.x[dim, step] >= -(model.obstacles[object][dim] + model.obstacle_width[dim]) - model.bobject[dim, object, 1, step] * model.M
    elif model.obstacles[object][-1] <= 1.5:
        return model.x[dim, step] >= -(model.obstacles[object][dim] + model.obstacle_top[dim]) - model.bobject[
            dim, object, 1, step] * model.M
    else:
        return model.x[dim, step] >= -(model.obstacles[object][dim] + model.gate_width[dim]) - model.bobject[dim, object, 1, step] * model.M
def avoid_objects_4(model, object, step, dim):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    if model.obstacles[object][-1] <= 0.5:
        return model.x[dim, step + 1] >= -(model.obstacles[object][dim] + model.obstacle_width[dim]) - model.bobject[dim, object, 1, step] * model.M
    elif model.obstacles[object][-1] <= 1.5:
        return model.x[dim, step + 1] >= -(model.obstacles[object][dim] + model.obstacle_top[dim]) - model.bobject[
            dim, object, 1, step] * model.M
    else:
        return model.x[dim, step + 1] >= -(model.obstacles[object][dim] + model.gate_width[dim]) - model.bobject[dim, object, 1, step] * model.M
def avoid_objects_5(model, object, step):
    sum = 0
    for dim in range(0, 3):
        for var in range(0, 2):
            sum += model.bobject[dim, object, var, step]
    return 5 >= sum

def fly_through_gates_easy(model, dim, gate):
    return model.x[dim, model.steps[gate]] == model.gates[gate][dim]

def start_at_start(model, dim):
    return model.x[dim, 0] == model.start[dim]
def end_at_goal(model, dim):
    return model.x[dim, model.n_steps-1] == model.goal[dim]


def length_rule_1(model, dim, step):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    return model.length[step] >= model.x[dim, step] - model.x[dim, step+1]
def length_rule_2(model, dim, step):
    if step == model.n_steps-1:
        return pyo.Constraint.Skip
    return model.length[step] >= model.x[dim, step+1] - model.x[dim, step]

def length_rule_3(model, step):
    return  model.length[step] >= 0.0001

def length_rule_4(model, step):
    return  model.length[step] <= 0.3

def objective_rule(model):
    sum = 0
    for step in range(0, model.n_steps-1):
        sum += model.length[step]
        if step < 10:
            sum -= 0.00001 * model.x[2,step]
    return sum


def initialize_constraints(model, three_degrees_of_freedom=False):
    # start position and goal position constraint
    model.start_rule = pyo.Constraint(model.dim_range, rule=start_at_start)
    model.end_rule = pyo.Constraint(model.dim_range, rule=end_at_goal)

    # rules to avoid obstacles
    model.avoid_obstacle_1 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_1)
    model.avoid_obstacle_2 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_2)
    model.avoid_obstacle_3 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_3)
    model.avoid_obstacle_4 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_4)
    model.avoid_obstacle_5 = pyo.Constraint(model.obstacle_range, model.step_range, rule=avoid_objects_5)

    # rules to control the step length
    model.length_1 = pyo.Constraint(model.dim_range, model.step_range, rule=length_rule_1)
    model.length_2 = pyo.Constraint(model.dim_range, model.step_range, rule=length_rule_2)
    model.length_3 = pyo.Constraint(model.step_range, rule=length_rule_3)
    model.length_4 = pyo.Constraint(model.step_range, rule=length_rule_4)

    #rules to control degrees of freedom
    if three_degrees_of_freedom == False:
        model.travel_direction_1 = pyo.Constraint(model.step_range, model.dim_range, rule=travel_only_one_direction_1)
        model.travel_direction_2 = pyo.Constraint(model.step_range, model.dim_range, rule=travel_only_one_direction_2)
        model.travel_direction_3 = pyo.Constraint(model.step_range, rule=travel_only_one_direction_3)

    # fly through the  gates at given order
    model.pass_gates_easy = pyo.Constraint(model.dim_range, model.gate_range, rule=fly_through_gates_easy)

    # model objective to minimze the total step length
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model

def update_model(model, start, goal, gates, obstacles):
    if len(gates) != len(model.gates) and len(obstacles) != len(model.obstacles):
        return False

    #update goal parameter
    model.gates = gates
    model.obstacles = model.obstacles
    model.start = start
    model.goal = goal

    #print('gates: \n', gates)
    #print('obstacles: \n', obstacles)
    #print('start: \n', start)
    #print('goal: \n', goal)

    #update constraints that are necessary to update
    model.del_component('start_rule')
    model.del_component('end_rule')
    model.del_component('avoid_obstacle_1')
    model.del_component('avoid_obstacle_3')
    model.del_component('avoid_obstacle_2')
    model.del_component('avoid_obstacle_4')
    #model.del_component('avoid_obstacle_5')
    model.del_component('pass_gates_easy')

    model.start_rule = pyo.Constraint(model.dim_range, rule=start_at_start)
    model.end_rule = pyo.Constraint(model.dim_range, rule=end_at_goal)
    model.avoid_obstacle_1 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_1)
    model.avoid_obstacle_2 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_2)
    model.avoid_obstacle_3 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_3)
    model.avoid_obstacle_4 = pyo.Constraint(model.obstacle_range, model.step_range, model.dim_range, rule=avoid_objects_4)
    #model.avoid_obstacle_5 = pyo.Constraint(model.obstacle_range, model.step_range, rule=avoid_objects_5)
    model.pass_gates_easy = pyo.Constraint(model.dim_range, model.gate_range, rule=fly_through_gates_easy)

    return model





def run_optimizer(model):
    optimizer = pyo.SolverFactory('gurobi')
    optimizer.options['SolutionLimit'] = 3
    optimizer.options['TimeLimit'] = 15
    optimizer.options['ResultFile'] = 'model.ilp'

    print("Calling Gurobi Optimizer to solve optimization problem...\n")
    #optimizer.solve(model, tee=True, keepfiles=True)
    optimizer.solve(model)

    waypoints = np.zeros((3, model.n_steps))
    for i in range(0, model.n_steps):
        for dim in range(0, model.dim):
            waypoints[dim, i] = pyo.value(model.x[dim, i])
    return waypoints.transpose()


def asynchronous_optimization(model):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    process_handler = executor.submit(run_optimizer, model)
    return process_handler







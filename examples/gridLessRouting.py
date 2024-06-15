
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import sys

if (len(sys.argv) < 2):
    filepath = None
else:
    filepath = sys.argv[1]
########################################################################################################################
# Mixed Integer Model Configuration
########################################################################################################################

ALLOW_THREE_DEGREES_OF_FREEDOM = False

ENFORCE_DISTANCE_FROM_OBSTACLES = False

AVOID_OBSTACLES = True

OPTIMIZE_GATE_SEQUENCE = False

########################################################################################################################
# Drone environment Information
########################################################################################################################
if filepath == None:
    start = [1.0, 1.0, 0.05]
    gate_list = [
        # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
        [0.45, -1.0, 0.525, 0, 0, 2.35, 1],
        [1.0, -1.55, 1.0, 0, 0, -0.78, 0],
        [0.0, 0.5, 0.525, 0, 0, 0, 1],
        [-0.5, -0.5, 1.0, 0, 0, 3.14, 0]
    ]

    obstacles = [
        # x, y, z, r, p, y
        [1.0, -0.5, 0.5, 0, 0, 0],
        [0.5, -1.5, 0.5, 0, 0, 0],
        [-0.5, 0, 0.5, 0, 0, 0],
        [0, 1.0, 0.5, 0, 0, 0]
    ]
else:
    data = np.load(filepath)
    gate_list = data['gate_list']
    obstacles = data['obstacles']
    start = data['start']

# additional fixed info
goal = [0, -1.5, 0.5]
obstacle_height = 1.05
gate_frame = 0.45
half_length = 0.1875
upper_constraints = [3, 3, 2]
lower_constraints = [-3, -3, 0]

########################################################################################################################
# Environment Pre Processing
########################################################################################################################


gates = []
gate_frames = []
 #add gate normal
for i in range(len(gate_list)):
    rot = gate_list[i][5]
    delta_x = np.cos(rot)
    delta_y = np.sin(rot)

    delta_x_ = np.cos(rot + np.pi/2)
    delta_y_ = np.sin(rot + np.pi/2)
    gates.append([gate_list[i][0] -delta_x_ * 0.1, gate_list[i][1] - delta_y_ * 0.1, gate_list[i][2], 0, 0, 0, 0])
    gates.append(gate_list[i])
    gates.append([gate_list[i][0] + delta_x_ * 0.1, gate_list[i][1] + delta_y_ * 0.1, gate_list[i][2], 0, 0, 0, 0])
    gate_frames.append([gate_list[i][0] - delta_x * 0.25, gate_list[i][1] - delta_y * 0.3, gate_list[i][2], 0, 0, 0])
    gate_frames.append([gate_list[i][0] + delta_x * 0.3, gate_list[i][1] + delta_y * 0.3, gate_list[i][2], 0, 0, 0])
    gate_frames.append([gate_list[i][0], gate_list[i][1], gate_list[i][2] + 0.3, 0, 0, 1])
    gate_frames.append([gate_list[i][0], gate_list[i][1], gate_list[i][2] - 0.3, 0, 0, 1])
steps = [30, 40, 50, 80, 90, 100, 130, 140, 150, 180, 190, 200]
for i in range(len(steps)):
    steps[i] = 2*steps[i]

obstacles = list(obstacles) + list(gate_frames)

########################################################################################################################
# MIP Model
########################################################################################################################

model = pyo.ConcreteModel()

model.obstacles = len(obstacles)
model.gates = len(gates)
model.Step = 450
model.M = 1000
model.dim = 3

model.obstacle_range = pyo.RangeSet(0,model.obstacles-1)
model.gate_range = pyo.RangeSet(0, model.gates-1)

model.n_steps = pyo.RangeSet(0, model.Step-1)
model.dim_range = pyo.RangeSet(0, model.dim-1)

model.obstacle_width = [0.1, 0.1, 0.8]
model.obstacle_top = [0.3, 0.3, 0.1]


model.x = pyo.Var(model.dim_range, model.n_steps, within=pyo.Reals, bounds=(lower_constraints[0],upper_constraints[0]))

model.bobject = pyo.Var(model.dim_range, model.obstacle_range, pyo.RangeSet(0,4), model.n_steps, within=pyo.Binary)
#model.bgate = pyo.Var(model.dim_range, model.gate_range, model.n_steps, within=pyo.Binary)

model.length = pyo.Var(model.n_steps, within=pyo.PositiveReals)

if ENFORCE_DISTANCE_FROM_OBSTACLES == True:
    model.obj_distance = pyo.Var(model.obstacle_range, model.dim_range, model.n_steps, within=pyo.PositiveReals)
    model.b_distance = pyo.Var(model.obstacle_range, model.dim_range, model.n_steps, within=pyo.Boolean)

    def obj_distance_rule_1(model, object, dim, step):
        return model.obj_distance[object, dim, step] >= model.x[dim, step] - obstacles[object][dim]
    def obj_distance_rule_2(model, object, dim, step):
        return model.obj_distance[object, dim, step] >= obstacles[object][dim] - model.x[dim, step]
    def obj_distance_rule_3(model, object, dim, step):
        return model.obj_distance[object, dim, step] == (model.x[dim, step] - obstacles[object][dim]) * model.b_distance[object, dim, step] + (obstacles[object][dim] -  model.x[dim, step]) *(1- model.b_distance[object, dim, step])

    def obj_allowed_distance(model, object, dim, step):
        if obstacles[object][-1] == 0:
            return model.obj_distance[object, dim, step] >= model.obstacle_width[dim]
        else:
            return model.obj_distance[object, dim, step] >= model.obstacle_top[dim]

    model.obstacle_distance_1 = pyo.Constraint(model.obstacle_range, model.dim_range, model.n_steps, rule=obj_distance_rule_1)
    model.obstacle_distance_2 = pyo.Constraint(model.obstacle_range, model.dim_range, model.n_steps, rule=obj_distance_rule_2)
    model.obstacle_distance_3 = pyo.Constraint(model.obstacle_range, model.dim_range, model.n_steps, rule=obj_distance_rule_3)
    #model.obstacle_distance_4 = pyo.Constraint(model.obstacle_range, model.dim_range, model.n_steps, rule=obj_allowed_distance)


if ALLOW_THREE_DEGREES_OF_FREEDOM == False:
    model.bx = pyo.Var(model.dim_range, model.n_steps, within=pyo.Binary)
    def travel_only_one_direction_1(model, step, dim):
        if step == model.Step-1:
            return pyo.Constraint.Skip
        return model.x[dim, step] <= model.x[dim, step + 1] + model.bx[dim, step] * model.M
    def travel_only_one_direction_2(model, step, dim):
        if step == model.Step-1:
            return pyo.Constraint.Skip
        return model.x[dim, step] >= model.x[dim, step+1] - model.bx[dim, step] * model.M
    def travel_only_one_direction_3(model, step):
        return model.bx[0, step] + model.bx[1, step] + model.bx[2, step] == 1

    model.travel_direction_1 = pyo.Constraint(model.n_steps, model.dim_range, rule=travel_only_one_direction_1)
    model.travel_direction_2 = pyo.Constraint(model.n_steps, model.dim_range, rule=travel_only_one_direction_2)
    model.travel_direction_3 = pyo.Constraint(model.n_steps, rule=travel_only_one_direction_3)


def avoid_objects_1(model, object, step, dim):
    if obstacles[object][-1] == 0:
        return model.x[dim, step] <= obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 0, step] * model.M
    else:
        return model.x[dim, step] <= obstacles[object][dim] - model.obstacle_top[dim] + model.bobject[dim, object, 0, step] * model.M
def avoid_objects_2(model, object, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    if obstacles[object][-1] == 0:
        return model.x[dim, step+1] <= obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 1, step] * model.M
    else:
        return model.x[dim, step+1] <= obstacles[object][dim] - model.obstacle_top[dim] + model.bobject[dim, object, 1, step] * model.M
def avoid_objects_3(model, object, step, dim):
    if obstacles[object][-1] == 0:
        return model.x[dim, step] >= obstacles[object][dim] + model.obstacle_width[dim] - model.bobject[dim, object, 2, step] * model.M
    else:
        return model.x[dim, step] >= obstacles[object][dim] + model.obstacle_top[dim] - model.bobject[dim, object, 2, step] * model.M
def avoid_objects_4(model, object, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    if obstacles[object][-1] == 0:
        return model.x[dim, step+1] >= obstacles[object][dim] + model.obstacle_width[dim] - model.bobject[dim, object, 3, step] * model.M
    else:
        return model.x[dim, step + 1] >= obstacles[object][dim] + model.obstacle_top[dim] - model.bobject[dim, object, 3, step] * model.M
def avoid_objects_5(model, object, step):
    sum = 0
    for dim in range(0,3):
        for var in range(0,4):
            sum += model.bobject[dim, object, var, step]
    return 10 == sum

if OPTIMIZE_GATE_SEQUENCE == True:
    model.gate_distance = pyo.Var(model.gate_range, model.dim_range, model.n_steps, within= pyo.PositiveReals)
    model.b_distance = pyo.Var(model.gate_range, model.dim_range, model.n_steps, within=pyo.Binary)
    model.b_gate = pyo.Var(model.gate_range, model.n_steps, within=pyo.Binary)
    def gate_distance_rule_1(model, gate, dim, step):
        return model.gate_distance[gate, dim, step] >= model.x[dim, step] - gates[gate][dim]
    def gate_distance_rule_2(model, gate, dim, step):
        return model.gate_distance[gate, dim, step] >= gates[gate][dim] - model.x[dim, step]
    def gate_distance_rule_3(model, gate, dim, step):
        return model.gate_distance[gate, dim, step] == (model.x[dim, step] - gates[gate][dim]) * model.b_distance[gate, dim, step] + (gates[gate][dim] -  model.x[dim, step]) *(1- model.b_distance[gate, dim, step])
    def gate_distance_zero_1(model, gate, step):
        return (model.gate_distance[gate, 0, step] + model.gate_distance[gate, 1, step] + model.gate_distance[gate, 2, step]) * model.b_gate[gate, step] == 0
    def gate_distance_zero_2(model, gate):
        sum = 0
        for step in range(0, model.Step):
            sum += model.b_gate[gate, step]
        return sum == 1

    model.gate_distance_1 = pyo.Constraint(model.gate_range, model.dim_range, model.n_steps, rule=gate_distance_rule_1)
    model.gate_distance_2 = pyo.Constraint(model.gate_range, model.dim_range, model.n_steps, rule=gate_distance_rule_2)
    model.gate_distance_3 = pyo.Constraint(model.gate_range, model.dim_range, model.n_steps, rule=gate_distance_rule_3)
    model.gate_pass_1 = pyo.Constraint(model.gate_range, model.n_steps, rule=gate_distance_zero_1)
    model.gate_pass_2 = pyo.Constraint(model.gate_range, rule=gate_distance_zero_2)
else:
    # finds ways through the gates in fixed order and at fixed step
    def fly_through_gates_easy(model, dim, gate):
        return model.x[dim, steps[gate]] == gates[gate][dim]
    # model.pass_gates_1 = pyo.Constraint(model.gate_range, model.n_steps, model.dim_range, rule=fly_through_gates_1)
    # model.pass_gates_2 = pyo.Constraint(model.gate_range, model.n_steps, model.dim_range, rule=fly_through_gates_2)
    # model.pass_gates_3 = pyo.Constraint(model.gate_range, rule=fly_through_gates_3)
    # model.pass_gates_4 = pyo.Constraint(model.gate_range, model.n_steps,  rule=fly_through_gates_4)
    # model.pass_gates_5 = pyo.Constraint(model.gate_range, model.n_steps,  rule=fly_through_gates_5)
    model.pass_gates_easy = pyo.Constraint(model.dim_range, model.gate_range, rule=fly_through_gates_easy)


def start_at_start(model, dim):
    return model.x[dim, 0] == start[dim]
def end_at_goal(model, dim):
    return model.x[dim, model.Step-1] == goal[dim]


def length_rule_1(model, dim, step):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.length[step] >= model.x[dim, step] - model.x[dim, step+1]
def length_rule_2(model, dim, step):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.length[step] >= model.x[dim, step+1] - model.x[dim, step]

def length_rule_5(model, step):
    if  step == model.Step-1:
        return pyo.Constraint.Skip
    else:
        return model.length[step]**2 >= (model.x[0, step+1] - model.x[0,step])**2 + (model.x[1, step+1] - model.x[1,step])**2 + (model.x[0, step+1] - model.x[0,step])**2

def length_rule_3(model, step):
    return  model.length[step] >= 0.001

def length_rule_4(model, step):
    return  model.length[step] <= 0.1

def objective_rule(model):
    sum = 0
    for step in range(0, model.Step-1):
        sum += model.length[step]
        #for object in range(0, model.obstacles):
            #sum -= 0.00001 * (model.obj_distance[object, 0, step] + model.obj_distance[object, 1, step] + model.obj_distance[object, 2, step])
        #avoid flying low over the ground at the beginning
        if i < 2:
            sum -= 0.00001 * model.x[2,step]
    return sum



model.start = pyo.Constraint(model.dim_range, rule=start_at_start)
model.end = pyo.Constraint(model.dim_range, rule=end_at_goal)

model.avoid_obstacle_1 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_1)
model.avoid_obstacle_2 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_2)
model.avoid_obstacle_3 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_3)
model.avoid_obstacle_4 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_4)
model.avoid_obstacle_5 = pyo.Constraint(model.obstacle_range, model.n_steps, rule=avoid_objects_5)

model.length_1 = pyo.Constraint(model.dim_range, model.n_steps, rule=length_rule_1)
model.length_2 = pyo.Constraint(model.dim_range, model.n_steps, rule=length_rule_2)
model.length_3 = pyo.Constraint(model.n_steps, rule=length_rule_3)
model.length_4= pyo.Constraint(model.n_steps, rule=length_rule_4)
#model.length_5 = pyo.Constraint(model.n_steps, rule=length_rule_5)


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


optimizer = pyo.SolverFactory('gurobi')
optimizer.options['SolutionLimit'] = 5
optimizer.options['TimeLimit'] = 60 * 60

print("Calling Gurobi Optimizer to solve optimization problem...\n")
result = optimizer.solve(model)# tee=True)
print(result)


waypoints = np.zeros((3,model.Step))
for i in range(0, model.Step):
    for dim in range(0, model.dim):
        waypoints[dim, i] = pyo.value(model.x[dim, i])
#waypoints = np.unique(waypoints, axis=1)



np.savetxt("optimal_waypoints.txt", waypoints)

if filepath == None:

    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot a 3D surface
    ax.scatter(waypoints[0, :], waypoints[1, :], waypoints[2, :])

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot
    plt.show()

sys.exit()
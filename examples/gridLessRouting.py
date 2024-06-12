

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# Drone environment Information
########################################################################################################################

goal = [0, -1.5, 0.5]
start = [1.0, 1.0, 0.05]

gate_list = [
    # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
    [0.45, -1.0, 0.525, 0, 0, 2.35, 1],
    [1.0, -1.55, 1.0, 0, 0, -0.78, 0],
    [0.0, 0.5, 0.525, 0, 0, 0, 1],
    [-0.5, -0.5, 1.0, 0, 0, 3.14, 0]
]

gates = []
 #add gate normal
for i in range(len(gate_list)):
    rot = gate_list[i][5]
    delta_x = np.cos(rot)
    delta_y = np.sin(rot)
    gates.append([gate_list[i][0] + delta_x * 0.01, gate_list[i][1] + delta_y * 0.01, gate_list[i][2], 0, 0, 0, 0])
    gates.append(gate_list[i])
    gates.append([gate_list[i][0] - delta_x * 0.01, gate_list[i][1] - delta_y * 0.01, gate_list[i][2], 0, 0, 0, 0])
steps = [30, 32, 34, 60, 62, 64, 90, 92, 94, 120, 122, 124]


gate_frame = 0.45
half_length = 0.1875

obstacles = [
    # x, y, z, r, p, y
    [1.0, -0.5, 0.5, 0, 0, 0],
    [0.5, -1.5, 0.5, 0, 0, 0],
    [-0.5, 0, 0.5, 0, 0, 0],
    [0, 1.0, 0.5, 0, 0, 0]
]
obstacle_height = 1.05

upper_constraints = [3, 3, 2]
lower_constraints = [-3, -3, 0]

########################################################################################################################
# MIP Model
########################################################################################################################

model = pyo.ConcreteModel()

model.obstacles = len(obstacles)
model.gates = len(gates)
model.Step = 300
model.M = 1000
model.dim = 3

model.obstacle_range = pyo.RangeSet(0,model.obstacles-1)
model.gate_range = pyo.RangeSet(0, model.gates-1)

model.n_steps = pyo.RangeSet(0, model.Step-1)
model.dim_range = pyo.RangeSet(0, model.dim-1)

model.obstacle_width = [0.02, 0.02, 0.8]


model.x = pyo.Var(model.dim_range, model.n_steps, within=pyo.Reals, bounds=(lower_constraints[0],upper_constraints[0]))
model.bx = pyo.Var(model.dim_range, model.n_steps, within=pyo.Binary)

model.bobject = pyo.Var(model.dim_range, model.obstacle_range, pyo.RangeSet(0,2), within=pyo.Binary)
model.bgate = pyo.Var(model.dim_range, model.gate_range, model.n_steps, within=pyo.Binary)

model.length = pyo.Var(model.n_steps, within=pyo.PositiveReals)




def travel_only_one_direction_1(model, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.x[dim, step] <= model.x[dim, step + 1] + model.bx[dim, step] * model.M
def travel_only_one_direction_2(model, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.x[dim, step+1] >= model.x[dim, step] - model.bx[dim, step] * model.M
def travel_only_one_direction_3(model, step):
    return model.bx[0, step] + model.bx[1, step] + model.bx[2, step] == 1

def avoid_objects_1(model, object, step, dim):
    return model.x[dim, step] <= obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 0] * model.M
def avoid_objects_2(model, object, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.x[dim, step+1] <= obstacles[object][dim] - model.obstacle_width[dim] + model.bobject[dim, object, 0] * model.M
def avoid_objects_3(model, object, step, dim):
    return model.x[dim, step] >= obstacles[object][dim] + model.obstacle_width[dim] - model.bobject[dim, object, 1] * model.M
def avoid_objects_4(model, object, step, dim):
    if step == model.Step-1:
        return pyo.Constraint.Skip
    return model.x[dim, step+1] >= obstacles[object][dim] + model.obstacle_width[dim] - model.bobject[dim, object, 1] * model.M
def avoid_objects_5(model, object):
    return 5 == model.bobject[0, object, 0] + model.bobject[0, object, 1] + model.bobject[1, object, 0] + model.bobject[1, object, 1] +  model.bobject[2, object, 0] + model.bobject[2, object, 1]


def fly_through_gates_1(model, gate, step, dim):
    return model.x[dim, step] <= gates[gate][dim]  + (1 - model.bgate[dim, gate, step]) * model.M
def fly_through_gates_2(model, gate, step, dim):
    return model.x[dim, step] >= gates[gate][dim]  - (1 - model.bgate[dim, gate, step]) * model.M
def fly_through_gates_3(model, gate):
    sum = 0
    for i in range(0, model.Step):
        sum += model.bgate[0, gate, i] + model.bgate[1, gate, i] + model.bgate[2, gate, 1]
    return sum >= 3*model.gates
def fly_through_gates_4(model, gate, step):
    return model.bgate[0, gate, step] == model.bgate[1, gate, step]
def fly_through_gates_5(model, gate, step):
    return model.bgate[2, gate, step] == model.bgate[1, gate, step]


def fly_through_gates_easy(model, dim, gate):
    return model.x[dim, steps[gate]] == gates[gate][dim]


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

def length_rule_3(model, step):
    return  model.length[step] >= 0.002

def length_rule_4(model, step):
    return  model.length[step] <= 0.1

def objective_rule(model):
    sum = 0
    for i in range(0, model.Step):
        sum += model.length[i]
    return sum



model.start = pyo.Constraint(model.dim_range, rule=start_at_start)
model.end = pyo.Constraint(model.dim_range, rule=end_at_goal)

model.travel_direction_1 = pyo.Constraint(model.n_steps, model.dim_range, rule=travel_only_one_direction_1)
model.travel_direction_2 = pyo.Constraint(model.n_steps, model.dim_range, rule=travel_only_one_direction_2)
model.travel_direction_3 = pyo.Constraint(model.n_steps, rule=travel_only_one_direction_3)

model.avoid_obstacle_1 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_1)
model.avoid_obstacle_2 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_2)
model.avoid_obstacle_3 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_3)
model.avoid_obstacle_4 = pyo.Constraint(model.obstacle_range, model.n_steps, model.dim_range, rule=avoid_objects_4)

#model.pass_gates_1 = pyo.Constraint(model.gate_range, model.n_steps, model.dim_range, rule=fly_through_gates_1)
#model.pass_gates_2 = pyo.Constraint(model.gate_range, model.n_steps, model.dim_range, rule=fly_through_gates_2)
#model.pass_gates_3 = pyo.Constraint(model.gate_range, rule=fly_through_gates_3)
#model.pass_gates_4 = pyo.Constraint(model.gate_range, model.n_steps,  rule=fly_through_gates_4)
#model.pass_gates_5 = pyo.Constraint(model.gate_range, model.n_steps,  rule=fly_through_gates_5)
model.pass_gates_easy = pyo.Constraint(model.dim_range, model.gate_range, rule=fly_through_gates_easy)


model.length_1 = pyo.Constraint(model.dim_range, model.n_steps, rule=length_rule_1)
model.length_2 = pyo.Constraint(model.dim_range, model.n_steps, rule=length_rule_2)
model.length_3 = pyo.Constraint(model.n_steps, rule=length_rule_3)
model.length_4= pyo.Constraint(model.n_steps, rule=length_rule_4)


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


optimizer = pyo.SolverFactory('gurobi')
#optimizer.options['PoolSolutions'] = 20
optimizer.options['TimeLimit'] = 45

print("Calling Gurobi Optimizer to solve optimization problem...\n")
result = optimizer.solve(model, tee=True)
print(result)


waypoints = np.zeros((3,model.Step))
for i in range(0, model.Step):
    for dim in range(0, model.dim):
        waypoints[dim, i] = pyo.value(model.x[dim, i])

np.savetxt("optimal_waypoints.txt", waypoints)

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
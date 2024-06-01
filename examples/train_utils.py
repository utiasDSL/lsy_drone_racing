import csv

def save_observations(obs_list, save_path, current_datetime):
    """
    Save the list of observations to a CSV file.

    Args:
        obs_list (list): List of observations.
        save_path (str): Path to save the CSV file.
        current_datetime (str): Current date and time.

    Returns:
        None
    """
    csv_file = save_path / f"observations_{current_datetime}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['drone_pos', 'drone_rpy', 'drone_vxyz', 'drone_vrpy', 'gates_xyz_yaw', 'gates_in_range', 'obstacles_xyz', 'obstacles_in_range', 'gate_id'])
        for obs in obs_list:
            writer.writerow([
                    obs['drone_pos'],
                    obs['drone_rpy'],
                    obs['drone_vxyz'],
                    obs['drone_vrpy'],
                    obs['gates_xyz_yaw'],
                    obs['gates_in_range'],
                    obs['obstacles_xyz'],
                    obs['obstacles_in_range'],
                    obs['gate_id']
                ])
            
def process_observation(x, print_flag=False):
    """
    Process the observation data and extract relevant information.

    Args:
        x (ndarray): The input observation data.
        print_flag (bool, optional): Flag to indicate whether to print the extracted information. 
                                     Defaults to False.

    Returns:
        dict: A dictionary containing the extracted information from the observation data.
            - 'drone_pos' (ndarray): The position of the drone [x, y, z].
            - 'drone_rpy' (ndarray): The roll, pitch, and yaw angles of the drone.
            - 'drone_vxyz' (ndarray): The velocity of the drone in the x, y, and z directions.
            - 'drone_vrpy' (ndarray): The angular velocity of the drone in roll, pitch, and yaw.
            - 'gates_xyz_yaw' (dict): The positions and yaw angles of the gates.
                - 'gate1' (ndarray): The position and yaw angle of gate 1.
                - 'gate2' (ndarray): The position and yaw angle of gate 2.
                - 'gate3' (ndarray): The position and yaw angle of gate 3.
                - 'gate4' (ndarray): The position and yaw angle of gate 4.
            - 'gates_in_range' (ndarray): The gates that are in range.
            - 'obstacles_xyz' (dict): The positions of the obstacles.
                - 'obstacle1' (ndarray): The position of obstacle 1.
                - 'obstacle2' (ndarray): The position of obstacle 2.
                - 'obstacle3' (ndarray): The position of obstacle 3.
                - 'obstacle4' (ndarray): The position of obstacle 4.
            - 'obstacles_in_range' (ndarray): The obstacles that are in range.
            - 'gate_id' (float): The ID of the gate.

    """
    observations = {
        'drone_pos': x[0][:3],
        'drone_rpy': x[0][3:6],
        'drone_vxyz': x[0][6:9],
        'drone_vrpy': x[0][9:12],
        'gates_xyz_yaw': {
            'gate1': x[0][12:16],
            'gate2': x[0][16:20],
            'gate3': x[0][20:24],
            'gate4': x[0][24:28]},
        'gates_in_range': x[0][28:32],
        'obstacles_xyz': {
            'obstacle1': x[0][32:36],
            'obstacle2': x[0][36:40],
            'obstacle3': x[0][40:44],
            'obstacle4': x[0][44:48]
        },
        'obstacles_in_range': x[0][44:48],
        'gate_id': x[0][48]
    }

    if print_flag == True:
        print("Drone Position: ", observations['drone_pos'])
        print("Drone Roll, Pitch, Yaw: ", observations['drone_rpy'])
        print("Drone Velocity (X, Y, Z): ", observations['drone_vxyz'])
        print("Drone Angular Velocity (Roll, Pitch, Yaw): ", observations['drone_vrpy'])
        print("Gates (X, Y, Z, Yaw): ", observations['gates_xyz_yaw'])
        print("Gates in Range: ", observations['gates_in_range'])
        print("Obstacles (X, Y, Z): ", observations['obstacles_xyz'])
        print("Obstacles in Range: ", observations['obstacles_in_range'])
        print("Gate ID: ", observations['gate_id'])

    return observations

import numpy as np
import pandas as pd


def function_filter(function_list):
    def is_function(record):
        return record["function"] in function_list

    return is_function


def add_obs_to_df(df: pd.DataFrame, obs, time_step=0):
    num_agents = len(obs["agent_loc"])
    num_targets = len(obs["target_loc"])
    agent_rows = [
        {
            "id": i,
            "type": "agent",
            "time": time_step,
            "x": location[0],
            "y": location[1],
            "z": location[1],
        }
        for i, location in zip(range(1, num_agents + 1), obs["agent_loc"])
    ]
    # currently only one environmental object really -- not sure the scene really counts yet
    env_rows = [
        {
            "id": 1,
            "type": "env",
            "time": time_step,
            "x": location[0],
            "y": location[1],
            "z": location[2],
        }
        for t, location in zip(range(1, num_targets + 1), obs["target_loc"])
    ]
    df = pd.concat([df, pd.DataFrame(agent_rows + env_rows)]).reset_index(drop=True)
    return df


def find_angle(first, second):
    print("first ", first)
    print("second ", second)
    print("dot ", np.dot(first, second))
    print("norm first ", np.linalg.norm(first))
    print("norm second ", np.linalg.norm(second))

    if np.linalg.norm(first) != 0.0 and np.linalg.norm(second) != 0.0:
        return np.arccos(
            np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        )
    else:
        return 0.0


def calc_angles(first, second):
    theta_x = 0
    theta_y = 0
    first_zero_x = np.array([0.0, first[1], first[2]])
    second_zero_x = np.array([0.0, second[1], second[2]])
    # print("angle = ", angle)
    theta_x = find_angle(first_zero_x, second_zero_x)

    first_zero_y = np.array([first[0], 0.0, first[2]])
    second_zero_y = np.array([second[0], 0.0, second[2]])
    theta_y = find_angle(first_zero_y, second_zero_y)

    first_zero_z = np.array([first[0], first[1], 0.0])
    second_zero_z = np.array([second[0], second[1], 0.0])
    theta_z = find_angle(first_zero_z, second_zero_z)
    print(theta_z)
    return theta_x, theta_y, theta_z

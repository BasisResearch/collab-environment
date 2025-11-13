import bisect
import struct

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger


def function_filter(function_list):
    def is_function(record):
        return record["function"] in function_list

    return is_function


def add_obs_to_df(
    df: pd.DataFrame, obs, time_step=0, variant_index_list=None, variant_type_list=None
):
    num_agents = len(obs["agent_loc"])
    num_targets = len(obs["target_loc"])
    logger.debug(f"time step = {time_step}")
    # add agent rows
    agent_rows = []
    for i, location, velocity in zip(
        range(1, num_agents + 1), obs["agent_loc"], obs["agent_vel"]
    ):
        row_dict = dict()
        row_dict["id"] = i
        # (f'id = {i}')
        variant_index = bisect.bisect_left(variant_index_list, i) - 1
        # print(f'variant index = {variant_index}')
        row_dict["type"] = "agent"  # type:ignore
        # print(f'species = {variant_type_list[variant_index]}')
        row_dict["species"] = variant_type_list[variant_index]
        row_dict["time"] = time_step
        row_dict["x"] = location[0]
        row_dict["y"] = location[1]
        row_dict["z"] = location[2]
        row_dict["v_x"] = velocity[0]
        row_dict["v_y"] = velocity[1]
        row_dict["v_z"] = velocity[2]

        for t, distance in zip(
            range(1, num_targets + 1), obs["distances_to_target_centers"][i - 1]
        ):
            row_dict[f"distance_target_center_{t}"] = distance

        for t, distance in zip(
            range(1, num_targets + 1),
            obs["distances_to_target_mesh_closest_points"][i - 1],
        ):
            row_dict[f"distance_to_target_mesh_closest_point_{t}"] = distance

        for t, closest_point in zip(
            range(1, num_targets + 1), obs["target_mesh_closest_points"][i - 1]
        ):
            row_dict[f"target_mesh_closest_point_{t}"] = closest_point

        row_dict["mesh_scene_distance"] = obs["mesh_scene_distance"][i - 1]

        row_dict["mesh_scene_closest_point"] = obs["mesh_scene_closest_points"][i - 1]

        agent_rows.append(row_dict)

    # agent_rows = [
    #     {
    #         "id": i,
    #         "type": "agent",
    #         "time": time_step,
    #         "x": location[0],
    #         "y": location[1],
    #         "z": location[2],
    #     }
    #     for i, location in zip(range(1, num_agents + 1), obs["agent_loc"])
    # ]

    # add environment rows

    # currently only one environmental object really -- not sure the scene really counts yet
    #  -- 080525
    # added row for each target -- need a sim test for this
    #
    env_rows = [
        {
            "id": t,  # should be the number of the target (fixed 081825 10:52PM)
            "type": "env",
            "time": time_step,
            "x": location[0],
            "y": location[1],
            "z": location[2],
        }
        for t, location in zip(range(1, num_targets + 1), obs["target_loc"])
    ]

    """
     -- 081825 10:50PM
    Fix that annoying deprecated warning about concatenating an empty DataFrame
    """
    if df is None:
        df = pd.DataFrame(agent_rows + env_rows)
    else:
        df = pd.concat([df, pd.DataFrame(agent_rows + env_rows)]).reset_index(drop=True)
    return df


def test_add_obs():
    obs = dict()
    obs["agent_loc"] = [[1, 2, 3]]
    obs["agent_vel"] = [[10, 20, 30]]
    obs["target_loc"] = [[1000, 2000, 3000]]
    obs["distances_to_target_centers"] = [[100.0]]
    obs["distances_to_target_mesh_closest_points"] = [[50.0]]
    obs["target_mesh_closest_points"] = [[100.0, 200.0, 300.0]]
    obs["mesh_scene_distance"] = [[25.0]]
    obs["mesh_scene_closest_points"] = [[1000.0, 2000.0, 3000.0]]
    df = add_obs_to_df(None, obs, 1)
    print("df\n", df[["id", "type", "mesh_scene_closest_point"]])

    agent1 = df.loc[(df["type"] == "agent") & (df["id"] == 1)]
    print("agent1 = \n", agent1)


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


def get_submesh_indices_from_ply(file_path):
    with open(file_path, "rb") as file:
        # Read the header
        header = []
        while True:
            line = file.readline().decode("utf-8").strip()
            header.append(line)
            if line == "end_header":
                break

        # Parse the header to find the number of vertices and properties
        vertex_count = 0
        # properties = []
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            # elif line.startswith("property"):
            #    properties.append(line.split()[2])  # Store property names

        # Read the vertex data
        # vertices = []
        keep_vertices = []
        # red_list = []
        for i in range(vertex_count):
            # custom_property_row = []
            # Read the vertex data according to the property types
            _ = file.read(3 * 8)  # Assuming first three properties are float (x, y, z)
            # _ = struct.unpack("<ddd", data)  # Little-endian float unpacking
            # vertices.append(vertex)

            # skip the normals
            _ = file.read(
                3 * 8
            )  # Assuming next three properties are float (n_x, n_y, n_z)

            # read the red
            data = file.read(1)  # Assuming indicator property is a byte
            red = struct.unpack("<B", data)[0]  # Little-endian float unpacking
            # red_list.append(red)
            if red > 0:
                keep_vertices.append(i)

            # skip the next two bytes for green and blue
            file.read(2)

    return keep_vertices


def plot_trajectories(df, env, frame_limit=None):
    # get the
    num_time_steps = df["time"].max()
    if frame_limit is not None:
        num_time_steps = min(frame_limit, num_time_steps)

    num_agents = df.loc[df["type"] == "agent"]["id"].max()
    agent_trajectories = []  # = np.zeros((num_time_steps+1, num_agents, 3))
    for i in tqdm(range(num_agents)):
        trajectory = []
        """
         -- 080825 11:58 AM
        Fix this. I stop at -1 because I am apparently missing a time step.

         -- 081125 2:30PM
        I think this got fixed.  
        """
        for t in range(num_time_steps):
            temp = df.loc[
                (df["type"] == "agent") & (df["time"] == t) & (df["id"] == (i + 1))
            ]
            trajectory.append(temp[["x", "y", "z"]].to_numpy()[0])
        agent_trajectories.append(trajectory)

    agent_trajectories = np.array(agent_trajectories)

    #
    #  -- 080825 9:56AM
    # Call reset with options set to plot the trajectories. This is a horrible design
    # that needs to be changed. I am doing this now to get a figure for the paper.
    # Later we will separate the rendering from the Gymnasium environment and be able
    # to call the renderer directly. Gymnasium has some restrictions because the
    # environment is in a wrapper -- need to understand wrappers better.
    #

    _, _ = env.reset(options=agent_trajectories)
    count = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # truncated indicates the user hit quit in the open3d visualizer
        _, _, terminated, truncated, _ = env.step(np.zeros((num_agents, 3)))
        count += 1
        logger.debug(f"count={count}")

    """
     -- 081825 9:01PM
    Not sure this should be closed because we may need it 
    for additional episodes. 
    """
    # env.close()


# def interpolate_color(start_color, end_color, steps):
#     """Interpolate between two RGB colors."""
#     # Number of steps for each transition
#     steps = 50
#
#     # Define colors
#     blue = (0, 0, 1.0)
#     cyan = (0, 1.0, 1.0)
#     green = (0, 1.0, 0)
#     yellow = (1.0, 1.0, 0)
#     red = (1.0, 0, 0)
#     # Generate the color list
#     color_list = (
#             interpolate_color(blue, cyan, steps) +
#             interpolate_color(cyan, green, steps) +
#             interpolate_color(green, yellow, steps) +
#             interpolate_color(yellow, red, steps)
#     )
#
#     return [
#         (
#             int(start_color[0] + (end_color[0] - start_color[0]) * (i / steps)),
#             int(start_color[1] + (end_color[1] - start_color[1]) * (i / steps)),
#             int(start_color[2] + (end_color[2] - start_color[2]) * (i / steps))
#         )
#         for i in range(steps + 1)
#     ]
#
#


if __name__ == "__main__":
    test_add_obs()

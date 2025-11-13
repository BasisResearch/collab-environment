import argparse

import pyarrow.parquet as pq
import numpy as np

from collab_env.data.file_utils import expand_path, get_project_root


def read_sim3D_episode_files(folder_path, agents_only=False):
    result_dataframe_list = []
    episode_file_list = list(folder_path.glob("episode*.parquet"))
    for episode_file in episode_file_list:
        trajectory_path = expand_path(episode_file, folder_path)
        df = pq.read_pandas(trajectory_path).to_pandas()
        # print(df.columns)
        if agents_only:
            df = df[df["type"] == "agent"]
            print("agent columns:", df.columns)
        result_dataframe_list.append(df)
    return result_dataframe_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="convert_3d_sim_to_gnn",
        description="Converts the 3D simulator output trajectories into the input file for the GNN, trains the GNN on those trajectories, performs a rollout, then converts the rollout trajectories into an input file for the 3D simulator to display the trajectories in the 3D environment",
        epilog="---",
    )
    parser.add_argument("-rf", "--run_folder")

    args = parser.parse_args()

    folder_name = "sim-output/" + args.run_folder
    folder_path = expand_path(
        #    f"{config['files']['trajectory_folder']}/{config['files']['trajectory_file']}",
        #    get_project_root(),
        folder_name,
        get_project_root(),
    )

    df_list = read_sim3D_episode_files(folder_path, agents_only=True)
    df = df_list[0]
    df["mean_distance_to_target_mesh"] = df.groupby("id")[
        "distance_to_target_mesh_closest_point_1"
    ].transform("mean")

    df["mean_distance_to_mesh_scene"] = df.groupby("id")[
        "mesh_scene_distance"
    ].transform("mean")
    print(df[df["id"] == 1])
    means_df = df.groupby("id", sort=True)["mesh_scene_distance"].mean().reset_index()
    print(means_df)

    grp = df.groupby("time")
    arrays = [g[["x", "y", "z"]].to_numpy() for _, g in grp]
    arr3d = np.stack(arrays, axis=0)  # ValueError if shapes differ
    times = list(grp.groups.keys())

    print("should be list of 3D vectors")
    matrix = np.array(
        [g[["x", "y", "z"]].to_numpy() for _, g in df.groupby("time", sort=False)]
    )
    print(matrix.shape)

    # arr3d: (T, N, 3)
    short_matrix = matrix[0:1, 0:3, :]
    diff = short_matrix[:, :, None, :] - matrix[:, None, :, :]  # (T, N, N, 3)
    dists = np.linalg.norm(diff, axis=-1)  # (T, N, N)
    print(short_matrix)
    print(dists)
    print(dists.shape)

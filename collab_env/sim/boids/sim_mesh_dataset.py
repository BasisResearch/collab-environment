import pandas as pd
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

from collab_env.data.file_utils import expand_path, get_project_root


class SimMeshDataset(Dataset):
    def __init__(
        self,
        folder_path,
        width,
        height,
        depth,
        full=False,
        agents_only=False,
        single_vector_for_all=False,
    ):
        self.sequences = []

        episode_file_list = list(folder_path.glob("episode*.parquet"))
        for episode_file in episode_file_list:
            trajectory_path = expand_path(episode_file, folder_path)
            """
            -- 101325 4:03PM
            TODO: Change this to read only the columns we need. 
            """
            df = pq.read_pandas(trajectory_path).to_pandas()
            # print(df.columns)
            if agents_only:
                df = df[df["type"] == "agent"]
                print("agent columns:", df.columns)
                print(
                    "mesh dist:",
                    df[
                        [
                            "time",
                            "id",
                            "mesh_scene_distance",
                            "mesh_scene_closest_point",
                        ]
                    ],
                )
            if full:
                """
                TOC -- 103025 
                Only dealing with the first target mesh right now. Need to fix this when there are more meshes -- though
                we may want to change the format of the dataframe for multiple targets, not sure. 
                """
                df[
                    [
                        "target_mesh_closest_point_x",
                        "target_mesh_closest_point_y",
                        "target_mesh_closest_point_z",
                    ]
                ] = pd.DataFrame(
                    df["target_mesh_closest_point_1"].to_list(), index=df.index
                )
                df[
                    [
                        "mesh_scene_closest_point_x",
                        "mesh_scene_closest_point_y",
                        "mesh_scene_closest_point_z",
                    ]
                ] = pd.DataFrame(
                    df["mesh_scene_closest_point"].to_list(), index=df.index
                )
                # print('mesh dist:', df[['time', 'id', 'mesh_scene_distance', 'mesh_scene_closest_point_x', 'mesh_scene_closest_point_y','mesh_scene_closest_point_z']])
                groups = df.groupby("time")[
                    [
                        "x",
                        "y",
                        "z",
                        "v_x",
                        "v_y",
                        "v_z",
                        "distance_to_target_mesh_closest_point_1",
                        "target_mesh_closest_point_x",
                        "target_mesh_closest_point_y",
                        "target_mesh_closest_point_z",
                        "mesh_scene_distance",
                        "mesh_scene_closest_point_x",
                        "mesh_scene_closest_point_y",
                        "mesh_scene_closest_point_z",
                    ]
                ]
            else:
                groups = df.groupby("time")[["x", "y", "z"]]
            # print(groups)
            position_groups = groups.apply(
                lambda g: torch.tensor(g.to_numpy(), dtype=torch.float32)
            )
            # print(type(position_groups))
            positions = torch.stack(position_groups.to_list()) / torch.tensor(
                [width, height, depth], dtype=torch.float32
            )

            if single_vector_for_all:
                # reshape to make all agents info into one list.
                positions = positions.reshape(positions.size(0), -1)

            # print(positions)
            # print(positions.shape)
            self.dataframe = df
            df["type"] = df["type"].astype("category")
            codes = df[df["time"] == 1]["type"].cat.codes
            # codes = pd.Categorical(cat).codes
            # print('codes values ', codes)
            # df_codes = df.assign(type_code=codes)
            #    groups = df_codes.groupby("time")['type_code']
            #    species_groups = groups.apply(lambda g: torch.tensor(g.to_numpy(), dtype=torch.long))
            species = torch.tensor(codes.values, dtype=torch.long)
            # species = torch.stack(species_groups.to_list())
            # print('species.shape: ', species.shape)
            self.sequences.append((positions, species))

    def __getitem__(self, item):
        return self.sequences[item]

    def __len__(self):
        return len(self.sequences)

    def get_dataframe(self):
        return self.dataframe


if __name__ == "__main__":
    print("testing SimMeshDataset")
    folder_path = expand_path(
        "sim-output/boids_sim_run-started-20251020-163830", get_project_root()
    )
    data = SimMeshDataset(
        folder_path,
        1500,
        1500,
        1500,
        full=False,
        agents_only=False,
        single_vector_for_all=False,
    )

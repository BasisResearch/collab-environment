import torch

import pyarrow.parquet as pq

from collab_env.gnn.build_dataset import compute_data_list_from_dataframe
from collab_env.sim.boids.sim_utils import add_obs_to_df


class GNN_Agents:
    def __init__(self, config, env):
        self.config = config

        # need these for building the data to pass to the model
        self.num_agents = config["simulator"]["num_agents"]
        self.num_time_steps = (
            config["simulator"]["num_frames"] + 1
        )  # add 1 for time step 0
        self.box_size = config["environment"]["box_size"]
        """
        TOC -- 010426 1:18AM
        
        The model must be saved in a file and the config must have the name of the file listed
        so we can load the model into the class. 
        
        The starting positions also must be specified in a file, this could be the validation
        prediction file, so we will read the positions at time step 0 and ignore the rest since
        the GNN will predict the remaining stime steps. 
        
        We should probably keep the entire dataframe for the validation predictions so we can 
        compare to the GNN predictions to get some sort of loss per time step from the rollout. 
        """
        model_file_name = config["learning_algorithm"]["model_file"]
        self.model = torch.load(model_file_name)
        self.model.eval()

        self.position_df = pq.read_pandas(
            config["learning_algorithm"]["position_file"]
        ).to_pandas()
        """
        TOC -- 010426 2:47AM
        Make sure this gets only the position for the agents at time 0. It gets them all right now. 
        """
        self.init_position = self.position_df.loc[
            (self.position_df["time"] == 0) & (self.position_df["type"] == "agent"),
            ["x", "y", "z"],
        ].to_numpy()

    def get_action_list(self, obs):
        df = add_obs_to_df(df=None, obs=obs)

        data_list = compute_data_list_from_dataframe(
            df.loc[df["type"] == "agent"],
            num_time_steps=2,  # needs to be 2 so that we get the first position added
            num_agents=self.num_agents,
            box_size=self.box_size,
            label_offset=0,  # needs to be 0 so that we don't go beyond the end, label is ignored in rollout
        )

        with torch.no_grad():
            prediction, _ = self.model(data_list[0])

        return prediction.detach().cpu().numpy()

    def update(self, time_step: int):
        # nothing to do
        pass

    def get_reset_options(self):
        # this should return dictionary that includes run_trajectories (value doesn't matter) and the initial position
        # for the agents.
        return {"run_trajectories": True, "agent_trajectories": self.init_position}

    def get_variant_types(self):
        return None, None

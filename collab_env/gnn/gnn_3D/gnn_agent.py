import torch

import pyarrow.parquet as pq

from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.gnn.gnn_3D.build_dataset import compute_data_list_from_dataframe
from collab_env.sim.boids.sim_utils import add_obs_to_df

"""
This is the GNN agent for computing rollouts with the simulator.
"""
class GNN_Agents:
    def __init__(self, simulator_config: dict, agent_config:dict, env):
        """
        :param simulator_config: the configuration dictionary for the simulator
        :param agent_config: the configuration dictionary specific to the agent.
        The agent config must contain the model file, the position file, the time step to of the initial position in the
        position file and the time step at which to end the rollout. The model file contains a full PyTorch model (this
        needs to be changed later for security and portability). The position file must contain the initial positions of
        the agents at the time step specified in the agent config. The episode file that was used to train the model
        could be used for the position file, but there could be situations where you would like to see where the model
        predicts agents to go from different initial positions.
        :param env: the environment, this is ignored for now and should probably be removed but boids_agents would also
        have to change if we remove it.


        """
        self.simulator_config = simulator_config

        # need these for building the data to pass to the model
        self.num_agents = simulator_config["simulator"]["num_agents"]
        self.num_time_steps = (
                simulator_config["simulator"]["num_frames"] + 1
        )  # add 1 for time step 0
        self.box_size = simulator_config["environment"]["box_size"]
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
        model_file_name = agent_config["model_file"]
        self.model = torch.load(expand_path(model_file_name, get_project_root()))
        self.model.eval()

        """
        TOC -- 010626 12:32PM
        There is probably an optimization to be used here with parquet to read only the rows and columns we need.
        """
        self.position_df = pq.read_pandas(
            agent_config["position_file"]
        ).to_pandas()
        start_time = agent_config["start_time"]

        self.init_position = self.position_df.loc[
            (self.position_df["time"] == start_time) & (self.position_df["type"] == "agent"),
            ["x", "y", "z"],
        ].to_numpy()

    def get_action_list(self, obs):
        df = add_obs_to_df(df=None, obs=obs)

        data_list = compute_data_list_from_dataframe(
            df.loc[df["type"] == "agent"],
            num_time_steps=1,
            num_agents=self.num_agents,
            box_size=self.box_size,
            label_offset=0,  # needs to be 0 so that we don't go beyond the end, label is ignored in rollout
        )

        with torch.no_grad():
            # we only created one graph, the one for the current observation.
            prediction, _ = self.model(data_list[0])

        return prediction.detach().cpu().numpy()


    def reset(self):
        # nothing to do
        pass

    def update(self, time_step: int):
        # nothing to do
        pass

    def get_reset_options(self):
        # this should return dictionary that includes run_trajectories (value doesn't matter) and the initial position
        # for the agents.
        return {"run_trajectories": True, "agent_trajectories": self.init_position}

    def get_variant_types(self):
        return None, None

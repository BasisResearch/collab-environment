import json
from typing import Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch

import pyarrow.parquet as pq

from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.gnn.gnn_3D.build_dataset import compute_data_list_from_dataframe
from collab_env.gnn.gnn_3D.simulator_agents_abstract import SimulatorAgents
from collab_env.sim.boids.sim_utils import add_obs_to_df

"""
This is the GNN agent for computing rollouts with the simulator.
"""


class GNN_Agents(SimulatorAgents):
    def __init__(
        self,
        simulator_config: dict,
        agent_config: dict,
        env: Any,  # this is ignored
        predictions_are_velocities: bool = False,
    ):
        """
        Args:
            simulator_config(dict): the configuration dictionary for the simulator
            agent_config(dict): the configuration dictionary specific to the agent. The agent config must contain
                the model file, the position file, the time step to of the initial position in the position file and the
                time step at which to end the rollout. The model file contains a full PyTorch model (this needs to be
                changed later for security and portability). The position file must contain the initial positions of the
                agents at the time step specified in the agent config. The episode file that was used to train the model
                could be used for the position file, but there could be situations where you would like to see where the
                model predicts agents to go from different initial positions.
            env(Any): the environment, this is ignored for now and should probably be removed but boids_agents would
                also have to change if we remove it from the abstract class.
        """

        self.simulator_config = simulator_config
        self.predictions_are_velocities = predictions_are_velocities

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
        TOC -- 010626 10:02PM
        TODO: Get the node features from the dataset metadata  
        """
        dataset_metadata_file = agent_config["dataset_metadata_file"]
        dataset_metadata_path = expand_path(dataset_metadata_file, get_project_root())
        with dataset_metadata_path.open("r", encoding="utf-8") as f:
            dataset_metadata = json.load(f)

        self.node_feature_columns = dataset_metadata["node_feature_columns"]
        self.time_window_length = dataset_metadata["time_window_length"]

        """
        TOC -- 010626 12:32PM
        There is probably an optimization to be used here with parquet to read only the rows and columns we need.
        """
        position_file_path = expand_path(
            agent_config["position_file"], get_project_root()
        )
        self.position_df = pq.read_pandas(position_file_path).to_pandas()

        start_time = agent_config["start_time"]
        self.init_position: np.ndarray[Any, Any] = self.get_position_from_df(start_time)
        self.last_position: np.ndarray[Any, Any] = self.init_position
        # lint didn't like when I set fixed_action to None, so I was forced to make it something
        # (and yeah, I know about Optional and it didn't work -- weird chain of assignments that forced
        # bad decisions all over the place)
        self.fixed_action: np.ndarray[Any, Any] = np.zeros(
            (1, 1)
        )  # Optional[np.ndarray[Any, Any]] = None
        self.time_step = -1  # use -1 to indicate that we haven't gotten a time step from a call to update() yet

        self.observation_dataframe: Optional[pd.DataFrame] = None

        self.predicting = False

    def get_position_from_df(self, time_step: int) -> np.ndarray:
        """
        Args:
            time_step (int): the time step in the dataframe for which we should return the position
        Returns:
            np.ndarray[Any, Any]: the positions of the agents at the specified time step.
        """
        return self.position_df.loc[
            (self.position_df["time"] == time_step)
            & (self.position_df["type"] == "agent"),
            ["x", "y", "z"],
        ].to_numpy()

    def get_action_list(self, obs: dict) -> np.ndarray:
        """
        Args:
            obs (dict): the observation from the simulator for a single time step.
        Returns:
            the actions for all the agents.
        """
        # print("get_action_list() called with obs  = \n", obs)
        # if self.observation_dataframe is not None:
        #     print("get_action_list() called with obs dataframe length = \n", len(self.observation_dataframe))
        #     print("get_action_list() called with obs dataframe max time = \n", self.observation_dataframe["time"].max())

        # add the current observation to the observation dataframe.
        self.observation_dataframe = add_obs_to_df(
            df=self.observation_dataframe, obs=obs, time_step=self.time_step
        )

        if not self.predicting:
            result = self.fixed_action
        else:
            data_list = compute_data_list_from_dataframe(
                self.observation_dataframe.loc[
                    self.observation_dataframe["type"] == "agent"
                ],
                num_time_steps=self.time_window_length,
                num_agents=self.num_agents,
                box_size=self.box_size,
                label_offset=0,  # needs to be 0 so that we don't go beyond the end, label is ignored in rollout
                node_feature_columns=self.node_feature_columns,
                time_window_length=self.time_window_length,
            )

            with torch.no_grad():
                # we only created one graph, the one for the current observation.
                prediction, _ = self.model(data_list[0])

                """
                TOC -- 010726 7:23PM
                Need to multiply this by the box size to rescale everything back to original simulator coordinates
                """
                result = prediction.detach().cpu().numpy() * self.box_size

                # if we are predicting velocities, we need to add the velocity to the last position
                if self.predictions_are_velocities:
                    result += self.last_position

                # remove the oldest observation from the observation dataframe so we are ready to fill in next time window
                self.observation_dataframe = self.observation_dataframe[
                    self.observation_dataframe["time"]
                    != self.observation_dataframe["time"].min()
                ]

        self.last_position = result
        return result

    def reset(self) -> None:
        """
        resets the agents to their initial state
        """
        self.last_position = self.init_position

    def update(self, time_step: int) -> None:
        """
        Called by the simulator at the beginning of each time step before get_action_list()
        Args:
            time_step (int): the current time step in the simulation
        """
        self.time_step = time_step

        # if we don't have enough time steps yet to fill a time window, then get another position from the position
        # dataframe and indicate that we are not predicting; otherwise indicate that we are ready to predict.
        if time_step < self.time_window_length:
            self.predicting = False
            # we want the position at the next time frame because that is the result of our action at this time frame
            self.fixed_action = self.get_position_from_df(time_step + 1)
        else:
            self.predicting = True

    def get_reset_options(self) -> dict:
        """
        Called by the simulator at the beginning of the simulation so the simulator
        can be reset with the proper options.

          Returns:
              a dictionary with the reset options to be used to initialize the simulator.
              The returned dictionary should includes run_trajectories (value doesn't matter)
              and the initial positions of the agents.
        """
        return {"run_trajectories": True, "agent_trajectories": self.init_position}

    def get_variant_types(self) -> Tuple[Any, Any]:
        """
        This isn't needed for the GNN agents, but it is needed for the BoidsAgents, so it is
        included in the abstract class and so must be included here.
        """
        return None, None

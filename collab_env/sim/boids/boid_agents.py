import numpy as np

import gymnasium as gym

from collab_env.sim.boids.boidsAgents import BoidsWorldAgent, Mesh_Avoidance


class BoidAgents:
    def __init__(self, config: dict, env: gym.Env):
        self.config = config
        self.env = env

        self.target_creation_time = config["simulator"]["target_creation_time"]
        self.target_weights = self.config["agent"]["target_weight"]
        self.num_targets = config["simulator"]["num_targets"]
        self.variant_index_list = []
        self.variant_type_list = []
        if "agent_variants" not in config["agent"]:
            """
            TOC -- 102825 10:55AM
            The variant index list keeps track of the indices that start each variant. 
            In the case of no variants, this will just be 0. 
            """
            # variant_num_list.append(config["simulator"]["num_agents"])
            self.variant_index_list.append(0)
            self.variant_type_list.append("unspecified")

            agent = BoidsWorldAgent(
                env=env,
                num_agents=config["simulator"]["num_agents"],
                num_targets=config["simulator"]["num_targets"],
                walking=config["simulator"]["walking"],
                has_mesh_scene=(config["meshes"]["mesh_scene"] != ""),
                min_ground_separation=config["agent"]["min_ground_separation"],
                min_separation=config["agent"]["min_separation"],
                neighborhood_dist=config["agent"]["neighborhood_dist"],
                random_weight=config["agent"]["random_weight"],
                ground_weight=config["agent"]["ground_weight"],
                separation_weight=config["agent"]["separation_weight"],
                alignment_weight=config["agent"]["alignment_weight"],
                cohesion_weight=config["agent"]["cohesion_weight"],
                target_weight=[0.0]
                * config["simulator"][
                    "num_targets"
                ],  # start at all 0's and add weights when created -- not a great design.
                max_speed=config["agent"]["max_speed"],
                min_speed=config["agent"]["min_speed"],
                max_force=config["agent"]["max_force"],
                random_walk=config["agent"]["random_walk"],
                mesh_avoidance_type=Mesh_Avoidance[
                    config["agent"]["mesh_avoidance"].upper()
                ],
            )
            self.agent_variant_list = [agent]
            self.agent_variants = []
        else:
            self.agent_variants = config.get("agent", {}).get("agent_variants", [])
            self.agent_variant_list = []
            num_agents_so_far = 0
            for variant in self.agent_variants:
                """
                TOC -- 102825 10:55AM
                The variant index list keeps track of the indices that start each variant. 
                This will be the number of agents we have created prior to this agent type, 
                which matches the initialize_index argument. 
                """
                self.variant_index_list.append(num_agents_so_far)
                self.variant_type_list.append(variant["type"])

                agent = BoidsWorldAgent(
                    env=env,
                    num_agents=variant["num_agents_of_type"],
                    initialize_index=num_agents_so_far,
                    num_targets=config["simulator"]["num_targets"],
                    walking=config["simulator"]["walking"],
                    has_mesh_scene=(config["meshes"]["mesh_scene"] != ""),
                    min_ground_separation=config["agent"]["min_ground_separation"]
                    if "min_ground_separation"
                    else variant["min_ground_separation"],
                    min_separation=config["agent"]["min_separation"]
                    if "min_separation" not in variant
                    else variant["min_separation"],
                    neighborhood_dist=config["agent"]["neighborhood_dist"]
                    if "neighborhood_dist" not in variant
                    else variant["neighborhood_dist"],
                    random_weight=config["agent"]["random_weight"]
                    if "random_weight" not in variant
                    else variant["random_weight"],
                    ground_weight=config["agent"]["ground_weight"]
                    if "ground_weight" not in variant
                    else variant["ground_weight"],
                    separation_weight=config["agent"]["separation_weight"]
                    if "separation_weight" not in variant
                    else variant["separation_weight"],
                    alignment_weight=config["agent"]["alignment_weight"]
                    if "alignment_weight" not in variant
                    else variant["alignment_weight"],
                    cohesion_weight=config["agent"]["cohesion_weight"]
                    if "cohesion_weight" not in variant
                    else variant["cohesion_weight"],
                    target_weight=[0.0]
                    * config["simulator"][
                        "num_targets"
                    ],  # start at all 0's and add weights when created -- not a great design.
                    max_speed=config["agent"]["max_speed"]
                    if "max_speed" not in variant
                    else variant["max_speed"],
                    min_speed=config["agent"]["min_speed"]
                    if "min_speed" not in variant
                    else variant["min_speed"],
                    max_force=config["agent"]["max_force"]
                    if "max_force" not in variant
                    else variant["max_force"],
                    random_walk=config["agent"]["random_walk"]
                    if "random_walk" not in variant
                    else variant["random_walk"],
                    mesh_avoidance_type=Mesh_Avoidance[
                        config["agent"]["mesh_avoidance"].upper()
                    ]
                    if "mesh_avoidance" not in variant
                    else variant["mesh_avoidance"],
                )
                num_agents_so_far += variant["num_agents_of_type"]
                self.agent_variant_list.append(agent)

        """
        TOC -- 010426 1:05AM
        This seems like it needs to handle multiple targets and it doesn't
        """
        for agent in self.agent_variant_list:
            agent.set_target_weight(0.0, 0)

    def update(self, time_step: int):
        """
        Updates the target weights based on the time step. In general,
        this method should do whatever update is necessary at each time
        step.
        """
        for t in range(self.num_targets):
            if time_step == self.target_creation_time[t]:
                """
                -- 082325 11:02PM
                I am only setting the first target weight here. That is a problem that
                needs to be fixed with multiple targets. 

                -- 082525 2:03PM 
                This was fixed.

                -- 091825 4:50PM
                Need to differentiate between variants. It would make sense to move this into the Agent class. 
                The original design of having it here was always bad.  
                """
                for v in range(len(self.agent_variant_list)):
                    self.agent_variant_list[v].set_target_weight(
                        self.target_weights[t]
                        if len(self.agent_variants) == 0
                        or "target_weight" not in self.agent_variants[v]
                        else self.agent_variants[v]["target_weight"],
                        t,
                    )

    def get_action_list(self, obs) -> np.ndarray:
        """
        TOC -- 010426
        Not sure I follow what is going on here. What is up with this
        np.concatenate() deal? Need to come back to this.
        """
        # each Agent chooses action
        action_list = []
        for agent in self.agent_variant_list:
            action_list.append(agent.get_action(obs))
        action_list = np.concatenate([np.asarray(a) for a in action_list])
        return action_list

    def get_variant_types(self):
        return self.variant_index_list, self.variant_type_list

    def get_reset_options(self):
        return None

import os
from glob import glob
import numpy as np

# import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path

from tests.sim_test_util import remove_run_folder, create_run_folder


def test_sim_output_consistency():
    """
    This will test the following:
    1 - the agents locations and velocity at a time step are consistent with the agent locations
    at the next time step

    2 - the distances reported to the targets and the scene are consistent with the locations reported
    for the agents and the closest points reported for the targets and the scene.

    Returns:

    """
    sim_runs_path = expand_path(
        "sim-output/tests-sim-runs-consistency", get_project_root()
    )
    remote_test = "CI" in os.environ
    if not remote_test:
        # Clear the sim-runs folder from previous tests -- a little dicey
        # If this is a remote test, these files shouldn't be here, so don't bother. They
        # get cleaned up at the end
        remove_run_folder(sim_runs_path)

    create_run_folder(sim_runs_path)

    program_path = expand_path(
        "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
    )

    # Test to see that the run_boids_simulator runs successfully with test config file
    logger.info(f"python {program_path} -cf tests/sim_test_consistent_config.yaml")
    result = os.system(
        f"python {program_path} -cf tests/sim_test_consistent_config.yaml"
    )
    assert result == 0

    # Get the output folders in the run path for this test
    """
    -- 082525 0929PM
    This is kind of a stupid way to find the files. 
    """
    folder_list = glob(f"{sim_runs_path}/boids_sim_run_consistent*")

    # get the parquet file in the first folder
    result_file_list = glob(f"{folder_list[0]}/*.parquet")
    logger.info(f"parquet folder list for {folder_list[0]} = {result_file_list}")

    # read the dataframe from the parquet file
    df = pq.read_pandas(result_file_list[0]).to_pandas()
    targets_df = df.loc[df["type"] == "env"]

    # get all the agent rows
    agents_df = df.loc[df["type"] == "agent"]
    num_agents = agents_df["id"].max()
    num_targets = targets_df["id"].max()

    for i in range(num_agents):
        # get the rows for agent i
        agent = agents_df.loc[df["id"] == i]

        # get the list of locations and velocities for agent i (one for each time step)
        location = agent[["x", "y", "z"]].to_numpy()
        velocity = agent[["v_x", "v_y", "v_z"]].to_numpy()

        # get the mesh scene distances and closest points for agent i
        mesh_scene_closest_point = agent["mesh_scene_closest_point"].to_numpy()
        mesh_scene_distance = agent["mesh_scene_distance"].to_numpy()

        # check each time step (should be able to do this with numpy instead of a for loop
        for time in range(len(location) - 1):
            if np.any(location[time + 1] - (location[time] + velocity[time + 1])):
                logger.debug("failed velocity change of position test ")
                logger.debug(location[time])
                logger.debug(velocity[time + 1])
                logger.debug(location[time + 1])
                logger.debug(location[time] + velocity[time + 1])
                logger.debug(location[time + 1] - (location[time] + velocity[time + 1]))
                assert False

            diff = (
                np.linalg.norm(location[time] - mesh_scene_closest_point[time])
                - mesh_scene_distance[time]
                < 0.001
            )
            if not diff.all():
                logger.debug("failed mesh scene distance test ")
                logger.debug(f"location\n {location[time]}")
                logger.debug(f"{mesh_scene_closest_point[time]}")
                logger.debug(f"{mesh_scene_distance[time]}")
                logger.debug(
                    f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time])}"
                )
                logger.debug(
                    f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time]) - mesh_scene_distance[time]}"
                )
                assert False

            for target in range(1, num_targets + 1):
                target_mesh_closest_point = agent[
                    f"target_mesh_closest_point_{target}"
                ].to_numpy()
                distance_to_target_mesh_closest_point = agent[
                    f"distance_to_target_mesh_closest_point_{target}"
                ].to_numpy()
                diff = (
                    np.linalg.norm(location[time] - target_mesh_closest_point[time])
                    - distance_to_target_mesh_closest_point[time]
                    < 0.001
                )
                # logger.info(f'target mesh diff = {np.linalg.norm(location[time] - target_mesh_closest_point[time]) - distance_to_target_mesh_closest_point[time]}')
                if not diff.all():
                    logger.debug("failed mesh scene distance test ")
                    logger.debug(f"location\n {location[time]}")
                    logger.debug(f"{target_mesh_closest_point[time]}")
                    logger.debug(f"{distance_to_target_mesh_closest_point[time]}")
                    logger.debug(
                        f"{np.linalg.norm(location[time] - distance_to_target_mesh_closest_point[time])}"
                    )
                    logger.debug(
                        f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time]) - mesh_scene_distance[time]}"
                    )
                    assert False

    if remote_test:
        # if this is not a remote test, we should clean up the files
        # clear the sim-runs folder from previous tests
        remove_run_folder(sim_runs_path)

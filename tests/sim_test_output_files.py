from loguru import logger
from collab_env.sim.boids.sim_utils import add_obs_to_df


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
    logger.info(f"df\n{df[['id', 'type', 'mesh_scene_closest_point']]}")

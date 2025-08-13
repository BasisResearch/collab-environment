import pandas as pd
from loguru import logger
from collab_env.sim.boids.sim_utils import add_obs_to_df


def test_add_obs():
    num_targets = 1
    distance_columns = [f"distance_target_{t}" for t in range(1, num_targets + 1)]
    pandas_columns = [
        "id",
        "type",
        "time",
        "x",
        "y",
        "z",
        "v_x",
        "v_y",
        "v_z",
    ] + distance_columns
    obs = dict()
    obs["agent_loc"] = [[1, 2, 3]]
    obs["agent_vel"] = [[10, 20, 30]]
    obs["target_loc"] = [[1000, 2000, 3000]]
    obs["distances_to_targets"] = [[100.0]]
    obs["target_closest_points"] = [[100.0, 200.0, 300.0]]
    df = pd.DataFrame(columns=pandas_columns)
    df = add_obs_to_df(df, obs, 1)
    logger.info(f"\n{df}")

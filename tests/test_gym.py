import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
import gymnasium as gym


def test_gym_env_init():
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode="human",
        num_agents=1,
        walking=False,
    )
    assert env is not None

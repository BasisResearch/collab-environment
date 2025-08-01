from gymnasium.envs.registration import register

register(
    id="gymnasium_env/BoidsWorldSimple-v0",
    entry_point="collab_env.sim.gymnasium_env.envs:BoidsWorldSimpleEnv",
)

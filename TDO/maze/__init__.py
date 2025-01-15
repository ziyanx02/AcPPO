from maze.envs.vanilla_reach import VanillaReachEnv, VanillaReachEnvCfg

ENV_DICT = {
    "VanillaReach": VanillaReachEnv,
}

CFG_DICT = {
    "VanillaReach": VanillaReachEnvCfg,
}

def get_env(env_name, num_envs, device="cuda"):
    env_cls = ENV_DICT[env_name]
    env_cfg = CFG_DICT[env_name]()
    env_cfg.num_envs = num_envs
    return env_cls(env_cfg, device)
from jsplendor.env.env import SelfPlayEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


def make_env(opponent_policy, reserve_masking, verbose_dict, rank: int, seed: int=0):
    """Create a wrapped, monitored environment"""
    def _init():
        env = SelfPlayEnv(
            opponent_policy=opponent_policy,
            reserve_masking=reserve_masking,
            verbose_dict=verbose_dict
        )
        env = Monitor(env)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init
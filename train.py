import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from jsplendor.env import JsplendorEnv
from jsplendor.utils import get_verbose_dict
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from jsplendor.models.transformer import TransformerFeatureExtractor

def make_env(rank: int, seed: int=0):
    train_verbose_dict = get_verbose_dict()

    def _init():
        env = JsplendorEnv(train_verbose_dict)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

def main(args):
    eval_verbose_dict = get_verbose_dict()
    eval_verbose_dict['player'] = False
    
    if args.debug:
        train_env = JsplendorEnv(get_verbose_dict())
    else:
        train_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

    eval_env = JsplendorEnv(eval_verbose_dict)
    exp = '250212'
    eval_log_dir = 'logs/{}'.format(exp)

    train_steps = 1e+8 # 100M
    n_steps = 4096*4
    eval_freq = n_steps

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir, eval_freq=eval_freq,
        n_eval_episodes=128, deterministic=False,
        render=False)

    policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            net_arch=[64],
            activation_fn=torch.nn.ReLU
    )

    model = PPO(
        policy=MaskedActorCriticPolicy,
        env=train_env,
        n_steps=n_steps,
        learning_rate=1e-6,
        batch_size=512,
        verbose=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log=eval_log_dir,
    )

    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)
    print('train done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num-cpu', type=int, default=8, help='Number of CPU cores to use')
    args = parser.parse_args()

    main(args)

import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer

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

    # Modify eval env to use multiple environments
    eval_env = SubprocVecEnv([make_env(i) for i in range(args.num_eval_cpu)])  # Multiple environments for evaluation
    eval_log_dir = 'logs/{}'.format(args.exp)

    train_steps = 1e+8 # 100M
    n_steps = args.n_steps
    eval_freq = n_steps

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir, 
        eval_freq=eval_freq,
        n_eval_episodes=128,  # This will be split across eval environments
        deterministic=False,
        render=False)

    policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            net_arch=[64],
            activation_fn=torch.nn.ReLU
    )

    if args.load_model:
        print(f"Loading pretrained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=eval_log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Load directly to target device
        )
        
        # Adjust n_steps and update related parameters
        model.n_steps = n_steps
        model.n_epochs = model.n_epochs  # Keep the same number of epochs
        model.batch_size = model.batch_size  # Keep the same batch size
        
        # Create new rollout buffer with correct dimensions
        model.rollout_buffer = RolloutBuffer(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gae_lambda=model.gae_lambda,
            gamma=model.gamma,
            n_envs=model.n_envs,
        )
        
        print("Model loaded successfully.")
        print(f"Continuing training in experiment: {args.exp}")
        print(f"Training parameters:")
        print(f" - Learning rate: {model.learning_rate}")
        print(f" - Batch size: {model.batch_size}")
        print(f" - N steps: {model.n_steps}")
        print(f" - Total timesteps to train: {train_steps}")
        print(f" - Number of environments: {args.num_cpu if not args.debug else 1}")
    else:
        model = PPO(
            policy=MaskedActorCriticPolicy,
            env=train_env,
            n_steps=n_steps,        # Reduced batch size for faster updates
            learning_rate=2e-6,  # Slightly increased for faster learning
            batch_size=2048,     # Increased to better utilize GPU memory
            verbose=False,
            policy_kwargs=policy_kwargs,
            tensorboard_log=eval_log_dir,
        )

    if args.debug:
        model.learn(total_timesteps=train_steps, progress_bar=False, callback=eval_callback)
    else:
        model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)
    print('train done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num_cpu', type=int, default=8, help='Number of CPU cores to use')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='250213', help='Experiment name for logging')
    parser.add_argument('--n_steps', type=int, default=4096*4, help='Number of steps per update')
    parser.add_argument('--num_eval_cpu', type=int, default=8, help='Number of CPU cores to use for evaluation')
    args = parser.parse_args()

    main(args)

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
    def _init():
        env = JsplendorEnv(get_verbose_dict())
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

def main(args):    
    if args.debug:
        train_env = JsplendorEnv(get_verbose_dict())
        eval_env = JsplendorEnv(get_verbose_dict())
    else:
        train_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
        eval_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])  # Multiple environments for evaluation


    # Modify eval env to use multiple environments
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

    # Create policy kwargs with min_prob and correct architecture
    policy_kwargs = {
        'min_prob': args.min_prob,
        'features_extractor_class': TransformerFeatureExtractor,
        'net_arch': dict(pi=[64], vf=[64]),  # Changed from list to dict
        'activation_fn': torch.nn.ReLU
    }

    if args.load_model:
        print(f"Loading pretrained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=eval_log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',  # Load directly to target device
            ent_coef=args.ent_coef
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
            batch_size=512,     # Increased to better utilize GPU memory
            policy_kwargs=policy_kwargs,
            tensorboard_log=eval_log_dir,
            ent_coef=args.ent_coef
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
    parser.add_argument('--exp', type=str, default='tmp', help='Experiment name for logging')
    parser.add_argument('--n_steps', type=int, default=16384, help='Number of steps per update')
    parser.add_argument('--min_prob', type=float, default=0,
                       help='Minimum probability for valid actions')
    parser.add_argument('--ent_coef', type=float, default=0,
                       help='Entropy coefficient for exploration')
    args = parser.parse_args()

    main(args)

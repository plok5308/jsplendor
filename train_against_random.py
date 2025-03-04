import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gym

from jsplendor.env.two_player_env import RandomStartTwoPlayerEnv
from jsplendor.models.transformer import TransformerFeatureExtractor
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy

class StepRewardWrapper(gym.Wrapper):
    """Wrapper that adds step-based reward while preserving original rewards"""
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add step-based reward only when winning
        if terminated and info.get('winner') == 'player':
            player_steps = info['steps']['player0'] if self.env.player_starts_first else info['steps']['player1']
            step_reward = 100 - player_steps
            reward = reward + step_reward  # Add to original reward
            info['step_reward'] = step_reward
            
        return obs, reward, terminated, truncated, info

def make_env(verbose_dict, rank: int, seed: int=0):
    """Create a wrapped, monitored environment"""
    def _init():
        env = RandomStartTwoPlayerEnv(
            opponent_policy=RandomPlayer(),
            verbose_dict=verbose_dict
        )
        env = StepRewardWrapper(env)  # Add step reward wrapper
        env = Monitor(env)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

def train(args):
    # Set up verbose dict
    verbose_dict = get_verbose_dict()
    if args.debug:
        verbose_dict['env'] = True
        verbose_dict['game'] = True
        verbose_dict['player'] = True
        verbose_dict['board'] = True
    
    # Set up environments with wrapper
    if args.debug:
        train_env = Monitor(StepRewardWrapper(RandomStartTwoPlayerEnv(RandomPlayer(), verbose_dict=verbose_dict)))
        eval_env = Monitor(StepRewardWrapper(RandomStartTwoPlayerEnv(RandomPlayer(), verbose_dict=verbose_dict)))
    else:
        train_env = SubprocVecEnv([make_env(verbose_dict, i) for i in range(args.num_cpu)])
        eval_env = Monitor(StepRewardWrapper(RandomStartTwoPlayerEnv(RandomPlayer(), verbose_dict=verbose_dict)))

    # Create log directory
    log_dir = f'logs/{args.exp}'
    os.makedirs(log_dir, exist_ok=True)

    # Create eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=100,
        deterministic=args.deterministic,
        render=False
    )

    # Create or load model
    if args.load_model:
        print(f"Loading pretrained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        policy_kwargs = {
            'features_extractor_class': TransformerFeatureExtractor,
            'net_arch': dict(pi=[64], vf=[64]),
            'activation_fn': torch.nn.ReLU,
            'temperature': 1.0,
            'top_k': 0,
            'top_p': 1.0
        }

        model = PPO(
            MaskedActorCriticPolicy,
            env=train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=2e-6,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # Train model
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=not args.debug,
        callback=eval_callback
    )
    
    # Save final model
    model.save(os.path.join(log_dir, "final_model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent against random player for JSplendor')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num_cpu', type=int, default=1, help='Number of CPU cores to use')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='against_random', help='Experiment name for logging')
    parser.add_argument('--total_timesteps', type=int, default=int(1e8), help='Total timesteps to train')
    parser.add_argument('--eval_freq', type=int, default=16384, help='Evaluation frequency')
    parser.add_argument('--n_steps', type=int, default=16384, help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=512, help='Size of the batch for training')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions during evaluation')
    args = parser.parse_args()

    model = train(args) 

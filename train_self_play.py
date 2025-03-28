import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.monitor import Monitor

from jsplendor.env.env import SelfPlayEnv
from jsplendor.models.linear2 import LinearFeatureExtractor
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from jsplendor.training.self_player_evaluation_callback import SelfPlayCallback
from jsplendor.env.utils import make_env

def create_model(env, args):
    # Select feature extractor based on model type
    if args.model_type == 'linear':
        feature_extractor = LinearFeatureExtractor
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create policy kwargs with selected feature extractor
    policy_kwargs = {
        'features_extractor_class': feature_extractor,
        'net_arch': dict(pi=[64], vf=[64]),  # Match feature dimension
        'activation_fn': torch.nn.ReLU,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 1.0
    }

    return PPO(
        MaskedActorCriticPolicy,
        env,
        n_steps=args.n_steps,
        batch_size=512,
        learning_rate=2e-6,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/{args.exp}",
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def train_self_play(args):
    # Set up verbose dict based on debug flag
    verbose_dict = get_verbose_dict()
    if args.debug:
        verbose_dict['env'] = True
        verbose_dict['game'] = True
        verbose_dict['player'] = True
        verbose_dict['board'] = True
    
    # Initialize opponent as random player
    opponent = RandomPlayer()
    best_models_dir = os.path.join('logs', args.exp, 'best_models')
    os.makedirs(best_models_dir, exist_ok=True)
    
    reserve_masking = args.reserve_masking
    
    # Set up environments - vectorized for training, single for evaluation
    if args.debug:
        train_env = SelfPlayEnv(opponent, reserve_masking, verbose_dict=verbose_dict)
        eval_env = SelfPlayEnv(opponent, reserve_masking, verbose_dict=verbose_dict)
    else:
        train_env = SubprocVecEnv([make_env(opponent, reserve_masking, verbose_dict, i) for i in range(args.num_cpu)])
        # Create a single environment for evaluation
        eval_env = Monitor(SelfPlayEnv(opponent, reserve_masking, verbose_dict=verbose_dict))

    eval_log_dir = f'logs/{args.exp}'
    os.makedirs(eval_log_dir, exist_ok=True)

    # Create self-play callback
    self_play_callback = SelfPlayCallback(
        eval_env=eval_env,
        opponent_builder=lambda: RandomPlayer(),  # Initial opponent builder
        reserve_masking=reserve_masking,
        verbose_dict=verbose_dict,
        best_models_dir=best_models_dir,
        n_eval_episodes=1000,
        deterministic=args.deterministic
    )

    if args.load_model:
        print(f"Loading pretrained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=eval_log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ent_coef=args.ent_coef
        )
        
        # Adjust n_steps and update related parameters for multi-env
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
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
    else:
        model = create_model(train_env, args)

    # Train model
    total_timesteps = args.n_steps * args.total_generations
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=not args.debug,
        callback=self_play_callback
    )
    
    print("\n" + "="*50)
    print('Training completed.')
    print(f"Completed {self_play_callback.generation} generations")
    print("="*50)
    
    # Save final model
    model.save(os.path.join(eval_log_dir, "final_model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor with self-play')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num_cpu', type=int, default=1, help='Number of CPU cores to use')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='tmp', help='Experiment name for logging')
    parser.add_argument('--total_generations', type=int, default=10000, help='Total number of generations to train')
    parser.add_argument('--ent_coef', type=float, default=0, help='Entropy coefficient for exploration')
    parser.add_argument('--n_steps', type=int, default=65536, help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=512, help='Size of the batch for training')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions during evaluation')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'linear'], default='linear',
                      help='Type of feature extractor to use')
    parser.add_argument('--reserve_masking', choices=['player', 'opponent', 'both'], default='both',
                      help='Type of masking to use for reserved cards')
    parser.add_argument('--gamma', type=float, default=0.8, 
                      help='Discount factor (lower values like 0.8 focus more on latter part of game)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, 
                      help='GAE lambda parameter for advantage estimation')
    args = parser.parse_args()

    model = train_self_play(args)

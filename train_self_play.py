import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer

from jsplendor.env.two_player_env import RandomStartTwoPlayerEnv
from jsplendor.models.transformer import TransformerFeatureExtractor
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy

def create_model(env, args):
    # Create policy kwargs with transformer and masking settings
    policy_kwargs = {
        'features_extractor_class': TransformerFeatureExtractor,
        'net_arch': dict(pi=[64], vf=[64]),
        'activation_fn': torch.nn.ReLU,
        'min_prob': args.min_prob,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 1.0
    }

    return PPO(
        MaskedActorCriticPolicy,  # Use masked policy instead of base policy
        env,
        n_steps=args.n_steps,
        batch_size=512,
        learning_rate=2e-6,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/{args.exp}",
        ent_coef=args.ent_coef,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def evaluate_agent(env, model, n_episodes=100, deterministic=True):
    """Run evaluation episodes and print results"""
    print("\nStarting evaluation...")
    print(f"Deterministic: {deterministic}")
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    not_terminated = 0
    total_games = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if done:
                total_games += 1
                if 'winner' in info:
                    if info['winner'] == 'player':
                        wins += 1
                    elif info['winner'] == 'opponent':
                        losses += 1
                    elif info['winner'] == 'not terminated':
                        not_terminated += 1
                    else:
                        draws += 1
                episode_rewards.append(total_reward)
                if 'steps' in info:
                    episode_lengths.append(max(info['steps'].values()))
    
    win_rate = wins/total_games if total_games > 0 else 0
    print("\nEvaluation Results:")
    print(f"Total games: {total_games}")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}, Not terminated: {not_terminated}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'not_terminated': not_terminated,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'win_rate': win_rate,
        'total_games': total_games
    }

class ModelPlayer:
    """Wrapper class for using a trained model as an opponent"""
    def __init__(self, model, deterministic=False):
        self.model = model
        self.deterministic = deterministic
    
    def __call__(self, observation):
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action

def make_env(opponent, verbose_dict, rank: int, seed: int=0):
    """Create a wrapped, monitored environment"""
    def _init():
        env = RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict)
        env = Monitor(env)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

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
    
    # Set up environments - vectorized for training, single for evaluation
    if args.debug:
        train_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
        eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
    else:
        train_env = SubprocVecEnv([make_env(opponent, verbose_dict, i) for i in range(args.num_cpu)])
        # Always use single environment for evaluation
        eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))

    eval_log_dir = f'logs/{args.exp}'
    os.makedirs(eval_log_dir, exist_ok=True)

    # Add eval callback with single environment
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=args.n_steps,
        n_eval_episodes=100,  # No need to adjust for num_cpu now
        deterministic=args.deterministic,
        render=False
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
            tensorboard_log=f"logs/{args.exp}",
            ent_coef=args.ent_coef,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    generation = 0
    while generation < args.total_generations:
        print("\n" + "="*50)
        print(f"Training Generation {generation}")
        print("="*50)
        
        # Train model with adjusted timesteps
        if args.debug:
            actual_timesteps = args.n_steps
        else:
            actual_timesteps = args.n_steps * args.num_cpu
        
        print(f"\nTraining for {actual_timesteps} timesteps...")
        model.learn(
            total_timesteps=actual_timesteps,
            progress_bar=not args.debug,
            callback=eval_callback
        )
        
        print("\nEvaluating against current opponent...")
        eval_results = evaluate_agent(eval_env, model, deterministic=args.deterministic)
        
        # If win rate is above threshold, save model and use it as new opponent
        if eval_results['win_rate'] > 0.6:
            print("\n" + "-"*50)
            print(f"Win rate {eval_results['win_rate']:.1%} exceeds threshold!")
            print(f"Saving model and updating opponent...")
            model_path = os.path.join(best_models_dir, f"model_gen_{generation}")
            model.save(model_path)
            
            # Create new opponent from current model
            opponent_model = PPO.load(model_path)
            opponent = ModelPlayer(opponent_model, deterministic=args.deterministic)
            
            # Update environments with new opponent
            if args.debug:
                train_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
                eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
            else:
                train_env = SubprocVecEnv([make_env(opponent, verbose_dict, i) for i in range(args.num_cpu)])
                # Keep evaluation environment single
                eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
            
            model.set_env(train_env)
            print(f"Now training against model from generation {generation}")
            print("-"*50)
        
        generation += 1
    
    print("\n" + "="*50)
    print('Training completed.')
    print(f"Completed {generation} generations")
    print("="*50)
    
    # Save final model
    model.save(os.path.join(eval_log_dir, "final_model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor with self-play')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num_cpu', type=int, default=8, help='Number of CPU cores to use')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='self_play6', help='Experiment name for logging')
    parser.add_argument('--total_generations', type=int, default=10000, help='Total number of generations to train')
    parser.add_argument('--ent_coef', type=float, default=0, help='Entropy coefficient for exploration')
    parser.add_argument('--n_steps', type=int, default=16384, help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=512, help='Size of the batch for training')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions during evaluation')
    args = parser.parse_args()

    model = train_self_play(args)

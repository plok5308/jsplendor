import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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

def train_self_play(args):
    # Set up verbose dict based on debug flag
    verbose_dict = get_verbose_dict()
    if args.debug:
        verbose_dict['env'] = True
        verbose_dict['game'] = True
        verbose_dict['player'] = True
        verbose_dict['board'] = True
    
    # Set up environments
    random_opponent = RandomPlayer()
    train_env = Monitor(RandomStartTwoPlayerEnv(random_opponent, verbose_dict=verbose_dict))
    eval_env = Monitor(RandomStartTwoPlayerEnv(random_opponent, verbose_dict=verbose_dict))

    # Set up eval callback
    eval_log_dir = f'logs/{args.exp}'
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=args.n_steps,
        n_eval_episodes=128,
        deterministic=False,
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
        print("Model loaded successfully.")
        print(f"Continuing training in experiment: {args.exp}")
    else:
        model = create_model(train_env, args)

    # Train model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=not args.debug
    )
    
    # Save final model
    model.save(os.path.join(eval_log_dir, "final_model"))
    print('Training completed.')
    
    return model

def evaluate_agent(env, n_episodes=100):
    """Run evaluation episodes and print results"""
    print("\nStarting evaluation...")
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action_mask = env.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)  # Random actions for now
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if done:
                if 'winner' in info:
                    if info['winner'] == 'player':
                        wins += 1
                    elif info['winner'] == 'opponent':
                        losses += 1
                    else:
                        draws += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{n_episodes} episodes")
            print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(f"Average length: {np.mean(episode_lengths):.1f} steps")
            print("-" * 50)
    
    print("\nEvaluation Results:")
    print(f"Win rate: {wins/n_episodes:.1%}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor with self-play')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='self_play', help='Experiment name for logging')
    parser.add_argument('--n_steps', type=int, default=16384, help='Number of steps per update')
    parser.add_argument('--total_timesteps', type=int, default=int(1e8), help='Total timesteps to train')
    parser.add_argument('--ent_coef', type=float, default=0, help='Entropy coefficient for exploration')
    parser.add_argument('--min_prob', type=float, default=0, help='Minimum probability for valid actions')
    args = parser.parse_args()

    model = train_self_play(args)

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

def evaluate_agent(env, model, n_episodes=100):
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
            # Pass action mask as part of the observation
            action_mask = env.get_action_mask()
            action, _ = model.predict(obs)  # Remove action_masks parameter
            
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
    
    win_rate = wins/n_episodes
    print("\nEvaluation Results:")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'win_rate': win_rate
    }

class ModelPlayer:
    """Wrapper class for using a trained model as an opponent"""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, observation):
        action, _ = self.model.predict(observation)
        return action

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
    
    # Set up environments
    train_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
    eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))

    # Set up eval callback
    eval_log_dir = f'logs/{args.exp}'
    os.makedirs(eval_log_dir, exist_ok=True)

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

    generation = 0
    while generation < args.total_generations:
        print(f"\nTraining generation {generation}")
        
        # Train model
        model.learn(
            total_timesteps=args.generation_timesteps,
            progress_bar=not args.debug
        )
        
        # Evaluate current model
        eval_results = evaluate_agent(eval_env, model)
        
        # If win rate is above threshold, save model and use it as new opponent
        if eval_results['win_rate'] > 0.6:
            print(f"Win rate {eval_results['win_rate']:.1%} exceeds threshold! Saving model and updating opponent...")
            model_path = os.path.join(best_models_dir, f"model_gen_{generation}")
            model.save(model_path)
            
            # Create new opponent from current model
            opponent_model = PPO.load(model_path)
            opponent = ModelPlayer(opponent_model)  # Wrap the model in ModelPlayer
            
            # Update environments with new opponent
            train_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
            eval_env = Monitor(RandomStartTwoPlayerEnv(opponent, verbose_dict=verbose_dict))
            # Update model's environment
            model.set_env(train_env)
            
            print(f"Now training against model from generation {generation}")
        
        generation += 1
    
    # Save final model
    model.save(os.path.join(eval_log_dir, "final_model"))
    print('Training completed.')
    print(f"Completed {generation} generations")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor with self-play')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='self_play4', help='Experiment name for logging')
    parser.add_argument('--total_generations', type=int, default=10000, help='Total number of generations to train')
    parser.add_argument('--generation_timesteps', type=int, default=65536, help='Timesteps to train per generation')
    parser.add_argument('--ent_coef', type=float, default=0, help='Entropy coefficient for exploration')
    parser.add_argument('--min_prob', type=float, default=0, help='Minimum probability for valid actions')
    parser.add_argument('--n_steps', type=int, default=16384, help='Number of steps per update')
    args = parser.parse_args()

    model = train_self_play(args)

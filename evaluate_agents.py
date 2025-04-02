import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from jsplendor.env import SelfPlayEnv
from jsplendor.models.linear2 import LinearFeatureExtractor
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from jsplendor.utils.action_description import get_action_description
from jsplendor.utils import TestLogger


def evaluate_match(agent1, agent2, env, n_episodes=100, deterministic1=True, deterministic2=True):
    """Evaluate matches between two agents using provided environment"""
    print("\nStarting evaluation...")
    print(f"Agent 1 deterministic: {deterministic1}")
    print(f"Agent 2 deterministic: {deterministic2}")
    
    wins1 = 0  # agent1 wins
    wins2 = 0  # agent2 wins
    draws = 0
    not_terminated = 0
    episode_lengths = []
    
    # Update environment's opponent policy
    env.env.opponent_policy = lambda x: agent2.predict(x, deterministic=deterministic2)[0]
    
    # Get device from agent1's policy
    device = next(agent1.policy.parameters()).device if not isinstance(agent1, RandomPlayer) else 'cpu'
    
    # Create logger if environment doesn't have one
    if args.verbose:
        logger = getattr(env.env, 'logger', TestLogger('logs/eval'))
    else:
        logger = None
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Take action
            action, _ = agent1.predict(obs, deterministic=deterministic1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # If verbose mode, wait for user input before continuing
            if args.verbose:
                input("Press Enter to continue...")
            
            if done:
                if 'winner' in info:
                    if info['winner'] == 'player':
                        wins1 += 1
                    elif info['winner'] == 'opponent':
                        wins2 += 1
                    elif info['winner'] == 'not terminated':
                        not_terminated += 1
                    else:
                        draws += 1
                if 'steps' in info:
                    episode_lengths.append(max(info['steps'].values()))
        
        if (episode + 1) % 10 == 0:  # Print progress every 10 games
            print(f"\nGames completed: {episode + 1}/{n_episodes}")
            print(f"Player wins: {wins1} ({wins1/(episode+1):.1%})")
            print(f"Opponent wins: {wins2} ({wins2/(episode+1):.1%})")
            print(f"Draws: {draws} ({draws/(episode+1):.1%})")
            print(f"Not terminated: {not_terminated} ({not_terminated/(episode+1):.1%})")
    
    print("\nFinal Results:")
    print(f"Player wins: {wins1} ({wins1/n_episodes:.1%})")
    print(f"Opponent wins: {wins2} ({wins2/n_episodes:.1%})")
    print(f"Draws: {draws} ({draws/n_episodes:.1%})")
    print(f"Not terminated: {not_terminated} ({not_terminated/n_episodes:.1%})")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'wins1': wins1,
        'wins2': wins2,
        'draws': draws,
        'not_terminated': not_terminated,
        'win_rate1': wins1/n_episodes,
        'win_rate2': wins2/n_episodes,
        'avg_length': np.mean(episode_lengths)
    }

def create_model(env, model_type='transformer'):
    """Create a new model with specified architecture"""
    if model_type == 'linear':
        feature_extractor = LinearFeatureExtractor
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    policy_kwargs = {
        'features_extractor_class': feature_extractor,
        'net_arch': dict(pi=[64], vf=[64]),
        'activation_fn': torch.nn.ReLU,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 1.0
    }

    return PPO(
        MaskedActorCriticPolicy,
        env,
        n_steps=16384,
        batch_size=512,
        learning_rate=2e-6,
        policy_kwargs=policy_kwargs,
        tensorboard_log="logs/eval",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate matches between JSplendor models')
    parser.add_argument('--player_model_path', type=str, default=None,
                       help='Path to player model')
    parser.add_argument('--opponent_model_path', type=str, default=None,
                       help='Path to opponent model')
    parser.add_argument('--n_episodes', type=int, default=1,
                       help='Number of episodes to evaluate')
    parser.add_argument('--deterministic1', action='store_true',
                       help='Use deterministic actions for first model')
    parser.add_argument('--deterministic2', action='store_true',
                       help='Use deterministic actions for second model')
    parser.add_argument('--reserve_masking', choices=['none', 'player', 'opponent', 'both'], default='both',
                      help='Type of masking to use for reserved cards')

    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose mode for detailed game information')
    args = parser.parse_args()
    
    # Set up environment
    verbose_dict = get_verbose_dict()
    if args.verbose:
        verbose_dict['env'] = True
        verbose_dict['game'] = True
        verbose_dict['player'] = True
        verbose_dict['board'] = True
    
    # Set up single environment
    env = Monitor(SelfPlayEnv(
        opponent_policy=RandomPlayer(),  # Will be updated later
        reserve_masking=args.reserve_masking,
        verbose_dict=verbose_dict,
        #player_starts_first=True
    ))
    
    # Load agents
    if args.player_model_path is not None:
        model1 = create_model(env, 'linear')
        agent1 = model1.load(args.player_model_path, env=env)
        print(f"Agent 1: {args.player_model_path}")
    else:
        agent1 = RandomPlayer()
        print("Agent 1: Random Player") 

    if args.opponent_model_path is not None:
        model2 = create_model(env, 'linear')
        agent2 = model2.load(args.opponent_model_path, env=env)
        print(f"Agent 2: {args.opponent_model_path}")
    else:
        agent2 = RandomPlayer()
        print("Agent 2: Random Player")
    
    # Run evaluation with single environment
    results = evaluate_match(agent1, agent2, env, args.n_episodes, 
                           args.deterministic1, args.deterministic2)


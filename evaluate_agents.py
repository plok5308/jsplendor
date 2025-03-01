import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from jsplendor.env.two_player_env import RandomStartTwoPlayerEnv
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer

def evaluate_agents(env, agent1=None, n_episodes=100):
    """Run evaluation episodes between agent1 and environment's opponent"""
    print("\nStarting evaluation...")
    episode_rewards = []
    episode_lengths = []
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    # Convert None to RandomPlayer
    if agent1 is None:
        agent1 = RandomPlayer()
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Track who starts first this episode
        agent1_starts = info.get('starts_first', True)
        
        while not done:
            # Get action from agent1
            action, _ = agent1.predict(obs, deterministic=True)
            action = int(action)  # Ensure integer action
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward = reward
            steps += 1
            
            if done:
                if 'winner' in info:
                    if info['winner'] == 'player':
                        agent1_wins += 1
                    elif info['winner'] == 'opponent':
                        agent2_wins += 1
                    else:
                        draws += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"\nCompleted {episode + 1}/{n_episodes} episodes")
            print(f"Agent 1 wins: {agent1_wins}, Agent 2 wins: {agent2_wins}, Draws: {draws}")
            print(f"Agent 1 win rate: {agent1_wins/(episode+1):.1%}")
            print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
            print("-" * 50)
    
    print("\nFinal Results:")
    print(f"Agent 1 wins: {agent1_wins}")
    print(f"Agent 2 wins: {agent2_wins}")
    print(f"Draws: {draws}")
    print(f"Agent 1 win rate: {agent1_wins/n_episodes:.1%}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'win_rate': agent1_wins/n_episodes,
        'avg_length': np.mean(episode_lengths)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate agents for JSplendor')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model1', type=str, help='Path to first model (None for random)')
    parser.add_argument('--model2', type=str, help='Path to second model (None for random)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--verbose', action='store_true', help='Show game logs')
    args = parser.parse_args()

    # Set up verbose dict
    verbose_dict = get_verbose_dict()
    if args.verbose:
        verbose_dict = {
            'env': True,
            'game': True,
            'player': True,
            'board': True
        }
        # Override episodes to 1 when verbose is enabled
        args.episodes = 1
        print("\nVerbose mode enabled - running single episode")
    
    # Load models if specified
    agent1 = PPO.load(args.model1) if args.model1 else None
    agent2 = PPO.load(args.model2) if args.model2 else RandomPlayer()
    
    # Create environment with agent2 as opponent
    eval_env = Monitor(RandomStartTwoPlayerEnv(
        opponent_policy=lambda x: agent2.predict(x)[0],
        verbose_dict=verbose_dict
    ))
    
    # Print agent descriptions
    print("\nAgent 1:", "Model" if agent1 else "Random")
    print("Agent 2:", "Model" if args.model2 else "Random")
    
    # Run evaluation
    results = evaluate_agents(eval_env, agent1=agent1, n_episodes=args.episodes) 
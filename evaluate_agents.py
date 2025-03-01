import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from jsplendor.env.two_player_env import RandomStartTwoPlayerEnv
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer

def evaluate_agents(agent1, agent2, n_episodes=1):
    env = Monitor(RandomStartTwoPlayerEnv(
        opponent_policy=lambda x: agent2.predict(x)[0],
        verbose_dict=verbose_dict
    ))
    
    # Print agent descriptions
    print("\nAgent 1:", "Model" if args.model1 else "Random")
    print("Agent 2:", "Model" if args.model2 else "Random")
    
    print("\nStarting evaluation...")
    episode_rewards = []
    episode_lengths = []
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    not_terminated = 0
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Track who starts first this episode
        agent1_starts = info.get('starts_first', True)
        #print(f"Agent 1 starts: {agent1_starts}")
        
        while not done:
            # Get action from agent1
            action, _ = agent1.predict(obs, deterministic=True)
            action = int(action)  # Ensure integer action
            
            #print('before action')
            #print(f"obs[action_mask]: {obs[-27:]}")
            #print(f"obs[player]: {obs[-27-26:-27-13]}")
            #print(f"obs[opponent]: {obs[-27-13:-27]}")
            obs, reward, terminated, truncated, info = env.step(action)

            #print('after action')
            #print(f"obs[action_mask]: {obs[-27:]}")
            #print(f"obs[player]: {obs[-27-26:-27-13]}")
            #print(f"obs[opponent]: {obs[-27-13:-27]}")

            done = terminated or truncated
            total_reward = reward
            
            if done:
                if 'winner' in info:
                    if info['winner'] == 'player':
                        agent1_wins += 1
                    elif info['winner'] == 'opponent':
                        agent2_wins += 1
                    elif info['winner'] == 'not terminated':
                        not_terminated += 1

                    else:
                        draws += 1
        
        episode_rewards.append(total_reward)
        if agent1_starts:
            episode_lengths.append(info['steps']['player0'])
        else:
            episode_lengths.append(info['steps']['player1'])
    
    print("\nFinal Results:")
    print(f"Agent 1 wins: {agent1_wins}")
    print(f"Agent 2 wins: {agent2_wins}")
    print(f"Draws: {draws}")
    print(f"Not terminated: {not_terminated}")
    print(f"Agent 1 win rate: {agent1_wins/n_episodes:.1%}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'not_terminated': not_terminated,
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
    agent1 = PPO.load(args.model1) if args.model1 else RandomPlayer()
    agent2 = PPO.load(args.model2) if args.model2 else RandomPlayer()
    
    results = evaluate_agents(agent1, agent2, n_episodes=args.episodes)


import os
import glob
import numpy as np
import argparse
from stable_baselines3 import PPO
from tqdm import tqdm
from jsplendor.env import SelfPlayEnv
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from stable_baselines3.common.monitor import Monitor

class EloPlayer:
    def __init__(self, name, model=None, initial_elo=1200):
        self.name = name
        self.model = model
        self.elo = initial_elo
        self.games_played = 0
        
    def predict(self, obs, deterministic=False):
        if isinstance(self.model, RandomPlayer):
            return self.model.predict(obs, deterministic)
        return self.model.predict(obs, deterministic=deterministic)

def expected_score(rating1, rating2):
    """Calculate expected score for player1"""
    return 1 / (1 + 10**((rating2 - rating1) / 400))

def update_elo(rating1, rating2, score, k=32):
    """Update Elo ratings after a game"""
    expected = expected_score(rating1, rating2)
    new_rating1 = rating1 + k * (score - expected)
    new_rating2 = rating2 + k * ((1 - score) - (1 - expected))
    return new_rating1, new_rating2

def evaluate_match(player1, player2, n_games=100, deterministic=False, reserve_masking='both'):
    """Evaluate a match between two players"""
    env = SelfPlayEnv(
        opponent_policy=lambda x: player2.predict(x, deterministic=deterministic)[0],
        verbose_dict=get_verbose_dict(),
        reserve_masking=reserve_masking
    )
    
    wins1 = 0  # player1 wins
    wins2 = 0  # player2 wins
    draws = 0
    not_terminated = 0
    print(f"\nEvaluating: {player1.name} vs {player2.name} (deterministic={deterministic})")
    
    for game in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = player1.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done and 'winner' in info:
                if info['winner'] == 'player':
                    wins1 += 1
                elif info['winner'] == 'opponent':
                    wins2 += 1
                elif info['winner'] == 'not terminated':
                    not_terminated += 1
                else:
                    draws += 1
                    
        if (game + 1) % 10 == 0:  # Print progress every 10 games
            games_completed = game + 1
            print(f"Games completed: {games_completed}/{n_games}")
            print(f"{player1.name} wins: {wins1} ({wins1/games_completed:.1%})")
            print(f"{player2.name} wins: {wins2} ({wins2/games_completed:.1%})")
            print(f"Draws: {draws} ({draws/games_completed:.1%})")
            print(f"Not terminated: {not_terminated} ({not_terminated/games_completed:.1%})")
            print("-" * 30)
    
    win_rate = wins1/n_games
    print(f"\nFinal results:")
    print(f"{player1.name} wins: {wins1} ({wins1/n_games:.1%})")
    print(f"{player2.name} wins: {wins2} ({wins2/n_games:.1%})")
    print(f"Draws: {draws} ({draws/n_games:.1%})")
    print(f"Not terminated: {not_terminated} ({not_terminated/n_games:.1%})")
    
    return {
        'win_rate1': wins1/n_games,
        'win_rate2': wins2/n_games,
        'draws': draws/n_games,
        'not_terminated': not_terminated/n_games
    }

def load_models(model_dir):
    """Load both transformer and linear models"""
    players = [EloPlayer("Random", RandomPlayer(), initial_elo=500)]
    
    model_paths = glob.glob(os.path.join(model_dir, "**", "*.zip"), recursive=True)
    for path in sorted(model_paths):
        name = f"{os.path.basename(path)}"
        print(f"Loading model: {name}")
        model = PPO.load(path)
        players.append(EloPlayer(name, model))
    
    return players

def calculate_elo_ratings(model_dir, n_games=20, deterministic=False, reserve_masking='both'):
    """Calculate Elo ratings for all models"""
    # Load all models
    players = load_models(model_dir)
    print(f"\nFound {len(players)-1} models to evaluate ({len(players)} players including Random)")
    
    # Play round-robin tournament
    n_players = len(players)
    total_matches = (n_players * (n_players - 1)) // 2
    print(f"\nStarting round-robin tournament:")
    print(f"Number of players: {n_players}")
    print(f"Total matches to play: {total_matches}")
    print(f"Games per match: {n_games}")
    print(f"Total games to play: {total_matches * n_games}")
    print(f"Random player starting Elo: {players[0].elo}")
    print(f"Using deterministic actions: {deterministic}")
    print(f"Reserve masking: {reserve_masking}")
    print("\nStarting matches...")
    
    match_count = 0
    for i in range(n_players):
        for j in range(i + 1, n_players):
            match_count += 1
            print(f"\nMatch {match_count}/{total_matches}")
            print(f"Current Elo - {players[i].name}: {players[i].elo:.1f}, {players[j].name}: {players[j].elo:.1f}")
            
            # Play match both ways (each player gets to go first)
            results1 = evaluate_match(players[i], players[j], n_games//2, deterministic, reserve_masking)
            results2 = evaluate_match(players[j], players[i], n_games//2, deterministic, reserve_masking)
            avg_score = (results1['win_rate1'] + (1 - results2['win_rate1'])) / 2
            
            # Update Elo ratings
            old_elo_i, old_elo_j = players[i].elo, players[j].elo
            players[i].elo, players[j].elo = update_elo(old_elo_i, old_elo_j, avg_score)
            players[i].games_played += n_games
            players[j].games_played += n_games
            
            # Print Elo changes
            print(f"\nElo changes:")
            print(f"{players[i].name}: {old_elo_i:.1f} -> {players[i].elo:.1f} ({players[i].elo - old_elo_i:+.1f})")
            print(f"{players[j].name}: {old_elo_j:.1f} -> {players[j].elo:.1f} ({players[j].elo - old_elo_j:+.1f})")
    
    # Sort and print results
    players.sort(key=lambda x: x.elo, reverse=True)
    print("\nFinal Elo Ratings:")
    print("-" * 60)
    print(f"{'Player':<30} {'Elo':>8} {'Games':>8}")
    print("-" * 60)
    for player in players:
        print(f"{player.name:<30} {player.elo:>8.1f} {player.games_played:>8}")
    
    return players

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--n_games', type=int, default=100,
                       help='Number of games to play per match')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--reserve_masking', choices=['none', 'player', 'opponent', 'both'],
                       default='both', help='Type of masking to use for reserved cards')
    
    args = parser.parse_args()
    
    # Calculate Elo ratings for all models in directory
    players = calculate_elo_ratings(
        args.model_dir, 
        args.n_games, 
        args.deterministic,
        args.reserve_masking
    )

if __name__ == "__main__":
    main() 

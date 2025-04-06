import pygame
import argparse
from copy import deepcopy
from jsplendor.env import SelfPlayEnv
from jsplendor.gui.game_gui import GameGUI
from jsplendor.utils import TestLogger
from stable_baselines3 import PPO
from jsplendor.models.random_player import RandomPlayer
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.linear2 import LinearFeatureExtractor
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
import torch

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
        'temperature': 0.5,  # Lower temperature for more focused probabilities
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

def main():
    parser = argparse.ArgumentParser(description='Run JSplendor GUI with two AI agents')
    parser.add_argument('--player_model_path', type=str, default=None,
                       help='Path to player model')
    parser.add_argument('--opponent_model_path', type=str, default=None,
                       help='Path to opponent model')
    parser.add_argument('--player_model_type', type=str, choices=['linear', 'random'], 
                       default='random', help='Type of player model')
    parser.add_argument('--opponent_model_type', type=str, choices=['linear', 'random'],
                       default='random', help='Type of opponent model')
    parser.add_argument('--deterministic1', action='store_true',
                       help='Use deterministic actions for player model')
    parser.add_argument('--deterministic2', action='store_true',
                       help='Use deterministic actions for opponent model')
    parser.add_argument('--log_dir', type=str, default='logs/gui_test',
                       help='Directory for logging')
    parser.add_argument('--reserve_masking', choices=['none', 'player', 'opponent', 'both'], default='none',
                      help='Type of masking to use for reserved cards')
                      
    args = parser.parse_args()

    # Validate arguments
    if args.player_model_type != 'random' and not args.player_model_path:
        parser.error("--player_model_path is required when player_model_type is not 'random'")
    if args.opponent_model_type != 'random' and not args.opponent_model_path:
        parser.error("--opponent_model_path is required when opponent_model_type is not 'random'")

    # Set all verbose settings to True
    verbose_dict = get_verbose_dict()
    verbose_dict['env'] = True
    verbose_dict['game'] = True
    verbose_dict['player'] = True
    verbose_dict['board'] = True

    # Create logger with specified log directory
    logger = TestLogger(args.log_dir)

    # Create environment first
    env = Monitor(SelfPlayEnv(
        opponent_policy=RandomPlayer(),  # Temporary opponent, will be updated
        reserve_masking=args.reserve_masking,
        verbose_dict=verbose_dict,
        player_starts_first=True,
    ))

    # Create/load first agent
    if args.player_model_type == 'random':
        agent1 = RandomPlayer()
        print("Agent 1: Random Player")
    else:
        model1 = create_model(env, args.player_model_type)
        print(f"Loading player model from {args.player_model_path}...")
        agent1 = model1.load(args.player_model_path, env=env)
        print(f"Agent 1: {args.player_model_type.capitalize()} Model")

    # Create/load second agent
    if args.opponent_model_type == 'random':
        agent2 = RandomPlayer()
        print("Agent 2: Random Player")
    else:
        model2 = create_model(env, args.opponent_model_type)
        print(f"Loading opponent model from {args.opponent_model_path}...")
        agent2 = model2.load(args.opponent_model_path, env=env)
        print(f"Agent 2: {args.opponent_model_type.capitalize()} Model")

    # Update environment with actual opponent
    env.env.opponent_policy = lambda x: agent2.predict(x, deterministic=args.deterministic2)[0]

    # Create GUI with both models and their paths, passing the environment
    gui = GameGUI(agent1, agent2, logger, verbose_dict, env=env)

    # Run the GUI
    try:
        gui.run()
    except Exception as e:
        logger.info(f"Error during GUI execution: {str(e)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main() 

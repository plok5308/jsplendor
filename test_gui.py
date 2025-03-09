import pygame
import argparse
from copy import deepcopy
from jsplendor.env import SelfPlayEnv
from jsplendor.gui.game_gui import AIGameGUI
from jsplendor.utils import TestLogger
from stable_baselines3 import PPO
from jsplendor.models.random_player import RandomPlayer
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.transformer import TransformerFeatureExtractor
from jsplendor.models.linear2 import LinearFeatureExtractor
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
import torch

def create_model(env, model_type='transformer'):
    """Create a new model with specified architecture"""
    if model_type == 'transformer':
        feature_extractor = TransformerFeatureExtractor
    elif model_type == 'linear':
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
    parser.add_argument('--model1_path', type=str,
                       help='Path to first model')
    parser.add_argument('--model2_path', type=str,
                       help='Path to second model')
    parser.add_argument('--model1_type', type=str, choices=['linear', 'random'], 
                       default='random', help='Type of first model')
    parser.add_argument('--model2_type', type=str, choices=['linear', 'random'],
                       default='random', help='Type of second model')
    parser.add_argument('--deterministic1', action='store_true',
                       help='Use deterministic actions for first model')
    parser.add_argument('--deterministic2', action='store_true',
                       help='Use deterministic actions for second model')
    parser.add_argument('--log_dir', type=str, default='logs/gui_test',
                       help='Directory for logging')
    args = parser.parse_args()

    # Validate arguments
    if args.model1_type != 'random' and not args.model1_path:
        parser.error("--model1_path is required when model1_type is not 'random'")
    if args.model2_type != 'random' and not args.model2_path:
        parser.error("--model2_path is required when model2_type is not 'random'")

    # Set all verbose settings to True
    verbose_dict = get_verbose_dict()
    verbose_dict['env'] = True
    verbose_dict['game'] = True
    verbose_dict['player'] = True
    verbose_dict['board'] = True

    # Create logger with specified log directory
    logger = TestLogger(args.log_dir)

    # Create/load first agent
    if args.model1_type == 'random':
        agent1 = RandomPlayer()
        print("Agent 1: Random Player")
    else:
        # Create dummy env just for model loading
        dummy_env = SelfPlayEnv(opponent_policy=RandomPlayer())
        model1 = create_model(dummy_env, args.model1_type)
        print(f"Loading model 1 from {args.model1_path}...")
        agent1 = model1.load(args.model1_path, env=dummy_env)
        print(f"Agent 1: {args.model1_type.capitalize()} Model")

    # Create/load second agent
    if args.model2_type == 'random':
        agent2 = RandomPlayer()
        print("Agent 2: Random Player")
    else:
        # Create dummy env just for model loading
        dummy_env = SelfPlayEnv(opponent_policy=RandomPlayer())
        model2 = create_model(dummy_env, args.model2_type)
        print(f"Loading model 2 from {args.model2_path}...")
        agent2 = model2.load(args.model2_path, env=dummy_env)
        print(f"Agent 2: {args.model2_type.capitalize()} Model")

    # Create GUI with both models and their paths
    gui = AIGameGUI(agent1, agent2, logger, verbose_dict)

    # Run the GUI
    try:
        gui.run()
    except Exception as e:
        logger.info(f"Error during GUI execution: {str(e)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main() 

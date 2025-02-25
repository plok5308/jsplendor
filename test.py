import torch
import numpy as np
import argparse
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from jsplendor.env import JsplendorEnv
from jsplendor.utils import get_verbose_dict, TestLogger
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from jsplendor.models.transformer import TransformerFeatureExtractor
from jsplendor.utils.logger import ActionLogger
from jsplendor.agent.test_agent import TestAgent

def main(args):
    np.random.seed(1)
    
    # Set verbose_dict for environment based on args.verbose
    verbose_dict = {
        'game': args.verbose,
        'player': args.verbose,
        'board': args.verbose,
        'env': args.verbose
    }

    # Set device to CPU
    device = "cpu"
    
    env = JsplendorEnv(verbose_dict)

    exp = '250225_double_coin'
    model_path = 'logs/{}/best_model'.format(exp)
    log_dir = 'logs/{}/'.format(exp)

    #model_path ='./pretrained/best_model'
    #log_dir = './pretrained/logs'

    # Setup logger with verbose=False to only show final stats
    logger = TestLogger(log_dir, verbose=False)  # Changed to False
    logger.log_config(args, model_path)

    # Load pretrained model and force it to CPU
    logger.info("Loading pretrained model...")
    ppo_model = PPO.load(model_path, env=env, device=device,
                    custom_objects={
                        'temperature': args.temperature,
                        'top_k': args.top_k,
                        'top_p': args.top_p
                    })
    logger.info('Model loaded successfully on CPU.')

    # Wrap PPO model with TestAgent
    model = TestAgent(env, ppo_model)

    game_n = args.num_games
    max_step = args.max_steps
    results = []

    for exp_i in tqdm(range(game_n), desc="Testing games"):
        obs, _ = env.reset(seed=exp_i)
        for i in range(max_step):
            # Get action probabilities
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                features = model.policy.extract_features(obs_tensor.unsqueeze(0))
                latent_pi, _ = model.policy.mlp_extractor(features)
                logits = model.policy.action_net(latent_pi)
                action_probs = torch.softmax(logits, dim=-1)
                action_probs = action_probs.squeeze(0).cpu().numpy()
                
                # Only log actions if verbose is True
                ActionLogger.log_action_probabilities(logger, env, action_probs, args.verbose)
            
            # Get and take action
            action, _state = model.predict(obs, deterministic=False)
            
            # Only log selected action if verbose is True
            ActionLogger.log_selected_action(logger, action, args.verbose)
            
            obs, reward, done, _, info = env.step(action)
            
            if done:
                results.append(i)
                if args.verbose:  # Only log individual game results if verbose
                    logger.log_game_result(exp_i, i, max_step)
                break

            if i == max_step - 1:
                results.append(i)
                if args.verbose:  # Only log individual game results if verbose
                    logger.log_game_result(exp_i, i, max_step)

    # Always show final statistics with verbose=True
    logger.verbose = True  # Temporarily enable verbose for final stats
    mean_steps, success_rate, fail_count = logger.log_statistics(results, game_n, max_step)
    logger.verbose = False  # Reset verbose setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PPO agent for JSplendor')
    parser.add_argument('--num_games', type=int, default=128, help='Number of games to test')
    parser.add_argument('--max_steps', type=int, default=127, help='Maximum steps per game')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Add sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of highest probability tokens to keep (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Cumulative probability cutoff for nucleus sampling (1.0 = disabled)')
    args = parser.parse_args()

    main(args)

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

def main(args):
    np.random.seed(1)
    verbose_dict = get_verbose_dict(args.verbose)

    # Set device to CPU
    device = "cpu"
    
    env = JsplendorEnv(verbose_dict)
    exp = '250215_small_step_penalty'
    model_path = 'logs/{}/best_model'.format(exp)
    log_dir = 'logs/{}/'.format(exp)

    #model_path = 'pretrained/best_model'
    #log_dir = 'pretrained/logs'

    # Setup logger
    logger = TestLogger(log_dir)
    logger.log_config(args, model_path)

    # Load pretrained model and force it to CPU
    logger.info("Loading pretrained model...")
    model = PPO.load(model_path, env=env, device=device)
    logger.info('Model loaded successfully on CPU.')

    game_n = args.num_games
    max_step = args.max_steps
    results = []
    
    for exp_i in tqdm(range(game_n), desc="Testing games"):
        obs, _ = env.reset(seed=exp_i)
        for i in range(max_step):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            
            if done:
                results.append(i)
                logger.log_game_result(exp_i, i, max_step)
                break

            if i == max_step - 1:
                results.append(i)
                logger.log_game_result(exp_i, i, max_step)

    # Log final statistics
    mean_steps, success_rate, fail_count = logger.log_statistics(results, game_n, max_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PPO agent for JSplendor')
    parser.add_argument('--num_games', type=int, default=32, help='Number of games to test')
    parser.add_argument('--max_steps', type=int, default=127, help='Maximum steps per game')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    main(args)

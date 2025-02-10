import torch
import numpy as np
import argparse
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from jsplendor.env import JsplendorEnv, FeatureExtractor
from jsplendor.utils import get_verbose_dict, TestLogger
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy

def main(args):
    np.random.seed(1)
    verbose_dict = get_verbose_dict(True)

    env = JsplendorEnv(verbose_dict)
    exp = 'transformer_model-vp_action_mask'
    model_path = 'logs/{}/best_model'.format(exp)

    # Setup logger
    logger = TestLogger(exp)
    logger.log_config(args, model_path)

    if args.load_model:
        logger.info("Loading pretrained model...")
        model = PPO.load(model_path, env=env)
    else:
        logger.info("Initializing new model...")
        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            net_arch=[64],
            activation_fn=torch.nn.ReLU
        )

        model = PPO(
            policy=MaskedActorCriticPolicy,
            env=env,
            verbose=False,
            policy_kwargs=policy_kwargs,
        )

    logger.info('Model loaded successfully.')

    game_n = args.num_games
    max_step = args.max_steps
    results = []
    
    for exp_i in tqdm(range(game_n), desc="Testing games"):
        obs, _ = env.reset(seed=exp_i)
        for i in range(max_step):
            # Get action from model
            action, _state = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _, info = env.step(action)
            
            if done:
                results.append(i)
                logger.log_game_result(exp_i, i, max_step)
                break

            if i == max_step - 1:
                results.append(i)
                logger.log_game_result(exp_i, i, max_step)

    # Log statistics and get summary for console
    mean_steps, success_rate, fail_count = logger.log_statistics(results, game_n, max_step)

    # No need for additional console printing as it's handled by the logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PPO agent for JSplendor')
    parser.add_argument('--load-model', action='store_true', help='Load pretrained model')
    parser.add_argument('--num-games', type=int, default=10, help='Number of games to test')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per game')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    args = parser.parse_args()

    main(args)

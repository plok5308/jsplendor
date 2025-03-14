import os
import torch
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer

from jsplendor.env.env import SelfPlayEnv
from jsplendor.models.transformer import TransformerFeatureExtractor
from jsplendor.models.linear2 import LinearFeatureExtractor
from jsplendor.utils.config import get_verbose_dict
from jsplendor.models.random_player import RandomPlayer
from jsplendor.policy.masked_policy import MaskedActorCriticPolicy
from jsplendor.env import StepRewardWrapper

def create_model(env, args):
    # Select feature extractor based on model type
    if args.model_type == 'transformer':
        feature_extractor = TransformerFeatureExtractor
    elif args.model_type == 'linear':
        feature_extractor = LinearFeatureExtractor
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create policy kwargs with selected feature extractor
    policy_kwargs = {
        'features_extractor_class': feature_extractor,
        'net_arch': dict(pi=[64], vf=[64]),  # Match feature dimension
        'activation_fn': torch.nn.ReLU,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 1.0
    }

    return PPO(
        MaskedActorCriticPolicy,
        env,
        n_steps=args.n_steps,
        batch_size=512,
        learning_rate=2e-6,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/{args.exp}",
        ent_coef=args.ent_coef,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def evaluate_agent(env, model, n_episodes=100, deterministic=True):
    """Run evaluation episodes and print results"""
    print("\nStarting evaluation...")
    print(f"Deterministic: {deterministic}")
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    not_terminated = 0
    total_games = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if done:
                total_games += 1
                if 'winner' in info:
                    if info['winner'] == 'player':
                        wins += 1
                    elif info['winner'] == 'opponent':
                        losses += 1
                    elif info['winner'] == 'not terminated':
                        not_terminated += 1
                    else:
                        draws += 1
                episode_rewards.append(total_reward)
                if 'steps' in info:
                    episode_lengths.append(max(info['steps'].values()))
    
    win_rate = wins/total_games if total_games > 0 else 0
    

    print("\nEvaluation Results:")
    print(f"Total games: {total_games}")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}, Not terminated: {not_terminated}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'not_terminated': not_terminated,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'win_rate': win_rate,
        'total_games': total_games
    }

class ModelPlayer:
    """Wrapper class for using a trained model as an opponent"""
    def __init__(self, model, deterministic=False):
        self.model = model
        self.deterministic = deterministic
    
    def __call__(self, observation):
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action

def make_env(opponent_policy, verbose_dict, rank: int, seed: int=0):
    """Create a wrapped, monitored environment"""
    def _init():
        env = SelfPlayEnv(
            opponent_policy=opponent_policy,
            verbose_dict=verbose_dict
        )
        env = StepRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

class SelfPlayCallback(EventCallback):
    def __init__(self, eval_env, opponent_builder, verbose_dict, best_models_dir, n_eval_episodes=100, deterministic=True):
        super().__init__(None, verbose=False)
        self.eval_env = eval_env
        self.opponent_builder = opponent_builder
        self.verbose_dict = verbose_dict
        self.best_models_dir = best_models_dir
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.generation = 0
        self.win_rate_threshold = 0.55
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % self.model.n_steps == 0:
            print("\nEvaluating against current opponent...")
            eval_results = evaluate_agent(self.eval_env, self.model, 
                                       n_episodes=self.n_eval_episodes,
                                       deterministic=self.deterministic)
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", eval_results['avg_reward'])
            self.logger.record("eval/mean_ep_length", eval_results['avg_length'])
            self.logger.record("eval/win_rate", eval_results['win_rate'])
            self.logger.record("eval/wins", eval_results['wins'])
            self.logger.record("eval/losses", eval_results['losses'])
            self.logger.record("eval/draws", eval_results['draws'])
            self.logger.record("eval/not_terminated", eval_results['not_terminated'])
            self.logger.record("eval/generation", self.generation)
            
            # Dump log info to disk
            self.logger.dump(self.n_calls)
            
            # Update best mean reward
            mean_reward = eval_results['avg_reward']
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
            
            self.last_mean_reward = mean_reward
            
            # If win rate exceeds threshold, update opponent
            if eval_results['win_rate'] > self.win_rate_threshold:
                print("\n" + "-"*50)
                print(f"Win rate {eval_results['win_rate']:.1%} exceeds threshold!")
                print(f"Saving model and updating opponent...")
                
                # Save current model
                model_path = os.path.join(self.best_models_dir, f"model_gen_{self.generation}")
                self.model.save(model_path)
                
                # Create and set new opponent
                opponent_model = PPO.load(model_path)
                new_opponent = ModelPlayer(opponent_model, deterministic=self.deterministic)
                
                # Update environments
                if isinstance(self.model.env, SubprocVecEnv):
                    self.model.env = SubprocVecEnv(
                        [make_env(new_opponent, self.verbose_dict, i) 
                         for i in range(self.model.env.num_envs)]
                    )
                else:
                    self.model.env = Monitor(
                        SelfPlayEnv(new_opponent, verbose_dict=self.verbose_dict)
                    )
                
                # Update eval environment
                self.eval_env = Monitor(
                    SelfPlayEnv(new_opponent, verbose_dict=self.verbose_dict)
                )
                
                print(f"Now training against model from generation {self.generation}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
                print("-"*50)
                
                self.generation += 1
                
        return True

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
    
    # Set up environments - vectorized for training, single for evaluation
    if args.debug:
        train_env = Monitor(SelfPlayEnv(opponent, verbose_dict=verbose_dict))
        eval_env = Monitor(SelfPlayEnv(opponent, verbose_dict=verbose_dict))
    else:
        train_env = SubprocVecEnv([make_env(opponent, verbose_dict, i) for i in range(args.num_cpu)])
        # Always use single environment for evaluation
        eval_env = Monitor(SelfPlayEnv(opponent, verbose_dict=verbose_dict))

    eval_log_dir = f'logs/{args.exp}'
    os.makedirs(eval_log_dir, exist_ok=True)

    # Create self-play callback
    self_play_callback = SelfPlayCallback(
        eval_env=eval_env,
        opponent_builder=lambda: RandomPlayer(),  # Initial opponent builder
        verbose_dict=verbose_dict,
        best_models_dir=best_models_dir,
        n_eval_episodes=1000,
        deterministic=args.deterministic
    )

    if args.load_model:
        print(f"Loading pretrained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=eval_log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ent_coef=args.ent_coef
        )
        
        # Adjust n_steps and update related parameters for multi-env
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.rollout_buffer = RolloutBuffer(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gae_lambda=model.gae_lambda,
            gamma=model.gamma,
            n_envs=model.n_envs,
        )
        
        print("Model loaded successfully.")
        print(f"Continuing training in experiment: {args.exp}")
    else:
        model = create_model(train_env, args)

    # Train model
    total_timesteps = args.n_steps * args.total_generations
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=not args.debug,
        callback=self_play_callback
    )
    
    print("\n" + "="*50)
    print('Training completed.')
    print(f"Completed {self_play_callback.generation} generations")
    print("="*50)
    
    # Save final model
    model.save(os.path.join(eval_log_dir, "final_model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for JSplendor with self-play')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single environment')
    parser.add_argument('--num_cpu', type=int, default=1, help='Number of CPU cores to use')
    parser.add_argument('--load_model', type=str, help='Path to pretrained model to continue training')
    parser.add_argument('--exp', type=str, default='tmp', help='Experiment name for logging')
    parser.add_argument('--total_generations', type=int, default=10000, help='Total number of generations to train')
    parser.add_argument('--ent_coef', type=float, default=0, help='Entropy coefficient for exploration')
    parser.add_argument('--n_steps', type=int, default=int(2**18), help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=512, help='Size of the batch for training')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions during evaluation')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'linear'], default='linear',
                      help='Type of feature extractor to use')
    args = parser.parse_args()

    model = train_self_play(args)

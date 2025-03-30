import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from jsplendor.env.env import SelfPlayEnv
from jsplendor.env.utils import make_env


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
        obs, info = env.reset()
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


class SelfPlayCallback(EventCallback):
    def __init__(self, eval_env, opponent_builder, reserve_masking, verbose_dict, best_models_dir, n_eval_episodes=100, win_rate_threshold=0.55, deterministic=True):
        super().__init__(None, verbose=False)
        self.eval_env = eval_env
        self.opponent_builder = opponent_builder
        self.reserve_masking = reserve_masking
        self.verbose_dict = verbose_dict
        self.best_models_dir = best_models_dir
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.generation = 0
        self.win_rate_threshold = win_rate_threshold
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.last_model_path = None

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
            
            # Save current model regardless of performance
            self.last_model_path = os.path.join(self.best_models_dir, f"model_gen_{self.generation}_last")
            self.model.save(self.last_model_path)
            
            # If win rate exceeds threshold, update opponent
            if eval_results['win_rate'] > self.win_rate_threshold:
                print("\n" + "-"*50)
                print(f"Win rate {eval_results['win_rate']:.1%} exceeds threshold!")
                print(f"Saving model and updating opponent...")
                
                # Save current model as best model
                model_path = os.path.join(self.best_models_dir, f"model_gen_{self.generation}")
                self.model.save(model_path)
                
                # Create and set new opponent using the best model
                opponent_model = PPO.load(model_path)
                new_opponent = ModelPlayer(opponent_model, deterministic=self.deterministic)
                
                # Update environments
                if isinstance(self.model.env, SubprocVecEnv):
                    self.model.env = SubprocVecEnv(
                        [make_env(new_opponent, self.reserve_masking, self.verbose_dict, i) 
                         for i in range(self.model.env.num_envs)]
                    )
                else:
                    self.model.env = Monitor(
                        SelfPlayEnv(new_opponent, self.reserve_masking, self.verbose_dict)
                    )
                
                # Update eval environment
                self.eval_env = Monitor(
                    SelfPlayEnv(new_opponent, self.reserve_masking, self.verbose_dict)
                )
                
                print(f"Now training against model from generation {self.generation}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
                print("-"*50)
                
                self.generation += 1
                
        return True


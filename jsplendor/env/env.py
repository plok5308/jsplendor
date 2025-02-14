import numpy as np
import gymnasium as gym
from gymnasium import spaces

from jsplendor.game import Game
from jsplendor.env.observation import get_observation_space, get_observation
from jsplendor.utils import get_verbose_dict


class JsplendorEnv(gym.Env):
    def __init__(self, verbose_dict=None):
        if verbose_dict is None:
            verbose_dict = get_verbose_dict()
        else:
            verbose_dict = verbose_dict

        self.game = Game(verbose_dict)
        action_n = self.game.player1.num_actions
        self.action_space = spaces.Discrete(action_n)
        
        # Create a single Box space that includes both observation and action mask
        obs_space = get_observation_space()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_space.shape[0] + action_n,),  # Combine obs and action mask shapes
            dtype=np.int32
        )
        self.verbose = verbose_dict['env']
        self.skip_sum = 0

        # parameters
        self.target_vp = 15
        self.max_step = 127
        
        # Penalties
        self.penalty = dict()
        self.penalty['invalid'] = 1.0  # Increased to discourage invalid moves more strongly
        #self.penalty['over_coin'] = 0.5  # Added penalty for inefficient coin management
        self.penalty['over_coin'] = 0  # Added penalty for inefficient coin management
        self.penalty['step_over'] = 10.0  # Kept the same
        
        # Rewards
        self.reward = dict()
        #self.reward['get_card'] = 1.0  # Added reward for acquiring cards
        #self.reward['noble_visit'] = 3.0  # New: reward for attracting nobles
        #self.reward['reach_goal'] = 100.0  # Increased base reward
        #self.reward['vp_progress'] = 2.0  # New: reward per victory point gained
        
        self.reward['get_card'] = 0  # Added reward for acquiring cards
        self.reward['noble_visit'] = 0  # New: reward for attracting nobles
        self.reward['reach_goal'] = 100.0  # Increased base reward
        self.reward['vp_progress'] = 0  # New: reward per victory point gained

        # Additional tracking
        self.previous_vp = 0  # New: track VP changes

    def get_action_mask(self):
        """Convert possible actions to a binary mask"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        possible_actions = self.get_possible_actions()
        mask[possible_actions] = 1.0
        return mask

    def step(self, action):
        terminated = False
        truncated = False

        reward, terminated, step_, is_noble_visit = self._run_action(action)

        if step_ >= self.max_step:
            reward = -1 * self.penalty['step_over']
            if self.verbose:
                print(f'Step {step_}: Max steps reached, terminating with reward {reward}')
            terminated = True
        else:
            if self.verbose:
                print(f'Step {step_}: Action {action}, Reward {reward}')

        observation = get_observation(self.game)
        action_mask = self.get_action_mask()
        
        # Concatenate observation and action mask
        obs = np.concatenate([observation, action_mask])
        
        info = {
            "is_noble_visit": is_noble_visit,
            "seed": getattr(self, 'seed', None)
        }

        return obs, reward, terminated, truncated, info

    def get_possible_actions(self):
        return self.game.player1.get_all_possible_actions(self.game.board)

    def _run_action(self, action):
        terminated = False
        action_result = self.game.run_with_action(action)
        step_ = action_result['step']
        reward = 0
        is_noble_visit = action_result['is_noble_visit']

        if action_result['is_skip']:
            self.skip_sum += 1
            reward = -1 * self.penalty['invalid']
            if self.verbose:
                print(f'Step {step_}: Invalid action, skipping')
        else:
            # Calculate VP progress
            vp_gained = action_result['victory_point'] - self.previous_vp
            reward += vp_gained * self.reward['vp_progress']
            self.previous_vp = action_result['victory_point']

            if self.verbose:
                print(f'Step {step_}: Current VP: {action_result["victory_point"]} (gained {vp_gained})')

            if action_result['victory_point'] >= self.target_vp:
                reward = self.reward['reach_goal'] - step_
                terminated = True
                if self.verbose:
                    print(f'Step {step_}: Target VP reached! Final reward: {reward}')
                    print(f'Step {step_}: Skip ratio: {self.skip_sum / step_:.2f}')

            # Print reward components if verbose
            if self.verbose:
                if action_result['is_get_card']:
                    print(f'Step {step_}: Got a card (+{self.reward["get_card"]})')
                
                if action_result['is_noble_visit']:
                    print(f'Step {step_}: Noble visited (+{self.reward["noble_visit"]})')
                
                if action_result['over_coin_count'] > 0:
                    penalty = self.penalty['over_coin'] * action_result['over_coin_count']
                    print(f'Step {step_}: Over coin count: {action_result["over_coin_count"]} (-{penalty})')

        # Adjust rewards
        if action_result['is_get_card']:
            reward += self.reward['get_card']
        
        if action_result['is_noble_visit']:
            reward += self.reward['noble_visit']
        
        if action_result['over_coin_count'] > 0:
            reward -= self.penalty['over_coin'] * action_result['over_coin_count']

        return reward, terminated, step_, is_noble_visit

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.skip_sum = 0
        self.previous_vp = 0
        self.game.reset()  # This resets the game, including noble cards
        
        observation = get_observation(self.game)
        action_mask = self.get_action_mask()
        
        # Concatenate observation and action mask
        obs = np.concatenate([observation, action_mask])
        
        info = {"seed": seed}
        return obs, info

    def render(self):
        pass

    def close(self):
        pass

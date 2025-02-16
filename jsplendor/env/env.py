import numpy as np
import gymnasium as gym
from gymnasium import spaces

from jsplendor.game import Game
from jsplendor.env.observation import get_observation_space, get_observation
from jsplendor.utils import get_verbose_dict
from jsplendor.utils.logger import TestLogger


class JsplendorEnv(gym.Env):
    def __init__(self, verbose_dict=None):
        if verbose_dict is None:
            verbose_dict = get_verbose_dict()
        
        self.verbose = verbose_dict['env']
        if self.verbose:
            self.logger = TestLogger('logs/env')
            
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
        self.skip_sum = 0

        # parameters
        self.target_vp = 15
        self.max_step = 127
        
        # Penalties
        self.penalty = dict()
        #self.penalty['invalid'] = 1.0  # Increased to discourage invalid moves more strongly
        self.penalty['step'] = 1
        #self.penalty['over_coin'] = 0.5  # Added penalty for inefficient coin management
        self.penalty['over_coin'] = 0  # Added penalty for inefficient coin management
        self.penalty['step_over'] = 100  # Kept the same
        
        # Rewards
        self.reward = dict()
        #self.reward['get_card'] = 1.0  # Added reward for acquiring cards
        #self.reward['noble_visit'] = 3.0  # New: reward for attracting nobles
        #self.reward['reach_goal'] = 100.0  # Increased base reward
        #self.reward['vp_progress'] = 2.0  # New: reward per victory point gained
        
        self.reward['get_card'] = 0  # Added reward for acquiring cards
        self.reward['noble_visit'] = 0  # New: reward for attracting nobles
        self.reward['reach_goal'] = 100  # 100->300
        self.reward['bonus'] = 10
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

        reward, terminated, step_, is_noble_visit, action_result = self._run_action(action)

        if step_ >= self.max_step:
            reward = -1 * self.penalty['step_over']
            terminated = True
        
        observation = get_observation(self.game)
        action_mask = self.get_action_mask()
        
        # Concatenate observation and action mask
        obs = np.concatenate([observation, action_mask])
        
        vp_gained = action_result['victory_point'] - self.previous_vp

        info = {
            "is_noble_visit": is_noble_visit,
            "is_get_card": action_result.get('is_get_card', False),
            "victory_point": action_result['victory_point'],
            "vp_gained": vp_gained,
            "step": step_,
            "seed": getattr(self, 'seed', None)
        }

        if self.verbose:
            self.logger.info(f"Step {step_}: Action {action}, Reward {reward:.2f}")
            if is_noble_visit:
                self.logger.info("Noble visited!")
        
        return obs, reward, terminated, truncated, info

    def get_possible_actions(self):
        return self.game.player1.get_all_possible_actions(self.game.board)

    def _run_action(self, action):
        terminated = False
        action_result = self.game.run_with_action(action)
        step_ = action_result['step']

        reward = 0
        reward -= self.penalty['step']

        is_noble_visit = action_result['is_noble_visit']

        # Calculate VP progress
        vp_gained = action_result['victory_point'] - self.previous_vp
        reward += vp_gained * self.reward['vp_progress']
        self.previous_vp = action_result['victory_point']

        if action_result['victory_point'] >= self.target_vp:
            reward = self.reward['reach_goal']
            terminated = True

        return reward, terminated, step_, is_noble_visit, action_result

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

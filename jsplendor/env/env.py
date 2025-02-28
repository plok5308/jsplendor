import numpy as np
import gymnasium as gym
from gymnasium import spaces

from jsplendor.game import Game
from jsplendor.env.observation import get_observation_space, get_observation, HIGH_VALUE
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
            high=HIGH_VALUE,
            shape=(obs_space.shape[0] + action_n,),  # 175 + 27 = 202
            dtype=np.int32
        )
        self.skip_sum = 0

        # parameters
        self.target_vp = 15
        self.max_step = 127
        self.bonus_step = 35
        self.l1_penalty_step = 13

        # Penalties
        self.penalty = dict()
        self.penalty['step'] = 1
        self.penalty['step2'] = 2
        self.penalty['over_coin'] = 1
        self.penalty['step_over'] = 100
        self.penalty['l1_card'] = 3
        
        # Rewards
        self.reward = dict()
        self.reward['reach_goal'] = 50
        self.reward['bonus'] = 10

        # Additional tracking
        self.previous_vp = 0  # New: track VP changes

    def get_action_mask(self):
        """Convert possible actions to a binary mask"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        possible_actions = self.get_possible_actions()
        mask[possible_actions] = 1.0
        return mask

    def step(self, action):
        if self.verbose:
            self.logger.info("\n" + "="*50)
            self.logger.info(f"Step {self.game.step + 1}")
            self.logger.info("-"*30)
        
        terminated = False
        truncated = False

        reward, terminated, step_, is_noble_visit, action_result = self._run_action(action)

        if step_ >= self.max_step:
            reward += -1 * self.penalty['step_over']
            terminated = True

        if (action_result['buy_l1_card'] and (step_ >= self.l1_penalty_step)):
            reward += -1 * self.penalty['l1_card']
        
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
            # Only log step-related information, not probabilities
            if terminated:
                self.logger.info("-"*30)
                if self.game.player1.sum_victory_point >= self.target_vp:
                    self.logger.info("Game finished! Victory achieved!")
                else:
                    self.logger.info("Game terminated (max steps reached)")
                self.logger.info("="*50 + "\n")
        
        return obs, reward, terminated, truncated, info

    def get_possible_actions(self):
        return self.game.player1.get_all_possible_actions(self.game.board)

    def _run_action(self, action):
        terminated = False
        action_result = self.game.run_with_action(action)
        step_ = action_result['step']

        reward = 0
        if step_ > self.bonus_step:
            over_step = step_ - self.bonus_step
            reward -= self.penalty['step2']
        else:
            reward -= self.penalty['step']

        if action_result['over_coin_count'] > 0:
            reward -= action_result['over_coin_count'] * self.penalty['over_coin']

        is_noble_visit = action_result['is_noble_visit']

        if action_result['victory_point'] >= self.target_vp:
            reward += self.reward['reach_goal']

            if step_ < self.bonus_step:
                bonus_scale = self.bonus_step - step_
                reward += bonus_scale * self.reward['bonus']

            terminated = True

        return reward, terminated, step_, is_noble_visit, action_result

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.skip_sum = 0
        self.previous_vp = 0
        self.game.reset()
        
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

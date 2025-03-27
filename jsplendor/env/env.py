import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy

from jsplendor.game import Game
from jsplendor.env.observation import get_observation_space, get_observation, HIGH_VALUE
from jsplendor.utils.config import get_verbose_dict
from jsplendor.utils import TestLogger


class SelfPlayEnv(gym.Env):
    """Two player environment with random starting positions"""
    def __init__(self, opponent_policy=None, verbose_dict=None):
        if verbose_dict is None:
            verbose_dict = get_verbose_dict()
        
        self.verbose = verbose_dict['env']
        if self.verbose:
            self.logger = TestLogger('logs/env')
            
        # Store opponent policy
        self.opponent_policy = opponent_policy
            
        # Initialize game
        self.game = Game(verbose_dict)  # This will create player1
        
        # Add second player if not already present
        if len(self.game.players) == 1:
            self.game.add_player("player2")
        
        # Verify both players are initialized
        assert len(self.game.players) == 2, "Game must have exactly 2 players"
                
        # Action and observation spaces
        action_n = self.game.players[0].num_actions
        self.action_space = spaces.Discrete(action_n)
        
        obs_space = get_observation_space(self.game)
        self.observation_space = spaces.Box(
            low=0,
            high=HIGH_VALUE,
            shape=(obs_space.shape[0],),  # Use dynamic shape from observation space
            dtype=np.int32
        )

        # Parameters
        self.target_vp = 15
        self.max_step = 300

    def get_action_mask(self, player_idx=0):
        """Get action mask for specified player"""
        return self.game.players[player_idx].get_all_possible_actions(self.game.board)  # Use player's method directly

    def get_env_observation(self, player_idx=0):
        """Get environment observation from specified player's perspective, including action mask"""
        # Get observation containing both players' states
        obs = get_observation(self.game, player_idx)

        # Get action mask for current player
        action_mask = self.get_action_mask(player_idx)
        
        # Combine observation and action mask
        final_obs = np.concatenate([obs, action_mask])
        
        return final_obs

    def _step_opponent(self, opponent_idx):
        opponent_player = self.game.players[opponent_idx]

        opponent_obs = self.get_env_observation(opponent_idx)
        opponent_action = self.opponent_policy(opponent_obs)
        opponent_vp, _, _, _ = opponent_player.do_action(self.game.board, opponent_action)
        return opponent_vp

    def step(self, action):
        # Execute learning player's action
        player_idx = 0 if self.player_starts_first else 1
        opponent_idx = 1 - player_idx
        current_player = self.game.players[player_idx]
        opponent_player = self.game.players[opponent_idx]
            
        # Execute player's action
        action_result = current_player.do_action(self.game.board, action)

        # Execute opponent's action for this turn
        if not self.player_starts_first:
            # Check game end before opponent's next action
            terminated, reward, info = self._check_game_end(
                current_player.sum_victory_point,
                opponent_player.sum_victory_point
            )
            if terminated:
                observation = self.get_env_observation(player_idx)
                return observation, reward, terminated, False, info

        # Execute opponent's action
        opponent_vp = self._step_opponent(opponent_idx)

        # Final game end check
        terminated, reward, info = self._check_game_end(
            current_player.sum_victory_point,
            opponent_player.sum_victory_point
        )
        
        # Get next observation
        observation = self.get_env_observation(player_idx)

        # Add steps info to info dictionary
        info['steps'] = {
            "player0": self.game.players[0].step,
            "player1": self.game.players[1].step
        }

        return observation, reward, terminated, False, info

    def _check_game_end(self, player_vp, opponent_vp):
        terminated = False
        reward = 0
        info = {}

        if opponent_vp is None:
            assert not (self.player_starts_first), "Opponent VP must be provided if player goes first"

        # Use max of both players' steps to check max steps
        max_steps = max(self.game.players[0].step, self.game.players[1].step)
        if max_steps >= self.max_step:
            terminated = True
            reward = -1
            info['winner'] = 'not terminated'
            info['reward'] = reward
            info['steps'] = {  # Add steps info here too
                "player0": self.game.players[0].step,
                "player1": self.game.players[1].step
            }
            return terminated, reward, info
        else:
            if (player_vp >= self.target_vp or opponent_vp >= self.target_vp):
                terminated = True
                if player_vp > opponent_vp:
                    reward = 1
                elif player_vp < opponent_vp:
                    reward = -1
                else:
                    reward = 0

                info['winner'] = 'player' if player_vp > opponent_vp else 'opponent'
                info['reward'] = reward
                info['steps'] = {  # Add steps info here too
                    "player0": self.game.players[0].step,
                    "player1": self.game.players[1].step
                }
            
            return terminated, reward, info

    def reset(self, seed=None, options=None):
        """Reset environment and randomly determine player order"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game first
        self.game.reset()
        
        # Make sure second player is added after reset
        if len(self.game.players) == 1:
            self.game.add_player("player2")
        
        # Randomly decide if trained agent starts first
        self.player_starts_first = bool(np.random.randint(2))

        player_idx = 0 if self.player_starts_first else 1
        
        # If player goes second, let opponent make first move
        if not self.player_starts_first:
            opponent_obs = self.get_env_observation(0)  # Get observation for opponent
            opponent_action = self.opponent_policy(opponent_obs)
            
            if self.verbose:
                self.logger.info("\n" + "="*50)
                self.logger.info("Opponent's First Move:")
                self.logger.info(f"Selected action: {opponent_action}")
                self.logger.info(f"Opponent coins: {self.game.players[0].coins}")
                self.logger.info(f"Valid actions: {np.where(self.get_action_mask(0))[0]}")
            
            # Execute opponent's action
            self.game.players[0].do_action(self.game.board, opponent_action)
        
        # Get observation from correct perspective
        observation = self.get_env_observation(player_idx)
        info = {
            "starts_first": self.player_starts_first,
            "steps": {  # Add both players' steps to info
                "player0": self.game.players[0].step,
                "player1": self.game.players[1].step
            }
        }
        
        if self.verbose:
            self.logger.info("\nReset State:")
            self.logger.info(f"Player starts first: {self.player_starts_first}")
            self.logger.info(f"Player idx: {player_idx}")
            self.logger.info(f"Steps - Player 0: {self.game.players[0].step}, Player 1: {self.game.players[1].step}")
            self.logger.info("="*50)
        
        return observation, info

    def render(self):
        pass

    def close(self):
        pass 
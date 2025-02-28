import numpy as np
import torch
from copy import deepcopy
from jsplendor.utils import Element
from stable_baselines3 import PPO

class TestAgent:
    """Agent that selects winning moves when available, otherwise follows PPO policy"""
    
    def __init__(self, env, ppo_model):
        self.env = env
        self.target_vp = env.target_vp
        self.policy = ppo_model.policy  # We only need the policy
        self.device = self.policy.device

    def predict(self, obs, deterministic=False):
        """Check for winning move, otherwise use PPO policy"""
        valid_actions = np.where(self.env.get_action_mask())[0]
        
        # Only check for winning moves after step 20
        if self.env.game.step >= 20:
            # First check if any action leads to immediate victory
            for action in valid_actions:
                if self._is_winning_action(action):
                    return int(action), None  # Convert to scalar integer
        
        # If no winning move or before step 20, use PPO policy
        with torch.no_grad():
            # Add batch dimension for transformer
            obs_tensor = torch.as_tensor(obs).to(self.device).unsqueeze(0)
            actions, _, _ = self.policy.forward(obs_tensor, deterministic)
            # Convert to scalar integer
            action = int(actions.cpu().numpy().squeeze())
            return action, None

    def _is_winning_action(self, action):
        """Check if an action leads to victory"""
        if action < 15:  # Only card buying actions can lead to victory
            return False
            
        # Create a copy of the game to simulate the action
        game_copy = deepcopy(self.env.game)
        
        # Simulate the action using game mechanics
        action_result = game_copy.run_with_action(action)
        
        return action_result['victory_point'] >= self.target_vp

    # Forward all other method calls to PPO policy
    def __getattr__(self, name):
        return getattr(self.policy, name)
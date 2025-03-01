import numpy as np

class RandomPlayer:
    """A player that selects random valid actions"""
    
    def __init__(self):
        self.name = "RandomPlayer"
    
    def __call__(self, obs):
        """Select a random valid action from the action mask
        
        Args:
            obs: Observation array where last 27 values are the action mask
        
        Returns:
            int: Selected action index
        """
        # Last 27 values are action mask
        action_mask = obs[-27:]
        valid_actions = np.where(action_mask > 0)[0]
        return np.random.choice(valid_actions) 
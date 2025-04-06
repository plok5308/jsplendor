import numpy as np

class RandomPlayer:
    """A player that selects random valid actions"""
    
    def __init__(self, action_num: int = 43):
        self.name = "RandomPlayer"
        self.action_num = action_num
    
    def __call__(self, observation):
        """Legacy method for compatibility"""
        return self.predict(observation)[0]

    def predict(self, observation, deterministic=True):
        """Match PPO agent's predict interface
        
        Args:
            observation: Environment observation
            deterministic: Ignored for random agent
            
        Returns:
            Tuple of (action, None) to match PPO interface
        """
        # Get action mask from observation
        if isinstance(observation, np.ndarray):
            action_mask = observation[-self.action_num:]  # Last 42 elements are action mask (updated for reserve actions)
        else:
            action_mask = observation['action_mask']
            
        # Get valid actions
        valid_actions = np.where(action_mask)[0]
        
        # Select random valid action
        action = int(np.random.choice(valid_actions))  # Convert to int
        
        return action, None  # Return None for state to match PPO interface
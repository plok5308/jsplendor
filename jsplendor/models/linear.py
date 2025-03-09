import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LinearFeatureExtractor(BaseFeaturesExtractor):
    """Simple linear feature extractor that matches transformer's input/output structure"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        # Initialize with observation space and features dimension
        super().__init__(observation_space, features_dim)
        
        # Store dimensions
        self.full_dim = observation_space.shape[0]
        self.obs_dim = self.full_dim - 27  # Remove action mask size
        
        # Simple linear layers
        self.linear = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
        )
        
        print('Linear feature extractor initialized with:')
        print(f' - Full input dim: {self.full_dim}')
        print(f' - Observation dim: {self.obs_dim}')
        print(f' - Output dim: {features_dim}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure input is float tensor
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Handle both full observations and pre-split observations
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim]  # Remove action mask
            
        # Process through linear layers
        features = self.linear(observations)
        
        return features
    
    def _get_dummy_data(self, observation_space, batch_size: int = 1) -> torch.Tensor:
        """Create dummy data for shape inference"""
        shape = (batch_size,) + observation_space.shape
        dummy_data = torch.ones(shape)
        return dummy_data 
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LinearFeatureExtractor(BaseFeaturesExtractor):
    """Simple linear feature extractor with skip connection"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, action_num: int = 43):
        # Initialize with observation space and features dimension
        super().__init__(observation_space, features_dim)
        
        print('Linear2 feature extractor initialized with:')
        # Store dimensions
        self.full_dim = observation_space.shape[0]
        self.obs_dim = self.full_dim - action_num  # Remove action mask size
        
        # First layer
        self.linear1 = nn.Linear(self.obs_dim, 256)  # Wider first layer
        self.norm1 = nn.LayerNorm(256)
        
        # Second layer
        self.linear2 = nn.Linear(256, 256)  # Keep width in hidden layer
        self.norm2 = nn.LayerNorm(256)
        
        # Output projection
        self.output = nn.Linear(256, features_dim)
        self.norm_out = nn.LayerNorm(features_dim)
        
        # Activation
        self.activation = nn.GELU()
        
        print('Linear feature extractor initialized with:')
        print(f' - Full input dim: {self.full_dim}')
        print(f' - Observation dim: {self.obs_dim}')
        print(f' - Hidden dim: 256')
        print(f' - Output dim: {features_dim}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure input is float tensor
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Handle both full observations and pre-split observations
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim]  # Remove action mask
        
        # First layer
        x = self.linear1(observations)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Second layer with skip connection
        identity = x
        x = self.linear2(x)
        x = self.norm2(x)
        x = x + identity  # Add skip connection
        x = self.activation(x)
        
        # Project to output dimension
        identity = x
        x = self.output(x)
        x = self.norm_out(x)
        x = x + identity
        x = self.activation(x)
        
        return x
    
    def _get_dummy_data(self, observation_space, batch_size: int = 1) -> torch.Tensor:
        """Create dummy data for shape inference"""
        shape = (batch_size,) + observation_space.shape
        dummy_data = torch.ones(shape)
        return dummy_data 
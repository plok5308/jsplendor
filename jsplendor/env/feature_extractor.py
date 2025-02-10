import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from x_transformers import Encoder


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=128)
        
        # Linear projection instead of embedding
        self.input_projection = nn.Linear(1, 128)  # Project each scalar to embedding dim
        
        self.transformer = Encoder(
            dim = 128, # embedding dimension
            depth = 4, # number of layers
            heads = 4 # number of attention heads
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape to [batch, sequence_length, 1]
        x = observations.unsqueeze(-1)
        
        # Project each scalar to embedding dimension
        x = self.input_projection(x)  # Shape: [batch, sequence_length, 128]
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Use CLS token (first position) as output
        x = x[:, 0, :]  # [batch, 128]
        return x

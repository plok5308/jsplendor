import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from x_transformers import TransformerWrapper, Encoder

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=32)
        
        # Layer normalization before transformer
        self.layer_norm = nn.LayerNorm(32)
        
        # Calculate total sequence length (observation + action mask)
        total_seq_len = observation_space.shape[0]  # This includes both obs and action mask
        
        self.transformer = TransformerWrapper(
            num_tokens = 32,    # vocabulary size
            max_seq_len = total_seq_len,
            attn_layers = Encoder(
                dim = 64,        
                depth = 2,       
                heads = 2,        
            )
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert to long for embedding layer
        x = observations.long()
        
        # Pass through transformer (handles embedding internally)
        x = self.transformer(x)
        
        # Use first token as output
        x = x[:, 0, :]
        return x 

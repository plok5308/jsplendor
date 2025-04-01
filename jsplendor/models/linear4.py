import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = self.linear(x)
        residual = self.norm(residual)
        residual = self.activation(residual)
        return x + residual


class BoardFeatureBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = ResidualBlock(output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.residual(x)
        return x


class PlayerFeatureBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = ResidualBlock(output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.residual(x)
        return x


class CombinedFeatureBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = x + residual

        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class LinearFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor for Splendor game state"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, action_num: int = 43):
        super().__init__(observation_space, features_dim)
        
        print('Linear4 feature extractor initialized with:')
        # Store dimensions
        self.full_dim = observation_space.shape[0]
        self.obs_dim = self.full_dim - action_num
        
        # Split observation into meaningful chunks
        self.board_size = 125  
        self.player_size = 155
        self.single_step_obs_size = self.board_size + self.player_size*2
        
        # Feature processing blocks
        self.feature_block1 = BoardFeatureBlock(self.board_size, 256)
        self.feature_block2 = PlayerFeatureBlock(self.player_size, 128)
        self.feature_block3 = PlayerFeatureBlock(self.player_size, 128)

        self.combined_block = CombinedFeatureBlock(1024, features_dim)
        
        print('Enhanced feature extractor initialized with:')
        print(f' - Full input dim: {self.full_dim}')
        print(f' - Observation dim: {self.obs_dim}')
        print(f' - Board size: {self.board_size}')
        print(f' - Player size: {self.player_size}')
        print(f' - Output dim: {features_dim}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype != torch.float32:
            observations = observations.float()
            
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim]

        # Reshape to [batch_size, channels, sequence_length]
        x = observations.view(-1, 2, self.single_step_obs_size)

        # Split observation into components
        obs1 = x[..., 0, :self.board_size] 
        obs2 = x[..., 0, self.board_size:self.board_size+self.player_size]
        obs3 = x[..., 0, self.board_size+self.player_size:]

        obs4 = x[..., 1, :self.board_size] 
        obs5 = x[..., 1, self.board_size:self.board_size+self.player_size]
        obs6 = x[..., 1, self.board_size+self.player_size:]
        
        # Process features through blocks
        obs1_features = self.feature_block1(obs1)
        obs2_features = self.feature_block2(obs2)
        obs3_features = self.feature_block3(obs3)

        obs4_features = self.feature_block1(obs4)
        obs5_features = self.feature_block2(obs5)
        obs6_features = self.feature_block3(obs6)
        
        # Combine and process features
        combined = torch.cat([obs1_features, obs2_features, obs3_features, obs4_features, obs5_features, obs6_features], dim=-1)
        output = self.combined_block(combined)
        
        return output
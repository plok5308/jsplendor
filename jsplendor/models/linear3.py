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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.residual = ResidualBlock(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.residual(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class LinearFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor for Splendor game state"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, action_num: int = 43):
        super().__init__(observation_space, features_dim)
        
        print('Linear3 feature extractor initialized with:')
        # Store dimensions
        self.full_dim = observation_space.shape[0]
        self.obs_dim = self.full_dim - action_num
        
        # Split observation into meaningful chunks
        self.board_size = 125  
        self.player_size = 155
        
        # Feature processing blocks
        self.board_block = BoardFeatureBlock(self.board_size, 256)
        self.player1_block = PlayerFeatureBlock(self.player_size, 128)
        self.player2_block = PlayerFeatureBlock(self.player_size, 128)
        self.combined_block = CombinedFeatureBlock(512, 512, features_dim)
        
        print('Enhanced feature extractor initialized with:')
        print(f' - Full input dim: {self.full_dim}')
        print(f' - Observation dim: {self.obs_dim}')
        print(f' - Board size: {self.board_size}')
        print(f' - Player size: {self.player_size}')
        print(f' - Output dim: {features_dim}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Handle input formatting
        if observations.dtype != torch.float32:
            observations = observations.float()
            
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim]
        
        # Split observation into components
        board_obs = observations[..., 1:self.board_size+1]  # Skip CLS token
        player1_obs = observations[..., self.board_size+1:self.board_size+1+self.player_size]
        player2_obs = observations[..., self.board_size+1+self.player_size:]
        
        # Process features through blocks
        board_features = self.board_block(board_obs)
        player1_features = self.player1_block(player1_obs)
        player2_features = self.player2_block(player2_obs)
        
        # Combine and process features
        combined = torch.cat([board_features, player1_features, player2_features], dim=-1)
        output = self.combined_block(combined)
        
        return output
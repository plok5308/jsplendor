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


# class CombinedFeatureBlock(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, input_dim)
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.linear2 = nn.Linear(input_dim, output_dim)
#         self.norm2 = nn.LayerNorm(output_dim)
#         self.activation = nn.GELU()
        
#     def forward(self, x):
#         residual = x
#         x = self.linear1(x)
#         x = self.norm1(x)
#         x = self.activation(x)
#         x = x + residual

#         x = self.linear2(x)
#         x = self.norm2(x)
#         x = self.activation(x)
#         return x


class Conv1DResidualBlock(nn.Module):
    def __init__(self, channels, seq_len, stride=2):
        super().__init__()
        kernel_size = 5  # Increased kernel size for wider receptive field
        padding = kernel_size // 2
        
        # Calculate output sequence length after stride
        output_seq_len = (seq_len + 2 * padding - kernel_size) // stride + 1
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.LayerNorm([channels, output_seq_len])
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm2 = nn.LayerNorm([channels, output_seq_len])
        self.activation = nn.GELU()
        
        # Downsample residual connection if using stride
        self.downsample = nn.Conv1d(channels, channels, kernel_size=1, stride=stride) if stride > 1 else None
        
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + residual)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor for Splendor game state"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, action_num: int = 43):
        super().__init__(observation_space, features_dim)
        
        print('Conv1D feature extractor initialized with:')
        # Store dimensions
        self.full_dim = observation_space.shape[0]
        self.obs_dim = self.full_dim - action_num
        self.single_step_obs_size = 435  # 3 x 435 features
        
        # Initial embedding with larger kernel
        kernel_size = 5
        padding = kernel_size // 2
        self.embed = nn.Conv1d(3, 64, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm([64, 435])
        
        # Calculate sequence lengths after each strided block
        current_size = 435
        seq_lengths = [current_size]
        for _ in range(8):
            current_size = (current_size + 2 * padding - kernel_size) // 2 + 1
            seq_lengths.append(current_size)
        
        print(f"Sequence lengths: {seq_lengths}")
        
        # Multiple residual blocks with stride
        self.residual_blocks = nn.ModuleList([
            Conv1DResidualBlock(64, seq_len, stride=2 if i < len(seq_lengths)-2 else 1)
            for i, seq_len in enumerate(seq_lengths[:-1])
        ])
        
        # Final processing
        self.final_linear = nn.Linear(64*4, features_dim)
        
        print(f' - Full input dim: {self.full_dim}')
        print(f' - Observation dim: {self.obs_dim}')
        print(f' - Output dim: {features_dim}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype != torch.float32:
            observations = observations.float()
            
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim]

        # Reshape to [batch_size, channels, sequence_length]
        x = observations.view(-1, 3, self.single_step_obs_size)
        
        # Initial embedding
        x = self.embed(x)  # Shape: [batch_size, 64, 435]
        x = self.norm(x)
        
        # Process through residual blocks with progressive downsampling
        for block in self.residual_blocks:
            x = block(x)

        x = x.view(-1, 64*4)
        
        # Final projection
        x = self.final_linear(x)  # Shape: [batch_size, features_dim]
        
        return x
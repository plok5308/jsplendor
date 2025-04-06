import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self attention
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class LinearFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 action_num: int = 43, history_length: int = 3, embed_dim: int = 192):
        super().__init__(observation_space, features_dim)
        
        # Dimensions
        self.full_dim = observation_space.shape[0]
        self.history_length = history_length
        self.obs_dim = (self.full_dim - action_num) // history_length
        self.embed_dim = embed_dim
        
        # Component sizes
        self.board_size = 125
        self.player_size = 155
        
        # Component embeddings (all to same dimension)
        self.board_embed = nn.Sequential(
            nn.Linear(self.board_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        self.player_embed = nn.Sequential(
            nn.Linear(self.player_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Position embeddings for different components and timesteps
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 3 * history_length, embed_dim)  # 3 components * history_length
        )
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            TransformerBlock(embed_dim),
            TransformerBlock(embed_dim)
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(embed_dim * 3 * history_length, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )
        
        print(f'Enhanced feature extractor initialized with:')
        print(f' - Embedding dimension: {embed_dim}')
        print(f' - History length: {history_length}')
        print(f' - Total sequence length: {3 * history_length}')
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Remove action mask if present
        if observations.shape[-1] == self.full_dim:
            observations = observations[..., :self.obs_dim * self.history_length]
        
        batch_size = observations.shape[0]
        
        # Reshape to (batch, history, obs_dim)
        observations = observations.view(batch_size, self.history_length, -1)
        
        # Process each timestep and component
        sequence = []
        
        for t in range(self.history_length):
            obs_t = observations[:, t]
            
            # Split and embed components
            board_t = obs_t[..., 1:self.board_size+1]
            player1_t = obs_t[..., self.board_size+1:self.board_size+1+self.player_size]
            player2_t = obs_t[..., self.board_size+1+self.player_size:]
            
            # Embed each component to same dimension
            board_embed = self.board_embed(board_t)
            player1_embed = self.player_embed(player1_t)
            player2_embed = self.player_embed(player2_t)
            
            # Add to sequence
            sequence.extend([board_embed, player1_embed, player2_embed])
        
        # Stack all embeddings into sequence
        # Shape: (batch_size, 3*history_length, embed_dim)
        sequence = torch.stack(sequence, dim=1)
        
        # Add positional embeddings
        sequence = sequence + self.pos_embedding
        
        # Apply transformer
        transformed = self.transformer(sequence)
        
        # Combine all features
        combined = transformed.flatten(start_dim=1)
        
        # Project to final dimension
        output = self.output(combined)
        
        return output
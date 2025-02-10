import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
import numpy as np

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs, deterministic=False):
        """Forward pass in all the networks (actor and critic)"""
        # Convert tensor observation to expected format
        if isinstance(obs, torch.Tensor):
            obs_dict = {
                'obs': obs[..., :-self.action_space.n],  # All features except action mask
                'action_mask': obs[..., -self.action_space.n:]  # Last n elements are action mask
            }
        else:
            obs_dict = obs

        # Extract observation and action mask
        action_mask = obs_dict['action_mask']
        obs = obs_dict['obs']

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Apply action mask
        distribution.distribution.logits = torch.where(
            action_mask.bool(),
            distribution.distribution.logits,
            torch.tensor(-1e+8).to(distribution.distribution.logits.device)
        )

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False):
        """Override the prediction method to handle action masking during rollouts"""
        # Convert numpy arrays to torch tensors
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation).float()
        
        # Convert tensor observation to expected format
        if isinstance(observation, torch.Tensor):
            obs_dict = {
                'obs': observation[..., :-self.action_space.n],
                'action_mask': observation[..., -self.action_space.n:]
            }
        else:
            obs_dict = observation

        action_mask = obs_dict['action_mask']
        obs = obs_dict['obs']

        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply action mask
        distribution.distribution.logits = torch.where(
            action_mask.bool(),
            distribution.distribution.logits,
            torch.tensor(-1e+8).to(distribution.distribution.logits.device)
        )

        if deterministic:
            # For deterministic actions, get the action with highest probability among valid actions
            logits = distribution.distribution.logits
            masked_logits = torch.where(
                action_mask.bool(),
                logits,
                torch.tensor(float('-inf')).to(logits.device)
            )
            action = torch.argmax(masked_logits, dim=1)
        else:
            action = distribution.sample()

        # Return the action as a torch tensor
        return action

    def evaluate_actions(self, obs, actions):
        """Evaluate actions according to the current policy"""
        # Convert tensor observation to expected format
        if isinstance(obs, torch.Tensor):
            obs_dict = {
                'obs': obs[..., :-self.action_space.n],
                'action_mask': obs[..., -self.action_space.n:]
            }
        else:
            obs_dict = obs

        action_mask = obs_dict['action_mask']
        obs = obs_dict['obs']

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)

        # Apply action mask
        distribution.distribution.logits = torch.where(
            action_mask.bool(),
            distribution.distribution.logits,
            torch.tensor(-1e+8).to(distribution.distribution.logits.device)
        )

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy 
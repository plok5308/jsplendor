import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
import numpy as np

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, min_prob=1e-3, temperature=1.0, top_k=0, top_p=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_prob = min_prob
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

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
        
        # Apply action mask with minimum probability
        masked_logits = torch.where(
            action_mask.bool(),
            torch.maximum(distribution.distribution.logits, 
                         torch.log(torch.tensor(self.min_prob).to(distribution.distribution.logits.device))),
            torch.tensor(-1e+8).to(distribution.distribution.logits.device)
        )
        distribution.distribution.logits = masked_logits

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False):
        """Override the prediction method to handle action masking and sampling methods"""
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

        # Apply temperature scaling
        logits = distribution.distribution.logits / self.temperature

        # Apply action mask
        logits = torch.where(
            action_mask.bool(),
            logits,
            torch.tensor(float('-inf')).to(logits.device)
        )

        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            # Apply top-k filtering
            if self.top_k > 0:
                values, indices = torch.topk(logits, min(self.top_k, logits.shape[-1]))
                logits[logits < values[..., [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

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
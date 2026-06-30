from __future__ import annotations

from typing import Any

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs


class DuelingQNetwork(BasePolicy):
    """Dueling Q network for discrete actions.

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)

        if len(net_arch) == 0:
            latent_dim = features_dim
            self.shared_net = nn.Identity()
        else:
            shared_layers = create_mlp(features_dim, net_arch[-1], net_arch[:-1], activation_fn)
            self.shared_net = nn.Sequential(*shared_layers)
            latent_dim = net_arch[-1]

        value_layers = create_mlp(latent_dim, 1, [], activation_fn)
        advantage_layers = create_mlp(latent_dim, action_dim, [], activation_fn)
        self.value_net = nn.Sequential(*value_layers)
        self.advantage_net = nn.Sequential(*advantage_layers)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        latent = self.shared_net(features)
        value = self.value_net(latent)
        advantage = self.advantage_net(latent)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class CustomDuelingDQNPolicy(DQNPolicy):
    """DQN policy that uses DuelingQNetwork for both online and target q nets."""

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)

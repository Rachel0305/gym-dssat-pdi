from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


STAGE_DAPS = (1, 30, 50, 65, 85, 110)


class StageQNetwork(nn.Module):
    """The unchanged 022-series 25-64-64-9 Q-network architecture."""

    def __init__(self, input_dim: int = 25, output_dim: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class StageSeparatedQEnsemble(nn.Module):
    """Six parameter-independent Q-networks dispatched by exact stage DAP."""

    def __init__(self, input_dim: int = 25, output_dim: int = 9) -> None:
        super().__init__()
        self.networks = nn.ModuleDict(
            {str(dap): StageQNetwork(input_dim=input_dim, output_dim=output_dim) for dap in STAGE_DAPS}
        )

    def network_for_stage(self, dap: int) -> StageQNetwork:
        dap = int(dap)
        if dap not in STAGE_DAPS:
            raise ValueError(f"DAP {dap} is not an executable fixed stage; expected one of {STAGE_DAPS}")
        return self.networks[str(dap)]

    def initialize_from_single_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        for dap in STAGE_DAPS:
            self.network_for_stage(dap).load_state_dict(state_dict, strict=True)

    def forward(self, observations: torch.Tensor, daps: torch.Tensor | Iterable[int]) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(f"observations must be rank 2, got shape={tuple(observations.shape)}")
        dap_tensor = torch.as_tensor(daps, device=observations.device, dtype=torch.long)
        if dap_tensor.ndim != 1 or len(dap_tensor) != len(observations):
            raise ValueError("daps must be a one-dimensional vector aligned with observations")
        unknown = sorted(set(int(value) for value in dap_tensor.detach().cpu().tolist()) - set(STAGE_DAPS))
        if unknown:
            raise ValueError(f"Unknown stage DAP values: {unknown}")
        outputs = observations.new_empty((len(observations), 9))
        for dap in STAGE_DAPS:
            mask = dap_tensor.eq(dap)
            if bool(mask.any()):
                outputs[mask] = self.network_for_stage(dap)(observations[mask])
        return outputs


def stage_parameter_count(model: StageSeparatedQEnsemble, dap: int) -> int:
    return sum(parameter.numel() for parameter in model.network_for_stage(dap).parameters())

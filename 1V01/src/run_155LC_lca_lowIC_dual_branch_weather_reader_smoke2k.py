"""155LC: transfer the frozen SY E2 weather reader to LCA/LC, 2K smoke only."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract


CONFIG = ROOT / "configs/155LC_lca_lowIC_dual_branch_weather_reader_smoke2k.json"
REFERENCE_CONFIG = ROOT / "configs/053_00_lca_lowIC_expanded_action_maskableppo.json"
PROMPT = ROOT / "prompts/2026-08-16_lc_E2_dual_branch_weather_reader_smoke2k.md"
EXPECTED_TASK_ID = "155LC"


class DualBranchBaseWeatherExtractor(BaseFeaturesExtractor):
    """Encode LC's 26 base values and the 12 E2 weather values independently."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        base_observation_dim: int = 26,
        weather_observation_dim: int = 12,
        base_latent_dim: int = 64,
        weather_latent_dim: int = 32,
    ) -> None:
        total_dim = int(np.prod(observation_space.shape))
        expected = int(base_observation_dim + weather_observation_dim)
        if total_dim != expected:
            raise ValueError(f"155LC expected {expected}=26+12 observation values, got {total_dim}")
        super().__init__(observation_space, features_dim=int(base_latent_dim + weather_latent_dim))
        self.base_observation_dim = int(base_observation_dim)
        self.weather_observation_dim = int(weather_observation_dim)
        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_observation_dim, int(base_latent_dim)),
            nn.LayerNorm(int(base_latent_dim)),
            nn.Tanh(),
        )
        self.weather_encoder = nn.Sequential(
            nn.Linear(self.weather_observation_dim, int(weather_latent_dim)),
            nn.LayerNorm(int(weather_latent_dim)),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1)
        base = flat[:, : self.base_observation_dim]
        weather = flat[:, self.base_observation_dim :]
        return torch.cat([self.base_encoder(base), self.weather_encoder(weather)], dim=1)


def architecture_kwargs() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    arch = cfg["policy_architecture"]
    return {
        "features_extractor_class": DualBranchBaseWeatherExtractor,
        "features_extractor_kwargs": {
            "base_observation_dim": int(arch["base_observation_dim"]),
            "weather_observation_dim": int(arch["weather_observation_dim"]),
            "base_latent_dim": int(arch["base_latent_dim"]),
            "weather_latent_dim": int(arch["weather_latent_dim"]),
        },
    }


@contextlib.contextmanager
def patched_maskableppo() -> Iterator[None]:
    """Inject the LC E2 extractor while preserving all existing PPO kwargs."""
    import sb3_contrib

    original = sb3_contrib.MaskablePPO
    extra_policy_kwargs = architecture_kwargs()

    class LCE2MaskablePPO(original):
        def __init__(self, policy, env, *args, **kwargs):
            policy_kwargs = copy.deepcopy(kwargs.pop("policy_kwargs", {}) or {})
            if "features_extractor_class" in policy_kwargs:
                raise ValueError("155LC refuses a second features_extractor_class")
            policy_kwargs.update(extra_policy_kwargs)
            kwargs["policy_kwargs"] = policy_kwargs
            super().__init__(policy, env, *args, **kwargs)

    LCE2MaskablePPO.__name__ = "LCE2MaskablePPO"
    LCE2MaskablePPO.__qualname__ = "LCE2MaskablePPO"
    LCE2MaskablePPO.__module__ = __name__
    sb3_contrib.MaskablePPO = LCE2MaskablePPO
    try:
        yield
    finally:
        sb3_contrib.MaskablePPO = original


def isolation_audit() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "lc_station_and_profile_registered": (cfg["station_code"], cfg["site"], cfg["input_profile"]) == ("LCA", "LC", "lowIC"),
        "seed_unchanged": cfg["seed"] == ref["seed"] == 0,
        "actions_unchanged": cfg["actions"] == ref["actions"],
        "years_unchanged": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"],
        "reward_and_safety_unchanged": cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "automatic_controls_false": cfg["scope"]["external_n"] is False and cfg["scope"]["native_dssat_automatic_irrigation"] is False,
        "base_observation_unchanged": cfg["observation_contract"]["base"] == ref["observation_contract"]["base"],
        "base_normalization_unchanged": cfg["observation_contract"]["normalization_enabled"] == ref["observation_contract"]["normalization_enabled"],
        "weather_contract_has_12_features": len(cfg["forecast_features"]["feature_names"]) == 12 and len(cfg["forecast_features"]["raw_feature_names"]) == 12,
        "weather_window_today_excluded": cfg["forecast_features"]["include_today_in_future_windows"] is False,
        "dual_branch_dimensions_registered": cfg["policy_architecture"]["base_observation_dim"] == 26 and cfg["policy_architecture"]["weather_observation_dim"] == 12 and cfg["policy_architecture"]["combined_features_dim"] == 96,
        "smoke_is_2k_only": cfg["training"] == {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]},
    }
    return {"reference_config": REFERENCE_CONFIG.relative_to(ROOT).as_posix(), "changed_factor": "SY_E2_weather_contract_and_policy_reader_transferred_to_LC", "checks": checks, "passed": all(checks.values())}


def architecture_smoke() -> dict[str, Any]:
    space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)
    extractor = DualBranchBaseWeatherExtractor(space)
    sample = torch.zeros((3, 38), dtype=torch.float32)
    sample[1, :26] = 1000.0
    sample[2, 26:] = 1.0
    with torch.no_grad():
        output = extractor(sample)
    finite = bool(torch.isfinite(output).all().item())
    return {"input_shape": list(sample.shape), "output_shape": list(output.shape), "features_dim": int(extractor.features_dim), "finite_for_zero_large_base_and_weather_samples": finite, "passed": bool(list(sample.shape) == [3, 38] and list(output.shape) == [3, 96] and finite)}


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    isolation = isolation_audit()
    arch_smoke = architecture_smoke()
    if not isolation["passed"] or not arch_smoke["passed"]:
        raise RuntimeError(json.dumps({"isolation": isolation, "architecture": arch_smoke}, ensure_ascii=False))
    patch_contract()
    with patched_maskableppo():
        result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, args.smoke, False)
    result["lc_e2_isolation_audit"] = isolation
    result["lc_e2_architecture_smoke"] = arch_smoke
    if args.smoke:
        out = ROOT / result["output_root"]
        audit_path = out / "155LC_architecture_isolation_audit.json"
        audit_path.write_text(json.dumps({"isolation": isolation, "architecture": arch_smoke}, ensure_ascii=False, indent=2), encoding="utf-8")
        result["lc_e2_architecture_isolation_audit_path"] = audit_path.relative_to(ROOT).as_posix()
        (out / "155LC_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

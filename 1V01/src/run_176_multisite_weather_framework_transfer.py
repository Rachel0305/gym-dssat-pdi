"""176: run the frozen SY weather reader on HLA/YCA/FQA with matched controls.

The runner is intentionally one-site-at-a-time.  It supports a 2K smoke and,
only after that site's paired forecast/no-forecast smoke gates pass, a 5K run.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import forecast_engineered_observation_056_057 as forecast
import ppo_safe_rendering
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_152E3_sya_dual_branch_noforecast_control import e3_audit, e3_calendar_audit, make_zero_weather_factory
from run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k import patched_maskableppo


PROMPT = ROOT / "prompts/2026-08-17_hl_yc_fq_sy_weather_framework_transfer_validation.md"
SPECS: dict[str, dict[str, Any]] = {
    "HLA": {"site": "HLA", "short": "HL", "base_dim": 24, "forecast_config": ROOT / "configs/170HLA_hla_lowIC_dual_branch_weather_reader_5k_seed1.json", "control_config": ROOT / "configs/171HLA_hla_lowIC_dual_branch_noforecast_control_5k_seed1.json"},
    "YCA": {"site": "YC", "short": "YC", "base_dim": 22, "forecast_config": ROOT / "configs/172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1.json", "control_config": ROOT / "configs/173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1.json"},
    "FQA": {"site": "FQ", "short": "FQ", "base_dim": 23, "forecast_config": ROOT / "configs/174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1.json", "control_config": ROOT / "configs/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1.json"},
}


class TransferDualBranchExtractor(BaseFeaturesExtractor):
    """Keep the 12-dim SY weather branch while respecting each site's base width."""

    def __init__(self, observation_space: gym.spaces.Box, base_observation_dim: int, weather_observation_dim: int = 12, base_latent_dim: int = 64, weather_latent_dim: int = 32) -> None:
        total_dim = int(np.prod(observation_space.shape))
        expected = int(base_observation_dim + weather_observation_dim)
        if total_dim != expected:
            raise ValueError(f"transfer extractor expected {expected}={base_observation_dim}+{weather_observation_dim}, got {total_dim}")
        super().__init__(observation_space, features_dim=int(base_latent_dim + weather_latent_dim))
        self.base_observation_dim = int(base_observation_dim)
        self.weather_observation_dim = int(weather_observation_dim)
        self.base_encoder = nn.Sequential(nn.Linear(self.base_observation_dim, int(base_latent_dim)), nn.LayerNorm(int(base_latent_dim)), nn.Tanh())
        self.weather_encoder = nn.Sequential(nn.Linear(self.weather_observation_dim, int(weather_latent_dim)), nn.LayerNorm(int(weather_latent_dim)), nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1)
        return torch.cat([self.base_encoder(flat[:, : self.base_observation_dim]), self.weather_encoder(flat[:, self.base_observation_dim :])], dim=1)


@contextlib.contextmanager
def patched_transfer_maskableppo(base_dim: int):
    import sb3_contrib

    original = sb3_contrib.MaskablePPO

    class TransferMaskablePPO(original):
        def __init__(self, policy, env, *args, **kwargs):
            policy_kwargs = copy.deepcopy(kwargs.pop("policy_kwargs", {}) or {})
            if "features_extractor_class" in policy_kwargs:
                raise ValueError("transfer runner refuses a second extractor")
            policy_kwargs.update({"features_extractor_class": TransferDualBranchExtractor, "features_extractor_kwargs": {"base_observation_dim": int(base_dim), "weather_observation_dim": 12, "base_latent_dim": 64, "weather_latent_dim": 32}})
            kwargs["policy_kwargs"] = policy_kwargs
            super().__init__(policy, env, *args, **kwargs)

    TransferMaskablePPO.__name__ = "TransferMaskablePPO"
    TransferMaskablePPO.__qualname__ = "TransferMaskablePPO"
    TransferMaskablePPO.__module__ = __name__
    sb3_contrib.MaskablePPO = TransferMaskablePPO
    try:
        yield
    finally:
        sb3_contrib.MaskablePPO = original


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_years(station: str) -> tuple[list[int], list[int]]:
    split = forecast.split_for_station(station)
    train = sorted(split.loc[split["split"].eq("train"), "year"].astype(int).tolist())
    valid = sorted(split.loc[split["split"].eq("validation"), "year"].astype(int).tolist())
    return train, valid


def validate_transfer_config(cfg: dict[str, Any], expected_task_id: str | None = None, return_audit: bool = False) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}")
    station = str(cfg["station_code"])
    if station not in SPECS or int(cfg.get("seed")) != 1:
        raise ValueError("Transfer runner only accepts HLA/YCA/FQA seed1 configs.")
    train, valid = expected_years(station)
    obs = cfg.get("observation_contract", {})
    arch = cfg.get("policy_architecture", {})
    checks = {
        "station_site_profile": (station, cfg.get("site"), cfg.get("input_profile")) == (station, SPECS[station]["site"], "originIC" if station == "FQA" else "lowIC"),
        "years_match_engine": cfg["scope"]["train_years"] == train and cfg["scope"]["validation_years"] == valid,
        "actions_match": cfg["actions"] == {"irrigation_levels_mm": [0.0, 15.0, 30.0, 45.0], "nitrogen_levels_kg_ha": [0.0, 40.0, 80.0, 120.0]},
        "reward_and_controls_unchanged": cfg["scope"]["reward_and_safety"] == "inherit_042_15_via_046_02_unchanged" and cfg["scope"]["external_n"] is False and cfg["scope"]["native_dssat_automatic_irrigation"] is False,
        "base_observation_unchanged": obs.get("base") == "046_02_raw_observation" and obs.get("normalization_enabled") is False,
        "architecture_matches": arch == {"name": "dual_branch_base_weather_layernorm", "base_observation_dim": SPECS[station]["base_dim"], "weather_observation_dim": 12, "base_latent_dim": 64, "weather_latent_dim": 32, "combined_features_dim": 96, "branch_activation": "Tanh", "branch_normalization": "LayerNorm", "downstream_net_arch": [64, 64]},
        "training_5k_contract": cfg["training"] == {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]},
    }
    if bool(obs.get("weather_forecast_enabled")):
        checks["weather_contract"] = cfg["forecast_features"].get("enabled") is True and len(cfg["forecast_features"].get("feature_names", [])) == 12 and obs.get("weather_forecast_mode") == "rolling_7d_actionable_prefix_encoding"
    else:
        checks["zero_weather_contract"] = cfg["forecast_features"].get("enabled") is False and obs.get("weather_forecast_mode") == "matched_architecture_zero_weather_control"
    if not all(checks.values()):
        raise ValueError(json.dumps({"task_id": cfg.get("task_id"), "checks": checks}, ensure_ascii=False))
    audit = {"reference_config": str(cfg.get("reference_run")), "changed_factor": "SY_weather_reader_enabled_vs_zero_weather_control", "checks": checks, "passed": True}
    return audit if return_audit else cfg


def site_preflight(cfg: dict[str, Any], prompt: Path) -> dict[str, Any]:
    station = str(cfg["station_code"])
    input_root = forecast.INPUT_PROFILES[str(cfg["input_profile"])]
    site_info = ppo_safe_rendering.SITE_INFO[station]
    source_mzx = input_root / site_info["short"] / site_info["template"]
    train, valid = expected_years(station)
    issues: list[str] = []
    if not input_root.exists():
        issues.append(f"input root missing: {input_root}")
    if not source_mzx.exists():
        issues.append(f"source MZX missing: {source_mzx}")
    if not prompt.exists():
        issues.append(f"prompt missing: {prompt}")
    if list(map(int, cfg["scope"]["train_years"])) != train:
        issues.append("train years mismatch")
    if list(map(int, cfg["scope"]["validation_years"])) != valid:
        issues.append("validation years mismatch")
    return {"task": f"{cfg['task_id']}_{cfg['task_name']}", "station_code": station, "site": cfg.get("site"), "input_profile": cfg["input_profile"], "resolved_input_root": input_root.relative_to(ROOT).as_posix(), "source_mzx": source_mzx.relative_to(ROOT).as_posix(), "config_train_years": train, "config_validation_years": valid, "reference_run": cfg.get("reference_run"), "issues": issues, "next_step_allowed": not issues}


def verify_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    smoke_cfg = forecast.prepare_config(cfg, smoke=True)
    out = forecast.output_root(smoke_cfg)
    result_path = out / f"{cfg['task_id']}_smoke_result.json"
    manifest_path = out / f"{cfg['task_id']}_run_manifest.json"
    checks: dict[str, Any] = {"smoke_result_exists": result_path.exists(), "smoke_manifest_exists": manifest_path.exists()}
    if all(checks.values()):
        result = read(result_path)
        manifest = read(manifest_path)
        gate = result.get("smoke_gate", result.get("action_forecast_gate", {}))
        checks.update({"status_completed_smoke": manifest.get("status") == "completed_smoke", "smoke_gate_passed": gate.get("next_step_allowed") is True, "isolation_passed": result.get("transfer_isolation_audit", {}).get("passed") is True})
    checks["next_step_allowed"] = all(bool(v) for k, v in checks.items() if k != "next_step_allowed")
    return checks


def write_record(cfg, pf, copied, audit, gate, phase, obs_smoke, calendar_audit, smoke_verification=None):
    out = forecast.output_root(cfg)
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [f"# {cfg['task_id']} {cfg['station_code']} SY weather framework transfer", "", f"- Phase: `{phase}`; output: `{out.relative_to(ROOT).as_posix()}`.", "- Reward, safety, action grid, station years and PPO settings are held fixed.", f"- Weather enabled: `{cfg['observation_contract']['weather_forecast_enabled']}`; mode: `{cfg['observation_contract']['weather_forecast_mode']}`.", "", "## gate", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    if smoke_verification is not None:
        lines.extend(["", "## smoke verification", "", *(f"- `{key}`: `{value}`" for key, value in smoke_verification.items())])
    lines.extend(["", "## action audit", "", audit.to_markdown(index=False), ""])
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run_one(station: str, kind: str, *, smoke: bool, formal: bool) -> dict[str, Any]:
    spec = SPECS[station]
    cfg_path = spec["forecast_config"] if kind == "forecast" else spec["control_config"]
    cfg = read(cfg_path)
    isolation = validate_transfer_config(cfg, return_audit=True)
    if formal:
        gate = verify_smoke(cfg)
        if not gate["next_step_allowed"]:
            raise RuntimeError(json.dumps(gate, ensure_ascii=False))
    patch_contract()
    originals = {"validate": forecast.validate_forecast_config, "preflight": forecast.preflight, "factory": forecast.make_forecast_env_factory, "calendar": forecast.forecast_calendar_audit, "audit": forecast.audit_actions_and_forecast_columns, "record": forecast.write_record, "verify": forecast.verify_completed_smoke}
    def validate(cfg2, expected_task_id=None):
        return validate_transfer_config(cfg2, expected_task_id)
    def preflight(cfg2, prompt2):
        return site_preflight(cfg2, prompt2)
    def verify(_cfg):
        return verify_smoke(cfg)
    forecast.validate_forecast_config = validate
    forecast.preflight = preflight
    forecast.verify_completed_smoke = verify
    if kind == "forecast":
        forecast.make_forecast_env_factory = originals["factory"]
        forecast.forecast_calendar_audit = originals["calendar"]
        forecast.audit_actions_and_forecast_columns = originals["audit"]
    else:
        forecast.make_forecast_env_factory = lambda base_make_env, _cfg: make_zero_weather_factory(base_make_env)
        forecast.forecast_calendar_audit = e3_calendar_audit
        forecast.audit_actions_and_forecast_columns = e3_audit
    forecast.write_record = write_record
    old_seed = forecast.engine.base03222.SEED
    forecast.engine.base03222.SEED = 1
    try:
        with patched_transfer_maskableppo(int(cfg["policy_architecture"]["base_observation_dim"])):
            result = forecast.run_forecast_experiment(cfg_path, PROMPT, str(cfg["task_id"]), False, smoke, formal)
    finally:
        forecast.engine.base03222.SEED = old_seed
        for name, value in originals.items():
            setattr(forecast, {"validate": "validate_forecast_config", "preflight": "preflight", "factory": "make_forecast_env_factory", "calendar": "forecast_calendar_audit", "audit": "audit_actions_and_forecast_columns", "record": "write_record", "verify": "verify_completed_smoke"}[name], value)
    result["transfer_isolation_audit"] = isolation
    result["transfer_kind"] = kind
    result["hidden_engine_seed_explicitly_set"] = True
    out = ROOT / str(result["output_root"])
    if smoke:
        result["next_step_allowed"] = bool(result.get("next_step_allowed", result.get("action_forecast_gate", {}).get("next_step_allowed", False))) and isolation["passed"]
        (out / f"{cfg['task_id']}_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (out / f"{cfg['task_id']}_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", choices=sorted(SPECS), required=True)
    parser.add_argument("--kind", choices=["forecast", "noforecast"], required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = parser.parse_args()
    run_one(args.station, args.kind, smoke=args.smoke, formal=args.train5k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""158LC: matched LC-E2 dual-branch control with zero weather signal.

The control keeps the LC-E2 architecture and all management logic, but fixes
the 12 weather-branch values to zero.  The same entry point supports a 2K
smoke and, only after that gate is present, the authorized 5K seed-0 run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_152E3_sya_dual_branch_noforecast_control import (
    ZeroWeatherDualBranchWrapper,
    e3_audit,
    e3_calendar_audit,
    make_zero_weather_factory,
)
from run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k import patched_maskableppo


CONFIG = ROOT / "configs/158LC_lca_lowIC_dual_branch_noforecast_control_5k_seed0.json"
REFERENCE_CONFIG = ROOT / "configs/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0.json"
PROMPT = ROOT / "prompts/2026-08-16_lc_noforecast_dual_branch_control_5k_seed0.md"
SMOKE_OUT = ROOT / "benchmark_results/158LC_lca_lowIC_dual_branch_noforecast_control_5k_seed0_smoke2k"
EXPECTED_TASK_ID = "158LC"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def control_isolation_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    ref = _read(REFERENCE_CONFIG)
    checks = {
        "station_profile_seed_match": (cfg["station_code"], cfg["site"], cfg["input_profile"], cfg["seed"]) == (ref["station_code"], ref["site"], ref["input_profile"], ref["seed"]),
        "actions_match": cfg["actions"] == ref["actions"],
        "years_match": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"],
        "reward_safety_match": cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "automatic_controls_false": cfg["scope"]["external_n"] is False and cfg["scope"]["native_dssat_automatic_irrigation"] is False,
        "architecture_matches_e2": cfg["policy_architecture"] == ref["policy_architecture"],
    }
    return {"reference_config": REFERENCE_CONFIG.relative_to(ROOT).as_posix(), "checks": checks, "passed": all(checks.values())}


def validate_zero_weather_config(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}")
    if (cfg.get("station_code"), cfg.get("site"), cfg.get("input_profile"), cfg.get("seed")) != ("LCA", "LC", "lowIC", 0):
        raise ValueError("158LC must remain LCA/LC lowIC seed0.")
    if cfg.get("actions", {}).get("irrigation_levels_mm") != [0.0, 15.0, 30.0, 45.0]:
        raise ValueError("158LC irrigation grid mismatch.")
    if cfg.get("actions", {}).get("nitrogen_levels_kg_ha") != [0.0, 40.0, 80.0, 120.0]:
        raise ValueError("158LC nitrogen grid mismatch.")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation" or bool(obs.get("normalization_enabled")):
        raise ValueError("158LC base observation contract mismatch.")
    if bool(obs.get("weather_forecast_enabled")) or obs.get("weather_forecast_mode") != "matched_architecture_zero_weather_control":
        raise ValueError("158LC must have no weather signal.")
    arch = cfg.get("policy_architecture", {})
    expected_arch = {
        "base_observation_dim": 26,
        "weather_observation_dim": 12,
        "base_latent_dim": 64,
        "weather_latent_dim": 32,
        "combined_features_dim": 96,
        "downstream_net_arch": [64, 64],
    }
    for key, value in expected_arch.items():
        if arch.get(key) != value:
            raise ValueError(f"158LC architecture mismatch at {key}: {arch.get(key)}")
    if cfg.get("training") != {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]}:
        raise ValueError("158LC training contract mismatch.")
    isolation = control_isolation_audit(cfg)
    if not isolation["passed"]:
        raise ValueError(json.dumps(isolation, ensure_ascii=False))
    return cfg


def smoke_verification() -> dict[str, Any]:
    result_path = SMOKE_OUT / "158LC_smoke_result.json"
    manifest_path = SMOKE_OUT / "158LC_run_manifest.json"
    checks: dict[str, Any] = {
        "smoke_result_exists": result_path.exists(),
        "smoke_manifest_exists": manifest_path.exists(),
    }
    if all(checks.values()):
        result = _read(result_path)
        manifest = _read(manifest_path)
        gate = result.get("smoke_gate", result.get("action_forecast_gate", {}))
        checks.update({
            "status_completed_smoke": manifest.get("status") == "completed_smoke",
            "smoke_gate_passed": gate.get("next_step_allowed") is True,
            "control_audit_passed": result.get("lc_noforecast_control", {}).get("passed") is True,
        })
    checks["next_step_allowed"] = all(bool(value) for key, value in checks.items() if key != "next_step_allowed")
    return checks


def write_record_lc(cfg, pf, copied, audit, gate, phase, obs_smoke, calendar_audit, smoke_verification=None):
    out = forecast.output_root(cfg)
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} LC matched-architecture 无预报对照记录",
        "",
        "## 设计",
        "",
        "- 保留 LC-E2 的 26 维基础分支 + 12 维天气分支，输出 96 维 latent。",
        "- 12 个天气分支输入固定为 0，不读取天气数据。",
        "- reward、动作网格、安全约束、lowIC 输入、年份和 PPO 设置保持不变。",
        f"- 当前阶段：`{phase}`；输出目录：`{out.relative_to(ROOT).as_posix()}`。",
        "",
        "## gate",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "## zero-weather audit", "", audit.to_markdown(index=False), "", "## 解释边界", "", "- 这是与 LC-E2 同结构的无天气信号对照，不是最终 forecast 优势证明。", "- 只有 E2 在同 checkpoint、同 seed 下优于本对照，才支持天气信号具有额外价值。", ""])
    if smoke_verification is not None:
        lines.extend(["## smoke verification", "", *(f"- `{key}`: `{value}`" for key, value in smoke_verification.items()), ""])
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run_phase(*, smoke: bool, formal: bool) -> dict[str, Any]:
    if smoke == formal:
        raise ValueError("choose exactly one of --smoke or --train5k")
    cfg = _read(CONFIG)
    validate_zero_weather_config(cfg, EXPECTED_TASK_ID)
    control_audit = control_isolation_audit(cfg)
    if formal:
        gate = smoke_verification()
        if not gate["next_step_allowed"]:
            raise RuntimeError(json.dumps(gate, ensure_ascii=False))

    patch_contract()
    originals = {
        "validate": forecast.validate_forecast_config,
        "factory": forecast.make_forecast_env_factory,
        "calendar": forecast.forecast_calendar_audit,
        "audit": forecast.audit_actions_and_forecast_columns,
        "record": forecast.write_record,
    }
    forecast.validate_forecast_config = validate_zero_weather_config
    forecast.make_forecast_env_factory = lambda base_make_env, _cfg: make_zero_weather_factory(base_make_env)
    forecast.forecast_calendar_audit = e3_calendar_audit
    forecast.audit_actions_and_forecast_columns = e3_audit
    forecast.write_record = write_record_lc
    try:
        with patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, False, smoke, formal)
    finally:
        forecast.validate_forecast_config = originals["validate"]
        forecast.make_forecast_env_factory = originals["factory"]
        forecast.forecast_calendar_audit = originals["calendar"]
        forecast.audit_actions_and_forecast_columns = originals["audit"]
        forecast.write_record = originals["record"]

    result["lc_noforecast_control"] = {
        "changed_factor": "weather_signal_zeroed_only_with_LC_E2_dual_branch_architecture",
        "reward_changed": False,
        "action_evaluation_changed": False,
        "passed": control_audit["passed"],
        "next_step_allowed": result.get("next_step_allowed", result.get("action_forecast_gate", {}).get("next_step_allowed", False)),
    }
    out = ROOT / str(result["output_root"])
    if smoke:
        (out / "158LC_control_audit.json").write_text(json.dumps(control_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "158LC_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if formal:
        (out / "158LC_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = parser.parse_args()
    run_phase(smoke=args.smoke, formal=args.train5k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""179/180: FQ SY-weather framework independent-seed 2K paired smoke."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import forecast_engineered_observation_056_057 as forecast
import run_141E1_sya_actionable_weather_encoding_smoke2k as e1
import run_176_multisite_weather_framework_transfer as transfer


PROMPT = ROOT / "prompts/2026-08-17_fq_wp_et_and_seed2_validation.md"
CONFIGS = {
    "forecast": ROOT / "configs/179FQA_fqa_originIC_dual_branch_weather_reader_2k_seed2.json",
    "noforecast": ROOT / "configs/180FQA_fqa_originIC_dual_branch_noforecast_control_2k_seed2.json",
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_seed2(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}")
    if str(cfg.get("station_code")) != "FQA" or str(cfg.get("site")) != "FQ" or str(cfg.get("input_profile")) != "originIC":
        raise ValueError("Seed2 validation must remain FQA originIC.")
    if int(cfg.get("seed")) != 2:
        raise ValueError(f"Expected explicit seed=2, got {cfg.get('seed')}")
    if cfg.get("training") != {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}:
        raise ValueError("Seed2 validation is limited to the 2K smoke contract.")
    # Reuse the already audited transfer validator against a temporary seed1/
    # 5K view, then return the true seed2 config to the training engine.
    checked = copy.deepcopy(cfg)
    checked["seed"] = 1
    checked["training"] = {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]}
    transfer.validate_transfer_config(checked, expected_task_id=expected_task_id)
    return cfg


def run_one(kind: str) -> dict[str, Any]:
    cfg_path = CONFIGS[kind]
    base = read(cfg_path)
    validate_seed2(base, expected_task_id=str(base["task_id"]))
    e1.patch_contract()
    originals = {
        "validate": forecast.validate_forecast_config,
        "preflight": forecast.preflight,
        "factory": forecast.make_forecast_env_factory,
        "calendar": forecast.forecast_calendar_audit,
        "audit": forecast.audit_actions_and_forecast_columns,
        "record": forecast.write_record,
    }

    def validate(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
        return validate_seed2(cfg, expected_task_id)

    forecast.validate_forecast_config = validate
    forecast.preflight = transfer.site_preflight
    if kind == "forecast":
        forecast.make_forecast_env_factory = originals["factory"]
        forecast.forecast_calendar_audit = originals["calendar"]
        forecast.audit_actions_and_forecast_columns = originals["audit"]
    else:
        forecast.make_forecast_env_factory = lambda base_make_env, _cfg: transfer.make_zero_weather_factory(base_make_env)
        forecast.forecast_calendar_audit = transfer.e3_calendar_audit
        forecast.audit_actions_and_forecast_columns = transfer.e3_audit
    forecast.write_record = transfer.write_record
    old_seed = forecast.engine.base03222.SEED
    forecast.engine.base03222.SEED = 2
    try:
        with transfer.patched_transfer_maskableppo(23):
            result = forecast.run_forecast_experiment(cfg_path, PROMPT, str(base["task_id"]), False, True, False)
    finally:
        forecast.engine.base03222.SEED = old_seed
        forecast.validate_forecast_config = originals["validate"]
        forecast.preflight = originals["preflight"]
        forecast.make_forecast_env_factory = originals["factory"]
        forecast.forecast_calendar_audit = originals["calendar"]
        forecast.audit_actions_and_forecast_columns = originals["audit"]
        forecast.write_record = originals["record"]
    result["explicit_config_seed"] = 2
    result["hidden_engine_seed_explicitly_set"] = True
    result["transfer_kind"] = kind
    out = ROOT / str(result["output_root"])
    result_path = out / f"{base['task_id']}_smoke_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    results = {kind: run_one(kind) for kind in ("forecast", "noforecast")}
    paired = {
        "prompt": PROMPT.relative_to(ROOT).as_posix(),
        "seed": 2,
        "station": "FQ",
        "checkpoint": 2000,
        "results": results,
        "forecast_next_step_allowed": bool(results["forecast"].get("next_step_allowed", False)),
        "noforecast_next_step_allowed": bool(results["noforecast"].get("next_step_allowed", False)),
        "paired_smoke_pass": bool(results["forecast"].get("next_step_allowed", False) and results["noforecast"].get("next_step_allowed", False)),
    }
    out = ROOT / "benchmark_results" / "179_180FQA_fqa_seed2_2k_paired"
    out.mkdir(parents=True, exist_ok=True)
    (out / "179_180FQA_seed2_paired_smoke_summary.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(paired, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

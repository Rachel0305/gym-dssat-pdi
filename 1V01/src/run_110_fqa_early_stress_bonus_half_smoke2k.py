"""110_00 isolated FQ 2K smoke with only the early water-stress bonus halved."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as reserve
import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as audit_base
import run_107_fqa_current_weather_observation_smoke2k as weather_base

CONFIG = ROOT / "configs/110_00_fqa_early_stress_bonus_half_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-15_fq_early_stress_bonus_half_smoke2k.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def preflight(cfg: dict) -> dict:
    issues: list[str] = []
    if not (INPUT_ROOT / "FQ/CNFQ0801.MZX").exists():
        issues.append("originIC FQ input/MZX missing")
    if cfg["training"] != {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}:
        issues.append("training must be exactly 2K with 1K/2K checkpoints")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    if cfg["observation_contract"]["current_weather_features"] != weather_base.WEATHER_NAMES:
        issues.append("current-weather feature contract mismatch")
    if cfg["observation_contract"]["weather_forecast_enabled"] is not False:
        issues.append("weather forecast must remain disabled")
    if float(cfg["reward_override"]["water_stress_relief_coef"]) != 5.0:
        issues.append("water_stress_relief_coef must be exactly 5.0")
    return {"task_id": cfg["task_id"], "station_code": "FQA", "site": "FQ", "seed": int(cfg["seed"]), "issues": issues, "next_step_allowed": not issues, "output_root": cfg["output_root"]}


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    pf = preflight(cfg)
    if "--dry-run" in sys.argv:
        print(json.dumps({**pf, "mode": "dry_run", "training_run": False, "smoke_only": True, "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return 0
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)

    effective_yaml = ROOT / "configs/110_00_effective_fqa_early_stress_bonus_half.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(cfg["seed"])
    ppo = dict(cfg["fixed_ppo"]); ppo.update(cfg.get("ppo_override", {})); yaml_cfg["ppo"].update(ppo)
    yaml_cfg["reward"].update(cfg["reward_override"])
    yaml_cfg["discrete_actions"]["irrigation_levels"] = LEVELS_I
    yaml_cfg["discrete_actions"]["nitrogen_levels"] = LEVELS_N
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")

    old_engine = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = {key: getattr(batch, key) for key in ["CONFIG", "OUT", "DOC", "SEED"]}
    old_site_names = dict(batch.SITE_NAMES)
    old_wrapper = batch.base.StressAwareDiscreteWrapper
    old_make_env = batch.base.make_env
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    doc = ROOT / "docs/110_00_fqa_early_stress_bonus_half_smoke2k_record.md"
    start = time.perf_counter(); rss0 = psutil.Process().memory_info().rss
    try:
        engine.TASK_ID = cfg["task_id"]; engine.TASK_NAME = cfg["task_name"]; engine.BASE_OUT = out; engine.BASE_DOC = doc; engine.PROMPT = PROMPT; engine.LOWIC_INPUT_ROOT = INPUT_ROOT; engine.STATION = "FQA"; engine.SITES = ["FQA"]; engine.BINARY_IRRIGATION_LEVELS = LEVELS_I; engine.BINARY_NITROGEN_LEVELS = LEVELS_N
        batch.SEED = int(cfg["seed"]); batch.CONFIG = effective_yaml; batch.OUT = out; batch.DOC = doc; batch.SITE_NAMES["FQA"] = "FQ"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
        base_make_env = batch.base.make_env

        def make_env_current(config, env_config, station, year, seed, run_tag, evaluation=False):
            wrapped = base_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
            return weather_base.CurrentWeatherObservationWrapper(wrapped, config, env_config, station, year)

        batch.base.make_env = make_env_current
        engine.run_training(2000, [1000, 2000], suffix="")
    finally:
        batch.base.make_env = old_make_env; batch.base.StressAwareDiscreteWrapper = old_wrapper; ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        for key, value in old_engine.items(): setattr(engine, key, value)
        for key, value in old_batch.items(): setattr(batch, key, value)
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_site_names)

    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("training did not produce expected checkpoint summary")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    summary.to_csv(out / "evaluation/110_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    audit, gate = audit_base.audit_25k(summary, out, 2000)
    audit.to_csv(out / "audits/110_00_action_audit_ckpt2000.csv", index=False, encoding="utf-8-sig")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    metric_cols = [c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]
    metrics = final[metric_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict()
    result = {**pf, "mode": "completed_2k_reward_ablation_smoke", "training_run": True, "effective_yaml": rel(effective_yaml), "metrics_mean_ckpt2000": metrics, "action_gate": gate, "elapsed_s": round(time.perf_counter() - start, 3), "rss_delta_bytes": psutil.Process().memory_info().rss - rss0, "WP_ET_status": "not_available_in_2K_summary; not inferred", "next_step_allowed": bool(gate.get("mechanism_pass"))}
    (out / "110_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "110_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 110_00 FQ early water-stress bonus half 2K smoke", "", "- 唯一干预：water_stress_relief_coef 10.0 -> 5.0。", "- 保持当前天气观测、PPO、动作格、DSSAT 自动灌溉和安全 mask 不变。", "", "## Result", "", *[f"- `{k}`: `{v}`" for k, v in result.items() if k in ["metrics_mean_ckpt2000", "action_gate", "elapsed_s", "rss_delta_bytes", "next_step_allowed"]], "", "- WP_ET 未从 2K summary 推断。", "- 不自动进入 5K/25K/100K。", ""]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

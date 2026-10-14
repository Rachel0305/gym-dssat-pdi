"""Task 173: task-172 direct-rain control with one irrigation-mask change."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_172_sya_weather_aug_rain_observation_smoke2k as task172


TASK_ID = "173"
TASK_NAME = "sya_weather_aug_rain_obs_irrigation_gap1_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-19_sya_weather_aug_rain_obs_irrigation_gap1_smoke2k.md"
STATION = "SYA"
SEED = 0
CHECKPOINT_STEPS = [1000, 2000]
VARIANTS = task172.VARIANTS
SCENARIO_KEYS = task172.SCENARIO_KEYS
IRRIGATION_GRID = task172.IRRIGATION_GRID
NITROGEN_GRID = task172.NITROGEN_GRID


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def configure_task172_globals() -> dict[str, Any]:
    old = {
        "TASK_ID": task172.TASK_ID,
        "TASK_NAME": task172.TASK_NAME,
        "OUT": task172.OUT,
        "DOC": task172.DOC,
        "PROMPT": task172.PROMPT,
    }
    task172.TASK_ID = TASK_ID
    task172.TASK_NAME = TASK_NAME
    task172.OUT = OUT
    task172.DOC = DOC
    task172.PROMPT = PROMPT
    return old


def restore_task172_globals(old: dict[str, Any]) -> None:
    for key, value in old.items():
        setattr(task172, key, value)


def patch_runtime_173(manifest: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    old_globals = configure_task172_globals()
    old_runtime = task172.patch_runtime(manifest)
    engine = task172.task168.engine
    base03222 = task172.task168.base03222
    old_load_config = engine.load_config

    def load_config_173() -> dict[str, Any]:
        config = old_load_config()
        config["ppo"]["ent_coef"] = 0.02
        config["action_safety"]["min_days_between_irrigation"] = 1
        config["action_safety"]["min_days_between_fertilization"] = 7
        config["action_safety"]["mask_sensitivity_override"] = {
            "changed_rule": "min_days_between_irrigation",
            "control_days": 7,
            "test_days": 1,
            "other_mask_rules_unchanged": True,
        }
        return config

    engine.load_config = load_config_173
    base03222.load_config = load_config_173
    return old_runtime, old_globals


def restore_runtime_173(old_runtime: dict[str, Any], old_globals: dict[str, Any]) -> None:
    task172.restore_runtime(old_runtime)
    restore_task172_globals(old_globals)


def evaluate_and_audit_173(split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv", keep_default_na=False)
    ok = inventory[inventory["run_status"].astype(str).str.startswith("ok")].copy()
    config = task172.task168.engine.load_config()
    env_config = task172.task168.base03222.direct_ppo.build_env_config(config, split)
    metric_rows: list[dict[str, Any]] = []
    for _, model_row in ok.iterrows():
        for scenario_key in split["year"].astype(int).tolist():
            result = task172.evaluate_checkpoint_actual(config, env_config, model_row, scenario_key)
            result.update({"weather_variant": SCENARIO_KEYS[scenario_key], "scenario_key": scenario_key, "source_year": 2005})
            metric_rows.append(result)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "evaluation" / "173_scenario_episode_metrics.csv", index=False, encoding="utf-8-sig")

    action_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in sorted(metrics["checkpoint_step"].astype(int).unique().tolist()):
        ckpt = metrics[metrics["checkpoint_step"].astype(int).eq(checkpoint)]
        base_metric = ckpt[ckpt["weather_variant"].eq("original")].iloc[0]
        base_daily = pd.read_csv(ROOT / str(base_metric["daily_csv_path"]), keep_default_na=False)
        for variant in VARIANTS:
            metric = ckpt[ckpt["weather_variant"].eq(variant)].iloc[0]
            daily = pd.read_csv(ROOT / str(metric["daily_csv_path"]), keep_default_na=False)
            pair = base_daily[["dap", "safe_action_amir", "safe_action_anfer"]].rename(columns={"safe_action_amir": "baseline_irrigation", "safe_action_anfer": "baseline_n"}).merge(
                daily[["dap", "safe_action_amir", "safe_action_anfer"]].rename(columns={"safe_action_amir": "variant_irrigation", "safe_action_anfer": "variant_n"}), on="dap", how="inner"
            )
            pair["checkpoint_step"] = checkpoint
            pair["weather_variant"] = variant
            pair["changed"] = ~np.isclose(pair["baseline_irrigation"], pair["variant_irrigation"]) | ~np.isclose(pair["baseline_n"], pair["variant_n"])
            action_rows.extend(pair.to_dict(orient="records"))
            after = pair[pair["dap"].gt(1)]
            summary_rows.append(
                {
                    "checkpoint_step": checkpoint,
                    "weather_variant": variant,
                    "scenario_key": int(metric["scenario_key"]),
                    "matched_decision_states": int(len(pair)),
                    "action_changed_rows": int(pair["changed"].sum()),
                    "action_change_rate": float(pair["changed"].mean()) if len(pair) else np.nan,
                    "action_change_rate_after_dap1": float(after["changed"].mean()) if len(after) else np.nan,
                    "all_actions_on_grid": bool(daily["safe_action_amir"].round(6).isin(IRRIGATION_GRID).all() and daily["safe_action_anfer"].round(6).isin(NITROGEN_GRID).all()),
                    "final_grnwt": float(metric.get("final_grnwt", np.nan)),
                    "total_irrigation": float(metric.get("total_irrigation", np.nan)),
                    "total_n": float(metric.get("total_n", np.nan)),
                }
            )
    action_summary = pd.DataFrame(summary_rows)
    pd.DataFrame(action_rows).to_csv(OUT / "evaluation" / "173_action_pair_detail.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(OUT / "evaluation" / "173_action_change_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, action_summary


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gate = task172.task168.verify_gate(cfg)
    manifest = task172.task168.read_manifest(cfg)
    split = task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "173_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    old_runtime, old_globals = patch_runtime_173(manifest)
    status = "completed"
    error = ""
    metrics = pd.DataFrame()
    action_summary = pd.DataFrame()
    try:
        task172.task168.base03222.main()
        metrics, action_summary = evaluate_and_audit_173(split)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        restore_runtime_173(old_runtime, old_globals)
    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    response_passed = bool(status == "completed" and not non_original.empty and bool(non_original["all_actions_on_grid"].all()) and float(pd.to_numeric(non_original["action_change_rate_after_dap1"], errors="coerce").fillna(0.0).max()) > 0.01)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "total_timesteps": 2000,
        "checkpoint_steps": [1000, 2000],
        "station": STATION,
        "scenario_count": len(VARIANTS),
        "simulation_years": [2005],
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "validation_executed": False,
        "direct_rain_observation": True,
        "reward_changed": False,
        "action_grid_changed": False,
        "action_mask_changed": True,
        "mask_only_change": "min_days_between_irrigation 7 -> 1; min_days_between_fertilization remains 7",
        "ppo_ent_coef": 0.02,
        "action_response_passed": response_passed,
        "outputs": {
            "training_inventory": rel(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "173_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "173_action_change_summary.csv"),
        },
        "error": error[-5000:] if error else "",
    }
    (OUT / "173_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                f"# {TASK_ID} SYA irrigation-mask sensitivity smoke",
                "",
                "- Control: task 172 direct-rain observation setup.",
                "- Only mask change: minimum irrigation interval 7 days -> 1 day.",
                "- Minimum fertilization interval remains 7 days; seasonal caps, DAP ranges, and late reserve remain unchanged.",
                f"- Training status: `{status}`; action response passed: `{response_passed}`.",
                f"- Result: `{rel(OUT / '173_result.json')}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if status != "completed":
        raise RuntimeError(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

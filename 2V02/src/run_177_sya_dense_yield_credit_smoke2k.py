"""Task 177: SYA daily grain-growth credit assignment 2K smoke."""

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

import run_174_sya_weather_aug_rain_obs_no_swfac_guardrail_smoke2k as task174


TASK_ID = "177"
TASK_NAME = "sya_dense_yield_credit_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_dense_yield_credit_smoke2k.md"
SEED = 0
CHECKPOINT_STEPS = [1000, 2000]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        value = float(arr[0]) if arr.size else default
        return value if np.isfinite(value) else default
    except Exception:
        return default


class DenseYieldCreditWrapper(task174.NoSwfacGuardrailLateMaskWrapper):
    """Task-174 control wrapper with only the yield-credit timing changed."""

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._dense_previous_grnwt = scalar(self.last_obs_dict.get("grnwt"), 0.0)
        return obs, info

    def step(self, action):
        previous = dict(self.last_obs_dict)
        previous_grnwt = scalar(previous.get("grnwt"), 0.0)
        previous_swfac = scalar(previous.get("swfac"), 0.0)
        previous_nstres = scalar(previous.get("nstres"), 0.0)

        obs, sparse_reward_reference, terminated, truncated, info = super().step(action)
        current = dict(self.last_obs_dict)
        current_grnwt = scalar(current.get("grnwt"), previous_grnwt)
        current_swfac = scalar(current.get("swfac"), previous_swfac)
        current_nstres = scalar(current.get("nstres"), previous_nstres)

        reward_cfg = self.config["reward"]
        safe_i = scalar(self.last_action_info.get("safe_action_amir"), 0.0)
        safe_n = scalar(self.last_action_info.get("safe_action_anfer"), 0.0)
        grnwt_increment = current_grnwt - previous_grnwt
        yield_coef = float(reward_cfg["yield_coef"])
        yield_component = yield_coef * grnwt_increment
        resource_cost = float(reward_cfg["nitrogen_cost"]) * safe_n + float(reward_cfg["water_cost"]) * safe_i
        water_relief = max(previous_swfac - current_swfac, 0.0)
        nitrogen_relief = max(previous_nstres - current_nstres, 0.0)
        stress_relief_bonus = (
            float(reward_cfg["water_stress_relief_coef"]) * safe_i * water_relief
            + float(reward_cfg["nitrogen_stress_relief_coef"]) * safe_n * nitrogen_relief
        )
        unscaled_reward = yield_component - resource_cost + stress_relief_bonus
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        reward = unscaled_reward * reward_scale

        self.last_action_info.update(
            {
                "yield_component": yield_component,
                "yield_component_mode": "daily_grnwt_increment",
                "grnwt_previous": previous_grnwt,
                "grnwt_current": current_grnwt,
                "grnwt_increment": grnwt_increment,
                "yield_component_sparse_terminal_reference": yield_coef * current_grnwt
                if bool(terminated or truncated)
                else 0.0,
                "reward_sparse_reference": float(sparse_reward_reference),
                "resource_cost": resource_cost,
                "water_stress_relief": water_relief,
                "nitrogen_stress_relief": nitrogen_relief,
                "stress_relief_bonus": stress_relief_bonus,
                "reward_unscaled": unscaled_reward,
                "reward_scale": reward_scale,
                "reward_stress_aware": reward,
                "dense_yield_credit_enabled": True,
            }
        )
        return obs, float(reward), terminated, truncated, info


def set_task174_globals() -> dict[str, Any]:
    old = {key: getattr(task174, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    task174.TASK_ID = TASK_ID
    task174.TASK_NAME = TASK_NAME
    task174.OUT = OUT
    task174.DOC = DOC
    task174.PROMPT = PROMPT
    return old


def restore_task174_globals(old: dict[str, Any]) -> None:
    for key, value in old.items():
        setattr(task174, key, value)


def patch_runtime_177(manifest: pd.DataFrame):
    old_task174_globals = set_task174_globals()
    old_runtime = task174.patch_runtime_174(manifest)
    base03222 = task174.task173.task172.task168.base03222
    old_wrapper = base03222.base.StressAwareDiscreteWrapper
    base03222.base.StressAwareDiscreteWrapper = DenseYieldCreditWrapper
    return old_task174_globals, old_runtime, old_wrapper


def restore_runtime_177(old_task174_globals: dict[str, Any], old_runtime: dict[str, Any], old_wrapper: Any) -> None:
    base03222 = task174.task173.task172.task168.base03222
    base03222.base.StressAwareDiscreteWrapper = old_wrapper
    task174.restore_runtime_174(old_runtime)
    restore_task174_globals(old_task174_globals)


def rename_outputs() -> None:
    evaluation = OUT / "evaluation"
    for source_name, target_name in [
        ("173_scenario_episode_metrics.csv", "177_scenario_episode_metrics.csv"),
        ("173_action_pair_detail.csv", "177_action_pair_detail.csv"),
        ("173_action_change_summary.csv", "177_action_change_summary.csv"),
    ]:
        source = evaluation / source_name
        target = evaluation / target_name
        if source.exists():
            source.rename(target)


def build_increment_summary() -> dict[str, Any]:
    paths = sorted((OUT / "daily_outputs" / "SYA").glob("*_daily.csv"))
    if not paths:
        return {"daily_files": 0}
    rows: list[dict[str, Any]] = []
    for path in paths:
        daily = pd.read_csv(path, keep_default_na=False)
        inc = pd.to_numeric(daily.get("grnwt_increment", 0), errors="coerce").fillna(0.0)
        yield_component = pd.to_numeric(daily.get("yield_component", 0), errors="coerce").fillna(0.0)
        rows.append(
            {
                "daily_file": rel(path),
                "weather_variant": str(daily.get("weather_variant", pd.Series([""])).iloc[0]),
                "checkpoint_step": int(float(daily.get("checkpoint_step", pd.Series([0])).iloc[0])),
                "n_rows": int(len(daily)),
                "grnwt_increment_mean": float(inc.mean()),
                "grnwt_increment_max": float(inc.max()),
                "grnwt_increment_min": float(inc.min()),
                "grnwt_increment_nonzero_rows": int((inc.abs() > 1e-9).sum()),
                "grnwt_increment_negative_rows": int((inc < -1e-9).sum()),
                "yield_component_mean": float(yield_component.mean()),
                "yield_component_max": float(yield_component.max()),
                "yield_component_sum": float(yield_component.sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "evaluation" / "177_grnwt_increment_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "daily_files": int(len(summary)),
        "grnwt_increment_mean_across_files": float(summary["grnwt_increment_mean"].mean()),
        "grnwt_increment_max_across_files": float(summary["grnwt_increment_max"].max()),
        "grnwt_increment_min_across_files": float(summary["grnwt_increment_min"].min()),
        "negative_increment_rows_total": int(summary["grnwt_increment_negative_rows"].sum()),
        "nonzero_increment_rows_total": int(summary["grnwt_increment_nonzero_rows"].sum()),
        "summary_csv": rel(OUT / "evaluation" / "177_grnwt_increment_summary.csv"),
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("total_timesteps") != 2000 or cfg.get("yield_coef") != 0.158 or cfg.get("reward_scale") != 0.001:
        raise ValueError("Task 177 configuration drifted from the pre-registered 2K credit-assignment smoke.")
    gate = task174.task173.task172.task168.verify_gate(cfg)
    manifest = task174.task173.task172.task168.read_manifest(cfg)
    split = task174.task173.task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "177_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    old_task174_globals, old_runtime, old_wrapper = patch_runtime_177(manifest)
    status = "completed"
    error = ""
    action_summary = pd.DataFrame()
    try:
        task174.task173.task172.task168.base03222.main()
        _, action_summary = task174.task173.evaluate_and_audit_173(split)
        rename_outputs()
        increment_summary = build_increment_summary()
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        increment_summary = {"daily_files": 0}
    finally:
        restore_runtime_177(old_task174_globals, old_runtime, old_wrapper)

    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    response_passed = bool(
        status == "completed"
        and not non_original.empty
        and bool(non_original["all_actions_on_grid"].all())
        and float(pd.to_numeric(non_original["action_change_rate_after_dap1"], errors="coerce").fillna(0.0).max()) > 0.01
    )
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "total_timesteps": 2000,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "station": "SYA",
        "scenario_count": 6,
        "simulation_years": [2005],
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "validation_executed": False,
        "direct_rain_observation": True,
        "reward_changed": True,
        "reward_change": "terminal_yield_to_daily_grnwt_increment",
        "yield_coef": 0.158,
        "reward_scale": 0.001,
        "action_mask_changed": False,
        "action_grid_changed": False,
        "ppo_ent_coef": 0.02,
        "action_response_passed": response_passed,
        "increment_summary": increment_summary,
        "outputs": {
            "training_inventory": rel(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "177_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "177_action_change_summary.csv"),
        },
        "error": error[-5000:] if error else "",
    }
    (OUT / "177_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                f"# {TASK_ID} SYA daily grain-growth credit assignment smoke",
                "",
                "- Control environment: task 174 direct-rain observation, original 7-day action mask, SWFAC guardrail disabled.",
                "- Only reward change: terminal yield component replaced by daily `0.158 * (GRNWT_t - GRNWT_(t-1))`.",
                "- Reward scale, resource costs, stress-relief coefficients, weather scenarios, PPO settings and seed unchanged.",
                f"- Training status: `{status}`; action response passed: `{response_passed}`.",
                f"- Result: `{rel(OUT / '177_result.json')}`.",
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

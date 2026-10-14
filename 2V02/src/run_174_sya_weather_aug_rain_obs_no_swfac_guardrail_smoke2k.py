"""Task 174: task-172 control with SWFAC guardrail reward disabled."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_173_sya_weather_aug_rain_obs_irrigation_gap1_smoke2k as task173
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036


TASK_ID = "174"
TASK_NAME = "sya_weather_aug_rain_obs_no_swfac_guardrail_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_weather_aug_rain_obs_no_swfac_guardrail_smoke2k.md"


class NoSwfacGuardrailLateMaskWrapper(base04036.LateIrrigationReserveMaskWrapper):
    """Keep the 040_36 mask but call the pre-guardrail reward step."""

    def step(self, action):
        prev_dap = self._dap()
        prev_i = float(self.safety_state.cumulative_irrigation)
        obs, reward, terminated, truncated, info = base04036.base04026.I240StagedIrrigationReserveMaskWrapper.step(self, action)
        current_i = float(self.safety_state.cumulative_irrigation)
        self.last_action_info.update(
            {
                "swfac_guardrail_enabled": False,
                "swfac_guardrail_disabled_for_task": "174",
                "late_irrigation_reserve_mask_enabled": True,
                "late_reserve_dap_end": base04036.LATE_RESERVE_DAP_END,
                "pre_late_cum_irrigation_cap_mm": base04036.PRE_LATE_CUM_IRRIGATION_CAP,
                "late_reserved_irrigation_mm": base04036.LATE_RESERVED_IRRIGATION_MM,
                "late_reserve_active_before_action": bool(prev_dap <= base04036.LATE_RESERVE_DAP_END),
                "late_reserve_irrigation_before_action_mm": prev_i,
                "late_reserve_irrigation_after_action_mm": current_i,
            }
        )
        return obs, reward, terminated, truncated, info


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def set_task_globals() -> dict:
    old = {key: getattr(task173, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    task173.TASK_ID = TASK_ID
    task173.TASK_NAME = TASK_NAME
    task173.OUT = OUT
    task173.DOC = DOC
    task173.PROMPT = PROMPT
    return old


def restore_task_globals(old: dict) -> None:
    for key, value in old.items():
        setattr(task173, key, value)


def patch_runtime_174(manifest: pd.DataFrame):
    old_task_globals = set_task_globals()
    old_runtime, old_task172_globals = task173.patch_runtime_173(manifest)
    engine = task173.task172.task168.engine
    base03222 = task173.task172.task168.base03222
    old_engine_load_config = engine.load_config
    old_base_load_config = base03222.load_config
    old_wrapper = base03222.base.StressAwareDiscreteWrapper

    def load_config_174():
        config = old_engine_load_config()
        config["ppo"]["ent_coef"] = 0.02
        config["action_safety"]["min_days_between_irrigation"] = 7
        config["action_safety"]["min_days_between_fertilization"] = 7
        config["reward"]["swfac_guardrail_process_penalty"] = {
            "enabled": False,
            "threshold": 0.05,
            "coef": 50.0,
            "formula": "disabled_for_task_174",
        }
        return config

    engine.load_config = load_config_174
    base03222.load_config = load_config_174
    base03222.base.StressAwareDiscreteWrapper = NoSwfacGuardrailLateMaskWrapper
    return {
        "runtime": old_runtime,
        "task172_globals": old_task172_globals,
        "task173_globals": old_task_globals,
        "base_wrapper": old_wrapper,
    }


def restore_runtime_174(old: dict) -> None:
    task173.restore_runtime_173(old["runtime"], old["task172_globals"])
    base03222 = task173.task172.task168.base03222
    base03222.base.StressAwareDiscreteWrapper = old["base_wrapper"]
    restore_task_globals(old["task173_globals"])


def rename_evaluation_outputs() -> None:
    evaluation = OUT / "evaluation"
    for source_name, target_name in [
        ("173_scenario_episode_metrics.csv", "174_scenario_episode_metrics.csv"),
        ("173_action_pair_detail.csv", "174_action_pair_detail.csv"),
        ("173_action_change_summary.csv", "174_action_change_summary.csv"),
    ]:
        source = evaluation / source_name
        target = evaluation / target_name
        if source.exists():
            source.rename(target)


def run(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gate = task173.task172.task168.verify_gate(cfg)
    manifest = task173.task172.task168.read_manifest(cfg)
    split = task173.task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "174_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    old = patch_runtime_174(manifest)
    status = "completed"
    error = ""
    action_summary = pd.DataFrame()
    try:
        task173.task172.task168.base03222.main()
        _, action_summary = task173.evaluate_and_audit_173(split)
        rename_evaluation_outputs()
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        restore_runtime_174(old)
    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    response_passed = bool(status == "completed" and not non_original.empty and bool(non_original["all_actions_on_grid"].all()) and float(pd.to_numeric(non_original["action_change_rate_after_dap1"], errors="coerce").fillna(0.0).max()) > 0.01)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "total_timesteps": 2000,
        "checkpoint_steps": [1000, 2000],
        "station": "SYA",
        "scenario_count": 6,
        "simulation_years": [2005],
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "validation_executed": False,
        "direct_rain_observation": True,
        "reward_changed": True,
        "swfac_guardrail_enabled": False,
        "action_mask_changed": False,
        "action_grid_changed": False,
        "ppo_ent_coef": 0.02,
        "action_response_passed": response_passed,
        "outputs": {
            "training_inventory": rel(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "174_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "174_action_change_summary.csv"),
        },
        "error": error[-5000:] if error else "",
    }
    (OUT / "174_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                f"# {TASK_ID} SYA SWFAC guardrail reward ablation",
                "",
                "- Control: task 172 direct-rain observation and original 7-day irrigation mask.",
                "- Only reward change: SWFAC guardrail process penalty disabled.",
                f"- Training status: `{status}`; action response passed: `{response_passed}`.",
                f"- Result: `{rel(OUT / '174_result.json')}`.",
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

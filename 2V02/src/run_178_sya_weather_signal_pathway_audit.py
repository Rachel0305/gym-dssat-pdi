"""Task 178: read-only weather observation and action-mask pathway audit."""

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

import run_177_sya_dense_yield_credit_smoke2k as task177


TASK_ID = "178"
TASK_NAME = "sya_weather_signal_pathway_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_weather_signal_pathway_audit.md"
MODEL_ROOT = ROOT / "benchmark_results" / "177_sya_dense_yield_credit_smoke2k" / "models" / "SYA"
CHECKPOINTS = [1000, 2000]
STATION = "SYA"
SEED = 0
VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
SCENARIO_KEYS = {2005: "original", 2006: "early_dry", 2007: "mid_dry", 2008: "late_dry", 2009: "early_wet", 2010: "mid_wet"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        number = float(arr[0]) if arr.size else default
        return number if np.isfinite(number) else default
    except Exception:
        return default


def action_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def mask_key(mask: np.ndarray) -> str:
    return "".join("1" if bool(value) else "0" for value in np.asarray(mask, dtype=bool).reshape(-1))


def set_task177_globals() -> dict[str, Any]:
    old = {key: getattr(task177, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    task177.TASK_ID = TASK_ID
    task177.TASK_NAME = TASK_NAME
    task177.OUT = OUT
    task177.DOC = DOC
    task177.PROMPT = PROMPT
    return old


def restore_task177_globals(old: dict[str, Any]) -> None:
    for key, value in old.items():
        setattr(task177, key, value)


def patch_runtime_178(manifest: pd.DataFrame):
    old_task177_globals = set_task177_globals()
    old_task174_globals, old_runtime, old_wrapper = task177.patch_runtime_177(manifest)
    return old_task177_globals, old_task174_globals, old_runtime, old_wrapper


def restore_runtime_178(old: tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]) -> None:
    old_task177_globals, old_task174_globals, old_runtime, old_wrapper = old
    task177.restore_runtime_177(old_task174_globals, old_runtime, old_wrapper)
    restore_task177_globals(old_task177_globals)


def model_path(checkpoint: int) -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{int(checkpoint)}.zip"


def run_one_episode(model, config: dict[str, Any], env_config: dict[str, Any], scenario_key: int, checkpoint: int, reference_rain: dict[int, float] | None, reference_masks: dict[int, str] | None, reference_actions: dict[int, int] | None) -> tuple[list[dict[str, Any]], dict[int, float], dict[int, str]]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    base03222 = task177.task174.task173.task172.task168.base03222
    variant = SCENARIO_KEYS[int(scenario_key)]
    env = base03222.base.make_env(
        config,
        env_config,
        STATION,
        int(scenario_key),
        SEED,
        f"{STATION}_2005_{variant}_{TASK_ID}_audit",
        evaluation=True,
    )
    rows: list[dict[str, Any]] = []
    local_rain: dict[int, float] = {}
    local_masks: dict[int, str] = {}
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            mask = np.asarray(get_action_masks(env), dtype=bool)
            dap = int(round(scalar(getattr(env, "last_obs_dict", {}).get("dap"), step_count)))
            obs_actual = np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs_actual.size < 1:
                raise ValueError("Observation is empty")
            actual_rain = float(obs_actual[-1])
            actual_action_raw, _ = model.predict(obs_actual, action_masks=mask, deterministic=True)
            actual_action = action_int(actual_action_raw)
            step_index = int(step_count)
            local_rain[step_index] = actual_rain
            local_masks[step_index] = mask_key(mask)

            if reference_rain is None:
                swapped_rain = actual_rain
            else:
                swapped_rain = float(reference_rain.get(step_index, actual_rain))
            obs_swapped = obs_actual.copy()
            obs_swapped[-1] = swapped_rain
            swapped_action_raw, _ = model.predict(obs_swapped, action_masks=mask, deterministic=True)
            swapped_action = action_int(swapped_action_raw)
            mask_code = mask_key(mask)
            reference_rain_value = swapped_rain
            reference_mask_code = reference_masks.get(step_index, mask_code) if reference_masks is not None else mask_code
            rows.append(
                {
                    "checkpoint_step": int(checkpoint),
                    "weather_variant": variant,
                    "step_index": step_index,
                    "scenario_key": int(scenario_key),
                    "dap": dap,
                    "observation_dim": int(obs_actual.size),
                    "rain_actual": actual_rain,
                    "rain_original_reference": reference_rain_value,
                    "rain_abs_difference": abs(actual_rain - reference_rain_value),
                    "mask_key_actual": mask_code,
                    "mask_key_original_reference": reference_mask_code,
                    "mask_valid_action_count": int(mask.sum()),
                    "mask_diff_vs_original": bool(mask_code != reference_mask_code),
                    "actual_action": actual_action,
                    "actual_action_change_vs_original": bool(
                        reference_actions is not None
                        and actual_action != reference_actions.get(step_index, actual_action)
                    ),
                    "rain_swap_action": swapped_action,
                    "rain_swap_action_changed": bool(actual_action != swapped_action),
                    "actual_action_was_applied": True,
                }
            )
            obs, reward, terminated, truncated, info = env.step(actual_action)
            done = bool(terminated or truncated)
            step_count += 1
    finally:
        env.close()
    return rows, local_rain, local_masks


def summarize(stepwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (checkpoint, variant), group in stepwise.groupby(["checkpoint_step", "weather_variant"], sort=True):
        non_original = variant != "original"
        rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "weather_variant": variant,
                "rows": int(len(group)),
                "rain_different_rows_vs_original": int((group["rain_abs_difference"] > 1e-9).sum()),
                "rain_abs_difference_max": float(group["rain_abs_difference"].max()),
                "mask_diff_rows_vs_original": int(group["mask_diff_vs_original"].sum()),
                "actual_action_change_rows_vs_original": int(group["actual_action_change_vs_original"].sum()),
                "rain_swap_action_change_rows": int(group["rain_swap_action_changed"].sum()),
                "rain_swap_action_change_rate": float(group["rain_swap_action_changed"].mean()),
                "mask_valid_action_count_min": int(group["mask_valid_action_count"].min()),
                "mask_valid_action_count_max": int(group["mask_valid_action_count"].max()),
            }
        )
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame, stepwise: pd.DataFrame) -> dict[str, Any]:
    non_original = summary[summary["weather_variant"] != "original"]
    rain_diff_rows = int(non_original["rain_different_rows_vs_original"].sum())
    mask_diff_rows = int(non_original["mask_diff_rows_vs_original"].sum())
    swap_changes = int(non_original["rain_swap_action_change_rows"].sum())
    actual_changes = int(non_original["actual_action_change_rows_vs_original"].fillna(0).sum())
    if rain_diff_rows == 0:
        diagnosis = "weather_observation_not_different"
    elif mask_diff_rows > 0:
        diagnosis = "weather_dependent_mask_or_state_path_requires_inspection"
    elif swap_changes > 0:
        diagnosis = "policy_is_sensitive_to_rain_channel_but_actual_scenario_actions_are_fixed"
    else:
        diagnosis = "policy_checkpoint_is_insensitive_to_rain_channel_under_fixed_mask"
    return {
        "diagnosis": diagnosis,
        "non_original_rain_different_rows": rain_diff_rows,
        "non_original_mask_diff_rows": mask_diff_rows,
        "non_original_rain_swap_action_changes": swap_changes,
        "non_original_actual_action_changes_vs_original": actual_changes,
        "rain_channel_nonconstant_passed": rain_diff_rows > 0,
        "mask_weather_invariance_passed": mask_diff_rows == 0,
        "rain_swap_policy_response_passed": swap_changes > 0,
        "note": "The swap uses the actual mask at the current state; it changes only the last rain observation dimension.",
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 178 is read-only and must not train.")
    gate = task177.task174.task173.task172.task168.verify_gate(cfg)
    manifest = task177.task174.task173.task172.task168.read_manifest(cfg)
    split = task177.task174.task173.task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    for checkpoint in CHECKPOINTS:
        if not model_path(checkpoint).exists():
            raise FileNotFoundError(model_path(checkpoint))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "178_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    from sb3_contrib import MaskablePPO

    all_rows: list[dict[str, Any]] = []
    old_runtime = patch_runtime_178(manifest)
    try:
        engine = task177.task174.task173.task172.task168.engine
        base03222 = task177.task174.task173.task172.task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        for checkpoint in CHECKPOINTS:
            model = MaskablePPO.load(str(model_path(checkpoint)), device="cpu")
            reference_rows, reference_rain, reference_masks = run_one_episode(
                model, runtime_config, env_config, 2005, checkpoint, None, None, None
            )
            for row in reference_rows:
                row["reference_episode"] = True
            all_rows.extend(reference_rows)
            reference_actions = {int(row["step_index"]): int(row["actual_action"]) for row in reference_rows}
            for scenario_key in sorted(SCENARIO_KEYS):
                if scenario_key == 2005:
                    continue
                rows, _, _ = run_one_episode(
                    model, runtime_config, env_config, scenario_key, checkpoint, reference_rain, reference_masks, reference_actions
                )
                for row in rows:
                    row["reference_episode"] = False
                all_rows.extend(rows)
    finally:
        restore_runtime_178(old_runtime)

    stepwise = pd.DataFrame(all_rows).sort_values(["checkpoint_step", "weather_variant", "dap"]).reset_index(drop=True)
    summary = summarize(stepwise)
    diagnosis = classify(summary, stepwise)
    stepwise.to_csv(OUT / "evaluation" / "178_stepwise_weather_signal_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "evaluation" / "178_pathway_summary.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "read_only_inference_audit",
        "status": "completed",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "station": STATION,
        "checkpoint_steps": CHECKPOINTS,
        "scenario_count": len(SCENARIO_KEYS),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "diagnosis": diagnosis,
        "outputs": {
            "stepwise": rel(OUT / "evaluation" / "178_stepwise_weather_signal_audit.csv"),
            "summary": rel(OUT / "evaluation" / "178_pathway_summary.csv"),
        },
    }
    (OUT / "178_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                "# Task 178：SYA 天气信号通路审计记录",
                "",
                "> 只读 inference audit；没有训练 PPO，也没有修改 task 177 的模型或输出。",
                "",
                f"- checkpoint：`{CHECKPOINTS}`；情景数：`{len(SCENARIO_KEYS)}`；物理 gate：`{bool(gate.get('physical_response_passed'))}`。",
                f"- diagnosis：`{diagnosis['diagnosis']}`。",
                f"- rain 通道实际差异行数：`{diagnosis['non_original_rain_different_rows']}`。",
                f"- 与 original 相比的 mask 差异行数：`{diagnosis['non_original_mask_diff_rows']}`。",
                f"- 固定 mask 下 rain swap 引起的 action 改变次数：`{diagnosis['non_original_rain_swap_action_changes']}`。",
                f"- 实际情景 action 与 original 的改变次数：`{diagnosis['non_original_actual_action_changes_vs_original']}`。",
                "",
                "## 输出",
                "",
                f"- `{rel(OUT / '178_result.json')}`",
                f"- `{rel(OUT / 'evaluation' / '178_stepwise_weather_signal_audit.csv')}`",
                f"- `{rel(OUT / 'evaluation' / '178_pathway_summary.csv')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    global OUT, DOC
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    if args.output_dir is not None:
        OUT = args.output_dir.resolve()
        DOC = ROOT / "docs" / f"{OUT.name}_record.md"
    try:
        print(json.dumps(run(config.resolve()), ensure_ascii=False, indent=2))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

"""Task 179: read-only rain-channel scale and policy-sensitivity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_178_sya_weather_signal_pathway_audit as task178


TASK_ID = "179"
TASK_NAME = "sya_rain_observation_scale_sensitivity_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_rain_observation_scale_sensitivity_audit.md"
MODEL_ROOT = ROOT / "benchmark_results" / "177_sya_dense_yield_credit_smoke2k" / "models" / "SYA"
CHECKPOINTS = [1000, 2000]
STATION = "SYA"
SEED = 0
SCENARIO_KEYS = {2005: "original", 2006: "early_dry", 2007: "mid_dry", 2008: "late_dry", 2009: "early_wet", 2010: "mid_wet"}
RAIN_SWEEP = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 250.0, 500.0]


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


def model_path(checkpoint: int) -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{int(checkpoint)}.zip"


def set_task178_globals() -> dict[str, Any]:
    old = {key: getattr(task178, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    task178.TASK_ID = TASK_ID
    task178.TASK_NAME = TASK_NAME
    task178.OUT = OUT
    task178.DOC = DOC
    task178.PROMPT = PROMPT
    return old


def restore_task178_globals(old: dict[str, Any]) -> None:
    for key, value in old.items():
        setattr(task178, key, value)


def run_episode(model, config: dict[str, Any], env_config: dict[str, Any], scenario_key: int, checkpoint: int, obs_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]) -> None:
    from sb3_contrib.common.maskable.utils import get_action_masks

    base03222 = task178.task177.task174.task173.task172.task168.base03222
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
    try:
        obs, info = env.reset()
        done = False
        step_index = 0
        while not done and step_index < int(config["runtime"]["max_steps"]):
            mask = np.asarray(get_action_masks(env), dtype=bool)
            dap = int(round(scalar(getattr(env, "last_obs_dict", {}).get("dap"), step_index)))
            actual_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            actual_rain = float(actual_obs[-1])
            actual_action_raw, _ = model.predict(actual_obs, action_masks=mask, deterministic=True)
            actual_action = action_int(actual_action_raw)
            obs_record = {
                "checkpoint_step": int(checkpoint),
                "weather_variant": variant,
                "scenario_key": int(scenario_key),
                "step_index": int(step_index),
                "dap": dap,
                "observation_dim": int(actual_obs.size),
                "rain_actual": actual_rain,
                "mask_valid_action_count": int(mask.sum()),
                "actual_action": actual_action,
            }
            for feature_index, feature_value in enumerate(actual_obs.tolist()):
                obs_record[f"obs_{feature_index}"] = float(feature_value)
            obs_rows.append(obs_record)
            for rain_value in RAIN_SWEEP:
                counterfactual = actual_obs.copy()
                counterfactual[-1] = float(rain_value)
                action_raw, _ = model.predict(counterfactual, action_masks=mask, deterministic=True)
                sweep_action = action_int(action_raw)
                sweep_rows.append(
                    {
                        "checkpoint_step": int(checkpoint),
                        "weather_variant": variant,
                        "scenario_key": int(scenario_key),
                        "step_index": int(step_index),
                        "dap": dap,
                        "rain_actual": actual_rain,
                        "rain_counterfactual": float(rain_value),
                        "mask_valid_action_count": int(mask.sum()),
                        "actual_action": actual_action,
                        "sweep_action": sweep_action,
                        "sweep_action_changed": bool(sweep_action != actual_action),
                    }
                )
            obs, reward, terminated, truncated, info = env.step(actual_action)
            done = bool(terminated or truncated)
            step_index += 1
    finally:
        env.close()


def build_feature_stats(obs_rows: pd.DataFrame) -> pd.DataFrame:
    # The rain channel is the final observation dimension by contract.
    feature_rows: list[dict[str, Any]] = []
    observation_dim = int(pd.to_numeric(obs_rows["observation_dim"], errors="coerce").max())
    for feature_index in range(observation_dim):
        column = f"obs_{feature_index}"
        series = pd.to_numeric(obs_rows[column], errors="coerce").dropna()
        feature_rows.append(
            {
                "feature": "rain" if feature_index == observation_dim - 1 else column,
                "dimension_index": feature_index,
                "count": int(series.size),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "nonzero_count": int((series.abs() > 1e-9).sum()),
            }
        )
    return pd.DataFrame(feature_rows)


def summarize(obs_rows: pd.DataFrame, sweep_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (checkpoint, variant), group in sweep_rows.groupby(["checkpoint_step", "weather_variant"], sort=True):
        actual = obs_rows[(obs_rows["checkpoint_step"] == checkpoint) & (obs_rows["weather_variant"] == variant)]
        actual_range_min = float(actual["rain_actual"].min())
        actual_range_max = float(actual["rain_actual"].max())
        group = group.copy()
        actual_range = group[
            (group["rain_counterfactual"] >= actual_range_min - 1e-9)
            & (group["rain_counterfactual"] <= actual_range_max + 1e-9)
        ]
        rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "weather_variant": variant,
                "state_rows": int(len(actual)),
                "actual_rain_min": actual_range_min,
                "actual_rain_max": actual_range_max,
                "actual_rain_std": float(actual["rain_actual"].std(ddof=0)),
                "sweep_distinct_actions": int(group["sweep_action"].nunique()),
                "sweep_action_change_rows": int(group["sweep_action_changed"].sum()),
                "sweep_action_change_rate": float(group["sweep_action_changed"].mean()),
                "within_actual_range_sweep_rows": int(len(actual_range)),
                "within_actual_range_action_change_rows": int(actual_range["sweep_action_changed"].sum()),
                "within_actual_range_action_change_rate": float(actual_range["sweep_action_changed"].mean()) if len(actual_range) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def diagnose(summary: pd.DataFrame) -> dict[str, Any]:
    non_original = summary[summary["weather_variant"] != "original"]
    actual_range_sweep_changes = int(non_original["within_actual_range_action_change_rows"].sum())
    full_sweep_changes = int(non_original["sweep_action_change_rows"].sum())
    actual_rain_span = float((non_original["actual_rain_max"] - non_original["actual_rain_min"]).max())
    if full_sweep_changes == 0:
        diagnosis = "policy_insensitive_even_to_full_rain_sweep"
    elif actual_range_sweep_changes == 0:
        diagnosis = "policy_changes_only_outside_actual_rain_range"
    else:
        diagnosis = "policy_is_sensitive_within_actual_rain_range"
    return {
        "diagnosis": diagnosis,
        "non_original_actual_rain_span_max": actual_rain_span,
        "non_original_full_sweep_action_changes": full_sweep_changes,
        "non_original_within_actual_range_action_changes": actual_range_sweep_changes,
        "rain_sweep_values": RAIN_SWEEP,
        "note": "All sweep predictions use the actual state and the actual action mask; only observation dimension 25 is replaced.",
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 179 is read-only and must not train.")
    gate = task178.task177.task174.task173.task172.task168.verify_gate(cfg)
    manifest = task178.task177.task174.task173.task172.task168.read_manifest(cfg)
    split = task178.task177.task174.task173.task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    for checkpoint in CHECKPOINTS:
        if not model_path(checkpoint).exists():
            raise FileNotFoundError(model_path(checkpoint))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "179_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    from sb3_contrib import MaskablePPO

    old_task178_globals = set_task178_globals()
    old_runtime = task178.patch_runtime_178(manifest)
    obs_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    try:
        engine = task178.task177.task174.task173.task172.task168.engine
        base03222 = task178.task177.task174.task173.task172.task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        for checkpoint in CHECKPOINTS:
            model = MaskablePPO.load(str(model_path(checkpoint)), device="cpu")
            for scenario_key in sorted(SCENARIO_KEYS):
                run_episode(model, runtime_config, env_config, scenario_key, checkpoint, obs_rows, sweep_rows)
    finally:
        task178.restore_runtime_178(old_runtime)
        restore_task178_globals(old_task178_globals)

    obs_frame = pd.DataFrame(obs_rows)
    sweep_frame = pd.DataFrame(sweep_rows)
    summary = summarize(obs_frame, sweep_frame)
    feature_stats = build_feature_stats(obs_frame)
    diagnosis = diagnose(summary)
    obs_frame.to_csv(OUT / "evaluation" / "179_observation_rain_states.csv", index=False, encoding="utf-8-sig")
    sweep_frame.to_csv(OUT / "evaluation" / "179_rain_sweep_actions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "evaluation" / "179_rain_scale_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    feature_stats.to_csv(OUT / "evaluation" / "179_observation_feature_stats.csv", index=False, encoding="utf-8-sig")
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
            "observation_states": rel(OUT / "evaluation" / "179_observation_rain_states.csv"),
            "rain_sweep_actions": rel(OUT / "evaluation" / "179_rain_sweep_actions.csv"),
            "summary": rel(OUT / "evaluation" / "179_rain_scale_sensitivity_summary.csv"),
            "feature_stats": rel(OUT / "evaluation" / "179_observation_feature_stats.csv"),
        },
    }
    (OUT / "179_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                "# Task 179：SYA rain 观测尺度与 PPO 策略敏感性审计记录",
                "",
                "> 只读 inference audit；没有训练 PPO，也没有修改任务 177 模型或环境。",
                "",
                f"- checkpoint：`{CHECKPOINTS}`；情景数：`{len(SCENARIO_KEYS)}`；物理 gate：`{bool(gate.get('physical_response_passed'))}`。",
                f"- diagnosis：`{diagnosis['diagnosis']}`。",
                f"- 实际 rain 范围内 action 改变次数：`{diagnosis['non_original_within_actual_range_action_changes']}`。",
                f"- 全部 `0–500` rain sweep 的 action 改变次数：`{diagnosis['non_original_full_sweep_action_changes']}`。",
                "",
                "## 输出",
                "",
                f"- `{rel(OUT / '179_result.json')}`",
                f"- `{rel(OUT / 'evaluation' / '179_rain_scale_sensitivity_summary.csv')}`",
                f"- `{rel(OUT / 'evaluation' / '179_observation_feature_stats.csv')}`",
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
    print(json.dumps(run(config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

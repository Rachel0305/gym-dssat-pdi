"""Task 180: low-cost action-conditioned payoff and threshold probe.

This is a read-only inference audit.  It replays the task-177 PPO-2000
prefix to selected DAPs and evaluates every action permitted by the actual
mask from the identical DSSAT state.  It does not train PPO and does not
alter reward, observations, masks, action grid, weather, or crop parameters.
"""

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

import run_178_sya_weather_signal_pathway_audit as task178


TASK_ID = "180"
TASK_NAME = "sya_action_conditioned_payoff_probe"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_action_conditioned_payoff_probe.md"
MODEL_ROOT = ROOT / "benchmark_results" / "177_sya_dense_yield_credit_smoke2k" / "models" / "SYA"
CHECKPOINT = 2000
STATION = "SYA"
SEED = 0
SCENARIO_KEYS = {
    2005: "original",
    2006: "early_dry",
    2007: "mid_dry",
    2008: "late_dry",
    2009: "early_wet",
    2010: "mid_wet",
}
TARGET_DAPS = [8, 31, 61]
STAGE_LABELS = {8: "early", 31: "middle", 61: "late_pre_reproductive"}


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


def model_path() -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{CHECKPOINT}.zip"


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


def current_dap(env, fallback: int) -> int:
    return int(round(scalar(getattr(env, "last_obs_dict", {}).get("dap"), fallback)))


def state_snapshot(env, obs: np.ndarray) -> dict[str, Any]:
    state = dict(getattr(env, "last_obs_dict", {}) or {})
    return {
        "dap": current_dap(env, -1),
        "rain": scalar(np.asarray(obs, dtype=np.float32).reshape(-1)[-1], 0.0),
        "swfac": scalar(state.get("swfac"), np.nan),
        "nstres": scalar(state.get("nstres"), np.nan),
        "grnwt": scalar(state.get("grnwt"), np.nan),
        "topwt": scalar(state.get("topwt"), np.nan),
        "xlai": scalar(state.get("xlai"), np.nan),
        "cumulative_irrigation": scalar(getattr(getattr(env, "safety_state", None), "cumulative_irrigation", np.nan), np.nan),
        "cumulative_nitrogen": scalar(getattr(getattr(env, "safety_state", None), "cumulative_nitrogen", np.nan), np.nan),
    }


def action_descriptor(env, action_index: int) -> tuple[float, float, str]:
    row = list(getattr(env, "grid"))[int(action_index)]
    if isinstance(row, dict):
        irrigation = scalar(row.get("amir", row.get("irrigation", 0.0)), 0.0)
        nitrogen = scalar(row.get("anfer", row.get("nitrogen", 0.0)), 0.0)
        descriptor = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    else:
        irrigation = scalar(row[0], 0.0)
        nitrogen = scalar(row[1], 0.0)
        descriptor = json.dumps(list(row), ensure_ascii=False, default=str)
    return irrigation, nitrogen, descriptor


def replay_to_target(model, config: dict[str, Any], env_config: dict[str, Any], scenario_key: int, target_dap: int) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    base03222 = task178.task177.task174.task173.task172.task168.base03222
    variant = SCENARIO_KEYS[int(scenario_key)]
    env = base03222.base.make_env(
        config,
        env_config,
        STATION,
        int(scenario_key),
        SEED,
        f"{STATION}_2005_{variant}_{TASK_ID}_prefix_t{target_dap}",
        evaluation=True,
    )
    prefix_actions: list[int] = []
    try:
        obs, info = env.reset()
        step_count = 0
        while step_count < int(config["runtime"]["max_steps"]):
            dap = current_dap(env, step_count)
            if dap == int(target_dap):
                mask = np.asarray(get_action_masks(env), dtype=bool)
                snapshot = state_snapshot(env, np.asarray(obs, dtype=np.float32))
                actual_raw, _ = model.predict(np.asarray(obs, dtype=np.float32), action_masks=mask, deterministic=True)
                return {
                    "scenario_key": int(scenario_key),
                    "weather_variant": variant,
                    "target_dap": int(target_dap),
                    "stage_label": STAGE_LABELS[int(target_dap)],
                    "prefix_actions": prefix_actions,
                    "prefix_length": len(prefix_actions),
                    "snapshot": snapshot,
                    "observation": np.asarray(obs, dtype=np.float32).reshape(-1),
                    "mask": mask,
                    "mask_key": mask_key(mask),
                    "model_action": action_int(actual_raw),
                }
            mask = np.asarray(get_action_masks(env), dtype=bool)
            action_raw, _ = model.predict(np.asarray(obs, dtype=np.float32), action_masks=mask, deterministic=True)
            action = action_int(action_raw)
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                raise RuntimeError(f"PPO prefix produced invalid action {action} at DAP {dap}; mask={mask_key(mask)}")
            prefix_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            step_count += 1
            if bool(terminated or truncated):
                break
    finally:
        env.close()
    raise RuntimeError(f"Target DAP {target_dap} was not reached for scenario {scenario_key}")


def replay_prefix_and_probe(
    model_state: dict[str, Any],
    config: dict[str, Any],
    env_config: dict[str, Any],
    action_index: int,
) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    base03222 = task178.task177.task174.task173.task172.task168.base03222
    scenario_key = int(model_state["scenario_key"])
    target_dap = int(model_state["target_dap"])
    variant = str(model_state["weather_variant"])
    env = base03222.base.make_env(
        config,
        env_config,
        STATION,
        scenario_key,
        SEED,
        f"{STATION}_2005_{variant}_{TASK_ID}_probe_t{target_dap}_a{action_index}",
        evaluation=True,
    )
    try:
        obs, info = env.reset()
        for prefix_action in model_state["prefix_actions"]:
            prefix_mask = np.asarray(get_action_masks(env), dtype=bool)
            if prefix_action < 0 or prefix_action >= len(prefix_mask) or not bool(prefix_mask[prefix_action]):
                raise RuntimeError(f"Prefix action became invalid during replay: {prefix_action}, mask={mask_key(prefix_mask)}")
            obs, _, terminated, truncated, _ = env.step(int(prefix_action))
            if bool(terminated or truncated):
                raise RuntimeError("DSSAT episode terminated before target DAP during prefix replay")

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        dap = current_dap(env, len(model_state["prefix_actions"]))
        mask = np.asarray(get_action_masks(env), dtype=bool)
        before = state_snapshot(env, obs)
        reference = model_state["snapshot"]
        consistency_columns = ["rain", "swfac", "nstres", "grnwt", "cumulative_irrigation", "cumulative_nitrogen"]
        consistency_errors = []
        for column in consistency_columns:
            before_value = scalar(before.get(column), np.nan)
            reference_value = scalar(reference.get(column), np.nan)
            if np.isfinite(before_value) and np.isfinite(reference_value):
                consistency_errors.append(abs(before_value - reference_value))
        state_consistency_error = max(consistency_errors) if consistency_errors else np.nan
        irrigation, nitrogen, descriptor = action_descriptor(env, int(action_index))
        candidate_valid = bool(0 <= int(action_index) < len(mask) and mask[int(action_index)])
        row: dict[str, Any] = {
            "checkpoint_step": CHECKPOINT,
            "scenario_key": scenario_key,
            "weather_variant": variant,
            "target_dap": target_dap,
            "stage_label": STAGE_LABELS[target_dap],
            "prefix_length": int(len(model_state["prefix_actions"])),
            "prefix_actions": "|".join(str(value) for value in model_state["prefix_actions"]),
            "dap_before": dap,
            "rain_before": before["rain"],
            "swfac_before": before["swfac"],
            "nstres_before": before["nstres"],
            "grnwt_before": before["grnwt"],
            "cumulative_irrigation_before": before["cumulative_irrigation"],
            "cumulative_nitrogen_before": before["cumulative_nitrogen"],
            "mask_key": mask_key(mask),
            "mask_valid_action_count": int(mask.sum()),
            "model_action_index": int(model_state["model_action"]),
            "candidate_action_index": int(action_index),
            "candidate_valid": candidate_valid,
            "candidate_irrigation_mm": irrigation,
            "candidate_nitrogen_kg_ha": nitrogen,
            "candidate_descriptor": descriptor,
            "state_consistency_max_abs_error": state_consistency_error,
        }
        if not candidate_valid:
            row.update({"probe_executed": False, "reward": np.nan})
            return row

        obs_after, reward, terminated, truncated, info = env.step(int(action_index))
        after = state_snapshot(env, np.asarray(obs_after, dtype=np.float32))
        action_info = dict(getattr(env, "last_action_info", {}) or {})
        row.update(
            {
                "probe_executed": True,
                "reward": float(reward),
                "reward_unscaled": scalar(action_info.get("reward_unscaled"), np.nan),
                "yield_component": scalar(action_info.get("yield_component"), np.nan),
                "resource_cost": scalar(action_info.get("resource_cost"), np.nan),
                "stress_relief_bonus": scalar(action_info.get("stress_relief_bonus"), np.nan),
                "grnwt_increment": scalar(action_info.get("grnwt_increment"), np.nan),
                "safe_action_irrigation_mm": scalar(action_info.get("safe_action_amir"), np.nan),
                "safe_action_nitrogen_kg_ha": scalar(action_info.get("safe_action_anfer"), np.nan),
                "swfac_after": after["swfac"],
                "nstres_after": after["nstres"],
                "grnwt_after": after["grnwt"],
                "rain_after": after["rain"],
                "dap_after": after["dap"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": json.dumps(info, ensure_ascii=False, default=str),
            }
        )
        return row
    finally:
        env.close()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (scenario_key, variant, target_dap), group in rows[rows["probe_executed"]].groupby(
        ["scenario_key", "weather_variant", "target_dap"], sort=True
    ):
        group = group.sort_values(["reward_unscaled", "candidate_action_index"], ascending=[False, True]).copy()
        noop = group[(group["candidate_irrigation_mm"].abs() < 1e-9) & (group["candidate_nitrogen_kg_ha"].abs() < 1e-9)]
        top = group.iloc[0]
        records.append(
            {
                "scenario_key": int(scenario_key),
                "weather_variant": str(variant),
                "target_dap": int(target_dap),
                "stage_label": str(top["stage_label"]),
                "candidate_count": int(len(group)),
                "mask_valid_action_count": int(group["mask_valid_action_count"].iloc[0]),
                "rain_before": float(group["rain_before"].iloc[0]),
                "swfac_before": float(group["swfac_before"].iloc[0]),
                "nstres_before": float(group["nstres_before"].iloc[0]),
                "top_action_index": int(top["candidate_action_index"]),
                "top_irrigation_mm": float(top["candidate_irrigation_mm"]),
                "top_nitrogen_kg_ha": float(top["candidate_nitrogen_kg_ha"]),
                "top_reward_unscaled": float(top["reward_unscaled"]),
                "top_reward": float(top["reward"]),
                "top_grnwt_increment": float(top["grnwt_increment"]),
                "noop_action_index": int(noop.iloc[0]["candidate_action_index"]) if len(noop) else -1,
                "noop_reward_unscaled": float(noop.iloc[0]["reward_unscaled"]) if len(noop) else np.nan,
                "top_gain_vs_noop_unscaled": float(top["reward_unscaled"] - noop.iloc[0]["reward_unscaled"]) if len(noop) else np.nan,
                "reward_spread_unscaled": float(group["reward_unscaled"].max() - group["reward_unscaled"].min()),
                "ranking_signature": ">".join(str(int(value)) for value in group["candidate_action_index"].tolist()),
                "all_candidates_state_consistent": bool(group["state_consistency_max_abs_error"].max() <= 1e-6),
            }
        )
    return pd.DataFrame(records)


def compare_to_original(summary: pd.DataFrame) -> pd.DataFrame:
    original = summary[summary["weather_variant"].eq("original")][["target_dap", "top_action_index", "ranking_signature", "top_reward_unscaled"]].rename(
        columns={
            "top_action_index": "original_top_action_index",
            "ranking_signature": "original_ranking_signature",
            "top_reward_unscaled": "original_top_reward_unscaled",
        }
    )
    compared = summary[summary["weather_variant"].ne("original")].merge(original, on="target_dap", how="left")
    compared["top_action_changed_vs_original"] = compared["top_action_index"].ne(compared["original_top_action_index"])
    compared["ranking_changed_vs_original"] = compared["ranking_signature"].ne(compared["original_ranking_signature"])
    compared["top_reward_delta_vs_original"] = compared["top_reward_unscaled"] - compared["original_top_reward_unscaled"]
    return compared


def diagnose(summary: pd.DataFrame, compared: pd.DataFrame) -> dict[str, Any]:
    non_original = summary[summary["weather_variant"].ne("original")]
    changed_top = int(compared["top_action_changed_vs_original"].sum()) if not compared.empty else 0
    changed_rank = int(compared["ranking_changed_vs_original"].sum()) if not compared.empty else 0
    unique_top_by_stage = {
        str(int(dap)): int(non_original[non_original["target_dap"].eq(dap)]["top_action_index"].nunique())
        for dap in TARGET_DAPS
    }
    if changed_top > 0 or changed_rank > 0:
        diagnosis = "weather_scenarios_change_local_action_payoff_ordering"
    elif int(non_original["top_action_index"].nunique()) > 1:
        diagnosis = "local_payoff_has_state_dependent_action_ordering_but_not_relative_to_original_at_each_dap"
    else:
        diagnosis = "local_action_payoff_ordering_is_fixed_across_tested_weather_scenarios"
    return {
        "diagnosis": diagnosis,
        "non_original_groups": int(len(non_original)),
        "top_action_changes_vs_original": changed_top,
        "full_ranking_changes_vs_original": changed_rank,
        "unique_top_actions_by_stage_non_original": unique_top_by_stage,
        "note": "This is a one-step local payoff probe. It cannot establish final-season optimality; delayed yield effects require a longer counterfactual rollout.",
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 180 is read-only and must not train.")
    target_daps = [int(value) for value in cfg.get("target_daps", TARGET_DAPS)]
    if target_daps != TARGET_DAPS:
        raise ValueError(f"Unexpected target DAPs: {target_daps}; pre-registered values are {TARGET_DAPS}.")
    gate = task178.task177.task174.task173.task172.task168.verify_gate(cfg)
    manifest = task178.task177.task174.task173.task172.task168.read_manifest(cfg)
    split = task178.task177.task174.task173.task172.task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    if not model_path().exists():
        raise FileNotFoundError(model_path())
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "180_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    from sb3_contrib import MaskablePPO

    old_task178_globals = set_task178_globals()
    old_runtime = task178.patch_runtime_178(manifest)
    rows: list[dict[str, Any]] = []
    try:
        engine = task178.task177.task174.task173.task172.task168.engine
        base03222 = task178.task177.task174.task173.task172.task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        model = MaskablePPO.load(str(model_path()), device="cpu")
        for scenario_key in sorted(SCENARIO_KEYS):
            for target_dap in TARGET_DAPS:
                state = replay_to_target(model, runtime_config, env_config, scenario_key, target_dap)
                valid_indices = np.flatnonzero(np.asarray(state["mask"], dtype=bool)).astype(int).tolist()
                if len(valid_indices) == 0:
                    raise RuntimeError(f"No valid actions at scenario={scenario_key}, target_dap={target_dap}")
                for action_index in valid_indices:
                    rows.append(replay_prefix_and_probe(state, runtime_config, env_config, action_index))
    finally:
        task178.restore_runtime_178(old_runtime)
        restore_task178_globals(old_task178_globals)

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    compared = compare_to_original(summary)
    diagnosis = diagnose(summary, compared)
    frame.to_csv(OUT / "evaluation" / "180_action_grid_probe.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "evaluation" / "180_action_grid_summary.csv", index=False, encoding="utf-8-sig")
    compared.to_csv(OUT / "evaluation" / "180_action_ranking_comparison.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "read_only_action_grid_probe",
        "status": "completed",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "station": STATION,
        "checkpoint_step": CHECKPOINT,
        "scenario_count": len(SCENARIO_KEYS),
        "target_daps": TARGET_DAPS,
        "candidate_rows": int(len(frame)),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "diagnosis": diagnosis,
        "outputs": {
            "action_grid_probe": rel(OUT / "evaluation" / "180_action_grid_probe.csv"),
            "action_grid_summary": rel(OUT / "evaluation" / "180_action_grid_summary.csv"),
            "ranking_comparison": rel(OUT / "evaluation" / "180_action_ranking_comparison.csv"),
        },
    }
    (OUT / "180_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                "# Task 180：SYA action-conditioned payoff / threshold probe",
                "",
                "> 只读 inference audit；没有训练 PPO，没有改变天气、观测、reward、mask 或 action grid。",
                "",
                f"- checkpoint：`{CHECKPOINT}`；情景数：`{len(SCENARIO_KEYS)}`；目标 DAP：`{TARGET_DAPS}`。",
                f"- 候选动作记录行数：`{len(frame)}`；physical gate：`{bool(gate.get('physical_response_passed'))}`。",
                f"- diagnosis：`{diagnosis['diagnosis']}`。",
                f"- 相对 original 的 top action 改变次数：`{diagnosis['top_action_changes_vs_original']}`。",
                f"- 相对 original 的完整 action ranking 改变次数：`{diagnosis['full_ranking_changes_vs_original']}`。",
                "",
                "## 解释边界",
                "",
                "候选动作只执行一步，因此结果用于判断局部 action-conditioned payoff 是否有差异；不能直接当作整季最优动作。若排序随情景改变，下一步应做固定前缀、长时域反事实 rollout；若排序不变，则应重点检查延迟产量信用分配、观测归一化和策略阈值。",
                "",
                "## 输出",
                "",
                f"- `{rel(OUT / '180_result.json')}`",
                f"- `{rel(OUT / 'evaluation' / '180_action_grid_probe.csv')}`",
                f"- `{rel(OUT / 'evaluation' / '180_action_grid_summary.csv')}`",
                f"- `{rel(OUT / 'evaluation' / '180_action_ranking_comparison.csv')}`",
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

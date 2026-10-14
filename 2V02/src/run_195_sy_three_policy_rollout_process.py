"""195: collect PPO rollout-process tables for three SY policy lines.

This is the three-policy counterpart of task 194's teaching table.  It loads
existing checkpoints, collects rollout transitions, computes returns and
advantages, and stops before any parameter update.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import ppo_safe_rendering
import run_sya_ppo_configured_046_02 as base04602
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine042
import run_184_sya_ppo_critic_td_action_value_audit as task184
import run_189_sya_gap1_early_irrigation_reward_critic_audit as task189
import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import DualBranchBaseWeatherExtractor


TASK_ID = "195"
TASK_NAME = "sy_three_policy_rollout_process"
N_STEPS = 144
SEED = 0

MODEL_04610 = ROOT / "benchmark_results/046_10_sya_originIC_expanded_action_maskableppo/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip"
MODEL_143E2 = ROOT / "benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt5000.zip"
MODEL_191 = ROOT / "benchmark_results/191_sya_balanced_irrigation_gap1_100k_learning_curve/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_label(action_index: int) -> str:
    irrigation = [0, 15, 30, 45][int(action_index) // 4]
    nitrogen = [0, 40, 80, 120][int(action_index) % 4]
    return f"I{irrigation}/N{nitrogen}"


def action_irrigation(action_index: int) -> float:
    return float([0, 15, 30, 45][int(action_index) // 4])


def action_nitrogen(action_index: int) -> float:
    return float([0, 40, 80, 120][int(action_index) % 4])


def probabilities_for_one_state(model: Any, observation: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    observation_batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    mask_batch = np.asarray(action_mask, dtype=bool).reshape(1, -1)
    observation_tensor = obs_as_tensor(observation_batch, model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(observation_tensor, action_masks=mask_batch)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
    return np.asarray(probabilities, dtype=float)


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return default
        result = float(array[0])
        return result if np.isfinite(result) else default
    except Exception:
        return default


class RolloutRecorder:
    def __init__(self, meta: dict[str, Any]):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                pre_observation = np.asarray(self.locals["self"]._last_obs).reshape(-1).copy()
                action = int(np.asarray(self.locals["actions"]).reshape(-1)[0])
                reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
                done = bool(np.asarray(self.locals["dones"]).reshape(-1)[0])
                value_before = float(np.asarray(self.locals["values"].detach().cpu()).reshape(-1)[0])
                log_probability_before = float(np.asarray(self.locals["log_probs"].detach().cpu()).reshape(-1)[0])
                action_mask = np.asarray(self.locals["action_masks"]).reshape(-1).astype(bool).copy()
                probability_before = probabilities_for_one_state(self.model, pre_observation, action_mask)
                dap = scalar(pre_observation[1] if pre_observation.size > 1 else np.nan)

                row = {
                    **outer.meta,
                    "rollout_step": len(outer.rows),
                    "dap": dap,
                    "legal_count": int(action_mask.sum()),
                    "action_index": action,
                    "action": action_label(action),
                    "selected_irrigation": action_irrigation(action),
                    "selected_nitrogen": action_nitrogen(action),
                    "reward": reward,
                    "done": done,
                    "value_before": value_before,
                    "log_probability_selected_before": log_probability_before,
                    "probability_selected_before": float(np.exp(log_probability_before)),
                    "observation_vector": json.dumps(np.round(pre_observation, 6).tolist()),
                    "action_mask": "".join("1" if flag else "0" for flag in action_mask.tolist()),
                    "all_action_probabilities_before": json.dumps(np.round(probability_before, 8).tolist()),
                }
                outer.rows.append(row)
                print(
                    f"{outer.meta['model_id']} {outer.meta['scenario_key']} "
                    f"step={row['rollout_step']:03d} dap={dap:6.1f} "
                    f"action={row['action']:>9s} reward={reward: .6f} "
                    f"V={value_before: .6f} legal={row['legal_count']}",
                    flush=True,
                )
                return True

        self.meta = meta
        self.rows: list[dict[str, Any]] = []
        self.callback = Callback()


def collect_table(model: Any, env: Any, meta: dict[str, Any]) -> pd.DataFrame:
    recorder = RolloutRecorder(meta)
    model.set_env(env)
    total_timesteps, callback = model._setup_learn(
        total_timesteps=N_STEPS,
        callback=recorder.callback,
        reset_num_timesteps=False,
        tb_log_name=f"{TASK_ID}_{meta['model_id']}",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    collected = model.collect_rollouts(
        model.get_env(),
        callback,
        model.rollout_buffer,
        n_rollout_steps=N_STEPS,
        use_masking=True,
    )
    callback.on_training_end()
    if not collected:
        raise RuntimeError(f"collect_rollouts returned False for {meta}")
    for index, row in enumerate(recorder.rows):
        row["return_used_for_training"] = float(model.rollout_buffer.returns[index, 0])
        row["advantage_used_for_training"] = float(model.rollout_buffer.advantages[index, 0])
    return pd.DataFrame(recorder.rows)


def build_04610_context(output_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    cfg = json.loads((ROOT / "configs/046_10_sya_originIC_expanded_action_maskableppo.json").read_text(encoding="utf-8"))
    old_engine = {
        "TASK_ID": engine042.TASK_ID,
        "TASK_NAME": engine042.TASK_NAME,
        "BASE_OUT": engine042.BASE_OUT,
        "BASE_DOC": engine042.BASE_DOC,
        "PROMPT": engine042.PROMPT,
        "LOWIC_INPUT_ROOT": engine042.LOWIC_INPUT_ROOT,
        "STATION": engine042.STATION,
        "SITES": list(engine042.SITES),
        "BINARY_IRRIGATION_LEVELS": list(engine042.BINARY_IRRIGATION_LEVELS),
        "BINARY_NITROGEN_LEVELS": list(engine042.BINARY_NITROGEN_LEVELS),
    }
    old_base = {
        "OUT": engine042.base03222.OUT,
        "DOC": engine042.base03222.DOC,
        "PROMPT": engine042.base03222.PROMPT,
        "SITES": list(engine042.base03222.SITES),
        "TOTAL_TIMESTEPS": engine042.base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(engine042.base03222.CHECKPOINT_STEPS),
        "load_config": engine042.base03222.load_config,
        "wrapper": engine042.base03222.base.StressAwareDiscreteWrapper,
        "copy2": engine042.base03222.shutil.copy2,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    engine042.TASK_ID = "046_10"
    engine042.TASK_NAME = "sya_originIC_expanded_action_maskableppo"
    engine042.BASE_OUT = output_dir / "runtime_04610"
    engine042.BASE_DOC = output_dir / "runtime_04610.md"
    engine042.LOWIC_INPUT_ROOT = base04602.INPUT_PROFILES["originIC"]
    engine042.STATION = "SYA"
    engine042.SITES = ["SYA"]
    engine042.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    engine042.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    engine042.patch_base_module(engine042.BASE_OUT, engine042.BASE_DOC, 100000, [100000])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine042.LOWIC_INPUT_ROOT
    runtime_config = engine042.load_config()
    split = engine042.base03222.load_split()
    split = split[split["station_code"].astype(str).eq("SYA")].copy()
    selection = engine042.base03222.build_selection(split)
    env_config = engine042.base03222.direct_ppo.build_env_config(runtime_config, selection)
    restore = {"old_engine": old_engine, "old_base": old_base, "old_input_root": old_input_root}
    return engine042.base03222, runtime_config, env_config, restore


def restore_04610_context(restore: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = restore["old_input_root"]
    for key, value in restore["old_base"].items():
        if key == "wrapper":
            engine042.base03222.base.StressAwareDiscreteWrapper = value
        elif key == "copy2":
            engine042.base03222.shutil.copy2 = value
        else:
            setattr(engine042.base03222, key, value)
    for key, value in restore["old_engine"].items():
        setattr(engine042, key, value)


def build_191_context(output_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any], pd.DataFrame, Any, Any, Any]:
    cfg = json.loads((ROOT / "configs/191_sya_balanced_irrigation_gap1_100k_learning_curve.json").read_text(encoding="utf-8"))
    task178 = task184.task178
    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)
    runtime_patch_owner, original_patch, original_restore = task189.patch_gap1_runtime()
    saved_runtime = runtime_patch_owner.patch_runtime_178(manifest)
    engine = task168.engine
    base03222 = task168.base03222
    runtime_config = engine.load_config()
    env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
    return base03222, runtime_config, env_config, split, runtime_patch_owner, original_patch, original_restore, saved_runtime


def restore_191_context(runtime_patch_owner: Any, original_patch: Any, original_restore: Any, saved_runtime: Any) -> None:
    runtime_patch_owner.restore_runtime_178(saved_runtime)
    task184.task178.patch_runtime_178 = original_patch
    task184.task178.restore_runtime_178 = original_restore


def build_143e2_context() -> tuple[Any, dict[str, Any], dict[str, Any], Any]:
    cfg = json.loads((ROOT / "configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json").read_text(encoding="utf-8"))
    patch_contract()
    old = {
        key: getattr(forecast.engine, key)
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
    forecast.engine.STATION = "SYA"
    forecast.engine.SITES = ["SYA"]
    forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
    runtime_config = forecast.engine.load_config()
    selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
    env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
    make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
    restore = {"old": old, "old_names": old_names, "old_root": old_root}
    return make_env, runtime_config, env_config, restore


def restore_143e2_context(restore: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = restore["old_root"]
    forecast.engine.base03222.SITE_NAMES.clear()
    forecast.engine.base03222.SITE_NAMES.update(restore["old_names"])
    for key, value in restore["old"].items():
        setattr(forecast.engine, key, value)


def load_maskable_model(model_path: Path):
    from sb3_contrib import MaskablePPO

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return MaskablePPO.load(str(model_path), device="cpu")


def load_e2_model(model_path: Path):
    model = load_maskable_model(model_path)
    extractor = model.policy.features_extractor
    if not (
        isinstance(extractor, DualBranchBaseWeatherExtractor)
        or type(extractor).__name__ == "DualBranchBaseWeatherExtractor"
    ):
        raise RuntimeError(f"Wrong E2 extractor: {type(extractor)}")
    return model


def collect_04610(output_dir: Path, years: list[int]) -> pd.DataFrame:
    base03222, runtime_config, env_config, restore = build_04610_context(output_dir)
    rows: list[pd.DataFrame] = []
    try:
        for year in years:
            env = base03222.base.make_env(runtime_config, env_config, "SYA", year, SEED, f"195_04610_{year}", evaluation=False)
            try:
                model = load_maskable_model(MODEL_04610)
                rows.append(
                    collect_table(
                        model,
                        env,
                        {
                            "model_id": "SY_no_forecast_04610_ckpt100000",
                            "source_run": "046_10",
                            "checkpoint_step": 100000,
                            "year": year,
                            "weather_variant": "real_validation_year",
                            "scenario_key": str(year),
                        },
                    )
                )
            finally:
                env.close()
    finally:
        restore_04610_context(restore)
    return pd.concat(rows, ignore_index=True)


def collect_143e2(years: list[int]) -> pd.DataFrame:
    make_env, runtime_config, env_config, restore = build_143e2_context()
    rows: list[pd.DataFrame] = []
    try:
        for year in years:
            env = make_env(runtime_config, env_config, "SYA", year, SEED, f"195_143E2_{year}", evaluation=False)
            try:
                model = load_e2_model(MODEL_143E2)
                rows.append(
                    collect_table(
                        model,
                        env,
                        {
                            "model_id": "SY_weather_E2_143E2_e2_s0_ckpt5000",
                            "source_run": "143E2_151E2N_e2_s0",
                            "checkpoint_step": 5000,
                            "year": year,
                            "weather_variant": "real_validation_year",
                            "scenario_key": str(year),
                        },
                    )
                )
            finally:
                env.close()
    finally:
        restore_143e2_context(restore)
    return pd.concat(rows, ignore_index=True)


def collect_191(output_dir: Path, variants: list[str]) -> pd.DataFrame:
    base03222, runtime_config, env_config, split, owner, original_patch, original_restore, saved_runtime = build_191_context(output_dir)
    rows: list[pd.DataFrame] = []
    try:
        selected = split[split["weather_variant"].astype(str).isin(variants)].copy()
        for scenario in selected.itertuples(index=False):
            year = int(getattr(scenario, "year"))
            variant = str(getattr(scenario, "weather_variant"))
            env = base03222.base.make_env(runtime_config, env_config, "SYA", year, SEED, f"195_191_{variant}", evaluation=False)
            try:
                model = load_maskable_model(MODEL_191)
                rows.append(
                    collect_table(
                        model,
                        env,
                        {
                            "model_id": "SY_one_year_six_weather_191_ckpt100000",
                            "source_run": "191",
                            "checkpoint_step": 100000,
                            "year": 2005,
                            "weather_variant": variant,
                            "scenario_key": variant,
                        },
                    )
                )
            finally:
                env.close()
    finally:
        restore_191_context(owner, original_patch, original_restore, saved_runtime)
    return pd.concat(rows, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["model_id", "scenario_key", "weather_variant"], dropna=False):
        model_id, scenario_key, weather_variant = keys
        legal = pd.to_numeric(group["legal_count"], errors="coerce")
        rows.append(
            {
                "model_id": model_id,
                "scenario_key": scenario_key,
                "weather_variant": weather_variant,
                "rows": int(len(group)),
                "mean_reward": float(pd.to_numeric(group["reward"], errors="coerce").mean()),
                "sum_reward": float(pd.to_numeric(group["reward"], errors="coerce").sum()),
                "mean_value_before": float(pd.to_numeric(group["value_before"], errors="coerce").mean()),
                "mean_return": float(pd.to_numeric(group["return_used_for_training"], errors="coerce").mean()),
                "mean_advantage": float(pd.to_numeric(group["advantage_used_for_training"], errors="coerce").mean()),
                "legal1_rows": int(legal.eq(1).sum()),
                "legal_ge2_rows": int(legal.ge(2).sum()),
                "positive_action_rows": int(
                    (
                        pd.to_numeric(group["selected_irrigation"], errors="coerce").abs().gt(1e-9)
                        | pd.to_numeric(group["selected_nitrogen"], errors="coerce").abs().gt(1e-9)
                    ).sum()
                ),
                "total_selected_irrigation": float(pd.to_numeric(group["selected_irrigation"], errors="coerce").sum()),
                "total_selected_nitrogen": float(pd.to_numeric(group["selected_nitrogen"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_id", "scenario_key"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode_name = "quick" if args.quick else "full"
    output_dir = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{mode_name}"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(output_dir)}")
    if output_dir.exists() and args.overwrite:
        raise RuntimeError("Overwrite is intentionally disabled for this research output; delete manually only if appropriate.")
    (output_dir / "evaluation").mkdir(parents=True, exist_ok=False)
    years = [2014] if args.quick else list(range(2014, 2024))
    variants = ["original"] if args.quick else ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]

    frames = [
        collect_04610(output_dir, years),
        collect_143e2(years),
        collect_191(output_dir, variants),
    ]
    all_rows = pd.concat(frames, ignore_index=True)
    all_path = output_dir / "evaluation" / "rollout_transitions_all_models.csv"
    all_rows.to_csv(all_path, index=False, encoding="utf-8-sig")
    legal_rows = all_rows[pd.to_numeric(all_rows["legal_count"], errors="coerce").ge(2)].copy()
    legal_path = output_dir / "evaluation" / "legal_decision_rows.csv"
    legal_rows.to_csv(legal_path, index=False, encoding="utf-8-sig")
    summary = summarize(all_rows)
    summary_path = output_dir / "evaluation" / "rollout_transition_summary_by_model.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    parameters = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "quick" if args.quick else "full",
        "training_executed": False,
        "model_train_executed": False,
        "n_steps_per_rollout": N_STEPS,
        "years_for_frozen_real_weather_lines": years,
        "weather_variants_for_191": variants,
        "outputs": {
            "rollout_transitions_all_models": rel(all_path),
            "legal_decision_rows": rel(legal_path),
            "summary": rel(summary_path),
        },
    }
    (output_dir / "ppo_rollout_collection_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{TASK_ID}_result.json").write_text(
        json.dumps({**parameters, "rows": int(len(all_rows)), "legal_rows": int(len(legal_rows))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**parameters, "rows": int(len(all_rows)), "legal_rows": int(len(legal_rows))}, ensure_ascii=False, indent=2))
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="collect only 04610-2014, 143E2-2014, and 191-original")
    parser.add_argument("--overwrite", action="store_true", help="guarded; currently refuses overwrite")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

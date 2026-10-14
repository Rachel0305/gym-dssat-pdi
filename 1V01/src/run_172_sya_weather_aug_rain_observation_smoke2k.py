"""Task 172: direct current-rain observation only, 2K PPO smoke.

This runner is intentionally isolated from tasks 168-170. It reuses the same
six 2005 weather variants, reward, action grid, mask, and PPO settings, but
appends the current raw DSSAT rain state to the observation. Evaluation
weather fields come from the actual current DSSAT state, not from an internal
scenario key interpreted as a physical year.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_168_sya_weather_aug_pilot_smoke2k_v2 as task168


TASK_ID = "172"
TASK_NAME = "sya_weather_aug_rain_observation_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-19_sya_weather_aug_rain_observation_smoke2k.md"
STATION = "SYA"
SITE = "SY"
SEED = 0
CHECKPOINT_STEPS = [1000, 2000]
VARIANTS = task168.VARIANTS
SCENARIO_KEYS = task168.SCENARIO_KEYS
IRRIGATION_GRID = {0.0, 15.0, 30.0, 45.0}
NITROGEN_GRID = {0.0, 40.0, 80.0, 120.0}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value, default=np.nan):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else default
    except Exception:
        return default


def state_weather(env) -> dict[str, Any]:
    state = getattr(env.unwrapped, "_state", {}) or {}
    yrdoy = int(state.get("yrdoy", 0) or 0)
    doy = int(yrdoy % 1000) if yrdoy else 0
    date = pd.Timestamp("2005-01-01") + pd.Timedelta(days=max(doy - 1, 0)) if doy else pd.NaT
    return {
        "source_year": 2005,
        "weather_yrdoy": yrdoy,
        "weather_date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
        "rain": scalar(state.get("rain"), np.nan),
        "srad": scalar(state.get("srad"), np.nan),
        "tmax": scalar(state.get("tmax"), np.nan),
        "tmin": scalar(state.get("tmin"), np.nan),
    }


class RainObservationWrapper(gym.Env):
    """Append the current raw DSSAT rain state; delegate reward and masks."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        low = np.concatenate([np.asarray(base_space.low, dtype=np.float32).reshape(-1), np.asarray([0.0], dtype=np.float32)])
        high = np.concatenate([np.asarray(base_space.high, dtype=np.float32).reshape(-1), np.asarray([500.0], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, obs):
        rain = state_weather(self.env)["rain"]
        if not np.isfinite(rain):
            rain = 0.0
        return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1), np.asarray([rain], dtype=np.float32)])

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def patch_runtime(manifest: pd.DataFrame) -> dict[str, Any]:
    old = task168.patch_runtime(manifest)
    engine = task168.engine
    base03222 = task168.base03222
    old_engine_load_config = engine.load_config
    old_base_load_config = base03222.load_config

    def load_config_172() -> dict[str, Any]:
        config = old_engine_load_config()
        config["ppo"]["ent_coef"] = 0.02
        return config

    engine.TASK_ID = TASK_ID
    engine.TASK_NAME = TASK_NAME
    engine.BASE_OUT = OUT
    engine.BASE_DOC = DOC
    engine.PROMPT = PROMPT
    engine.load_config = load_config_172
    engine.patch_base_module(OUT, DOC, 2000, CHECKPOINT_STEPS)
    base03222.load_config = load_config_172
    old_scenario_make_env = base03222.base.make_env

    def rain_scenario_make_env(config, env_config, station, year, seed, run_tag, evaluation=False):
        return RainObservationWrapper(
            old_scenario_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
        )

    base03222.base.make_env = rain_scenario_make_env
    old["task172_engine_load_config"] = old_engine_load_config
    old["task172_base_load_config"] = old_base_load_config
    return old


def restore_runtime(old: dict[str, Any]) -> None:
    old_engine_load_config = old.pop("task172_engine_load_config", None)
    old_base_load_config = old.pop("task172_base_load_config", None)
    task168.restore_runtime(old)
    if old_engine_load_config is not None:
        task168.engine.load_config = old_engine_load_config
    if old_base_load_config is not None:
        task168.base03222.load_config = old_base_load_config


def evaluate_checkpoint_actual(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series, scenario_key: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    step = int(row["checkpoint_step"])
    model_path = ROOT / str(row["model_path"])
    variant = SCENARIO_KEYS[int(scenario_key)]
    daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_2005_{variant}_seed{SEED}_ckpt{step}_daily.csv"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        return {"station_code": STATION, "year": int(scenario_key), "seed": SEED, "checkpoint_step": step, "run_status": "missing_model"}

    model = MaskablePPO.load(str(model_path), device="cpu")
    env = task168.base03222.base.make_env(
        config, env_config, STATION, int(scenario_key), SEED,
        f"{STATION}_2005_{variant}_{TASK_ID}_eval", evaluation=True,
    )
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = task168.base03222.base.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap"), step_count)
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) else step_count
            weather = state_weather(env)
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest_after = task168.base03222.base.latest_observation_dict(env, obs, info)
            records.append(
                {
                    "station_code": STATION,
                    "year": int(scenario_key),
                    "source_year": 2005,
                    "weather_variant": variant,
                    "seed": SEED,
                    "checkpoint_step": step,
                    "algorithm": "MaskablePPO",
                    "date": weather["weather_date"],
                    "dap": dap,
                    "rain": weather["rain"],
                    "srad": weather["srad"],
                    "tmax": weather["tmax"],
                    "tmin": weather["tmin"],
                    "weather_yrdoy": weather["weather_yrdoy"],
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "xlai": scalar(latest_after.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": bool(terminated or truncated),
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
            done = bool(terminated or truncated)
    finally:
        env.close()
    daily = pd.DataFrame(records)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    out = task168.base03222.base.summarize_daily("MaskablePPO", daily, daily_path, model_path)
    out.update({"station_code": STATION, "year": int(scenario_key), "seed": SEED, "checkpoint_step": step, "run_status": "ok"})
    out.update(task168.base03222.lc_multi.stress_summary(daily))
    return out


def evaluate_and_audit(split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv", keep_default_na=False)
    ok = inventory[inventory["run_status"].astype(str).str.startswith("ok")].copy()
    config = task168.engine.load_config()
    env_config = task168.base03222.direct_ppo.build_env_config(config, split)
    metric_rows: list[dict[str, Any]] = []
    for _, model_row in ok.iterrows():
        for scenario_key in split["year"].astype(int).tolist():
            result = evaluate_checkpoint_actual(config, env_config, model_row, scenario_key)
            result.update({"weather_variant": SCENARIO_KEYS[scenario_key], "scenario_key": scenario_key, "source_year": 2005})
            metric_rows.append(result)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "evaluation" / "172_scenario_episode_metrics.csv", index=False, encoding="utf-8-sig")

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
    pd.DataFrame(action_rows).to_csv(OUT / "evaluation" / "172_action_pair_detail.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(OUT / "evaluation" / "172_action_change_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, action_summary


def run(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "172_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    old = patch_runtime(manifest)
    status = "completed"
    error = ""
    metrics = pd.DataFrame()
    action_summary = pd.DataFrame()
    try:
        task168.base03222.main()
        metrics, action_summary = evaluate_and_audit(split)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        restore_runtime(old)
    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    response_passed = bool(status == "completed" and not non_original.empty and bool(non_original["all_actions_on_grid"].all()) and float(pd.to_numeric(non_original["action_change_rate_after_dap1"], errors="coerce").fillna(0.0).max()) > 0.01)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "total_timesteps": 2000,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "station": STATION,
        "scenario_count": len(VARIANTS),
        "simulation_years": [2005],
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "validation_executed": False,
        "direct_rain_observation": True,
        "reward_changed": False,
        "action_mask_changed": False,
        "action_grid_changed": False,
        "action_response_passed": response_passed,
        "outputs": {
            "training_inventory": rel(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "172_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "172_action_change_summary.csv"),
        },
        "error": error[-5000:] if error else "",
    }
    (OUT / "172_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                f"# {TASK_ID} SYA direct-rain observation PPO smoke",
                "",
                "- Only environment change: append current raw DSSAT `rain` to the observation.",
                "- Reward, action grid, action mask, PPO settings, and six RAIN-only 2005 scenarios unchanged from task 170.",
                "- Daily weather fields are read from the actual current DSSAT state and include `source_year=2005` and `weather_variant`.",
                f"- Training status: `{status}`; action response passed: `{response_passed}`.",
                f"- Result: `{rel(OUT / '172_result.json')}`.",
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

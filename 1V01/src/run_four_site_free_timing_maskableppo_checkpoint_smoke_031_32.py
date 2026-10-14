from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_32_four_site_free_timing_maskableppo_checkpoint_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_32_four_site_free_timing_maskableppo_checkpoint_smoke"
DOC = ROOT / "docs" / "031_32_four_site_free_timing_maskableppo_checkpoint_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models", "daily_outputs", "evaluation", "logs", "tensorboard"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def site_years(meta: dict[str, Any]) -> list[tuple[str, int]]:
    return [(str(x["station_code"]), int(x["year"])) for x in meta["target_site_years"]]


def make_selection(meta: dict[str, Any]) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    frames: list[pd.DataFrame] = []
    for station, year in site_years(meta):
        row = pool[(pool["station_code"].eq(station)) & (pool["year"].astype(int).eq(year))].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected exactly one scenario row for {station}{year}, found {len(row)}")
        row["selected_for_train"] = True
        row["selected_for_eval"] = True
        row["selection_reason"] = "031_32_four_site_free_timing_maskableppo_checkpoint_smoke"
        frames.append(row)
    return pd.concat(frames, ignore_index=True)


def resolved_config(base_config: dict[str, Any], meta: dict[str, Any], seed: int, max_train_steps: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = int(max_train_steps)
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    return cfg


def checkpoint_path(station: str, year: int, seed: int, step: int) -> Path:
    return OUT / "models" / station / f"{station}_{year}_maskableppo_seed{seed}_ckpt{step}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, station: str, year: int, seed: int, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step and x not in self.outer.saved_steps]
                for target in pending:
                    path = checkpoint_path(self.outer.station, self.outer.year, self.outer.seed, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(target)
                return True

        self.station = station
        self.year = int(year)
        self.seed = int(seed)
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


def train_one(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, checkpoint_steps: list[int]) -> list[dict[str, Any]]:
    from sb3_contrib import MaskablePPO

    rows: list[dict[str, Any]] = []
    env = None
    try:
        (OUT / "tensorboard" / station).mkdir(parents=True, exist_ok=True)
        env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_32_seed{seed}_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **base.ppo_kwargs(config),
        )
        cb = FixedStepCheckpointCallback(station, year, seed, checkpoint_steps)
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False, callback=cb.callback)
        for step in checkpoint_steps:
            path = checkpoint_path(station, year, seed, int(step))
            rows.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "seed": int(seed),
                    "checkpoint_step": int(step),
                    "run_status": "ok" if path.exists() else "missing",
                    "model_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "",
                    "model_sha256": sha256_file(path) if path.exists() else "",
                }
            )
    except Exception:
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "seed": int(seed),
                "checkpoint_step": "",
                "run_status": "failed",
                "notes": traceback.format_exc()[-3000:],
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return rows


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, step: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    model_path = checkpoint_path(station, year, seed, step)
    if not model_path.exists():
        return {
            "station_code": station,
            "year": int(year),
            "seed": int(seed),
            "checkpoint_step": int(step),
            "run_status": "missing_model",
        }
    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_32_seed{seed}_ckpt{step}_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = base.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "seed": int(seed),
                    "checkpoint_step": int(step),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": base.scalar(wrow.get("rain"), np.nan),
                    "srad": base.scalar(wrow.get("srad"), np.nan),
                    "tmax": base.scalar(wrow.get("tmax"), np.nan),
                    "tmin": base.scalar(wrow.get("tmin"), np.nan),
                    "swfac": base.scalar(latest.get("swfac")),
                    "nstres": base.scalar(latest.get("nstres")),
                    "topwt": base.scalar(latest.get("topwt")),
                    "grnwt": base.scalar(latest.get("grnwt")),
                    "xlai": base.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    except Exception:
        return {
            "station_code": station,
            "year": int(year),
            "seed": int(seed),
            "checkpoint_step": int(step),
            "run_status": "failed",
            "notes": traceback.format_exc()[-3000:],
        }
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_dir = OUT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{station}_{year}_seed{seed}_ckpt{step}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    row = base.summarize_daily(daily, daily_path, "eval", model_path)
    row.update(
        {
            "station_code": station,
            "year": int(year),
            "seed": int(seed),
            "checkpoint_step": int(step),
            "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
            "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    return row


def write_record(meta: dict[str, Any], checkpoint_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    eval_cols = [
        "station_code",
        "year",
        "seed",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "PFP_N",
        "early_dap1_10_irrigation",
        "early_dap1_10_n",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "action_sequence",
    ]
    lines = [
        "# 031_32 Four-site free-timing MaskablePPO checkpoint smoke record",
        "",
        "## Scope",
        "",
        "- Four target stations: HLA2015, FQA2016, LCA2010, YCA2014.",
        "- Seed0 only.",
        "- Short checkpoint smoke only; no scientific success claim.",
        "- Free timing; no expert-DAP windows.",
        "- Reward/action/constraints unchanged from the frozen 031 MaskablePPO line.",
        "",
        "## Frozen reward and constraints",
        "",
        "Reward: `0.001 * (0.158 * final_yield_at_harvest - 1.1 * irrigation - 1.58 * nitrogen)`.",
        "",
        "Action grid: irrigation `[0,6,12,18,24]`; nitrogen `[0,40,80,120,160]`.",
        "",
        "Caps/guards: I<=160, N<=250, min interval 7 days, irrigation DAP1-120, nitrogen DAP1-90.",
        "",
        "## Checkpoint inventory",
        "",
        checkpoint_df.to_string(index=False) if not checkpoint_df.empty else "No checkpoint rows.",
        "",
        "## Evaluation summary",
        "",
        eval_df[[c for c in eval_cols if c in eval_df.columns]].to_string(index=False) if not eval_df.empty else "No eval rows.",
        "",
        "## Pass/fail",
        "",
    ]
    ok = (not checkpoint_df.empty and not eval_df.empty and checkpoint_df["run_status"].eq("ok").all() and eval_df["run_status"].eq("ok").all())
    lines.append(f"- all_four_station_smoke_pass = {bool(ok)}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "031_32 only verifies that four target stations can execute the frozen free-timing MaskablePPO pipeline. Long training and all-year transfer require a separate approved task.",
        ]
    )
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    args = parser.parse_args()

    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    if args.mode != str(meta.get("run_mode", "smoke")):
        raise RuntimeError(f"031_32 currently only supports configured run_mode={meta.get('run_mode')}")

    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    base_config = direct_ppo.load_yaml(base_config_path)
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)
    selection = make_selection(meta)
    selection.to_csv(OUT / "configs" / "031_32_four_site_selection.csv", index=False, encoding="utf-8-sig")

    run_cfg = meta["smoke"]
    checkpoint_steps = [int(x) for x in run_cfg["checkpoint_steps"]]
    seeds = [int(x) for x in run_cfg["seeds"]]
    max_train_steps = int(run_cfg["max_train_steps"])

    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(base_config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_32_resolved_env_config.yaml")

    checkpoint_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for station, year in site_years(meta):
        for seed in seeds:
            config = resolved_config(base_config, meta, seed, max_train_steps)
            direct_ppo.write_yaml(config, OUT / "configs" / f"031_32_resolved_model_{station}_{year}_seed{seed}.yaml")
            checkpoint_rows.extend(train_one(config, env_config, station, year, seed, checkpoint_steps))
            for step in checkpoint_steps:
                eval_rows.append(evaluate_checkpoint(config, env_config, station, year, seed, int(step)))

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    eval_df = pd.DataFrame(eval_rows)
    checkpoint_path_csv = OUT / "evaluation" / "031_32_checkpoint_inventory.csv"
    eval_path_csv = OUT / "evaluation" / "031_32_smoke_eval_summary.csv"
    checkpoint_df.to_csv(checkpoint_path_csv, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_path_csv, index=False, encoding="utf-8-sig")
    write_record(meta, checkpoint_df, eval_df)

    result = {
        "task": "031_32_four_site_free_timing_maskableppo_checkpoint_smoke",
        "mode": args.mode,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_inventory": str(checkpoint_path_csv.relative_to(ROOT)).replace("\\", "/"),
        "eval_summary": str(eval_path_csv.relative_to(ROOT)).replace("\\", "/"),
        "all_four_station_smoke_pass": bool(
            not checkpoint_df.empty
            and not eval_df.empty
            and checkpoint_df["run_status"].eq("ok").all()
            and eval_df["run_status"].eq("ok").all()
        ),
    }
    (OUT / "031_32_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        cols = ["station_code", "year", "seed", "checkpoint_step", "run_status", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "action_sequence"]
        print(eval_df[[c for c in cols if c in eval_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

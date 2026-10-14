from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_literature_aligned_dqn_free_timing_smoke_031_09 as lit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_13_literature_dqn_interval7_ncost2x.yaml"
OUT = ROOT / "benchmark_results" / "031_13_literature_dqn_interval7_ncost2x"
DOC = ROOT / "docs" / "031_13_literature_dqn_interval7_ncost2x_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
SEEDS = [0, 1, 2]


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "tensorboard/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_13_literature_dqn_interval7_ncost2x"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def train_seed(config: dict, env_config: dict, seed: int, year: int) -> dict[str, Any]:
    from stable_baselines3 import DQN

    cfg = dict(config)
    cfg["seed"] = int(seed)
    station = "SYA"
    model_path = OUT / "models" / station / f"literature_dqn_interval7_ncost2x_seed{seed}.zip"
    status = "ok"
    notes = ""
    env = None
    try:
        env = lit.make_env(cfg, env_config, station, year, seed, f"{station}_{year}_031_13_seed{seed}_train", evaluation=False)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **lit.dqn_kwargs(cfg),
        )
        model.learn(total_timesteps=int(cfg["total_timesteps"]), reset_num_timesteps=True, progress_bar=False)
        model.save(str(model_path.with_suffix("")))
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {
        "algorithm": "literature_DQN_interval7_ncost2x",
        "station_code": station,
        "train_years": str(year),
        "seed": seed,
        "total_timesteps": int(cfg["total_timesteps"]),
        "run_status": status,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
        "notes": notes[-2000:] if notes else "",
    }


def evaluate_seed(config: dict, env_config: dict, seed: int, year: int, model_path: Path) -> dict[str, Any]:
    from stable_baselines3 import DQN

    cfg = dict(config)
    cfg["seed"] = int(seed)
    station = "SYA"
    model = DQN.load(str(model_path))
    weather = direct_ppo.weather_for_daily(cfg)
    env = lit.make_env(cfg, env_config, station, year, seed, f"{station}_{year}_031_13_seed{seed}_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = direct_ppo.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and step_count < int(cfg["runtime"]["max_steps"]):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "seed": int(seed),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_dir = OUT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{year}_seed{seed}_eval_literature_dqn_interval7_ncost2x_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    early_i = float(irr[dap <= 10].sum()) if len(daily) else np.nan
    early_n = float(n[dap <= 10].sum()) if len(daily) else np.nan
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "algorithm": "literature_DQN_interval7_ncost2x",
        "policy_name": "lit_dqn_interval7_ncost2x_5k",
        "station_code": station,
        "year": int(year),
        "seed": int(seed),
        "split": "eval",
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "literature_reward_sum": float(pd.to_numeric(daily.get("literature_reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": early_i,
        "early_dap1_10_n": early_n,
        "reached_both_caps_by_dap10": bool(early_i >= 159.999 and early_n >= 249.999),
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "action_sequence": action_sequence,
        "model_path": str(model_path.relative_to(ROOT)),
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection_path = OUT / "configs" / "031_13_sy2014_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    direct_ppo.ensure_dirs()
    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_13_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)
    year = 2014

    train_rows = []
    eval_rows = []
    for seed in SEEDS:
        train_row = train_seed(config, env_config, seed, year)
        train_rows.append(train_row)
        if train_row["run_status"] == "ok":
            eval_rows.append(evaluate_seed(config, env_config, seed, year, ROOT / train_row["model_path"]))
        else:
            eval_rows.append({"algorithm": "literature_DQN_interval7_ncost2x", "seed": seed, "split": "eval", "run_status": "failed", "notes": train_row["notes"]})

    train_summary = pd.DataFrame(train_rows)
    eval_summary = pd.DataFrame(eval_rows)
    train_path = OUT / "evaluation" / "training_run_summary.csv"
    eval_path = OUT / "evaluation" / "eval_summary.csv"
    train_summary.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_summary.to_csv(eval_path, index=False, encoding="utf-8-sig")

    old_path = ROOT / "benchmark_results" / "031_11_literature_aligned_dqn_min_interval_seed12" / "031_11_seed012_comparison.csv"
    comparison = eval_summary.copy()
    comparison["source"] = "031_13"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old["source"] = old.get("source", "031_10_031_11")
        comparison = pd.concat([old, comparison], ignore_index=True, sort=False)
    comparison_path = OUT / "031_13_vs_031_10_031_11_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 031_13 Literature DQN interval7 ncost2x record",
        "",
        "## Scope",
        "",
        "- SYA2014, seeds 0/1/2.",
        "- Frozen 031_10 DQN design.",
        "- Single changed variable: literature reward nitrogen cost 0.79 -> 1.58.",
        "- N250 cap remains available; this is not a site-year-specific cap change.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False),
        "",
        "## Comparison with 031_10/031_11",
        "",
        comparison.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "This is a single pre-registered nitrogen-cost test, not a parameter scan. Results should not be tuned in place.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "task": "031_13_literature_dqn_interval7_ncost2x",
        "training_run": True,
        "site_year": "SYA2014",
        "seeds": SEEDS,
        "changed_variable": "reward.nitrogen_cost 0.79 -> 1.58",
        "train_summary": str(train_path.relative_to(ROOT)),
        "eval_summary": str(eval_path.relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "031_13_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(eval_summary.to_string(index=False))


if __name__ == "__main__":
    main()


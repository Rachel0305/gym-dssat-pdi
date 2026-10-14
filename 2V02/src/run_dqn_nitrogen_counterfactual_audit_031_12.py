from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_10_literature_aligned_dqn_min_interval_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_12_dqn_nitrogen_counterfactual_audit"
DOC = ROOT / "docs" / "031_12_dqn_nitrogen_counterfactual_audit_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
SEED_DAILY = {
    0: ROOT / "benchmark_results" / "031_10_literature_aligned_dqn_min_interval_smoke" / "daily_outputs" / "SYA" / "2014_eval_literature_aligned_dqn_daily.csv",
    1: ROOT / "benchmark_results" / "031_11_literature_aligned_dqn_min_interval_seed12" / "daily_outputs" / "SYA" / "2014_seed1_eval_literature_aligned_dqn_daily.csv",
    2: ROOT / "benchmark_results" / "031_11_literature_aligned_dqn_min_interval_seed12" / "daily_outputs" / "SYA" / "2014_seed2_eval_literature_aligned_dqn_daily.csv",
}


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = False
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_12_dqn_nitrogen_counterfactual_audit"
    return row


def normalize_action(config: dict, amir: float, anfer: float) -> np.ndarray:
    i_max = float(config["action_scale"]["daily_irrigation_max"])
    n_max = float(config["action_scale"]["daily_n_max"])
    return np.asarray(
        [
            np.clip(2.0 * (float(amir) / i_max) - 1.0, -1.0, 1.0),
            np.clip(2.0 * (float(anfer) / n_max) - 1.0, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def original_schedule(daily_path: Path) -> tuple[dict[int, float], dict[int, float]]:
    df = pd.read_csv(daily_path)
    dap = pd.to_numeric(df["dap"], errors="coerce").astype(int)
    irr = pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0)
    nit = pd.to_numeric(df["safe_action_anfer"], errors="coerce").fillna(0.0)
    i_schedule = {int(d): float(v) for d, v in zip(dap, irr) if float(v) > 0}
    n_schedule = {int(d): float(v) for d, v in zip(dap, nit) if float(v) > 0}
    return i_schedule, n_schedule


def scaled_original_n(n_schedule: dict[int, float], total: float) -> dict[int, float]:
    current = float(sum(n_schedule.values()))
    if current <= 0:
        return {}
    factor = float(total) / current
    out = {dap: amount * factor for dap, amount in n_schedule.items()}
    # Correct tiny floating mismatch on final event.
    if out:
        keys = sorted(out)
        diff = float(total) - sum(out.values())
        out[keys[-1]] += diff
    return out


def variant_n_schedule(variant: str, original_n: dict[int, float]) -> dict[int, float]:
    if variant == "original_replay":
        return dict(original_n)
    if variant == "stage_spread_N250":
        return {1: 50.0, 30: 50.0, 50: 50.0, 65: 50.0, 85: 50.0}
    if variant == "stage_spread_N200":
        return {1: 40.0, 30: 40.0, 50: 40.0, 65: 40.0, 85: 40.0}
    if variant == "stage_spread_N160":
        return {1: 40.0, 30: 40.0, 50: 40.0, 65: 40.0}
    if variant == "early_scaled_N200":
        return scaled_original_n(original_n, 200.0)
    if variant == "early_scaled_N160":
        return scaled_original_n(original_n, 160.0)
    raise ValueError(variant)


def run_variant(seed: int, variant: str, config: dict, env_config: dict, weather: pd.DataFrame) -> dict[str, Any]:
    station = "SYA"
    year = 2014
    i_schedule, original_n = original_schedule(SEED_DAILY[seed])
    n_schedule = variant_n_schedule(variant, original_n)
    env = direct_ppo.make_training_env(config, env_config, station, year, seed, f"{station}_{year}_031_12_seed{seed}_{variant}")
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = direct_ppo.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = direct_ppo.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            amir = float(i_schedule.get(dap, 0.0))
            anfer = float(n_schedule.get(dap, 0.0))
            action = normalize_action(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest_after = direct_ppo.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "station_code": station,
                    "year": year,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "xlai": scalar(latest_after.get("xlai")),
                    "requested_amir": amir,
                    "requested_anfer": anfer,
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
    daily_path = OUT / "daily_outputs" / station / f"2014_seed{seed}_{variant}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nit = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(nit.sum())
    nonzero = daily[(irr > 0) | (nit > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "seed": seed,
        "variant": variant,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((nit > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[nit > 0, "dap"].iloc[0]) if (nit > 0).any() else np.nan,
        "action_sequence": action_sequence,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def main() -> None:
    ensure_dirs()
    config = direct_ppo.load_yaml(BASE_CONFIG)
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT))
    config["total_timesteps"] = 0
    config["action_scale"] = {"daily_irrigation_max": 24.0, "daily_n_max": 160.0}
    config["reward"] = {
        "reward_type": "delta_grnwt_minus_water_nitrogen_cost_for_replay_only",
        "topwt_delta_coef": 0.0,
        "grnwt_delta_coef": 1.0,
        "water_cost": 1.0,
        "nitrogen_cost": 5.0,
    }
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_12_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_12_resolved_env_config.yaml")
    (OUT / "configs" / "031_12_replay_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    weather = direct_ppo.weather_for_daily(config)

    variants = [
        "original_replay",
        "stage_spread_N250",
        "stage_spread_N200",
        "stage_spread_N160",
        "early_scaled_N200",
        "early_scaled_N160",
    ]
    rows = []
    for seed in [0, 1, 2]:
        for variant in variants:
            rows.append(run_variant(seed, variant, config, env_config, weather))
    summary = pd.DataFrame(rows)
    summary_path = OUT / "evaluation" / "031_12_nitrogen_counterfactual_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pivot = summary.pivot_table(index="variant", values=["final_grnwt", "total_n", "profit_simple", "PFP_N", "nstres_days_gt_0p05"], aggfunc=["mean", "min", "max"])
    pivot_path = OUT / "evaluation" / "031_12_nitrogen_counterfactual_variant_aggregate.csv"
    pivot.to_csv(pivot_path, encoding="utf-8-sig")

    lines = [
        "# 031_12 DQN nitrogen counterfactual audit record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- SYA2014 only.",
        "- Seeds 0/1/2 from 031_10/031_11.",
        "- Seed-specific DQN irrigation schedules are kept unchanged.",
        "- Only nitrogen timing/total amount is replaced.",
        "- Current constraints retained: N<=250, DAP>90 no N, 7-day same-resource interval.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Variant aggregate",
        "",
        pivot.to_string(),
        "",
        "## Interpretation boundary",
        "",
        "This is a DSSAT counterfactual audit. It can test whether the learned early N250 schedule is necessary under fixed DQN irrigation, but it is not a new trained policy.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "task": "031_12_dqn_nitrogen_counterfactual_audit",
        "training_run": False,
        "dssat_forward_runs": int(len(rows)),
        "summary": str(summary_path.relative_to(ROOT)),
        "aggregate": str(pivot_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "031_12_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(summary[["seed", "variant", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "PFP_N", "nstres_days_gt_0p05"]].to_string(index=False))


if __name__ == "__main__":
    main()


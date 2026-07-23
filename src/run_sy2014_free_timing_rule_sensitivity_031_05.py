from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_04_free_daily_original_reward_5k_ppo.yaml"
OUT = ROOT / "benchmark_results" / "031_05_sy2014_free_timing_rule_sensitivity"
DOC = ROOT / "docs" / "031_05_sy2014_free_timing_rule_sensitivity_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_05_rule_timing_sensitivity_SY2014"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def normalized_action(config: dict, amir: float, anfer: float) -> np.ndarray:
    i_max = float(config["action_scale"]["daily_irrigation_max"])
    n_max = float(config["action_scale"]["daily_n_max"])
    amir_norm = 2.0 * (float(amir) / i_max) - 1.0
    anfer_norm = 2.0 * (float(anfer) / n_max) - 1.0
    return np.asarray([np.clip(amir_norm, -1.0, 1.0), np.clip(anfer_norm, -1.0, 1.0)], dtype=np.float32)


def scheduled_amount(schedule: dict[int, tuple[float, float]], dap: int) -> tuple[float, float]:
    return schedule.get(int(dap), (0.0, 0.0))


def rule_action(rule: str, dap: int, latest: dict[str, Any], used_i: float, used_n: float) -> tuple[float, float]:
    if rule == "early_dump":
        return 40.0, 80.0

    if rule in {"uniform_spread", "expert_window_budget"}:
        schedule = {
            1: (30.0, 50.0),
            30: (30.0, 50.0),
            50: (30.0, 50.0),
            65: (30.0, 50.0),
            85: (20.0, 50.0),
            110: (20.0, 0.0),
        }
        return scheduled_amount(schedule, dap)

    if rule == "stress_triggered":
        swfac = scalar(latest.get("swfac"), 0.0)
        nstres = scalar(latest.get("nstres"), 0.0)
        amir = 40.0 if swfac > 0.05 and used_i < 160.0 else 0.0
        anfer = 80.0 if nstres > 0.05 and dap <= 90 and used_n < 250.0 else 0.0
        return amir, anfer

    if rule == "delayed_late":
        n_schedule = {83: 40.0, 84: 40.0, 85: 40.0, 86: 40.0, 87: 40.0, 88: 40.0, 89: 10.0}
        i_schedule = {102: 20.0, 103: 20.0, 104: 20.0, 105: 20.0, 106: 20.0, 107: 20.0, 108: 20.0, 109: 20.0}
        return i_schedule.get(dap, 0.0), n_schedule.get(dap, 0.0)

    raise ValueError(rule)


def run_rule(rule: str, config: dict, env_config: dict, weather: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    station = "SYA"
    year = 2014
    env = direct_ppo.make_training_env(config, env_config, station, year, seed, f"{station}_{year}_{rule}")
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
            used_i = float(getattr(env, "safety_state").cumulative_irrigation)
            used_n = float(getattr(env, "safety_state").cumulative_n)
            amir, anfer = rule_action(rule, dap, latest, used_i, used_n)
            action = normalized_action(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest_after = direct_ppo.latest_observation_dict(env, obs, info)
            action_info = dict(env.last_action_info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "rule": rule,
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
                    "reward": float(reward),
                    **action_info,
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_path = OUT / "daily_outputs" / station / f"2014_{rule}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    final_topwt = float(pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    profit_simple = final_grnwt - total_i - 5.0 * total_n if np.isfinite(final_grnwt) else np.nan
    return {
        "rule": rule,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "final_topwt": final_topwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": profit_simple,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def main() -> None:
    ensure_dirs()
    config = direct_ppo.load_yaml(CONFIG)
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT))
    config["total_timesteps"] = 0
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_05_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_05_resolved_env_config.yaml")
    weather = direct_ppo.weather_for_daily(config)

    rules = ["early_dump", "uniform_spread", "expert_window_budget", "stress_triggered", "delayed_late"]
    rows = [run_rule(rule, config, env_config, weather, seed=0) for rule in rules]
    summary = pd.DataFrame(rows)
    summary_path = OUT / "evaluation" / "031_05_rule_timing_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    yield_range = float(summary["final_grnwt"].max() - summary["final_grnwt"].min()) if not summary.empty else np.nan
    profit_range = float(summary["profit_simple"].max() - summary["profit_simple"].min()) if not summary.empty else np.nan
    result = {
        "task": "031_05_sy2014_free_timing_rule_sensitivity",
        "training_or_dssat_run": True,
        "training_run": False,
        "summary": str(summary_path.relative_to(ROOT)),
        "yield_range_kg_ha": yield_range,
        "profit_range": profit_range,
        "rows": summary.to_dict(orient="records"),
        "interpretation": "Fixed timing sensitivity check for SY2014 before free-timing RL reward redesign.",
    }
    (OUT / "031_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# 031_05 SY2014 free-timing rule sensitivity record",
        "",
        "## Purpose",
        "",
        "Before changing reward or training free-timing RL, test whether SY2014 is sensitive to operation timing under the same I160/N250 caps.",
        "",
        "## Rules",
        "",
        "- `early_dump`: use water/N as early as possible.",
        "- `uniform_spread`: spread I160/N250 over fixed stage-like DAPs.",
        "- `expert_window_budget`: same DAP timing as the expert-style window reference under I160/N250; not the official expert dose.",
        "- `stress_triggered`: only act when SWFAC or NSTRES > 0.05.",
        "- `delayed_late`: intentionally late water/N timing.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Timing sensitivity",
        "",
        f"- Yield range across rules: {yield_range:.2f} kg/ha.",
        f"- Simple profit range across rules: {profit_range:.2f}.",
        "",
        "## Interpretation boundary",
        "",
        "This is not RL training. It only checks whether timing choices create enough outcome contrast for a free-timing RL testbed.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "031_05_sy2014_free_timing_rule_sensitivity_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"yield_range_kg_ha={yield_range:.2f}")
    print(f"profit_range={profit_range:.2f}")


if __name__ == "__main__":
    main()

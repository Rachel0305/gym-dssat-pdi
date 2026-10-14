from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as smoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results" / "032_01_lc2010_rule_timing_counterfactual"
DOC = ROOT / "docs" / "032_01_lc2010_rule_timing_counterfactual_record.md"


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/LCA", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def planned_action(rule: str, dap: int, latest: dict[str, Any]) -> tuple[float, float]:
    if rule == "early_frontload":
        schedule = {
            1: (30.0, 120.0),
            8: (0.0, 120.0),
            15: (30.0, 0.0),
        }
        return schedule.get(dap, (0.0, 0.0))
    if rule == "ppo_03200_replay":
        schedule = {
            1: (30.0, 120.0),
            8: (0.0, 120.0),
            9: (15.0, 0.0),
            16: (15.0, 0.0),
        }
        return schedule.get(dap, (0.0, 0.0))
    if rule == "uniform_spread":
        schedule = {
            1: (15.0, 40.0),
            22: (15.0, 40.0),
            43: (15.0, 40.0),
            64: (15.0, 40.0),
            85: (15.0, 40.0),
            106: (15.0, 0.0),
        }
        return schedule.get(dap, (0.0, 0.0))
    if rule == "midseason_shift":
        schedule = {
            36: (30.0, 80.0),
            50: (30.0, 80.0),
            64: (30.0, 80.0),
        }
        return schedule.get(dap, (0.0, 0.0))
    if rule == "late_delayed":
        schedule = {
            70: (30.0, 80.0),
            77: (30.0, 80.0),
            84: (30.0, 80.0),
            91: (30.0, 0.0),
        }
        return schedule.get(dap, (0.0, 0.0))
    if rule == "stress_triggered":
        swfac = scalar(latest.get("swfac"), 0.0)
        nstres = scalar(latest.get("nstres"), 0.0)
        amir = 30.0 if swfac > 0.05 else 0.0
        anfer = 80.0 if nstres > 0.05 and dap <= 90 else 0.0
        return amir, anfer
    raise ValueError(rule)


def nearest_action_index(config: dict, amir: float, anfer: float) -> int:
    grid = smoke.action_grid(config)
    distances = [
        abs(float(a["amir"]) - float(amir)) + abs(float(a["anfer"]) - float(anfer))
        for a in grid
    ]
    return int(np.argmin(distances))


def run_rule(rule: str, config: dict, env_config: dict, weather: pd.DataFrame) -> dict[str, Any]:
    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    seed = int(config["seed"])
    env = smoke.make_env(config, env_config, station, year, seed, f"{station}_{year}_032_01_{rule}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = smoke.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            amir, anfer = planned_action(rule, dap, latest)
            action_index = nearest_action_index(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(action_index)
            done = bool(terminated or truncated)
            latest_after = smoke.latest_observation_dict(env, obs, info)
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
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_path = OUT / "daily_outputs" / station / f"{year}_{rule}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "rule": rule,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple_032": final_grnwt - 1.1 * total_i - 1.58 * total_n if np.isfinite(final_grnwt) else np.nan,
        "reward_stress_aware_sum": float(pd.to_numeric(daily.get("reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "stress_relief_bonus_sum_unscaled": float(pd.to_numeric(daily.get("stress_relief_bonus", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": float(irr[dap <= 10].sum()) if len(daily) else np.nan,
        "early_dap1_10_n": float(n[dap <= 10].sum()) if len(daily) else np.nan,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "action_sequence": action_sequence,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def write_record(summary: pd.DataFrame) -> None:
    yield_range = float(summary["final_grnwt"].max() - summary["final_grnwt"].min()) if not summary.empty else np.nan
    reward_range = float(summary["reward_stress_aware_sum"].max() - summary["reward_stress_aware_sum"].min()) if not summary.empty else np.nan
    best_reward = summary.sort_values("reward_stress_aware_sum", ascending=False).iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# 032_01 LC2010 rule-timing counterfactual audit record",
        "",
        "## Scope",
        "",
        "- LCA2010 only.",
        "- No PPO/DQN training.",
        "- Same coarse grid, caps, 7-day interval, and stress-aware reward accounting as 032_00.",
        "- Deterministic rule policies only.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Ranges",
        "",
        f"- Yield range across rules: {yield_range:.2f} kg/ha.",
        f"- 032 reward-sum range across rules: {reward_range:.6f}.",
        f"- Highest 032 reward rule: {best_reward.get('rule', 'NA')}.",
        "",
        "## Interpretation boundary",
        "",
        "- This audit diagnoses the LC2010 timing/reward landscape.",
        "- It does not select a final management policy and does not train RL.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copyfile(ROOT / "prompts" / "032_01_lc2010_rule_timing_counterfactual.md", OUT / "configs" / "032_01_prompt.md")
    config = direct_ppo.load_yaml(CONFIG)
    selection = smoke.make_selection(config)
    selection.to_csv(OUT / "configs" / "032_01_lc2010_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_01_resolved_env_config.yaml")
    weather = direct_ppo.weather_for_daily(config)
    rules = [
        "early_frontload",
        "uniform_spread",
        "midseason_shift",
        "late_delayed",
        "stress_triggered",
        "ppo_03200_replay",
    ]
    summary = pd.DataFrame([run_rule(rule, config, env_config, weather) for rule in rules])
    summary_path = OUT / "evaluation" / "032_01_rule_timing_counterfactual_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_record(summary)
    result = {
        "task": "032_01_lc2010_rule_timing_counterfactual",
        "training_run": False,
        "dssat_forward_runs": len(rules),
        "summary": str(summary_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "032_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


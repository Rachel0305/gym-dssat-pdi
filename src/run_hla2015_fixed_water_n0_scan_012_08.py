from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_hla2015_fixed_action_counterfactual_012_06 import (
    OUT_DIR as UNUSED_012_06_OUT_DIR,
    YEAR,
    make_env,
    parse_events,
    harvest_yields_from_mgmt,
)
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import install_official_reward_module, prepare_case_at


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_fixed_water_n0_scan_012_08"

SCENARIOS = {
    "fixed_I0_N0": [],
    "fixed_I30_N0": [46],
    "fixed_I60_N0": [46, 53],
    "fixed_I90_N0": [46, 53, 78],
    "fixed_I120_N0": [46, 53, 78, 92],
}


def action_for_dap(dap: float, irrigation_daps: list[int]) -> int:
    return 1 if int(round(float(dap))) in set(irrigation_daps) else 0


def run_scenario(env_args: dict, scenario: str, irrigation_daps: list[int]) -> dict:
    case_dir = OUT_DIR / scenario
    case_dir.mkdir(parents=True, exist_ok=True)
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    env = make_env(env_args)
    rows = []
    try:
        obs, info = env.reset()
        latest = latest_observation_dict(env, obs, info)
        for step in range(260):
            dap = scalar(latest.get("dap", 0.0)) or 0.0
            action = action_for_dap(dap, irrigation_daps)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "action_index": env.last_action_index,
                    "safe_amir": env.last_safe_real_action.get("amir", 0.0),
                    "safe_anfer": env.last_safe_real_action.get("anfer", 0.0),
                    "reward": reward,
                    "delta_grnwt": env.last_reward_components.get("delta_grnwt", 0.0),
                    "water_cost_term": env.last_reward_components.get("water_cost_term", 0.0),
                    "nitrogen_cost_term": env.last_reward_components.get("nitrogen_cost_term", 0.0),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / f"{scenario}_daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "scenario": scenario,
        "year": YEAR,
        "irrigation_daps": irrigation_daps,
        "harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]),
        "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]),
        "irrigation_total": float(daily["safe_amir"].sum()),
        "nitrogen_total": float(daily["safe_anfer"].sum()),
        "economic_reward_total": float(daily["reward"].sum()),
        "input_cost_total": float(daily["water_cost_term"].sum() + daily["nitrogen_cost_term"].sum()),
        "max_swfac": float(daily["swfac"].max()),
        "max_nstres": float(daily["nstres"].max()),
        **events,
    }
    (case_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    install_official_reward_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_dir = OUT_DIR / "_input_case"
    prepare_case_at(YEAR, prep_dir)
    env_args = json.loads((prep_dir / "env_args.json").read_text(encoding="utf-8"))
    summaries = []
    frames = []
    for scenario, daps in SCENARIOS.items():
        print(f"running {scenario}", flush=True)
        summaries.append(run_scenario(env_args, scenario, daps))
        frames.append(pd.read_csv(OUT_DIR / scenario / f"{scenario}_daily.csv"))
    summary = pd.DataFrame(summaries)
    daily = pd.concat(frames, ignore_index=True, sort=False)
    summary.to_csv(OUT_DIR / "hla2015_fixed_water_n0_scan_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "hla2015_fixed_water_n0_scan_daily.csv", index=False, encoding="utf-8-sig")
    print(summary[["scenario", "harvest_yield_kg_ha", "irrigation_total", "economic_reward_total", "max_swfac", "max_nstres"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_discrete_action_probe_012_01 import (
    IRRIGATION_WINDOWS,
    NITROGEN_WINDOWS,
    DiscreteBudgetedDailyActionWrapper,
)
from run_hla2010_dqn_economic_reward_probe_012_03 import EconomicRewardWrapper
from run_hla_official_reward_restart_smoke import (
    LazyScalarGymDssatWrapper,
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)


OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2015_fixed_action_counterfactual_012_06"
)

YEAR = 2015
WATER_COST = 1.0
NITROGEN_COST = 5.0


SCENARIOS: dict[str, dict[str, list[int]]] = {
    "fixed_I0_N0": {
        "irrigation_daps": [],
        "nitrogen_daps": [],
    },
    "fixed_I120_N0": {
        "irrigation_daps": [28, 49, 78, 92],
        "nitrogen_daps": [],
    },
    "fixed_I120_N50": {
        "irrigation_daps": [28, 49, 78, 92],
        "nitrogen_daps": [35],
    },
    "fixed_I120_N100": {
        "irrigation_daps": [28, 49, 78, 92],
        "nitrogen_daps": [35, 60],
    },
    "fixed_I120_N150": {
        "irrigation_daps": [28, 49, 78, 92],
        "nitrogen_daps": [35, 60, 67],
    },
}


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def make_env(env_args: dict):
    import gym

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    base = DiscreteBudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(raw),
        irrigation_windows=IRRIGATION_WINDOWS,
        nitrogen_windows=NITROGEN_WINDOWS,
    )
    return EconomicRewardWrapper(base, water_cost=WATER_COST, nitrogen_cost=NITROGEN_COST)


def action_for_dap(dap: float, irrigation_daps: list[int], nitrogen_daps: list[int]) -> int:
    rounded = int(round(float(dap)))
    do_i = rounded in set(irrigation_daps)
    do_n = rounded in set(nitrogen_daps)
    if do_i and do_n:
        return 3
    if do_i:
        return 1
    if do_n:
        return 2
    return 0


def run_scenario(base_env_args: dict, scenario: str, irrigation_daps: list[int], nitrogen_daps: list[int]) -> dict:
    case_dir = OUT_DIR / scenario
    case_dir.mkdir(parents=True, exist_ok=True)
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    env = make_env(base_env_args)
    rows = []
    try:
        obs, info = env.reset()
        latest = latest_observation_dict(env, obs, info)
        for step in range(260):
            dap = scalar(latest.get("dap", 0.0)) or 0.0
            action = action_for_dap(dap, irrigation_daps, nitrogen_daps)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "action_index": env.last_action_index,
                    "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": env.used_irrigation,
                    "used_nitrogen": env.used_nitrogen,
                    "reward": reward,
                    "delta_grnwt": env.last_reward_components.get("delta_grnwt", np.nan),
                    "water_cost_term": env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": env.last_reward_components.get("nitrogen_cost_term", np.nan),
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
        "water_cost": WATER_COST,
        "nitrogen_cost": NITROGEN_COST,
        "irrigation_daps": irrigation_daps,
        "nitrogen_daps": nitrogen_daps,
        "harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "irrigation_total": float(daily["safe_amir"].sum()) if not daily.empty else None,
        "nitrogen_total": float(daily["safe_anfer"].sum()) if not daily.empty else None,
        "economic_reward_total": float(daily["reward"].sum()) if not daily.empty else None,
        "input_cost_total": float(daily["water_cost_term"].sum() + daily["nitrogen_cost_term"].sum()) if not daily.empty else None,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else None,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else None,
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
    daily_frames = []
    for scenario, cfg in SCENARIOS.items():
        print(f"running {scenario}", flush=True)
        summary = run_scenario(env_args, scenario, cfg["irrigation_daps"], cfg["nitrogen_daps"])
        summaries.append(summary)
        daily_path = OUT_DIR / scenario / f"{scenario}_daily.csv"
        daily_frames.append(pd.read_csv(daily_path))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "hla2015_fixed_action_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    all_daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    all_daily.to_csv(OUT_DIR / "hla2015_fixed_action_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    print(summary_df[["scenario", "harvest_yield_kg_ha", "irrigation_total", "nitrogen_total", "economic_reward_total", "max_swfac", "max_nstres"]].to_string(index=False))


if __name__ == "__main__":
    main()

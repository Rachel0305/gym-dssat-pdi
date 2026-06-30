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

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import (
    LazyScalarGymDssatWrapper,
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_fixed_irrigation_n_scan_012_02"

IRRIGATION_DAP = 30
IRRIGATION_MM = 30.0
N_SCHEDULES = {
    "N0": [],
    "N50": [30],
    "N100": [30, 56],
    "N150": [30, 56, 63],
}
N_PER_EVENT = 50.0


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


def run_case(year: int, label: str, fert_daps: list[int]) -> dict:
    install_official_reward_module()
    import gym

    case_dir = OUT_DIR / str(year) / label
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    env = LazyScalarGymDssatWrapper(raw)
    rows = []
    used_i = 0.0
    used_n = 0.0
    try:
        obs, info = env.reset()
        for step in range(240):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step))))
            real = {
                "amir": IRRIGATION_MM if dap_before == IRRIGATION_DAP else 0.0,
                "anfer": N_PER_EVENT if dap_before in fert_daps else 0.0,
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            used_i += real["amir"]
            used_n += real["anfer"]
            latest = latest_observation_dict(env, obs, info)
            done = bool(terminated or truncated)
            rows.append(
                {
                    "step": step,
                    "dap_before": dap_before,
                    "dap": scalar(latest.get("dap")),
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "used_irrigation": used_i,
                    "used_nitrogen": used_n,
                    "reward": reward,
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
    daily.to_csv(case_dir / "daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "year": year,
        "label": label,
        "irrigation_schedule": {IRRIGATION_DAP: IRRIGATION_MM},
        "fertilizer_daps": fert_daps,
        "fertilizer_per_event": N_PER_EVENT,
        "planned_irrigation_total": IRRIGATION_MM,
        "planned_fertilizer_total": len(fert_daps) * N_PER_EVENT,
        "mgmtevent_irrigation_total": events.get("irrigation_total_mgmtevent"),
        "mgmtevent_fertilizer_total": events.get("fertilizer_total_mgmtevent"),
        "mgmtevent_irrigation_events": events.get("irrigation_events_mgmtevent"),
        "mgmtevent_fertilizer_events": events.get("fertilizer_events_mgmtevent"),
        "harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else None,
        "mean_swfac": float(daily["swfac"].mean()) if not daily.empty else None,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else None,
        "mean_nstres": float(daily["nstres"].mean()) if not daily.empty else None,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else None,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for label, fert_daps in N_SCHEDULES.items():
        print(f"running {label}", flush=True)
        summaries.append(run_case(2010, label, fert_daps))
    df = pd.DataFrame(summaries)
    df.to_csv(OUT_DIR / "hla2010_fixed_dqn_irrigation_n_scan_summary.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(json.dumps({"out_dir": str(OUT_DIR)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

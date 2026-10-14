from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import (
    num,
    parse_summary_out,
    select_matching_row,
)
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared


OUT_ROOT = ROOT / "benchmark_results" / "021_18"
ENV_SOURCE = (
    ROOT
    / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1/dqn_env_args.json"
)
SMOKE_REFERENCE_YIELD = 11199.0
OFFICIAL_EXPERT_YIELD = 11077.0
OFFICIAL_EXPERT_PFP = 36.92333333333333

IRRIGATION = {
    "I120_ref": {1: 30.0, 8: 15.0, 22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0, 79: 15.0},
    "I90_reduced": {8: 15.0, 22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0, 79: 15.0},
    "I75_no_early": {22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0, 79: 15.0},
    "I75_no_late": {8: 15.0, 22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0},
    "I60_reduced": {22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0},
    "I30_reduced": {56: 15.0, 79: 15.0},
}
N_TIMES = {
    "early": [1, 8, 15, 29],
    "mid": [29, 42, 56, 70],
    "late": [69, 79, 89, 99],
}
N_AMOUNTS = {
    "N150": [50.0, 50.0, 50.0, 0.0],
    "N200": [50.0, 100.0, 50.0, 0.0],
    "N250": [50.0, 100.0, 100.0, 0.0],
    "N300": [50.0, 100.0, 100.0, 50.0],
}


def nitrogen_schedule(timing: str, total: str) -> dict[int, float]:
    return {
        dap: amount
        for dap, amount in zip(N_TIMES[timing], N_AMOUNTS[total])
        if amount > 0
    }


def validate_schedule(irrigation: dict[int, float], nitrogen: dict[int, float]) -> None:
    if sum(irrigation.values()) > 120.0 + 1e-9 or sum(nitrogen.values()) > 300.0 + 1e-9:
        raise ValueError("Seasonal budget exceeded")
    if any(v not in {15.0, 30.0} for v in irrigation.values()):
        raise ValueError("Irrigation amount outside DQN action levels")
    if any(v not in {50.0, 100.0} for v in nitrogen.values()):
        raise ValueError("Nitrogen amount outside DQN action levels")
    daps = sorted(set(irrigation) | set(nitrogen))
    if any(dap < 1 or dap > 120 for dap in daps):
        raise ValueError("DAP outside frozen decision window")
    if any(b - a < 7 for a, b in zip(daps, daps[1:])):
        raise ValueError(f"Shared operation interval <7 d: {daps}")


def make_env_args(run_dir: Path) -> dict[str, Any]:
    args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    return args


def copy_snapshot(env: Any, destination: Path) -> None:
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if not tmp or not Path(tmp).exists():
        raise RuntimeError("PDI temporary output directory unavailable")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(Path(tmp), destination)


def run_one(name: str, irrigation: dict[int, float], nitrogen: dict[int, float], phase: str) -> dict[str, Any]:
    validate_schedule(irrigation, nitrogen)
    run_dir = OUT_ROOT / "runs" / name
    if run_dir.exists():
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"Incomplete existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    schedule: dict[int, dict[str, float]] = {}
    for dap, amount in irrigation.items():
        schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})["amir"] += amount
    for dap, amount in nitrogen.items():
        schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})["anfer"] += amount
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    env_args = make_env_args(run_dir)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")

    env = shared.make_raw_env(env_args)
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    fired: set[int] = set()
    try:
        obs, info = env.reset()
        for step in range(420):
            before = latest_observation_dict(env, obs, info)
            dap_action = int(round(scalar(before.get("dap", step)) or 0))
            real = schedule[dap_action] if dap_action in schedule and dap_action not in fired else {"amir": 0.0, "anfer": 0.0}
            if dap_action in schedule:
                fired.add(dap_action)
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            row = {
                "scenario": name, "step": step, "dap_action": dap_action,
                "dap": scalar(latest.get("dap")), "yrdoy": scalar(latest.get("yrdoy")),
                "rain": scalar(latest.get("rain")), "grnwt": scalar(latest.get("grnwt")),
                "topwt": scalar(latest.get("topwt")), "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")), "irrigation_mm_action": float(real["amir"]),
                "fertilizer_kg_ha_action": float(real["anfer"]), "raw_reward": repr(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
            }
            daily_rows.append(row)
            if real["amir"] > 0 or real["anfer"] > 0:
                action_rows.append(row.copy())
            if terminated or truncated:
                break
    finally:
        copy_snapshot(env, run_dir / "pdi_tmp_snapshot_eval")
        env.close()

    daily = pd.DataFrame(daily_rows)
    actions = pd.DataFrame(action_rows)
    daily.to_csv(run_dir / "daily_values.csv", index=False)
    actions.to_csv(run_dir / "requested_actions.csv", index=False)
    snapshot = run_dir / "pdi_tmp_snapshot_eval"
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    plantgro.to_csv(run_dir / "plantgro_parsed.csv", index=False)
    gwad = pd.to_numeric(plantgro["GWAD"], errors="coerce").dropna()
    cwad = pd.to_numeric(plantgro["CWAD"], errors="coerce").dropna()
    final_gwad, final_cwad = float(gwad.iloc[-1]), float(cwad.iloc[-1])
    expected_i, expected_n = float(sum(irrigation.values())), float(sum(nitrogen.values()))
    summary_rows = parse_summary_out(snapshot / "Summary.OUT")
    srow, match_score, row_index = select_matching_row(summary_rows, final_gwad, expected_i, expected_n)
    ircm, nicm, nucm, etcp = num(srow, "IRCM"), num(srow, "NICM"), num(srow, "NUCM"), num(srow, "ETCP")
    ypem, ypim, ypnam, ypnum = num(srow, "YPEM"), num(srow, "YPIM"), num(srow, "YPNAM"), num(srow, "YPNUM")
    result = {
        "scenario": name, "phase": phase, "final_gwad": final_gwad, "final_cwad": final_cwad,
        "requested_irrigation_total": float(actions["irrigation_mm_action"].sum()),
        "requested_nitrogen_total": float(actions["fertilizer_kg_ha_action"].sum()),
        "summary_irrigation_total": ircm, "summary_nitrogen_total": nicm,
        "nitrogen_uptake_kg_ha": nucm, "etcp_mm": etcp,
        "WP_ET_kg_m3": ypem * 0.1 if ypem is not None and ypem >= 0 else np.nan,
        "WP_ET_recalculated": final_gwad / etcp / 10 if etcp and etcp > 0 else np.nan,
        "IWP_gross_kg_m3": ypim * 0.1 if ircm and ircm > 0 and ypim is not None and ypim >= 0 else np.nan,
        "PFP_N_kg_kg": ypnam if nicm and nicm > 0 and ypnam is not None and ypnam >= 0 else np.nan,
        "NUtE_kg_kg": ypnum if nucm and nucm > 0 and ypnum is not None and ypnum >= 0 else np.nan,
        "PNB_N": nucm / nicm if nicm and nicm > 0 and nucm is not None else np.nan,
        "summary_match_score": match_score, "summary_row_index": row_index,
        "missing_schedule_daps": json.dumps(sorted(set(schedule) - fired)),
        "yield_pass": bool(final_gwad >= OFFICIAL_EXPERT_YIELD),
        "pfp_pass": bool(nicm and nicm > 0 and ypnam is not None and ypnam >= OFFICIAL_EXPERT_PFP),
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    return result


def save_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "021_18_candidate_summary.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates("scenario", keep="last").sort_values(["phase", "scenario"])
    new.to_csv(path, index=False)
    return new


def run_smoke() -> pd.DataFrame:
    row = run_one("smoke_I120_early_N300", IRRIGATION["I120_ref"], nitrogen_schedule("early", "N300"), "smoke")
    if abs(row["final_gwad"] - SMOKE_REFERENCE_YIELD) > 2.0:
        raise RuntimeError(f"Smoke yield mismatch: {row['final_gwad']} vs {SMOKE_REFERENCE_YIELD}")
    if row["summary_irrigation_total"] != 120.0 or row["summary_nitrogen_total"] != 300.0:
        raise RuntimeError("Smoke Summary.OUT resource totals mismatch")
    if json.loads(row["missing_schedule_daps"]):
        raise RuntimeError("Smoke has missing scheduled actions")
    return save_summary([row])


def run_stage_a() -> pd.DataFrame:
    rows = []
    for timing in N_TIMES:
        for total in N_AMOUNTS:
            rows.append(run_one(f"A_I120_{timing}_{total}", IRRIGATION["I120_ref"], nitrogen_schedule(timing, total), "stage_a"))
    return save_summary(rows)


def run_stage_b() -> pd.DataFrame:
    summary = pd.read_csv(OUT_ROOT / "021_18_candidate_summary.csv")
    stage_a = summary[summary["phase"].eq("stage_a")].copy()
    if len(stage_a) != 12:
        raise RuntimeError("Stage A incomplete; expected 12 candidates")
    stage_a["rank_score"] = (
        stage_a["yield_pass"].astype(int) * 1_000_000
        + stage_a["pfp_pass"].astype(int) * 100_000
        + stage_a["final_gwad"]
    )
    top = stage_a.sort_values(["rank_score", "PFP_N_kg_kg"], ascending=False).head(3)
    rows = []
    for scenario in top["scenario"]:
        _, _, timing, total = scenario.split("_")
        for i_name in ("I90_reduced", "I60_reduced", "I30_reduced"):
            rows.append(run_one(f"B_{i_name}_{timing}_{total}", IRRIGATION[i_name], nitrogen_schedule(timing, total), "stage_b"))
    return save_summary(rows)


def run_refinement() -> pd.DataFrame:
    rows = []
    for total in ("N200", "N250", "N300"):
        for i_name in ("I75_no_early", "I75_no_late"):
            rows.append(
                run_one(
                    f"R_{i_name}_mid_{total}",
                    IRRIGATION[i_name],
                    nitrogen_schedule("mid", total),
                    "adaptive_refinement",
                )
            )
    return save_summary(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "stage-a", "stage-b", "refinement"], required=True)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUT_ROOT / "021_18_search_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"env_source": str(ENV_SOURCE.relative_to(ROOT)), "irrigation": IRRIGATION, "n_times": N_TIMES, "n_amounts": N_AMOUNTS, "success": {"yield_min": OFFICIAL_EXPERT_YIELD, "pfp_n_min": OFFICIAL_EXPERT_PFP}}, indent=2), encoding="utf-8")
    if args.phase == "smoke":
        result = run_smoke()
    elif args.phase == "stage-a":
        result = run_stage_a()
    elif args.phase == "stage-b":
        result = run_stage_b()
    else:
        result = run_refinement()
    print(result.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()

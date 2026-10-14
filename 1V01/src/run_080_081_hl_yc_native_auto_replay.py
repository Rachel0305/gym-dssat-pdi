"""Isolated 10K PPO replay against native DSSAT automatic irrigation.

This runner imports the stable site-local replay helpers but does not modify
their source files. It intentionally excludes dssat_auto_external_n.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "HLA": ROOT / "src/054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_five_scenario_figures.py",
    "YCA": ROOT / "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
}
SCENARIOS = ["null", "recorded_farmer_template", "official_extension_expert", "dssat_auto", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "official_extension_expert": "Official expert",
    "dssat_auto": "Native DSSAT auto irrigation",
    "rl_candidate": "PPO",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_module(station: str):
    path = MODULES[station]
    spec = importlib.util.spec_from_file_location(f"native_replay_{station}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.STATION = station
    module.SITE = "HL" if station == "HLA" else "YC"
    module.SCENARIOS = list(SCENARIOS)
    module.LABELS = dict(LABELS)
    module.PPO_REPLAY_YIELD_ATOL_KG_HA = 1.0
    return module


def snapshot_map(module, baseline_root: Path, ppo_snapshot: Path, year: int) -> dict[str, Path]:
    station = module.STATION
    root = baseline_root / "snapshots" / station / str(year)
    return {
        "null": root / "null",
        "recorded_farmer_template": root / "recorded_farmer_template",
        "official_extension_expert": root / "official_extension_expert",
        "dssat_auto": root / "dssat_auto",
        "rl_candidate": ppo_snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--years", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    station = str(cfg["station_code"])
    if station not in MODULES or cfg.get("external_n_control"):
        raise ValueError("only HLA/YCA native-automatic-irrigation replay is allowed")
    ppo_root = ROOT / cfg["ppo_run_dir"]
    baseline_root = ROOT / cfg["baseline_root"]
    out = ROOT / cfg["output_root"]
    checkpoint = int(cfg["checkpoint"])
    years = [int(x) for x in args.years.split(",") if x.strip()] or list(map(int, cfg["validation_years"]))
    module = load_module(station)
    required = [ppo_root, baseline_root]
    for year in years:
        required.extend(
            baseline_root / "snapshots" / station / str(year) / scenario / "Summary.OUT"
            for scenario in ["null", "recorded_farmer_template", "official_extension_expert", "dssat_auto"]
        )
    missing = [rel(path) for path in required if not path.exists()]
    payload: dict[str, Any] = {
        "task": cfg["task_id"],
        "station": station,
        "checkpoint": checkpoint,
        "years": years,
        "ppo_run_dir": rel(ppo_root),
        "baseline_root": rel(baseline_root),
        "output_root": rel(out),
        "scenarios": SCENARIOS,
        "external_n_control": False,
        "missing_inputs": missing,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if missing:
        raise FileNotFoundError(missing[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / cfg_path.name).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    # The 078/079 isolated runners preserve the shared 042_10 engine inventory
    # filenames. Keep the new 080/081 task identity in the output, but pass the
    # source inventory identity and the training scope expected by the legacy
    # replay helper.
    replay_cfg = dict(cfg)
    replay_cfg["task_id"] = str(cfg.get("ppo_task_id", "042_10"))
    replay_cfg["task_name"] = str(cfg.get("ppo_task_name", "native_auto_replay_source"))
    replay_cfg["scope"] = {"validation_years": years}
    replay_cfg["training"] = {"total_timesteps": 25000, "checkpoint_steps": [5000, 10000, 25000]}
    for year in years:
        ppo_snapshot = module.replay_ppo_snapshot(replay_cfg, checkpoint, year, out, ppo_root)
        snap = snapshot_map(module, baseline_root, ppo_snapshot, year)
        absent = [rel(path) for path in snap.values() if not (path / "Summary.OUT").exists()]
        if absent:
            raise FileNotFoundError(absent[0])
        merged = module.add_unified_reward(
            pd.concat([module.daily_from_snapshot(path, year, scenario) for scenario, path in snap.items()], ignore_index=True)
        )
        frames.append(merged)
        for scenario, sub in merged.groupby("scenario"):
            last = sub.iloc[-1]
            metrics = module.baseline.metrics_from_snapshot(snap[scenario], float(last["grain_yield_kg_ha"]))
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            summaries.append(
                {
                    "station_code": station,
                    "site": module.SITE,
                    "year": year,
                    "scenario": scenario,
                    "grain_yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "biomass_kg_ha": float(last["biomass_kg_ha"]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": n_total,
                    "etcp_mm": float(metrics["etcp_mm"]),
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) and n_total > 0 else np.nan,
                    "max_wspd": float(pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").max()),
                    "max_nstd": float(pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").max()),
                    "unified_reward": float(last["unified_cumulative_reward"]),
                    "snapshot_path": rel(snap[scenario]),
                }
            )
    daily = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    table_dir = out / "tables"
    table_dir.mkdir(exist_ok=True)
    daily.to_csv(table_dir / f"{cfg['task_id']}_native_auto_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / f"{cfg['task_id']}_native_auto_summary.csv", index=False, encoding="utf-8-sig")
    gaps: list[dict[str, Any]] = []
    for year, subset in summary.groupby("year"):
        ppo = subset[subset["scenario"].eq("rl_candidate")].iloc[0]
        four = subset[~subset["scenario"].eq("rl_candidate")]
        for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            best = pd.to_numeric(four[metric], errors="coerce").max(skipna=True)
            val = float(ppo[metric]) if pd.notna(ppo[metric]) else np.nan
            gaps.append({"year": int(year), "metric": metric, "ppo_value": val, "four_baseline_max": best, "ppo_minus_four_max": val - best if pd.notna(best) and pd.notna(val) else np.nan})
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(table_dir / f"{cfg['task_id']}_native_auto_gaps.csv", index=False, encoding="utf-8-sig")
    payload.update(
        {
            "status": "completed",
            "summary_csv": rel(table_dir / f"{cfg['task_id']}_native_auto_summary.csv"),
            "daily_csv": rel(table_dir / f"{cfg['task_id']}_native_auto_daily.csv"),
            "gaps_csv": rel(table_dir / f"{cfg['task_id']}_native_auto_gaps.csv"),
            "ppo_win_counts": {metric: int((pd.to_numeric(gaps_df.loc[gaps_df["metric"].eq(metric), "ppo_minus_four_max"], errors="coerce") > 0).sum()) for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]},
        }
    )
    (out / f"{cfg['task_id']}_result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / f"{cfg['task_id']}_record.md").write_text(
        "\n".join([
            f"# {cfg['task_id']} {module.SITE} native automatic irrigation replay",
            "",
            f"- PPO checkpoint: `{checkpoint}`; validation years: `{years}`.",
            "- Scenarios: null, recorded farmer template, official extension expert, native DSSAT automatic irrigation, PPO.",
            "- `dssat_auto_external_n` and all external nitrogen controls were excluded.",
            f"- Summary: `{payload['summary_csv']}`; gaps: `{payload['gaps_csv']}`.",
            f"- PPO win counts versus the best non-PPO scenario: `{payload['ppo_win_counts']}`.",
        ]) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

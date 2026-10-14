"""Sequential, checkpoint-verified PPO ETCP replay for the 055_03 HL/FQ figures."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hl_fq_8seed_cross_site.analyze_hl_fq_8seed import config_and_output, selected_summary

OUT = ROOT / "results/hl_fq_8seed_cross_site/055_03_five_scenario/wp_et_replay"
RUNNERS = {
    "HL": ROOT / "src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py",
    "FQ": ROOT / "src/051_fqa_originIC_site_transfer/run_051_03_fqa_originIC_five_scenario_figures.py",
}


def load_runner(site: str):
    spec = importlib.util.spec_from_file_location(f"five_scenario_{site.lower()}", RUNNERS[site])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["HL", "FQ"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--years", default="2014,2015,2016,2017,2018,2019,2020,2021,2022,2023")
    args = parser.parse_args()
    site, seed = args.site, args.seed
    if seed not in (range(8) if site == "HL" else range(7)):
        raise ValueError("Only completed historical-weather seeds can be replayed")
    years = [int(x) for x in args.years.split(",")]
    if not years or any(y not in range(2014, 2024) for y in years):
        raise ValueError("Invalid validation years")
    cfg_path, ppo_root = config_and_output(site, seed)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    run = load_runner(site)
    station = "HLA" if site == "HL" else "FQA"
    expected = selected_summary(ppo_root, station, seed)
    work = OUT / site / f"seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for year in years:
        metric_file = work / f"{year}_wp_et.json"
        if metric_file.is_file():
            record = json.loads(metric_file.read_text(encoding="utf-8"))
            if record.get("year") == year and record.get("seed") == seed:
                records.append(record)
                print(f"SKIP {site} seed{seed} {year}: existing verified metric", flush=True)
                continue
        snapshot = run.replay_ppo_snapshot(cfg, 100000, year, work, ppo_root)
        daily = run.daily_from_snapshot(snapshot, year, "rl_candidate")
        final_yield = float(daily.grain_yield_kg_ha.iloc[-1])
        measured = run.baseline.metrics_from_snapshot(snapshot, final_yield)
        saved_yield = float(expected[year]["final_grnwt"])
        if abs(final_yield - saved_yield) > 1.0:
            raise ValueError(f"Yield endpoint mismatch: {site} seed{seed} {year}")
        saved_i = float(expected[year]["total_irrigation"])
        saved_n = float(expected[year]["total_n"])
        if abs(measured["actual_irrigation_mm"] - saved_i) > 0.51 or abs(measured["actual_nitrogen_kg_ha"] - saved_n) > 0.51:
            raise ValueError(f"Water/N endpoint mismatch: {site} seed{seed} {year}")
        record = {
            "site": site, "seed": seed, "year": year,
            "saved_yield_kg_ha": saved_yield, "replay_yield_kg_ha": final_yield,
            "saved_irrigation_mm": saved_i, "replay_irrigation_mm": measured["actual_irrigation_mm"],
            "saved_n_kg_ha": saved_n, "replay_n_kg_ha": measured["actual_nitrogen_kg_ha"],
            "etcp_mm": measured["etcp_mm"], "WP_ET_kg_m3": measured["WP_ET_kg_m3"],
            "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
            "metric_source": "Summary.OUT ETCP from deterministic 100K checkpoint replay",
        }
        metric_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        print(f"DONE {site} seed{seed} {year}: ETCP={record['etcp_mm']:.1f}, WP_ET={record['WP_ET_kg_m3']:.3f}", flush=True)
    with (work / "wp_et_replay_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

"""Isolated 2018 replay of each archived FQA WGEN 100K model on 088 correction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/fqa_multiyear_wgen_ppo_052_8seed/evaluate_100k.py"
OUT = ROOT / "results/fqa_wgen_052_corrected2018_231"
CORRECTED = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction/corrected_input"
MANIFEST = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction/088_00_correction_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    weather = CORRECTED / "FQ/CNFQ1801.WTH"
    digest = hashlib.sha256(weather.read_bytes()).hexdigest()
    if digest != manifest["corrected_weather_sha256"]:
        raise RuntimeError("Corrected weather hash mismatch")
    spec = importlib.util.spec_from_file_location("fqa_wgen_eval_052", SRC)
    assert spec and spec.loader
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    target = OUT / f"seed_{args.seed:02d}"
    summary_file = target / "corrected2018_metrics.csv"
    if summary_file.exists():
        raise FileExistsError(summary_file)
    ev.INPUT_ROOT = CORRECTED
    ev.VALIDATION = OUT
    ev.ppo_safe_rendering.MULTISITE_INPUT_ROOT = CORRECTED
    config, env_config = ev.configure_eval()
    ev.engine.LOWIC_INPUT_ROOT = CORRECTED
    model_file = ev.model_path(args.seed)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_file), device="cpu")
    snapshot = ev.run_year(
        args.seed, 2018, model, config, env_config,
        target / "snapshot", ev.sha256(model_file),
    )
    raw = ev.daily_from_snapshot(snapshot, 2018, f"ppo_seed_{args.seed}")
    final_yield = float(pd.to_numeric(raw.grain_yield_kg_ha, errors="coerce").iloc[-1])
    measured = ev.baseline.metrics_from_snapshot(snapshot, final_yield)
    target.mkdir(parents=True, exist_ok=True)
    raw.to_csv(target / "corrected2018_daily.csv", index=False, encoding="utf-8-sig")
    row = {
        "site": "FQ", "station_code": "FQA", "seed": args.seed, "year": 2018,
        "checkpoint_step": 100000, "scenario": f"ppo_seed_{args.seed}",
        "scenario_label": f"PPO seed {args.seed}", "grain_yield_kg_ha": final_yield,
        "total_irrigation_mm": measured["actual_irrigation_mm"],
        "total_n_kg_ha": measured["actual_nitrogen_kg_ha"],
        "etcp_mm": measured["etcp_mm"], "WP_ET_kg_m3": measured["WP_ET_kg_m3"],
        "PFP_N_kg_grain_per_kg_N": measured["PFP_N_kg_kg"],
        "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
        "model_sha256": ev.sha256(model_file), "weather_sha256": digest,
        "correction_manifest": MANIFEST.relative_to(ROOT).as_posix(),
    }
    pd.DataFrame([row]).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(json.dumps(row, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

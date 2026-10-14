"""Phase-0 read-only full-metric replay for existing HL/YC checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
YEARS = list(range(2014, 2024))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def view_for_candidate(source: Path, candidate_id: str, formal_source: str) -> Path:
    """Create a small read-only artifact view; model paths remain host-relative."""
    view = ROOT / "benchmark_results" / "_066_phase0_views" / candidate_id
    if view.exists():
        shutil.rmtree(view)
    (view / "evaluation").mkdir(parents=True)
    for item in (source / "evaluation").glob("*.csv"):
        shutil.copy2(item, view / "evaluation" / item.name)
    formal = source / formal_source
    if not formal.exists():
        raise FileNotFoundError(formal)
    # The reporting module only uses this file as a provenance presence check.
    view_name = "054_00_formal_result.json" if candidate_id.startswith("hl") else "055_00_formal_result.json"
    shutil.copy2(formal, view / view_name)
    return view


def run_candidate(spec: dict[str, Any]) -> pd.DataFrame:
    station = spec["station"]
    if station == "HLA":
        mod = load_module(ROOT / "src/054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_five_scenario_figures.py", f"phase0_hl_{spec['id']}")
    else:
        mod = load_module(ROOT / "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py", f"phase0_yc_{spec['id']}")
    cfg = mod.read_json(mod.DEFAULT_CONFIG)
    auto_cfg = mod.read_json(mod.DEFAULT_AUTO_CONFIG)
    ppo_root = Path(spec["ppo_root"])
    if not ppo_root.is_absolute():
        ppo_root = ROOT / ppo_root
    ppo_root = ppo_root.resolve()
    if spec.get("view", False):
        ppo_root = view_for_candidate(ppo_root, spec["id"], spec["formal_source"])
    baseline_root = mod.BASELINE_ROOT
    auto_root = mod.auto_root(auto_cfg, "")
    out = mod.output_root(cfg, auto_cfg, int(spec["checkpoint"]), f"phase0_{spec['id']}", "")
    if out.exists():
        raise FileExistsError(out)
    (out / "tables").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        ppo_snapshot = mod.replay_ppo_snapshot(cfg, int(spec["checkpoint"]), year, out, ppo_root)
        snapshots = mod.build_snapshot_map(baseline_root, auto_root, ppo_snapshot, year)
        for scenario, snapshot in snapshots.items():
            daily = mod.daily_from_snapshot(snapshot, year, scenario)
            last = daily.iloc[-1]
            metrics = mod.baseline.metrics_from_snapshot(snapshot, float(last["grain_yield_kg_ha"]))
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            pfp = float(metrics["PFP_N_kg_kg"]) if n_total > 0 and pd.notna(metrics["PFP_N_kg_kg"]) else float("nan")
            rows.append(
                {
                    "candidate": spec["id"],
                    "station": station,
                    "checkpoint": int(spec["checkpoint"]),
                    "year": year,
                    "scenario": scenario,
                    "yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "WP_ET": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N": pfp,
                    "irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "nitrogen_kg_ha": n_total,
                    "snapshot": rel(snapshot),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "tables" / "phase0_season_summary.csv", index=False, encoding="utf-8-sig")
    means = frame.groupby(["candidate", "station", "checkpoint", "scenario"], as_index=False)[
        ["yield_kg_ha", "WP_ET", "PFP_N", "irrigation_mm", "nitrogen_kg_ha"]
    ].mean()
    means.to_csv(out / "tables" / "phase0_means.csv", index=False, encoding="utf-8-sig")
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["all", "hl_base25", "hl_lr25", "yc_base25", "yc_lr5", "yc_lr10", "yc_lr25"], default="all")
    parser.add_argument("--attempt", default="")
    args = parser.parse_args()
    specs = [
        {"id": "hl_base25", "station": "HLA", "checkpoint": 25000, "ppo_root": "benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo", "view": False},
        {"id": "hl_lr25", "station": "HLA", "checkpoint": 25000, "ppo_root": "benchmark_results/064_01_hla_lowIC_lr1e4_rescue25k", "view": True, "formal_source": "064_01_rescue_result.json"},
        {"id": "yc_base25", "station": "YCA", "checkpoint": 25000, "ppo_root": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo", "view": False},
        {"id": "yc_lr5", "station": "YCA", "checkpoint": 5000, "ppo_root": "benchmark_results/065_01_yca_lowIC_lr1e4_rescue25k_maskableppo_smoke2k", "view": True, "formal_source": "065_execution.json"},
        {"id": "yc_lr10", "station": "YCA", "checkpoint": 10000, "ppo_root": "benchmark_results/065_01_yca_lowIC_lr1e4_rescue25k_maskableppo_smoke2k", "view": True, "formal_source": "065_execution.json"},
        {"id": "yc_lr25", "station": "YCA", "checkpoint": 25000, "ppo_root": "benchmark_results/065_01_yca_lowIC_lr1e4_rescue25k_maskableppo_smoke2k", "view": True, "formal_source": "065_execution.json"},
    ]
    selected = specs if args.candidate == "all" else [item for item in specs if item["id"] == args.candidate]
    if args.attempt:
        selected = [dict(item, id=f"{item['id']}_{args.attempt}") for item in selected]
    outputs: list[dict[str, Any]] = []
    for spec in selected:
        means = run_candidate(spec)
        ppo = means[means.scenario.eq("rl_candidate")].iloc[0]
        outputs.append({"candidate": spec["id"], **{key: (None if pd.isna(ppo[key]) else float(ppo[key])) for key in ["yield_kg_ha", "WP_ET", "PFP_N", "irrigation_mm", "nitrogen_kg_ha"]}})
    out = ROOT / "benchmark_results" / "066_phase0_full_metric_replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase0_summary.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

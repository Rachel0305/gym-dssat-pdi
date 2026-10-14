"""Read-only screen for SY years with existing irrigation response evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/122_00_sya_irrigation_responsive_state_screen.json"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / cfg["source_pool"]
    required = [
        "station_code", "year", "scenario_type", "has_water_stress",
        "irrigation_responsive", "swfac_stress_days_gt_0p05", "max_swfac",
        "yield_gain_from_irrigation_at_same_N", "recommended_for_ppo_eval",
    ]
    if args.dry_run:
        issues = [] if source.exists() else [f"source_missing:{rel(source)}"]
        print(json.dumps({"task": f"{cfg['task_id']}_{cfg['task_name']}", "issues": issues, "training": False, "dssat_run": False, "source": rel(source)}, ensure_ascii=False, indent=2))
        return 0 if not issues else 2

    out = ROOT / cfg["output_root"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    pool = pd.read_csv(source, keep_default_na=False)
    missing = [col for col in required if col not in pool.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    sy = pool[pool["station_code"].astype(str).eq(str(cfg["station_code"]))].copy()
    for col in ["year", "swfac_stress_days_gt_0p05", "max_swfac", "yield_gain_from_irrigation_at_same_N", "profit_gain_from_irrigation_low_water_cost", "recommended_for_ppo_eval"]:
        if col in sy.columns:
            sy[col] = pd.to_numeric(sy[col], errors="coerce") if col != "recommended_for_ppo_eval" else sy[col]
    sy["water_stress_candidate"] = sy["has_water_stress"].map(truthy) | sy["swfac_stress_days_gt_0p05"].fillna(0).gt(0) | sy["max_swfac"].fillna(0).gt(0.05)
    sy["irrigation_response_candidate"] = sy["irrigation_responsive"].map(truthy) | sy["yield_gain_from_irrigation_at_same_N"].fillna(0).gt(0)
    sy["validation_candidate"] = sy["recommended_for_ppo_eval"].map(truthy)
    sy["priority_score"] = (
        sy["irrigation_response_candidate"].astype(int) * 1000
        + sy["water_stress_candidate"].astype(int) * 500
        + sy["validation_candidate"].astype(int) * 100
        + sy["yield_gain_from_irrigation_at_same_N"].fillna(0).clip(lower=0) * 1
        + sy["swfac_stress_days_gt_0p05"].fillna(0) * 2
        + sy["max_swfac"].fillna(0) * 50
    )
    sy = sy.sort_values(["priority_score", "year"], ascending=[False, True]).reset_index(drop=True)
    candidates = sy[sy["water_stress_candidate"] & sy["validation_candidate"]].copy()
    sy.to_csv(out / "122_00_sya_all_year_screen.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out / "122_00_sya_validation_candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "status": "completed_read_only",
        "station_code": "SYA",
        "n_sy_years": int(len(sy)),
        "n_water_stress_candidates": int(sy["water_stress_candidate"].sum()),
        "n_irrigation_response_candidates": int(sy["irrigation_response_candidate"].sum()),
        "n_validation_candidates": int(len(candidates)),
        "priority_years": [int(x) for x in candidates["year"].head(6).tolist()],
        "next_step": "run PPO-linked rolling weather oracle on validation candidates" if len(candidates) else "stop_missing_candidates",
        "output_all": rel(out / "122_00_sya_all_year_screen.csv"),
        "output_candidates": rel(out / "122_00_sya_validation_candidates.csv"),
    }
    (out / "122_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

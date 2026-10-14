"""Repair FQ2023 native-auto daily events from the last DSSAT @RUN block."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hl_fq_8seed_cross_site import plot_hl_fq_five_scenario_05503_style as viz
from hl_fq_8seed_cross_site.build_native_auto_corrected2018_figures import (
    BASELINE_ROOT, OUT, load_runner, snapshot_daily,
)

BACKUP = ROOT / "backups/hl_fq_05503_20261003/fq2023_auto_before_last_run_fix"


def backup(path: Path) -> None:
    dest = BACKUP / path.relative_to(OUT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(path, dest)


def main() -> None:
    viz.BASELINE_SCENARIOS[2] = "dssat_auto"
    viz.BASELINE_LABELS["dssat_auto"] = "DSSAT auto irrigation (N=0)"
    runner = load_runner("FQ")
    snapshot = BASELINE_ROOT["FQ"] / "snapshots/FQA/2023/dssat_auto"
    for seed in range(7):
        prefix = f"fq_seed{seed}"
        folder = OUT / "FQ" / f"seed{seed}"
        tables = folder / "tables"
        daily_path = tables / f"{prefix}_five_scenario_daily.csv"
        event_path = tables / f"{prefix}_five_scenario_management_events.csv"
        figure_path = folder / "figures" / f"{prefix}_2023_five_scenario_daily.png"
        for path in (daily_path, event_path, figure_path):
            backup(path)
        daily = pd.read_csv(daily_path, keep_default_na=False)
        before = daily.loc[daily.year.eq(2023) & daily.scenario.eq("dssat_auto")]
        fixed = snapshot_daily(runner, snapshot, "FQ", 2023, "dssat_auto", seed)
        if len(before) != len(fixed):
            raise ValueError(f"Daily row count changed: seed{seed}")
        if abs(float(fixed.irrigation_executed_mm.sum()) - 39.0) > 0.2:
            raise ValueError(f"Final-run irrigation does not close to Summary.OUT: seed{seed}")
        daily = pd.concat([daily.loc[~(daily.year.eq(2023) & daily.scenario.eq("dssat_auto"))], fixed], ignore_index=True)
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        events = daily.loc[
            pd.to_numeric(daily.irrigation_executed_mm, errors="coerce").gt(0)
            | pd.to_numeric(daily.nitrogen_executed_kg_ha, errors="coerce").gt(0),
            ["site", "station_code", "year", "scenario", "scenario_label", "date", "doy", "dap",
             "irrigation_executed_mm", "nitrogen_executed_kg_ha"],
        ].copy()
        events.to_csv(event_path, index=False, encoding="utf-8-sig")
        viz.plot_daily_year("FQ", seed, 2023, daily, folder / "figures", True)
        print(f"DONE FQ seed{seed} 2023 native auto: {fixed.irrigation_executed_mm.sum():.1f} mm", flush=True)


if __name__ == "__main__":
    main()

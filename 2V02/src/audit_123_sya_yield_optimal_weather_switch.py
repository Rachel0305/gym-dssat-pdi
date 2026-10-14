"""Read-only post-hoc audit of yield-optimal action switches in 123."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/123_00_sya_2017_weather_responsive_dap_scan_rerun1"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    rows_path = OUT / "123_00_actionability_rows.csv"
    df = pd.read_csv(rows_path)
    valid = df[(df["status"].eq("ok")) & (df["legal_at_target"].astype(bool))].copy()
    valid["end_grnwt"] = pd.to_numeric(valid["end_grnwt"], errors="coerce")
    best = valid.loc[valid.groupby(["year", "target_dap", "scenario"])["end_grnwt"].idxmax(), ["year", "target_dap", "scenario", "candidate_irrigation_mm", "end_grnwt"]].copy()
    best = best.rename(columns={"candidate_irrigation_mm": "yield_optimal_irrigation_mm", "end_grnwt": "yield_optimal_grnwt"})
    best.to_csv(OUT / "123_00_yield_optimal_action_by_scenario.csv", index=False, encoding="utf-8-sig")
    pivot = best.pivot_table(index=["year", "target_dap"], columns="scenario", values="yield_optimal_irrigation_mm", aggfunc="first").reset_index()
    pivot.to_csv(OUT / "123_00_yield_optimal_action_scenario_pivot.csv", index=False, encoding="utf-8-sig")
    scenario_cols = [c for c in ["actual", "dry", "wet"] if c in pivot.columns]
    switch_rows = pivot[pivot[scenario_cols].nunique(axis=1).gt(1)].copy() if scenario_cols else pivot.iloc[0:0].copy()
    summary = {
        "task": "123_00_sya_2017_yield_optimal_weather_switch_audit",
        "source": rel(rows_path),
        "n_dap_states": int(len(pivot)),
        "n_weather_switch_states": int(len(switch_rows)),
        "weather_switch_detected": bool(len(switch_rows) > 0),
        "switch_daps": [int(x) for x in switch_rows["target_dap"].tolist()],
        "yield_optimal_action_table": rel(OUT / "123_00_yield_optimal_action_by_scenario.csv"),
        "yield_optimal_action_pivot": rel(OUT / "123_00_yield_optimal_action_scenario_pivot.csv"),
        "interpretation": "weather changes the yield-optimal one-step irrigation action in selected 2017 states; rolling-policy validation is still required",
    }
    (OUT / "123_00_yield_optimal_weather_switch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

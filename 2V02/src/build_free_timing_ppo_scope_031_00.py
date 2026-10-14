from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "benchmark_results" / "030_00_weather_qc" / "030_00_weather_year_usability.csv"
OUT = ROOT / "benchmark_results" / "031_00_free_timing_ppo_scope"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(QC)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    scope = df[(df["selected_for_next_rl"].eq("yes")) & (df["year"] >= 2000)].copy()
    scope = scope.sort_values(["station", "year"]).reset_index(drop=True)
    scope["free_timing_allowed"] = "yes"
    scope["expert_dap_windows_removed"] = "yes"
    scope["min_operation_interval_days"] = 7
    scope["fertilization_latest_dap"] = 90
    scope["scope_reason"] = "usable_weather_year_from_030_00_and_year_ge_2000"
    scope["weather_use_note"] = scope.apply(
        lambda r: "use_030_00_corrected_copy" if int(r["corrected_value_count"]) > 0 else "use_raw_weather",
        axis=1,
    )

    cols = [
        "station",
        "year",
        "free_timing_allowed",
        "expert_dap_windows_removed",
        "min_operation_interval_days",
        "fertilization_latest_dap",
        "weather_use_note",
        "source_group",
        "filename",
        "path",
        "n_rows_year",
        "corrected_value_count",
        "scope_reason",
    ]
    scope[cols].to_csv(OUT / "031_00_station_year_scope.csv", index=False, encoding="utf-8-sig")

    by_station = {
        station: [int(y) for y in group["year"].tolist()]
        for station, group in scope.groupby("station", sort=True)
    }
    result = {
        "task": "031_00_free_timing_ppo_scope",
        "source": str(QC.relative_to(ROOT)),
        "n_station_years": int(len(scope)),
        "stations": sorted(scope["station"].unique().tolist()),
        "years_by_station": by_station,
        "excluded_policy": "year_lt_2000_or_weather_unusable",
        "min_operation_interval_days": 7,
        "expert_dap_windows_removed": True,
        "training_or_dssat_run": False,
    }
    (OUT / "031_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 031_00 Free-timing PPO scope record",
        "",
        "## Outcome",
        "",
        f"- Station-years selected: {len(scope)}",
        "- Years before 2000 are excluded.",
        "- LC1998 remains excluded by 030_00 weather QC.",
        "- LC1999 is excluded by the new year >= 2000 rule.",
        "- HLA2010 is retained, using the corrected 030_00 weather copy.",
        "",
        "## Decision rule",
        "",
        "- PPO will not be restricted to expert DAP windows.",
        "- PPO will have daily decision opportunities during the management season.",
        "- A 7-day minimum interval between non-zero field operations is retained as an operational feasibility constraint.",
        "- This interval does not define expert timing; it only prevents unrealistic daily repeated operations.",
        "",
        "## Station-year list",
        "",
    ]
    for station, years in by_station.items():
        lines.append(f"- {station}: {', '.join(map(str, years))}")
    lines.extend(
        [
            "",
            "## Next task",
            "",
            "031_01 should implement a small free-timing PPO smoke test before any all-year run.",
        ]
    )
    (OUT / "031_00_free_timing_ppo_scope_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

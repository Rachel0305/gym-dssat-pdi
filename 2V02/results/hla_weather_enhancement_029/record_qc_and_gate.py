"""Record pre-WGEN fitting QA and the 029 stop gate; never launch a simulator."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hla_weather_enhancement_029"
NAMES = ["gpcc_raw", "cpc_raw", "gpcc_biascorr", "cpc_biascorr", "ensemble_biascorr"]


def main() -> None:
    rows = []
    for name in NAMES:
        path = OUT / "weather_fitting" / name / "fitting_weather.csv"
        with path.open(newline="", encoding="utf-8-sig") as f:
            days = list(csv.DictReader(f))
        dates = [dt.date.fromisoformat(r["DATE"]) for r in days]
        numeric = [{k: float(r[k]) for k in ("RAIN", "SRAD", "TMAX", "TMIN")} for r in days]
        unique = len(dates) == len(set(dates))
        continuous = all((b - a).days == 1 for a, b in zip(dates, dates[1:]))
        finite = all(math.isfinite(v) for row in numeric for v in row.values())
        physical = all(r["RAIN"] >= 0 and r["SRAD"] >= 0 and r["TMAX"] >= r["TMIN"] for r in numeric)
        annual = defaultdict(list)
        for day, data in zip(dates, numeric):
            annual[day.year].append(data)
        year_summary = []
        for year, vals in sorted(annual.items()):
            dry_max = dry_now = 0
            for data in vals:
                dry_now = dry_now + 1 if data["RAIN"] <= 0 else 0
                dry_max = max(dry_max, dry_now)
            year_summary.append({"year": year, "days": len(vals),
                                 "rain_total_mm": round(sum(v["RAIN"] for v in vals), 6),
                                 "wet_days_gt0": sum(v["RAIN"] > 0 for v in vals),
                                 "max_daily_rain_mm": max(v["RAIN"] for v in vals),
                                 "longest_dry_spell_days": dry_max})
        rows.append({"scenario": name.upper(), "row_count": len(days),
                     "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
                     "unique_dates": unique, "daily_continuous": continuous,
                     "finite_values": finite, "basic_physical": physical,
                     "zero_monthly_constant_warning": [
                         f"{year}-{month:02d}-{field}" for year in annual for month in range(1, 13)
                         for field in ("RAIN", "SRAD", "TMAX", "TMIN")
                         if len({r[field] for d, r in zip(dates, numeric) if d.year == year and d.month == month}) == 1
                     ],
                     "annual": year_summary,
                     "pre_wgen_pass": len(days) == 3653 and unique and continuous and finite and physical,
                     "candidate_qa_status": "NOT_RUN_NO_WGEN_CANDIDATES"})
    qc = {"task": "029", "site": "HLA", "qa_scope": "FITTING_INPUT_ONLY_NOT_SYNTHETIC_CANDIDATES",
          "scenarios": rows, "all_pre_wgen_pass": all(x["pre_wgen_pass"] for x in rows),
          "observed_2014_2023_used": False,
          "candidate_qa_status": "NOT_RUN_NO_WGEN_CANDIDATES"}
    (OUT / "weather_fitting" / "fitting_weather_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_dir = OUT / "wgen_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "task": "029", "site": "HLA",
        "user_selection": "Five isolated limited-candidate sensitivity tracks",
        "planned_tracks": [{"scenario": name.upper(), "candidate_id": "candidate_001",
                            "seed": None, "seed_status": "NOT_ASSIGNED_NO_VERIFIED_WGEN_RUNTIME_CONTEXT",
                            "status": "NOT_GENERATED"} for name in NAMES],
        "candidate_count_generated": 0,
        "wgen_executed": False,
        "blocking_reason": "No project-local verified official full-calendar-year WGEN/WeatherMan entry point. YC known runtime generates crop-season daily-state fragments inside DSSAT crop simulation, and 004_01/004_16 did not validate a full-year export path. The AGENTS.md project-only boundary precludes inspecting an external installation.",
        "failed_candidate_files": [],
        "candidate_qa_status": "NOT_RUN",
        "no_seed_resampling_or_candidate_selection": True,
    }
    (candidate_dir / "candidate_plan_and_blocker.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {
        "task": "029", "site": "HLA", "purpose": "WEATHER_ENHANCEMENT_RESUME",
        "source_audit_pass": False,
        "source_audit_detail": "Five 024 reference scenarios are frozen and structurally complete, but 111 rain days have no formally accepted HLA daily source; 2013 climate magnitude remains suspicious.",
        "weather_modified": False, "wgen_executed": False, "dssat_executed": False, "ppo_executed": False,
        "phase_A": "PASS_SCENARIO_SOURCE_FREEZE_NOT_FORMAL_SOURCE_ACCEPTANCE",
        "phase_B": "PASS_FIVE_SCENARIO_FITTING_AND_STATIC_CLI_CANDIDATES",
        "phase_C": "BLOCKED_NO_VERIFIED_PROJECT_LOCAL_FULL_YEAR_WGEN_RUNTIME",
        "phase_D": "NOT_RUN_NO_GENERATED_CANDIDATES",
        "candidate_pool_status": "BLOCKED_NO_GENERATED_CANDIDATES",
        "candidate_count_generated": 0, "ready_for_ppo": False,
        "recommendation": "Keep five fitting/CLI tracks as exploratory sensitivity inputs. Provide a project-local verified official full-year WGEN/WeatherMan executable and version plus seed/context contract; first generate one isolated candidate per scenario and run structural, climate and extreme-event QA. Do not enter PPO until the source and candidate gates pass."
    }
    (OUT / "final_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pre_wgen_pass": qc["all_pre_wgen_pass"], "candidate_generated": 0,
                      "ready_for_ppo": False}))


if __name__ == "__main__":
    main()

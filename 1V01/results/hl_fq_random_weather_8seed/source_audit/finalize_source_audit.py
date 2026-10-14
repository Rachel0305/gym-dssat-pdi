"""Combine local/NASA and official GHCN source checks into the final gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from audit_sources import OUT, SITES


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    stage1 = read_json("source_audit.json")
    rain = read_json("ghcn_rain_station_audit.json")
    coverage = read_json("ghcn_multi_station_coverage.json")
    crosscheck = read_json("rainfall_candidate_crosscheck.json")
    final = {
        "task": "HL/FQ WGEN random-weather 8-seed source audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weather_source_audit": "BLOCKED",
        "final_decision": "BLOCKED_BY_DATA_PROVENANCE",
        "stop_before_wgen": True,
        "fitting_weather_materialized": False,
        "cli_generated": False,
        "wgen_started": False,
        "runtime_smoke_started": False,
        "ppo_started": False,
        "validation_years_read_or_used": False,
        "original_workbooks_modified": False,
        "historical_wth_modified": False,
        "official_cma_daily_source": {
            "dataset_catalogue": "https://m.data.cma.cn/data/cdcindex/cid/6d1b5efbdcbf9a58.html",
            "dataset_status": "IDENTITY_AUTHENTICATED_ACCESS_REQUIRED; not present in project inputs",
            "authenticated_download_attempted": False,
            "reason_not_downloaded": "The official portal requires a verified real-name account; no such project-local data or authorized download session was available in this audit.",
        },
        "yc_reference_route": {
            "status": "REUSABLE_REFERENCE_CONFIRMED",
            "evidence": [
                "docs/yc_cli_generation_and_weather_qc.md",
                "results/yc_random_weather_ppo/004_03/experiment_log.md",
                "results/yc_random_weather_ppo/004_03/config/weather_runtime_correction.json",
                "results/yc_random_weather_ppo/004_03/config/experiment_design.json",
            ],
            "runtime_mode": "Task-local FileX copy with METHODS.WTHER=W; inject scheduled weather seed into PDI rseed1 before each episode; hash runtime daily RAIN/SRAD/TMAX/TMIN.",
            "weather_seed_contract": "training 1001-1080; held-out 1081-1100; separate PPO seed; fixed common training pool",
            "successful_ppo_scope": "3 PPO seeds per regime at 100000 requested steps, 100080 actual; not an 8-seed result",
            "weather_qa_scope_note": "004_02 passed a common 106-day crop-season window; it explicitly deferred full-year WGEN validation.",
        },
        "sites": {},
    }

    for site, spec in SITES.items():
        local = stage1["sites"][site]
        rain_site = rain["sites"][site]
        cov_site = coverage["sites"][site]
        cross_site = crosscheck["sites"][site]
        unresolved = cov_site["uncovered_local_rain_days"]
        final["sites"][site] = {
            "status": "BLOCKED",
            "fit_period": f"{spec['start']}-01-01/{spec['end']}-12-31",
            "expected_days": int(local["fit_period"]["expected_days"]),
            "raw_source_files": local["files"],
            "existing_wth": local["weather_header"],
            "raw_local_variable_audit": local["raw_variables"],
            "nasa_power_screen": {
                "period": f"{spec['start']}-01-01/{spec['end']}-12-31",
                "temperature_and_srad_candidate_gates": {
                    variable: local["raw_variables"][variable]["nasa_candidate_gate"]
                    for variable in ("TMAX", "TMIN", "SRAD")
                },
                "rain_candidate_gate": "FAIL; NASA POWER precipitation is not used for gap filling",
                "nasa_rain_event_agreement": local["raw_variables"]["RAIN"]["nasa_precipitation_screen"]["event_agreement"],
                "candidate_gapfill_rows": sum(local["raw_variables"][v]["candidate_fill_or_replacement_rows"] for v in ("TMAX", "TMIN", "SRAD")),
            },
            "official_ghcn_rain_screen": {
                "screen_radius_km": rain["inventory"]["search_radius_km"] if "search_radius_km" in rain["inventory"] else 300.0,
                "primary_station": rain_site["selected_station"],
                "qualified_candidates": cov_site["qualified_stations_available"],
                "local_nonmissing_days": cross_site["local_nonmissing_days"],
                "local_blank_days": cross_site["local_missing_days"],
                "wet_dry_agreement_all_days_blank_as_dry": cross_site["wet_dry_agreement_all_external_days_assuming_blank_is_dry"],
                "wet_dry_agreement_recorded_local_days_only": cross_site["wet_dry_agreement_only_recorded_local_days"],
                "qualified_station_gapfill_rows": cov_site["candidate_gap_rows"],
                "gapfill_station_counts": cov_site["gap_fill_rows_by_station"],
                "maximum_station_distance_km_used": cov_site["maximum_gap_fill_station_distance_km"],
                "unresolved_local_rain_days": unresolved,
                "unresolved_day_count": len(unresolved),
                "unresolved_years": sorted({day[:4] for day in unresolved}),
                "candidate_gapfills_promoted": False,
            },
            "rainfall_composite_sensitivity": cross_site,
            "fitting_weather_complete": False,
            "stop_reason": "RAIN has unresolved dates in 2013 after all screened <=300 km official GHCN candidates; no authenticated CMA daily data is present. No verified rule authorizes assigning zero to these residual dates.",
            "wgen_started": False,
            "cli_generated": False,
            "training_pool_members": 0,
            "heldout_pool_members": 0,
            "runtime_smoke": "NOT_RUN",
            "ppo_seeds_completed": 0,
        }

    output = OUT / "weather_source_audit_final.json"
    output.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_decision": final["final_decision"], "stop_before_wgen": final["stop_before_wgen"],
                      "sites": {s: {"status": d["status"], "fit_period": d["fit_period"],
                                    "expected_days": d["expected_days"],
                                    "unresolved_rain_days": d["official_ghcn_rain_screen"]["unresolved_day_count"],
                                    "unresolved_years": d["official_ghcn_rain_screen"]["unresolved_years"]}
                                for s, d in final["sites"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

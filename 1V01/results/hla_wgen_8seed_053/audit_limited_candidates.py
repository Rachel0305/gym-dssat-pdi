"""Audit the five isolated HLA 2007 seed-101 crop-season captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCENARIOS = ("gpcc_raw", "cpc_raw", "gpcc_biascorr", "cpc_biascorr", "ensemble_biascorr")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        root = BASE / "limited_candidates" / scenario / "2007_seed101"
        provenance = json.loads((root / "input_provenance.json").read_text(encoding="utf-8"))
        result = json.loads((root / "runtime/attempt_01/result.json").read_text(encoding="utf-8"))
        weather = root / "runtime/attempt_01/runtime_weather_daily.csv"
        integrity = weather.is_file() and digest(weather) == result.get("weather_sha256")
        physical = result.get("physical_sanity", {}).get("status") == "PASS"
        seed = result.get("rseed1") == provenance["weather_seed"] == 101
        row = {
            "scenario": scenario.upper(),
            "status": result.get("status"),
            "runtime_cli_matches_frozen": result.get("runtime_cli_sha256") == provenance["cli_sha256"],
            "weather_seed_confirmed": seed and bool(result.get("pdi_rseed1_configured")),
            "daily_weather_hash_matches": integrity,
            "physical_screen_pass": physical,
            "complete_four_fields": result.get("weather_fields_complete"),
            "days": result.get("weather_row_count"),
            "season_rainfall_mm": result.get("weather_summary", {}).get("season_rainfall_mm"),
            "max_daily_rainfall_mm": result.get("weather_summary", {}).get("maximum_daily_rainfall_mm"),
            "peak_process_tree_rss_mb": result.get("peak_process_tree_rss_mb"),
            "weather_sha256": result.get("weather_sha256"),
        }
        row["limited_capture_pass"] = all(
            (row["status"] == "RUNTIME_WEATHER_CAPTURED", row["runtime_cli_matches_frozen"],
             row["weather_seed_confirmed"], row["daily_weather_hash_matches"],
             row["physical_screen_pass"], row["complete_four_fields"],
             isinstance(row["days"], int) and row["days"] > 0)
        )
        rows.append(row)
    output = {
        "task": "053_hla_wgen_limited_candidates",
        "site": "HLA",
        "crop_year": 2007,
        "weather_seed": 101,
        "scenarios": rows,
        "all_five_limited_captures_pass": all(row["limited_capture_pass"] for row in rows),
        "full_training_weather_pool_qc_pass": False,
        "native_field_coordinate_gate_pass": False,
        "ready_for_100k": False,
    }
    target = BASE / "limited_candidate_audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_five_limited_captures_pass": output["all_five_limited_captures_pass"], "days": [r["days"] for r in rows], "ready_for_100k": False}))


if __name__ == "__main__":
    main()

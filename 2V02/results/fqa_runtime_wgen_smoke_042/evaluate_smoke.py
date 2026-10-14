"""Read-only evaluation of the two archived FQA runtime WGEN smokes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATTEMPTS = ("attempt_01", "attempt_02_same_seed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    if (OUT / "final_gate.json").exists():
        raise FileExistsError("final gate already exists; refusing overwrite")
    provenance = json.loads((OUT / "input_provenance.json").read_text(encoding="utf-8"))
    attempts = []
    for name in ATTEMPTS:
        folder = OUT / "runtime" / name
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        weather = folder / "runtime_weather_daily.csv"
        snapshot = folder / "runtime_snapshot"
        if not weather.is_file():
            raise FileNotFoundError(weather)
        if sha(weather) != result.get("weather_sha256"):
            raise ValueError(f"weather hash mismatch: {name}")
        rows = weather.read_text(encoding="utf-8").splitlines()
        if not rows or rows[0] != "DATE,DOY,RAIN,SRAD,TMAX,TMIN" or len(rows) != result.get("weather_row_count", -1) + 1:
            raise ValueError(f"daily archive schema/count mismatch: {name}")
        cli = snapshot / "CNFQ.CLI"
        filex = snapshot / "fileX.MZX"
        inp = snapshot / "DSSAT48.INP"
        warning = snapshot / "WARNING.OUT"
        if not all(path.is_file() for path in (cli, filex, inp, warning)):
            raise FileNotFoundError(f"incomplete runtime snapshot: {name}")
        inp_text = inp.read_text(encoding="latin-1")
        warning_text = warning.read_text(encoding="latin-1")
        field = inp_text.split("*FIELDS", 1)[1].split("*INITIAL CONDITIONS", 1)[0]
        missing_coords = bool(re.search(r"-999\.00000\s+-99\.00000\s+-99\.00", field))
        attempts.append({
            "attempt": name,
            "status": result.get("status"),
            "year": result.get("year"),
            "treatment": result.get("treatment"),
            "rseed1": result.get("rseed1"),
            "steps": result.get("steps"),
            "weather_rows": result.get("weather_row_count"),
            "weather_archive": str(weather.relative_to(ROOT)).replace("\\", "/"),
            "weather_sha256": sha(weather),
            "runtime_cli_sha256": sha(cli),
            "runtime_filex_sha256": sha(filex),
            "runtime_inp_sha256": sha(inp),
            "runtime_wther": result.get("runtime_filex_wther"),
            "runtime_wsta": result.get("runtime_filex_wsta"),
            "pdi_seed_confirmed": result.get("pdi_rseed1_configured"),
            "physical_sanity": result.get("physical_sanity", {}).get("status"),
            "peak_process_tree_rss_mb": result.get("peak_process_tree_rss_mb"),
            "native_field_coordinates_missing": missing_coords,
            "ipfld_coordinate_warnings": all(fragment in warning_text for fragment in ("Error reading latitude", "Error reading longitude", "Error reading elevation")),
            "cli_named_in_native_inp": "WEATHERC       CNFQ.CLI" in inp_text,
        })
    weather_equal = (OUT / "runtime/attempt_01/runtime_weather_daily.csv").read_bytes() == (OUT / "runtime/attempt_02_same_seed/runtime_weather_daily.csv").read_bytes()
    runtime_capture_pass = all(
        a["status"] == "RUNTIME_WEATHER_CAPTURED" and a["weather_rows"] == a["steps"]
        and a["weather_rows"] > 0 and a["runtime_wther"] == "W" and a["runtime_wsta"] == "CNFQ0701"
        and a["pdi_seed_confirmed"] and a["rseed1"] == 101 and a["physical_sanity"] == "PASS"
        and a["cli_named_in_native_inp"] and a["runtime_cli_sha256"] == provenance["file_hashes"][3]["sha256"]
        for a in attempts
    )
    coordinate_pass = all(not a["native_field_coordinates_missing"] and not a["ipfld_coordinate_warnings"] for a in attempts)
    output = {
        "task": "042", "site": "FQA", "scope": "2007 treatment 1, one weather seed, two fresh processes, no PPO",
        "attempts": attempts,
        "runtime_weather_capture": "PASS" if runtime_capture_pass else "FAIL",
        "same_seed_daily_reproducibility": "PASS_THIS_CONTEXT" if runtime_capture_pass and weather_equal else "FAIL_OR_UNVERIFIED",
        "native_field_coordinates": "PASS" if coordinate_pass else "FAIL_MISSING",
        "weather_archived_for_both_attempts": True,
        "seed_alone_is_not_realization_identity": True,
        "ready_for_formal_weather_pool": bool(runtime_capture_pass and weather_equal and coordinate_pass),
        "ready_for_ppo": False,
        "next_step": "Resolve or explicitly scope the common native FIELD coordinate issue; then run limited archived multi-seed weather QA before any full pool or PPO.",
    }
    (OUT / "final_gate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("runtime_weather_capture", "same_seed_daily_reproducibility", "native_field_coordinates", "ready_for_formal_weather_pool")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

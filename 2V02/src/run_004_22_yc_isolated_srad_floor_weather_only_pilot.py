#!/usr/bin/env python3
"""Read-only Phase 1 source/build gate for YC 004_22. Never invokes WGEN/DSSAT."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot"
CONTAINER = "nifty_taussig"
EXPECTED_PACKAGE = "4.8.0.24+2~a4ae650-1"
SOURCE_ROOT = "/home/gymdev/gym_dssat_pdi/dssat-pdi-4.8.0.24+2~a4ae650"


def run_read_only(args: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=45, check=False)
        return {
            "command": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def output_hash(stdout: str, label: str) -> str | None:
    prefix = f"{label}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):].split()
            return value[0] if value else None
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inspection = run_read_only([
        "docker", "inspect", CONTAINER,
        "--format", "{{.Id}}|{{.Image}}|{{json .Mounts}}",
    ])
    shell_script = r'''set +e
printf 'CONTAINER_PWD='; pwd
printf 'PACKAGE_VERSION='; dpkg-query -W -f='${Version}' dssat-pdi 2>/dev/null; echo
printf 'EXPECTED_SOURCE_ROOT='; test -d /home/gymdev/gym_dssat_pdi/dssat-pdi-4.8.0.24+2~a4ae650 && echo PRESENT || echo MISSING
printf 'EXPECTED_WGEN_SOURCE='; test -f /home/gymdev/gym_dssat_pdi/dssat-pdi-4.8.0.24+2~a4ae650/Weather/WGEN.for && echo PRESENT || echo MISSING
printf 'EXECUTABLE_SHA256='; sha256sum /opt/dssat_pdi/dscsm048 2>/dev/null
printf 'WRAPPER_SHA256='; sha256sum /opt/dssat_pdi/run_dssat 2>/dev/null
printf 'SOURCE_CANDIDATES_BEGIN\n'
find /home/gymdev /opt/dssat_pdi /usr/src -type f \( -iname 'WGEN.for' -o -iname '*DailyWeatherCheck*' \) -print 2>/dev/null | head -100
printf 'SOURCE_CANDIDATES_END\n'
printf 'CHECK_ROUTINE_MATCHES_BEGIN\n'
grep -Ril --include='*.for' --include='*.f' --include='*.f90' 'DailyWeatherCheck' /home/gymdev /opt/dssat_pdi /usr/src 2>/dev/null | head -100
printf 'CHECK_ROUTINE_MATCHES_END\n'
printf 'BINARY_SOURCE_STRINGS_BEGIN\n'
strings /opt/dssat_pdi/dscsm048 2>/dev/null | grep -E 'Weather/WGEN\.for|DailyWeatherCheck|SRAD < 0\.1|dssat-pdi-4\.8\.0\.24' | head -40
printf 'BINARY_SOURCE_STRINGS_END\n'
printf 'DEBUG_SECTIONS_BEGIN\n'
readelf -S /opt/dssat_pdi/dscsm048 2>/dev/null | grep -E 'debug|symtab' | head -20
printf 'DEBUG_SECTIONS_END\n'
'''
    runtime = run_read_only(["docker", "exec", CONTAINER, "sh", "-lc", shell_script])
    transcript = "\n".join([
        "# docker inspect",
        json.dumps(inspection, ensure_ascii=False, indent=2),
        "# container read-only source/build inspection",
        json.dumps(runtime, ensure_ascii=False, indent=2),
    ])
    (OUT / "source_gate_raw_output.txt").write_text(transcript + "\n", encoding="utf-8")

    stdout = str(runtime.get("stdout") or "")
    package_match = re.search(r"(?m)^PACKAGE_VERSION=(.*)$", stdout)
    package_version = package_match.group(1).strip() if package_match else None
    wgen_present = "EXPECTED_WGEN_SOURCE=PRESENT" in stdout
    candidates = []
    in_candidates = False
    for line in stdout.splitlines():
        if line == "SOURCE_CANDIDATES_BEGIN":
            in_candidates = True
            continue
        if line == "SOURCE_CANDIDATES_END":
            in_candidates = False
            continue
        if in_candidates and line.strip():
            candidates.append(line.strip())
    check_matches = []
    in_checks = False
    for line in stdout.splitlines():
        if line == "CHECK_ROUTINE_MATCHES_BEGIN":
            in_checks = True
            continue
        if line == "CHECK_ROUTINE_MATCHES_END":
            in_checks = False
            continue
        if in_checks and line.strip():
            check_matches.append(line.strip())
    has_check_source = bool(check_matches)
    package_ok = package_version == EXPECTED_PACKAGE
    gate_pass = bool(inspection.get("returncode") == 0 and runtime.get("returncode") == 0
                     and package_ok and wgen_present and has_check_source)
    decision = {
        "task_id": "004_22",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_status": "READY_FOR_PHASE_2_REVIEW" if gate_pass else "BLOCKED_SOURCE_GATE",
        "source_build_gate": "PASS" if gate_pass else "FAIL",
        "expected_package_version": EXPECTED_PACKAGE,
        "observed_package_version": package_version,
        "expected_debug_source_root": SOURCE_ROOT,
        "expected_WGEN_for_present": wgen_present,
        "DailyWeatherCheck_source_matches": check_matches,
        "DailyWeatherCheck_source_present": has_check_source,
        "source_candidates": candidates,
        "runtime_executable_sha256": output_hash(stdout, "EXECUTABLE_SHA256"),
        "runtime_wrapper_sha256": output_hash(stdout, "WRAPPER_SHA256"),
        "container_inspection_sha256": sha256_text(str(inspection.get("stdout", ""))),
        "source_gate_reason": ("Exact patched source and post-check implementation are available." if gate_pass
                               else "Exact patched WGEN/DailyWeatherCheck source-to-installed-build binding is incomplete."),
        "weather_generation_started": False,
        "dssat_crop_simulation_started": False,
        "ppo_started": False,
        "canonical_runtime_modified": False,
    }
    (OUT / "source_gate.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "025_00"
INPUT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
MZX = INPUT / "CNSY1201.MZX"
SITE_CONFIG = ROOT / "configs" / "sites" / "sya.yaml"
PROTOCOL = ROOT / "configs" / "deterministic_optimization_protocol_025.yaml"
STRICT = ROOT / "benchmark_results" / "022_01" / "022_01_strict_success_candidates.csv"
DIAG = INPUT / "sy_2012_2014_ic_diagnosis_016_01_runs" / "sy_2012_2014_ic_diagnosis_summary.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    config = yaml.safe_load(SITE_CONFIG.read_text(encoding="utf-8"))
    strict = pd.read_csv(STRICT)
    text = MZX.read_text(encoding="utf-8", errors="replace")
    diagnosis = pd.read_csv(DIAG)

    checks = {
        "protocol_yaml_loaded": protocol.get("protocol_id") == "deterministic_water_nitrogen_optimization_025",
        "strict_candidate_count_18": len(strict) == 18,
        "strict_candidate_scenarios_unique": strict["scenario"].nunique() == 18,
        "mzx_exists": MZX.exists(),
        "mzx_2012_treatment_ic1": "1 1 1 0 Sim2012" in text and " 1  1  0  1  1" in text,
        "mzx_2014_treatment_ic2": "2 1 1 0 Sim2014" in text and " 1  2  0  2  2" in text,
        "weather_2012_exists": (INPUT / "CNSY1201.WTH").exists(),
        "weather_2014_exists": (INPUT / "CNSY1401.WTH").exists(),
        "soil_exists": (INPUT / "SOIL.SOL").exists(),
        "cultivar_exists": (INPUT / "MZCER048.CUL").exists(),
        "historical_2012_forward_completed": bool(
            len(diagnosis.loc[(diagnosis["case"] == "sy2012_ic1") & diagnosis["terminated"].astype(bool)]) == 1
        ),
        "site_config_registers_2014": 2014 in config["site"].get("supported_years", []),
        "site_config_registers_2012": 2012 in config["site"].get("supported_years", []),
    }
    readiness_rows = [
        {"item": key, "passed": bool(value), "category": "blocking_input" if key in {
            "strict_candidate_count_18", "strict_candidate_scenarios_unique", "mzx_exists",
            "mzx_2012_treatment_ic1", "mzx_2014_treatment_ic2", "weather_2012_exists",
            "soil_exists", "cultivar_exists", "historical_2012_forward_completed"
        } else "registration_or_protocol"}
        for key, value in checks.items()
    ]
    pd.DataFrame(readiness_rows).to_csv(OUT / "025_00_sy_crossyear_readiness.csv", index=False)

    blocking = [row["item"] for row in readiness_rows if row["category"] == "blocking_input" and not row["passed"]]
    protocol_ready = not blocking
    registration_complete = bool(checks["site_config_registers_2012"])
    branch = (
        "A_protocol_ready_input_validation_required"
        if protocol_ready and not registration_complete
        else ("A_protocol_and_registration_ready" if protocol_ready else "C_missing_blocking_input")
    )
    result = {
        "status": "completed",
        "branch": branch,
        "protocol_ready": protocol_ready,
        "sy2014_strict_candidate_count": int(len(strict)),
        "sy2012_historical_forward_evidence": bool(checks["historical_2012_forward_completed"]),
        "sy2012_site_config_registered": registration_complete,
        "sy2012_four_baselines_current_input_complete": False,
        "candidate_transfer_allowed_now": False,
        "next_step": "025_01_sy2012_input_provenance_and_four_baseline_forward_validation" if protocol_ready else "repair_missing_input",
        "mzx_sha256": sha256(MZX),
        "weather_2012_sha256": sha256(INPUT / "CNSY1201.WTH"),
        "formal_optimization_runs": 0,
        "dssat_calls": 0,
        "learning_training_steps": 0,
        "files_modified_in_input_package": 0,
        "claim_boundary": "SY2014 results remain a retrospective oracle until unchanged schedules pass SY2012.",
    }
    (OUT / "025_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

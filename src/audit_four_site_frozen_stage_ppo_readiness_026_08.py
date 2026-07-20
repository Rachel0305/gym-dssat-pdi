from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "026_08_attempt2"
MODELS = {
    0: ROOT / "benchmark_results" / "026_03" / "checkpoint_000120.zip",
    1: ROOT / "benchmark_results" / "026_02" / "checkpoint_000060.zip",
    2: ROOT / "benchmark_results" / "026_04" / "checkpoint_000240.zip",
}
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
HLA_ROOT = ROOT / "DSSAT_auto_validation" / "HLA_2004"
HLA_INPUTS = {
    2007: HLA_ROOT / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2007" / "input",
    2010: HLA_ROOT / "hla2010_nstep_dqn_seed0_020_08" / "nstep5_seed0_50000steps" / "input",
    2015: HLA_ROOT / "hla2010_nstep_to_hla2015_transfer_020_09" / "test_case_hla2015" / "input",
    2016: HLA_ROOT / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2016" / "input",
    2022: HLA_ROOT / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2022" / "input",
}
HLA_SUMMARY = HLA_ROOT / "hla_five_scenario_nstep_020_11" / "020_11_hla_five_scenario_summary.csv"
CROSS_SUMMARY = ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12" / "020_13_yc_fq_seed0_seed1_summary.csv"
EXPERT_SUMMARY = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_03_clean_multisite_comparison_with_extension_expert.csv"
SY_RESULT = ROOT / "benchmark_results" / "026_07_attempt2" / "026_07_result.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_checks(path: Path, weather_hint: str | None = None) -> dict[str, Any]:
    files = list(path.iterdir()) if path.is_dir() else []
    mzx = [item for item in files if item.suffix.upper() == ".MZX"]
    weather = [item for item in files if item.suffix.upper() == ".WTH"]
    if weather_hint:
        weather_match = any(item.name.upper() == weather_hint.upper() for item in weather)
    else:
        weather_match = len(weather) >= 1
    return {
        "input_dir_exists": path.is_dir(),
        "mzx_count": len(mzx),
        "weather_count": len(weather),
        "target_weather_present": weather_match,
        "soil_present": any(item.name.upper() == "SOIL.SOL" for item in files),
        "cultivar_present": any(item.name.upper() == "MZCER048.CUL" for item in files),
    }


def baseline_set_present(site: str, year: int) -> bool:
    if site == "HLA":
        frame = pd.read_csv(HLA_SUMMARY)
        families = set(frame.loc[frame["year"].eq(year), "scenario_family"].fillna("null").astype(str))
        return {"null", "dssat_auto", "extension_expert"}.issubset(families)
    if site in {"YC", "FQ"}:
        frame = pd.read_csv(CROSS_SUMMARY)
        families = set(frame.loc[frame["site"].eq(site) & frame["year"].eq(year), "scenario_family"].fillna("null").astype(str))
        return {"null", "dssat_auto", "extension_expert"}.issubset(families)
    if site == "LC":
        frame = pd.read_csv(EXPERT_SUMMARY)
        labels = set(frame.loc[frame["site"].eq("LC") & frame["year"].eq(year), "scenario_group"].astype(str))
        return {"Null", "DSSAT auto", "Official extension expert fixed DAP"}.issubset(labels)
    return False


def add_case(rows: list[dict[str, Any]], site: str, year: int, adapter: str, input_dir: Path, weather: str | None) -> None:
    checks = input_checks(input_dir, weather)
    checks["baseline_set_present"] = baseline_set_present(site, year)
    checks["adapter_source_exists"] = (ROOT / adapter).exists()
    required = [
        checks["input_dir_exists"],
        checks["mzx_count"] >= 1,
        checks["target_weather_present"],
        checks["soil_present"],
        checks["cultivar_present"],
        checks["baseline_set_present"],
        checks["adapter_source_exists"],
    ]
    rows.append(
        {
            "site": site,
            "year": year,
            "adapter": adapter,
            "input_dir": str(input_dir.relative_to(ROOT)) if input_dir.exists() else str(input_dir),
            **checks,
            "eligible_for_site_smoke": all(required),
            "training_steps_planned": 0,
        }
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    required = [SCALER, HLA_SUMMARY, CROSS_SUMMARY, EXPERT_SUMMARY, SY_RESULT, *MODELS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen-model/readiness evidence: {missing}")
    sy_payload = json.loads(SY_RESULT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for year, input_dir in HLA_INPUTS.items():
        add_case(rows, "HLA", year, "src/run_hla_five_scenario_completion_020_11.py", input_dir, None)
    add_case(
        rows,
        "YC",
        2014,
        "src/run_yc_fq_frozen_nstep_cross_site_020_12.py",
        ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC",
        "CNYC1401.WTH",
    )
    add_case(
        rows,
        "FQ",
        2016,
        "src/run_yc_fq_frozen_nstep_cross_site_020_12.py",
        ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "FQ",
        "CNFQ1601.WTH",
    )
    add_case(
        rows,
        "LC",
        2010,
        "src/run_lc_fixed_input_year_screening_017_11.py",
        ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "LC",
        "CNLC1001.WTH",
    )
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / "026_08_case_readiness.csv", index=False)
    model_hashes = {str(seed): sha256(path) for seed, path in MODELS.items()}
    payload = {
        "status": "completed",
        "branch": "A_all_cases_ready_for_serial_smoke" if bool(frame["eligible_for_site_smoke"].all()) else "B_some_cases_blocked",
        "case_count": len(frame),
        "ready_count": int(frame["eligible_for_site_smoke"].sum()),
        "blocked_cases": frame.loc[~frame["eligible_for_site_smoke"], ["site", "year"]].to_dict("records"),
        "model_hashes": model_hashes,
        "scaler_sha256": sha256(SCALER),
        "sy_026_07_branch": sy_payload.get("branch"),
        "dssat_calls": 0,
        "training_steps": 0,
        "next_step": "Run one site at a time: HLA, YC, FQ, LC; smoke before frozen-model evaluation.",
    }
    (OUT / "026_08_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

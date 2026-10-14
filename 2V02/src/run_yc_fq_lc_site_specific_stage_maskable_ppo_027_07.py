from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import sb3_contrib
import stable_baselines3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_extension_expert_baseline_018_03 as extension
import run_fq_all_year_screen_and_dqn_transfer_014_01 as fq
import run_fq_yc_new_cultivar_forward_screening_013_01 as input_tools
import run_lc_fixed_input_year_screening_017_11 as lc
from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    FEASIBILITY_BONUS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
)


DEFAULT_OUT = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo"
REFERENCE_SUMMARY = ROOT / "benchmark_results" / "027_05" / "027_05_dqn_five_scenario_summary.csv"
EXPERT_SCHEDULE = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_03_extension_expert_schedule.csv"
)
SCENARIOS = ("null", "recorded_farmer", "dssat_auto", "official_extension_expert")
CHECKPOINTS = (0, 60, 120, 180, 240)
OBS_BEFORE = ["cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad"]
OBS_AFTER = ["swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai"]
STD_EPSILON = 1e-6
COMPACT_OBSERVATION_SCHEMA = OBS_BEFORE + ["sw"] + OBS_AFTER


def json_default(value: Any) -> Any:
    """Serialize numpy scalars without weakening the scientific checks."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        allow_nan=True,
        default=json_default,
    )


@dataclass(frozen=True)
class SiteSpec:
    code: str
    station: str
    year: int
    treatment: int
    input_root: Path
    mzx_name: str
    weather_name: str
    soil_id: str
    cultivar: str
    observation_dimension: int
    soil_water_layers: int
    official_stage_daps: tuple[int, ...]
    executable_stage_daps: tuple[int, ...]
    audited_max_dap: int
    initial_condition_pointer: int
    initial_condition_date: str
    simulation_start_date: str
    planting_date: str
    field_pointer: int
    field_weather_station: str


INPUT_PARENT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
OFFICIAL_HUANGHUAI_STAGES = (7, 30, 45, 60, 80, 100)
SPECS = {
    "YC": SiteSpec(
        "YC", "Yucheng", 2014, 2, INPUT_PARENT / "YC", "CNYC0801.MZX",
        "CNYC1401.WTH", "YC99001200", "ZD0985", 22, 6,
        OFFICIAL_HUANGHUAI_STAGES, OFFICIAL_HUANGHUAI_STAGES, 104,
        1, "08153", "14152", "14168", 2, "CNYC1401",
    ),
    "FQ": SiteSpec(
        "FQ", "Fengqiu", 2016, 2, INPUT_PARENT / "FQ", "CNFQ0801.MZX",
        "CNFQ1601.WTH", "FQ99001200", "FQ0985", 23, 7,
        OFFICIAL_HUANGHUAI_STAGES, OFFICIAL_HUANGHUAI_STAGES[:-1], 96,
        1, "07152", "16153", "16162", 2, "CNFQ1601",
    ),
    "LC": SiteSpec(
        "LC", "Luancheng", 2010, 3, INPUT_PARENT / "LC", "CNLC0801.MZX",
        "CNLC1001.WTH", "LC990012007", "XY0004", 26, 10,
        OFFICIAL_HUANGHUAI_STAGES, OFFICIAL_HUANGHUAI_STAGES[:-1], 93,
        3, "10121", "10121", "10171", 1, "CNLC0801",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("latin-1", errors="ignore")).hexdigest()


def expected_labels(spec: SiteSpec) -> list[str]:
    return OBS_BEFORE + [f"sw_layer_{index}" for index in range(1, spec.soil_water_layers + 1)] + OBS_AFTER


def required_source_paths(spec: SiteSpec) -> dict[str, Path]:
    return {
        "source_mzx": spec.input_root / spec.mzx_name,
        "weather": spec.input_root / spec.weather_name,
        "soil": spec.input_root / "SOIL.SOL",
        "cultivar": spec.input_root / "MZCER048.CUL",
    }


def source_hashes(spec: SiteSpec) -> dict[str, str]:
    paths = required_source_paths(spec)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source input(s): " + ", ".join(missing))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()}
    if spec.code == "LC":
        hashes["derived:lc.fixed_source_text()"] = text_sha256(lc.fixed_source_text())
    return hashes


def prepare_text(spec: SiteSpec, scenario: str) -> str:
    if spec.code == "YC":
        source = (spec.input_root / spec.mzx_name).read_text(encoding="latin-1", errors="ignore")
        if scenario in {"official_extension_expert", "linked_train", "linked_smoke"}:
            return input_tools.set_management_for_treatment(source, spec.treatment, "L", "L")
        source_scenario = "recorded" if scenario == "recorded_farmer" else scenario
        return input_tools.prepare_text_for_scenario(source, spec.treatment, source_scenario)
    if spec.code == "FQ":
        if scenario in {"official_extension_expert", "linked_train", "linked_smoke"}:
            source_scenario = "dqn_linked_free_daily"
        elif scenario == "recorded_farmer":
            source_scenario = "recorded_shifted"
        else:
            source_scenario = scenario
        return fq.prepare_text_for_shifted_scenario(spec.year, source_scenario)
    if spec.code == "LC":
        source = lc.fixed_source_text()
        if scenario in {"official_extension_expert", "linked_train", "linked_smoke"}:
            return input_tools.set_management_for_treatment(source, spec.treatment, "L", "L")
        source_scenario = "recorded" if scenario == "recorded_farmer" else scenario
        text = input_tools.prepare_text_for_scenario(source, spec.treatment, source_scenario)
        if source_scenario == "dssat_auto":
            text = input_tools.set_treatment_pointers(text, spec.treatment, "0", "0")
        return text
    raise ValueError(spec.code)


def section_lines(text: str, marker: str) -> list[str]:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith(marker.upper())),
        None,
    )
    if start is None:
        raise RuntimeError(f"Missing MZX section {marker}")
    rows: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("*"):
            break
        rows.append(line)
    return rows


def treatment_values(text: str, treatment: int) -> dict[str, str]:
    rows = section_lines(text, "*TREATMENTS")
    header_line = next((line for line in rows if line.strip().startswith("@N R O C TNAME")), None)
    if header_line is None:
        raise RuntimeError("Missing treatment header")
    header = [re.sub(r"\.+$", "", token).upper() for token in header_line.split()]
    header[0] = "N"
    for line in rows:
        parts = line.split()
        if parts and parts[0] == str(treatment) and not line.lstrip().startswith("@"):
            if len(parts) < len(header):
                raise RuntimeError(f"Treatment {treatment} row is shorter than header")
            return dict(zip(header, parts))
    raise RuntimeError(f"Treatment {treatment} not found")


def prepared_input_provenance(spec: SiteSpec, text: str, scenario: str) -> dict[str, Any]:
    treatment = treatment_values(text, spec.treatment)
    field_pointer = int(treatment.get("FL", "-1"))
    initial_pointer = int(treatment.get("IC", "-1"))
    planting_pointer = int(treatment.get("MP", "-1"))
    cultivar_pointer = int(treatment.get("CU", "-1"))
    field_rows = section_lines(text, "*FIELDS")
    field = next(
        (line.split() for line in field_rows if line.split() and line.split()[0] == str(field_pointer)),
        None,
    )
    if field is None or len(field) < 4:
        raise RuntimeError(f"{spec.code}: target field row was not found")
    initial_rows = section_lines(text, "*INITIAL CONDITIONS")
    initial_match = next(
        (
            re.match(rf"^\s*{initial_pointer}\s+MZ\s+(\d{{5}})\b", line)
            for line in initial_rows
            if re.match(rf"^\s*{initial_pointer}\s+MZ\s+(\d{{5}})\b", line)
        ),
        None,
    )
    planting_rows = section_lines(text, "*PLANTING DETAILS")
    planting_match = next(
        (
            re.match(rf"^\s*{planting_pointer}\s+(\d{{5}})\s+(\d{{5}})\b", line)
            for line in planting_rows
            if re.match(rf"^\s*{planting_pointer}\s+(\d{{5}})\s+(\d{{5}})\b", line)
        ),
        None,
    )
    cultivar_rows = section_lines(text, "*CULTIVARS")
    cultivar_match = next(
        (
            re.match(rf"^\s*{cultivar_pointer}\s+MZ\s+(\S+)", line)
            for line in cultivar_rows
            if re.match(rf"^\s*{cultivar_pointer}\s+MZ\s+(\S+)", line)
        ),
        None,
    )
    general_match = next(
        (
            re.match(rf"^\s*{spec.treatment}\s+GE\s+\d+\s+\d+\s+S\s+(\d{{5}})\b", line)
            for line in text.splitlines()
            if re.match(rf"^\s*{spec.treatment}\s+GE\s+\d+\s+\d+\s+S\s+(\d{{5}})\b", line)
        ),
        None,
    )
    management_match = next(
        (
            re.match(rf"^\s*{spec.treatment}\s+MA\s+\S+\s+(\S+)\s+(\S+)\s+", line)
            for line in text.splitlines()
            if re.match(rf"^\s*{spec.treatment}\s+MA\s+\S+\s+(\S+)\s+(\S+)\s+", line)
        ),
        None,
    )
    options_match = next(
        (
            re.match(rf"^\s*{spec.treatment}\s+OP\s+(\S+)\s+(\S+)\s+", line)
            for line in text.splitlines()
            if re.match(rf"^\s*{spec.treatment}\s+OP\s+(\S+)\s+(\S+)\s+", line)
        ),
        None,
    )
    if not all((initial_match, planting_match, cultivar_match, general_match, management_match, options_match)):
        raise RuntimeError(f"{spec.code}: one or more provenance rows could not be parsed")
    expected_modes = {
        "null": ("N", "N"),
        "recorded_farmer": ("R", "R"),
        "dssat_auto": ("A", "A"),
        "official_extension_expert": ("L", "L"),
        "linked_train": ("L", "L"),
        "linked_smoke": ("L", "L"),
    }
    if scenario == "null" or (scenario == "dssat_auto" and spec.code == "LC"):
        expected_mi_mf = ("0", "0")
    else:
        expected_mi_mf = (str(spec.treatment), str(spec.treatment))
    irrigation_mode, fertilizer_mode = management_match.group(1), management_match.group(2)
    weather_station = field[2]
    soil_id = next((token for token in field if token == spec.soil_id), "")
    values = {
        "treatment": int(spec.treatment),
        "treatment_name": treatment.get("TNAME", ""),
        "irrigation_pointer": treatment.get("MI"),
        "fertilizer_pointer": treatment.get("MF"),
        "cultivar_pointer": cultivar_pointer,
        "field_pointer": field_pointer,
        "initial_condition_pointer": initial_pointer,
        "planting_pointer": planting_pointer,
        "initial_condition_date": initial_match.group(1),
        "simulation_start_date": general_match.group(1),
        "planting_date": planting_match.group(1),
        "weather_station": weather_station,
        "soil_id": soil_id,
        "cultivar_code": cultivar_match.group(1),
        "irrigation_mode": irrigation_mode,
        "fertilizer_mode": fertilizer_mode,
        "water_simulation": options_match.group(1),
        "nitrogen_simulation": options_match.group(2),
    }
    checks = {
        "treatment_exact": int(treatment.get("N", "-1")) == spec.treatment,
        "treatment_name_exact": values["treatment_name"] == f"Sim{spec.year}",
        "management_pointers_exact": (
            values["irrigation_pointer"], values["fertilizer_pointer"]
        ) == expected_mi_mf,
        "field_pointer_exact": field_pointer == spec.field_pointer,
        "initial_condition_pointer_exact": values["initial_condition_pointer"] == spec.initial_condition_pointer,
        "initial_condition_date_exact": values["initial_condition_date"] == spec.initial_condition_date,
        "simulation_start_date_exact": values["simulation_start_date"] == spec.simulation_start_date,
        "planting_date_exact": values["planting_date"] == spec.planting_date,
        "field_weather_station_exact": values["weather_station"] == spec.field_weather_station,
        "soil_id_exact": values["soil_id"] == spec.soil_id,
        "cultivar_exact": values["cultivar_code"] == spec.cultivar,
        "management_modes_exact": (irrigation_mode, fertilizer_mode) == expected_modes[scenario],
        "water_and_nitrogen_enabled": values["water_simulation"] == "Y" and values["nitrogen_simulation"] == "Y",
    }
    return {"scenario": scenario, "values": values, "checks": checks, "pass": all(checks.values())}


def copy_inputs(spec: SiteSpec, run_dir: Path, scenario: str, seed: int) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    filex = input_dir / f"{spec.code}{spec.year}_{scenario}.MZX"
    prepared_text = prepare_text(spec, scenario)
    filex.write_text(prepared_text, encoding="latin-1", errors="ignore")
    for source in sorted(spec.input_root.iterdir()):
        if source.is_file() and source.name != spec.mzx_name:
            shutil.copyfile(source, input_dir / source.name)
    aux_suffixes = {".CUL", ".SOL", ".WTH", ".MZA", ".MZT", ".CLI", ".PRM", ".WDB"}
    auxiliary = [str(path) for path in sorted(input_dir.iterdir()) if path.suffix.upper() in aux_suffixes]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": int(seed),
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": int(spec.treatment),
        "auxiliary_file_paths": auxiliary,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    provenance = prepared_input_provenance(spec, prepared_text, scenario)
    required = required_source_paths(spec)
    copied_hash_checks: dict[str, bool] = {}
    copied_hashes: dict[str, str] = {}
    for label in ("weather", "soil", "cultivar"):
        copied = input_dir / required[label].name
        copied_hashes[label] = sha256(copied)
        copied_hash_checks[label] = copied_hashes[label] == sha256(required[label])
    all_auxiliary_copy_checks: dict[str, bool] = {}
    for source in sorted(spec.input_root.iterdir()):
        if source.is_file() and source.name != spec.mzx_name:
            copied = input_dir / source.name
            all_auxiliary_copy_checks[source.name] = copied.exists() and sha256(copied) == sha256(source)
    provenance.update(
        {
            "derived_mzx": str(filex.relative_to(ROOT)),
            "derived_mzx_sha256": sha256(filex),
            "copied_required_hashes": copied_hashes,
            "copied_required_hashes_match_source": copied_hash_checks,
            "all_auxiliary_copies_match_source": all_auxiliary_copy_checks,
        }
    )
    provenance["pass"] = bool(
        provenance["pass"]
        and all(copied_hash_checks.values())
        and all(all_auxiliary_copy_checks.values())
    )
    (run_dir / "input_provenance.json").write_text(json_text(provenance, indent=2), encoding="utf-8")
    if not provenance["pass"]:
        raise RuntimeError(f"{spec.code}/{scenario}: derived-input provenance audit failed")
    (run_dir / "env_args.json").write_text(
        json_text(env_args, indent=2), encoding="utf-8"
    )
    return env_args


def no_op(raw_env: Any) -> np.ndarray:
    return normalize_action(
        raw_env.formator.action_names,
        raw_env.formator.action_space_dict,
        {"amir": 0.0, "anfer": 0.0},
    )


def real_action(raw_env: Any, irrigation: float, nitrogen: float) -> np.ndarray:
    return normalize_action(
        raw_env.formator.action_names,
        raw_env.formator.action_space_dict,
        {"amir": float(irrigation), "anfer": float(nitrogen)},
    )


def expert_schedule(spec: SiteSpec) -> pd.DataFrame:
    schedule = pd.read_csv(EXPERT_SCHEDULE)
    rows = schedule[(schedule["site"].eq(spec.code)) & (schedule["year"].eq(spec.year))].copy()
    if rows["dap"].astype(int).tolist() != list(spec.official_stage_daps):
        raise ValueError(f"{spec.code}: official schedule mismatch: {rows['dap'].tolist()}")
    return rows


def live_formator_names(raw_env: Any) -> list[str]:
    value = getattr(raw_env.unwrapped, "observation_variables", None)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    for attribute in ("observation_names", "observations", "observation_variables"):
        value = getattr(raw_env.formator, attribute, None)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return []


def assert_live_observation_schema(raw_env: Any, spec: SiteSpec) -> list[str]:
    names = live_formator_names(raw_env)
    if names != COMPACT_OBSERVATION_SCHEMA:
        raise RuntimeError(
            f"{spec.code}: live compact observation schema mismatch; "
            f"expected {COMPACT_OBSERVATION_SCHEMA}, got {names}"
        )
    return names


def summary_row(
    snapshot: Path,
    final_yield: float,
    expected_irrigation: float,
    expected_nitrogen: float,
) -> tuple[dict[str, Any], float, int]:
    rows = parse_summary_out(snapshot / "Summary.OUT")
    candidates = []
    for index, row in enumerate(rows):
        hwam = num(row, "HWAM")
        ircm = num(row, "IRCM")
        nicm = num(row, "NICM")
        if hwam is not None and ircm is not None and nicm is not None:
            score = (
                abs(float(hwam) - float(final_yield))
                + abs(float(ircm) - float(expected_irrigation))
                + abs(float(nicm) - float(expected_nitrogen))
            )
            candidates.append((score, -index, row, index))
    if not candidates:
        raise ValueError(f"No usable Summary.OUT row in {snapshot}")
    score, _, row, index = min(candidates, key=lambda item: (item[0], item[1]))
    checks = {
        "yield": abs(float(num(row, "HWAM")) - float(final_yield)) <= 2.0,
        "irrigation": abs(float(num(row, "IRCM")) - float(expected_irrigation)) <= 2.0,
        "nitrogen": abs(float(num(row, "NICM")) - float(expected_nitrogen)) <= 2.0,
    }
    if not all(checks.values()):
        raise ValueError(
            f"Summary row mismatch at {snapshot}: expected Y/I/N="
            f"{final_yield}/{expected_irrigation}/{expected_nitrogen}, got "
            f"{num(row, 'HWAM')}/{num(row, 'IRCM')}/{num(row, 'NICM')}"
        )
    return row, float(score), int(index)


def strict_metrics_from_snapshot(
    snapshot: Path,
    final_yield: float,
    expected_irrigation: float,
    expected_nitrogen: float,
) -> dict[str, Any]:
    row, score, row_index = summary_row(
        snapshot, final_yield, expected_irrigation, expected_nitrogen
    )
    ircm, nicm = float(num(row, "IRCM")), float(num(row, "NICM"))
    etcp = num(row, "ETCP")
    ypem = num(row, "YPEM")
    ypnam = num(row, "YPNAM")
    if etcp is None or etcp <= 0:
        raise ValueError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and ypem >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and ypnam >= 0 else math.nan
    return {
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": float(etcp),
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": score,
        "summary_row_index": row_index,
        "management_or_action_n_total": float(expected_nitrogen),
        "n_accounting_difference_vs_summary": float(expected_nitrogen) - nicm,
        "i_accounting_difference_vs_summary": float(expected_irrigation) - ircm,
    }


def snapshot_from_env(raw_env: Any) -> Path:
    folder = getattr(raw_env.unwrapped, "_tmp_folder", None)
    if not folder:
        raise RuntimeError("DSSAT temporary folder is unavailable")
    snapshot = Path(folder)
    if not (snapshot / "Summary.OUT").exists():
        raise FileNotFoundError(snapshot / "Summary.OUT")
    return snapshot


def reference_baselines(spec: SiteSpec) -> pd.DataFrame:
    source = pd.read_csv(REFERENCE_SUMMARY, keep_default_na=False)
    rows = source[
        source["site"].eq(spec.code)
        & source["year"].eq(spec.year)
        & source["scenario"].isin(SCENARIOS)
    ].copy()
    if len(rows) != 4 or set(rows["scenario"]) != set(SCENARIOS):
        raise ValueError(f"{spec.code}: 027_05 four-baseline reference is incomplete")
    return rows


def collect_baseline_scenario(
    spec: SiteSpec,
    scenario: str,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    env_args = copy_inputs(spec, run_dir, scenario, seed=0)
    raw_env = extension.make_raw_env(env_args)
    reference = reference_baselines(spec).set_index("scenario").loc[scenario]
    scheduled = extension.split_irrigation_events(expert_schedule(spec)) if scenario == "official_extension_expert" else {}
    stage_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    fired: set[int] = set()
    final_state: dict[str, Any] = {}
    tmp: Path | None = None
    try:
        obs, info = raw_env.reset()
        formator_names = assert_live_observation_schema(raw_env, spec)
        for step in range(420):
            state = latest_observation_dict(raw_env, obs, info)
            dap = int(round(float(scalar(state.get("dap"), step))))
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            if vector.size != spec.observation_dimension or not np.isfinite(vector).all():
                raise ValueError(
                    f"{spec.code}/{scenario}/DAP{dap}: expected finite {spec.observation_dimension}D observation, got {vector.shape}"
                )
            if dap in spec.executable_stage_daps and not any(row["dap"] == dap for row in stage_rows):
                stage_rows.append(
                    {
                        "site": spec.code,
                        "year": spec.year,
                        "scenario": scenario,
                        "dap": dap,
                        **{label: float(value) for label, value in zip(expected_labels(spec), vector)},
                    }
                )
            request = scheduled.get(dap, {"amir": 0.0, "anfer": 0.0}) if dap not in fired else {"amir": 0.0, "anfer": 0.0}
            if dap in scheduled:
                fired.add(dap)
            daily_rows.append(
                {
                    "site": spec.code,
                    "year": spec.year,
                    "scenario": scenario,
                    "step": step,
                    "dap": dap,
                    "grnwt": float(scalar(state.get("grnwt"), 0.0)),
                    "topwt": float(scalar(state.get("topwt"), 0.0)),
                    "swfac": float(scalar(state.get("swfac"), math.nan)),
                    "nstres": float(scalar(state.get("nstres"), math.nan)),
                    "irrigation_action": float(request.get("amir", 0.0)),
                    "nitrogen_action": float(request.get("anfer", 0.0)),
                }
            )
            obs, _, terminated, truncated, info = raw_env.step(
                real_action(raw_env, request.get("amir", 0.0), request.get("anfer", 0.0))
            )
            final_state = latest_observation_dict(raw_env, obs, info)
            if terminated or truncated:
                break
        else:
            raise RuntimeError(f"{spec.code}/{scenario}: season did not terminate within 420 daily steps")
        if [row["dap"] for row in stage_rows] != list(spec.executable_stage_daps):
            raise RuntimeError(
                f"{spec.code}/{scenario}: expected stages {spec.executable_stage_daps}, got {[row['dap'] for row in stage_rows]}"
            )
        tmp = snapshot_from_env(raw_env)
        snapshot = run_dir / "pdi_tmp_snapshot_eval"
        shutil.copytree(tmp, snapshot)
    finally:
        raw_env.close()
    final_yield = float(scalar(final_state.get("grnwt"), math.nan))
    final_biomass = float(scalar(final_state.get("topwt"), math.nan))
    selected, _, _ = summary_row(
        snapshot,
        final_yield,
        float(reference["irrigation_event_total_mm"]),
        float(reference["nitrogen_event_total_kg_ha"]),
    )
    irrigation = float(num(selected, "IRCM") or 0.0)
    nitrogen = float(num(selected, "NICM") or 0.0)
    metrics = strict_metrics_from_snapshot(snapshot, final_yield, irrigation, nitrogen)
    result = {
        "site": spec.code,
        "station": spec.station,
        "year": spec.year,
        "scenario": scenario,
        "final_yield": final_yield,
        "final_biomass": final_biomass,
        "irrigation_total": irrigation,
        "nitrogen_total": nitrogen,
        "stage_state_count": len(stage_rows),
        "max_dap_observed": max(int(row["dap"]) for row in daily_rows),
        "live_compact_observation_names_json": json_text(formator_names),
        "live_compact_observation_schema_exact": formator_names == COMPACT_OBSERVATION_SCHEMA,
        "observation_dimension": spec.observation_dimension,
        "snapshot_path": str(snapshot.relative_to(ROOT)),
        **metrics,
    }
    return stage_rows, daily_rows, result


def fit_scaler(spec: SiteSpec, states: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = expected_labels(spec)
    values = states[labels].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    near_constant = std < STD_EPSILON
    scale = std.copy()
    scale[near_constant] = 1.0
    normalized = (values - mean) / scale
    variable = ~near_constant
    reconstructed = normalized * scale + mean
    validation = {
        "state_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "near_constant_count": int(near_constant.sum()),
        "max_abs_normalized_mean_nonconstant": float(np.max(np.abs(normalized[:, variable].mean(axis=0)))) if variable.any() else 0.0,
        "max_abs_normalized_std_minus_one_nonconstant": float(np.max(np.abs(normalized[:, variable].std(axis=0) - 1.0))) if variable.any() else 0.0,
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - values))),
    }
    table = pd.DataFrame(
        {
            "observation_index": np.arange(spec.observation_dimension),
            "observation_label": labels,
            "mean": mean,
            "scale_std_or_one": scale,
            "near_constant": near_constant,
        }
    )
    return table, validation


def baseline_reference_audit(
    reruns: pd.DataFrame, references: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    mapping = {
        "final_yield": ("final_grain_kg_ha", 2.0),
        "final_biomass": ("final_biomass_kg_ha", 2.0),
        "irrigation_total": ("irrigation_event_total_mm", 1.1),
        "nitrogen_total": ("nitrogen_event_total_kg_ha", 1.1),
        "etcp_mm": ("etcp_mm", 0.2),
        "WP_ET_kg_m3": ("wp_et_kg_m3", 0.02),
        "PFP_N_kg_kg": ("pfp_n_kg_kg", 0.05),
    }
    fresh = reruns.set_index("scenario")
    old = references.set_index("scenario")
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for actual_column, (reference_column, tolerance) in mapping.items():
            actual = pd.to_numeric(pd.Series([fresh.loc[scenario, actual_column]]), errors="coerce").iloc[0]
            reference = pd.to_numeric(pd.Series([old.loc[scenario, reference_column]]), errors="coerce").iloc[0]
            comparable = bool(pd.notna(actual) and pd.notna(reference))
            error = abs(float(actual) - float(reference)) if comparable else math.nan
            passed = bool(error <= tolerance) if comparable else bool(actual_column == "PFP_N_kg_kg")
            rows.append(
                {
                    "scenario": scenario,
                    "metric": actual_column,
                    "fresh_value": float(actual) if pd.notna(actual) else math.nan,
                    "reference_value": float(reference) if pd.notna(reference) else math.nan,
                    "abs_error": error,
                    "tolerance": tolerance,
                    "comparable": comparable,
                    "pass": passed,
                }
            )
    table = pd.DataFrame(rows)
    return table, bool(table["pass"].all())


class SiteStageEnv027(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: SiteSpec,
        raw_env: Any,
        scaler: pd.DataFrame,
        null_yield: float,
        gate_yield: float,
        phase: str,
    ) -> None:
        super().__init__()
        scaler = scaler.sort_values("observation_index")
        if scaler["observation_index"].tolist() != list(range(spec.observation_dimension)):
            raise ValueError(f"{spec.code}: scaler index mismatch")
        if scaler["observation_label"].astype(str).tolist() != expected_labels(spec):
            raise ValueError(f"{spec.code}: scaler label mismatch")
        self.spec = spec
        self.raw_env = raw_env
        self.live_compact_observation_names = assert_live_observation_schema(raw_env, spec)
        self.mean = scaler["mean"].to_numpy(dtype=np.float32)
        self.scale = scaler["scale_std_or_one"].to_numpy(dtype=np.float32)
        self.null_yield = float(null_yield)
        self.gate_yield = float(gate_yield)
        self.phase = phase
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(spec.observation_dimension,), dtype=np.float32
        )
        self.obs: np.ndarray | None = None
        self.state: dict[str, Any] = {}
        self.stage_index = 0
        self.used_i = 0.0
        self.used_n = 0.0
        self.invalid_attempts = 0
        self.valid_steps = 0
        self.stage_rows: list[dict[str, Any]] = []
        self.all_stage_rows: list[dict[str, Any]] = []
        self.completed_episodes: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

    def _dap(self) -> int:
        return int(round(float(scalar(self.state.get("dap"), -999))))

    def _raw(self, irrigation: float, nitrogen: float) -> np.ndarray:
        return real_action(self.raw_env, irrigation, nitrogen)

    def _scaled(self) -> np.ndarray:
        vector = np.asarray(self.obs, dtype=np.float32).reshape(-1)
        if vector.size != self.spec.observation_dimension or not np.isfinite(vector).all():
            raise ValueError(f"{self.spec.code}: invalid observation {vector.shape}")
        return ((vector - self.mean) / self.scale).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        dap = self.spec.executable_stage_daps[self.stage_index]
        remaining_i = IRRIGATION_BUDGET - self.used_i
        remaining_n = NITROGEN_BUDGET - self.used_n
        mask = np.zeros(9, dtype=bool)
        for index, request in ACTION_TABLE_9.items():
            if dap >= 90 and float(request["anfer"]) > 0:
                continue
            if float(request["amir"]) <= remaining_i + 1e-9 and float(request["anfer"]) <= remaining_n + 1e-9:
                mask[index] = True
        if not mask[0]:
            raise RuntimeError("No-op must remain valid")
        return mask

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.obs, info = self.raw_env.reset()
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        first_dap = self.spec.executable_stage_daps[0]
        while self._dap() < first_dap:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
            if terminated or truncated:
                raise RuntimeError(f"{self.spec.code}: season ended before DAP{first_dap}")
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if self._dap() != first_dap:
            raise RuntimeError(f"{self.spec.code}: expected DAP{first_dap}, got DAP{self._dap()}")
        self.stage_index = 0
        self.used_i = self.used_n = 0.0
        self.invalid_attempts = 0
        self.stage_rows = []
        self.last_result = None
        return self._scaled(), {"dap": self._dap(), "action_mask": self.action_masks().copy()}

    def step(self, action_index: int):
        action_index = int(action_index)
        stage_index = int(self.stage_index)
        stage_dap = self.spec.executable_stage_daps[stage_index]
        mask = self.action_masks().copy()
        if action_index < 0 or action_index >= 9 or not bool(mask[action_index]):
            self.invalid_attempts += 1
            raise ValueError(f"{self.spec.code}: invalid/masked action {action_index} at DAP{self._dap()}")
        if self._dap() != stage_dap:
            raise RuntimeError(f"{self.spec.code}: expected DAP{stage_dap}, got DAP{self._dap()}")
        request = ACTION_TABLE_9[action_index]
        executed_i = min(float(request["amir"]), max(0.0, IRRIGATION_BUDGET - self.used_i))
        executed_n = min(float(request["anfer"]), max(0.0, NITROGEN_BUDGET - self.used_n))
        used_i_before, used_n_before = self.used_i, self.used_n
        self.used_i += executed_i
        self.used_n += executed_n
        resource_reward = -(executed_i + 5.0 * executed_n) / 1000.0
        self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(executed_i, executed_n))
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        next_stage = (
            self.spec.executable_stage_daps[stage_index + 1]
            if stage_index + 1 < len(self.spec.executable_stage_daps)
            else None
        )
        while not (terminated or truncated) and next_stage is not None and self._dap() < next_stage:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if next_stage is not None and not (terminated or truncated) and self._dap() != next_stage:
            raise RuntimeError(f"{self.spec.code}: DSSAT skipped DAP{next_stage}; current DAP{self._dap()}")
        if next_stage is not None and (terminated or truncated):
            raise RuntimeError(f"{self.spec.code}: season ended before required DAP{next_stage}")
        self.stage_index += 1
        if self.stage_index == len(self.spec.executable_stage_daps) and not (terminated or truncated):
            while not (terminated or truncated):
                self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
                self.state = latest_observation_dict(self.raw_env, self.obs, info)
        done = bool(terminated or truncated)
        reward = resource_reward
        yield_gain = gate_bonus = terminal_reward = 0.0
        if done:
            if self.stage_index != len(self.spec.executable_stage_daps):
                raise RuntimeError(f"{self.spec.code}: season ended before all executable stages")
            final_yield = float(scalar(self.state.get("grnwt"), 0.0))
            final_biomass = float(scalar(self.state.get("topwt"), 0.0))
            yield_gain = max(0.0, final_yield - self.null_yield)
            gate_bonus = FEASIBILITY_BONUS if final_yield >= self.gate_yield else 0.0
            terminal_reward = (yield_gain + gate_bonus) / 1000.0
            reward += terminal_reward
            self.last_result = {
                "final_yield": final_yield,
                "final_biomass": final_biomass,
                "irrigation_total": self.used_i,
                "nitrogen_total": self.used_n,
            }
            obs_out = np.zeros(self.spec.observation_dimension, dtype=np.float32)
            info_out = {**self.last_result, "terminal_observation_available": False}
        else:
            obs_out = self._scaled()
            info_out = {"dap": self._dap(), "action_mask": self.action_masks().copy()}
        row = {
            "phase": self.phase,
            "episode_index": len(self.completed_episodes),
            "stage_index": stage_index,
            "dap": stage_dap,
            "action_index": action_index,
            "mask_valid": bool(mask[action_index]),
            "valid_actions": ",".join(map(str, np.flatnonzero(mask).tolist())),
            "requested_irrigation": float(request["amir"]),
            "requested_nitrogen": float(request["anfer"]),
            "executed_irrigation": executed_i,
            "executed_nitrogen": executed_n,
            "used_irrigation_before": used_i_before,
            "used_nitrogen_before": used_n_before,
            "remaining_irrigation_before": IRRIGATION_BUDGET - used_i_before,
            "remaining_nitrogen_before": NITROGEN_BUDGET - used_n_before,
            "resource_reward": resource_reward,
            "yield_gain": yield_gain,
            "gate_bonus": gate_bonus,
            "terminal_reward": terminal_reward,
            "reward": reward,
        }
        self.stage_rows.append(row)
        self.all_stage_rows.append(row)
        self.valid_steps += 1
        if done:
            if self.last_result is None:
                raise RuntimeError("Terminal episode is missing last_result")
            self.completed_episodes.append(
                {
                    "phase": self.phase,
                    "episode_index": len(self.completed_episodes),
                    **self.last_result,
                    "resource_reward_total": float(sum(item["resource_reward"] for item in self.stage_rows)),
                    "yield_gain_total": float(sum(item["yield_gain"] for item in self.stage_rows)),
                    "gate_bonus_total": float(sum(item["gate_bonus"] for item in self.stage_rows)),
                    "terminal_reward_total": float(sum(item["terminal_reward"] for item in self.stage_rows)),
                    "episode_total_reward": float(sum(item["reward"] for item in self.stage_rows)),
                    "stage_count": len(self.stage_rows),
                    "action_sequence": ",".join(str(item["action_index"]) for item in self.stage_rows),
                }
            )
        return obs_out, float(reward), bool(terminated), bool(truncated), info_out

    def close(self) -> None:
        self.raw_env.close()


def make_stage_env(
    spec: SiteSpec,
    run_dir: Path,
    scaler: pd.DataFrame,
    reward: dict[str, Any],
    seed: int,
    phase: str,
) -> SiteStageEnv027:
    env_args = copy_inputs(spec, run_dir, "linked_train" if phase != "pretrain_noop_smoke" else "linked_smoke", seed)
    return SiteStageEnv027(
        spec,
        extension.make_raw_env(env_args),
        scaler,
        float(reward["local_null_yield"]),
        float(reward["local_feasibility_yield"]),
        phase,
    )


def run_noop_smoke(
    spec: SiteSpec,
    run_dir: Path,
    scaler: pd.DataFrame,
    reward: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    env = make_stage_env(spec, run_dir, scaler, reward, seed=9000, phase="pretrain_noop_smoke")
    try:
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            if not bool(env.action_masks()[0]):
                raise RuntimeError("No-op is masked during smoke")
            obs, step_reward, terminated, truncated, _ = env.step(0)
            total += float(step_reward)
            done = bool(terminated or truncated)
        episode = dict(env.completed_episodes[-1])
        payload = {
            "observation_dimension": int(np.asarray(obs).size),
            "final_yield": episode["final_yield"],
            "final_biomass": episode["final_biomass"],
            "irrigation_total": episode["irrigation_total"],
            "nitrogen_total": episode["nitrogen_total"],
            "episode_total_reward": total,
            "invalid_attempts": env.invalid_attempts,
            "stage_count": episode["stage_count"],
            "action_sequence": episode["action_sequence"],
        }
        return payload, pd.DataFrame(env.all_stage_rows)
    finally:
        env.close()


def run_readiness(spec: SiteSpec, site_root: Path) -> dict[str, Any]:
    out = site_root / "readiness"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    hashes_before = source_hashes(spec)
    references = reference_baselines(spec)
    all_states: list[dict[str, Any]] = []
    all_daily: list[dict[str, Any]] = []
    reruns: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        states, daily, result = collect_baseline_scenario(
            spec, scenario, out / "baseline_runs" / scenario
        )
        all_states.extend(states)
        all_daily.extend(daily)
        reruns.append(result)
    states_df = pd.DataFrame(all_states)
    daily_df = pd.DataFrame(all_daily)
    reruns_df = pd.DataFrame(reruns)
    states_df.to_csv(out / "scaler_source_stage_states.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(out / "four_baseline_daily_audit.csv", index=False, encoding="utf-8-sig")
    reruns_df.to_csv(out / "four_baseline_fresh_rerun.csv", index=False, encoding="utf-8-sig")
    references.to_csv(out / "four_baseline_027_05_reference.csv", index=False, encoding="utf-8-sig")
    scaler, scaler_validation = fit_scaler(spec, states_df)
    scaler.to_csv(out / "observation_scaler.csv", index=False, encoding="utf-8-sig")
    local_null = float(reruns_df.loc[reruns_df["scenario"].eq("null"), "final_yield"].iloc[0])
    local_gate = float(
        reruns_df.loc[
            reruns_df["scenario"].isin(["dssat_auto", "official_extension_expert"]), "final_yield"
        ].max()
    )
    reward = {
        "site": spec.code,
        "year": spec.year,
        "local_null_yield": local_null,
        "local_feasibility_yield": local_gate,
        "water_cost": 1.0,
        "nitrogen_cost": 5.0,
        "feasibility_bonus": FEASIBILITY_BONUS,
        "irrigation_budget": IRRIGATION_BUDGET,
        "nitrogen_budget": NITROGEN_BUDGET,
        "recorded_used_in_reward": False,
        "precision_source": "fresh deterministic 027_07 rerun",
    }
    (out / "reward_config.json").write_text(
        json_text(reward, indent=2), encoding="utf-8"
    )
    smoke, smoke_rows = run_noop_smoke(spec, out / "smoke" / "all_noop", scaler, reward)
    smoke_rows.to_csv(out / "smoke_stage_actions.csv", index=False, encoding="utf-8-sig")
    reference_yield = references.set_index("scenario")["final_grain_kg_ha"].astype(float)
    actual_yield = reruns_df.set_index("scenario")["final_yield"].astype(float)
    reference_errors = {scenario: abs(actual_yield.loc[scenario] - reference_yield.loc[scenario]) for scenario in SCENARIOS}
    reference_audit, reference_audit_pass = baseline_reference_audit(reruns_df, references)
    reference_audit.to_csv(out / "baseline_reference_errors.csv", index=False, encoding="utf-8-sig")
    expected_state_count = 4 * len(spec.executable_stage_daps)
    checks = {
        "four_baselines_fresh": len(reruns_df) == 4 and set(reruns_df["scenario"]) == set(SCENARIOS),
        "baseline_yield_reference_error_le_2": max(reference_errors.values()) <= 2.0,
        "all_baseline_endpoint_reference_checks_pass": reference_audit_pass,
        "all_baseline_metrics_finite": bool(
            np.isfinite(reruns_df[["final_yield", "final_biomass", "WP_ET_kg_m3"]].to_numpy(dtype=float)).all()
        ),
        "stage_state_count_exact": len(states_df) == expected_state_count,
        "observation_dimension_exact": scaler_validation["dimension"] == spec.observation_dimension,
        "observation_labels_exact": scaler["observation_label"].astype(str).tolist() == expected_labels(spec),
        "live_compact_observation_schema_exact": bool(
            reruns_df["live_compact_observation_schema_exact"].astype(bool).all()
        ),
        "baseline_n_accounting_difference_le_2": bool(
            reruns_df["n_accounting_difference_vs_summary"].abs().le(2.0).all()
        ),
        "baseline_i_accounting_difference_le_2": bool(
            reruns_df["i_accounting_difference_vs_summary"].abs().le(2.0).all()
        ),
        "scaler_mean_pass": scaler_validation["max_abs_normalized_mean_nonconstant"] < 1e-4,
        "scaler_std_pass": scaler_validation["max_abs_normalized_std_minus_one_nonconstant"] < 1e-4,
        "scaler_inverse_pass": scaler_validation["reconstruction_max_abs_error"] < 2e-3,
        "official_stage_source_six_points": spec.official_stage_daps == OFFICIAL_HUANGHUAI_STAGES,
        "all_executable_stages_preharvest": max(spec.executable_stage_daps) <= spec.audited_max_dap,
        "unreachable_dap100_not_moved": (
            spec.code == "YC" or (100 not in spec.executable_stage_daps and spec.audited_max_dap < 100)
        ),
        "smoke_observation_dimension": smoke["observation_dimension"] == spec.observation_dimension,
        "smoke_stage_count": smoke["stage_count"] == len(spec.executable_stage_daps),
        "smoke_all_noop": smoke["action_sequence"] == ",".join("0" for _ in spec.executable_stage_daps),
        "smoke_zero_resources": smoke["irrigation_total"] == 0 and smoke["nitrogen_total"] == 0,
        "smoke_null_yield_close": abs(smoke["final_yield"] - local_null) <= 2.0,
        "smoke_zero_reward": abs(smoke["episode_total_reward"]) <= 1e-10,
        "smoke_zero_invalid": smoke["invalid_attempts"] == 0,
        "source_hashes_unchanged": source_hashes(spec) == hashes_before,
    }
    ready = all(checks.values())
    payload = {
        "status": "completed",
        "branch": "A_ready_for_seed0" if ready else "B_readiness_failed",
        "site": spec.code,
        "year": spec.year,
        "training_steps": 0,
        "model_count": 0,
        "input_hashes": hashes_before,
        "official_stage_daps": list(spec.official_stage_daps),
        "executable_stage_daps": list(spec.executable_stage_daps),
        "audited_max_dap": spec.audited_max_dap,
        "observation_dimension": spec.observation_dimension,
        "soil_water_layers": spec.soil_water_layers,
        "scaler_validation": scaler_validation,
        "reward_config": reward,
        "fresh_baselines": reruns_df.to_dict("records"),
        "baseline_reference_errors": reference_errors,
        "baseline_reference_audit": reference_audit.to_dict("records"),
        "smoke": smoke,
        "checks": checks,
        "next_step_allowed": ready,
    }
    (out / "readiness_result.json").write_text(
        json_text(payload, indent=2), encoding="utf-8"
    )
    return payload


def load_readiness(spec: SiteSpec, site_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    readiness_dir = site_root / "readiness"
    payload = json.loads((readiness_dir / "readiness_result.json").read_text(encoding="utf-8"))
    if payload.get("branch") != "A_ready_for_seed0":
        raise RuntimeError(f"{spec.code}: readiness did not authorize training")
    scaler = pd.read_csv(readiness_dir / "observation_scaler.csv")
    baselines = pd.read_csv(readiness_dir / "four_baseline_fresh_rerun.csv", keep_default_na=False)
    return payload, scaler, baselines


def model_finite(model: MaskablePPO) -> bool:
    return all(bool(np.isfinite(parameter.detach().cpu().numpy()).all()) for parameter in model.policy.parameters())


def metrics_and_passes(
    spec: SiteSpec,
    env: SiteStageEnv027,
    episode: dict[str, Any],
    baselines: pd.DataFrame,
) -> dict[str, Any]:
    snapshot = snapshot_from_env(env.raw_env)
    metrics = strict_metrics_from_snapshot(
        snapshot,
        float(episode["final_yield"]),
        float(episode["irrigation_total"]),
        float(episode["nitrogen_total"]),
    )
    auto_expert = baselines[baselines["scenario"].isin(["dssat_auto", "official_extension_expert"])]
    yield_gate = float(auto_expert["final_yield"].max())
    wp_gate = float(auto_expert["WP_ET_kg_m3"].max())
    pfp_gate = float(pd.to_numeric(auto_expert["PFP_N_kg_kg"], errors="coerce").max())
    pfp = float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) else math.nan
    primary = bool(
        float(episode["final_yield"]) >= yield_gate
        and float(metrics["WP_ET_kg_m3"]) >= wp_gate
        and math.isfinite(pfp)
        and pfp >= pfp_gate
    )
    all_yield_max = float(baselines["final_yield"].max())
    all_wp_max = float(baselines["WP_ET_kg_m3"].max())
    all_pfp_series = pd.to_numeric(baselines["PFP_N_kg_kg"], errors="coerce").dropna()
    all_pfp_max = float(all_pfp_series.max()) if not all_pfp_series.empty else math.nan
    yield_winner = float(episode["final_yield"]) > all_yield_max
    wp_winner = float(metrics["WP_ET_kg_m3"]) > all_wp_max
    pfp_winner = math.isfinite(pfp) and math.isfinite(all_pfp_max) and pfp > all_pfp_max
    return {
        **metrics,
        "primary_yield_gate": yield_gate,
        "primary_wp_gate": wp_gate,
        "primary_pfp_gate": pfp_gate,
        "primary_pass": primary,
        "all_four_yield_max": all_yield_max,
        "all_four_wp_max": all_wp_max,
        "all_four_pfp_positive_n_max": all_pfp_max,
        "yield_gap_vs_all_four_max": float(episode["final_yield"]) - all_yield_max,
        "wp_gap_vs_all_four_max": float(metrics["WP_ET_kg_m3"]) - all_wp_max,
        "pfp_gap_vs_all_four_positive_n_max": pfp - all_pfp_max if math.isfinite(pfp) and math.isfinite(all_pfp_max) else math.nan,
        "strict_yield_winner_all_four": yield_winner,
        "strict_wp_winner_all_four": wp_winner,
        "strict_pfp_winner_all_four_positive_n": pfp_winner,
        "advisor_any_metric_strict_winner": bool(yield_winner or wp_winner or pfp_winner),
        "advisor_close_other_metrics_not_adjudicated": True,
    }


def evaluate(
    model: MaskablePPO,
    env: SiteStageEnv027,
    checkpoint: int,
    baselines: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_start = len(env.all_stage_rows)
    episode_start = len(env.completed_episodes)
    obs, _ = env.reset()
    done = False
    while not done:
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = bool(terminated or truncated)
    if len(env.completed_episodes) != episode_start + 1:
        raise RuntimeError("Evaluation did not produce exactly one completed season")
    episode = dict(env.completed_episodes[-1])
    result = {
        "checkpoint": int(checkpoint),
        **episode,
        **metrics_and_passes(env.spec, env, episode, baselines),
    }
    stages = [{"checkpoint": int(checkpoint), **row} for row in env.all_stage_rows[row_start:]]
    if len(stages) != len(env.spec.executable_stage_daps):
        raise RuntimeError(f"Checkpoint {checkpoint} evaluation has {len(stages)} stages")
    return result, stages


def select_by_reward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [row for row in rows if int(row["checkpoint"]) > 0]
    return max(trained, key=lambda row: (float(row["episode_total_reward"]), -int(row["checkpoint"])))


def run_seed(spec: SiteSpec, site_root: Path, seed: int) -> dict[str, Any]:
    readiness, scaler, baselines = load_readiness(spec, site_root)
    seed_dir = site_root / f"seed{seed}"
    if seed_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {seed_dir}")
    seed_dir.mkdir(parents=True)
    hashes_before = source_hashes(spec)
    reward = readiness["reward_config"]
    train_env = make_stage_env(spec, seed_dir / "runtime_train", scaler, reward, seed, f"train_seed{seed}")
    eval_env = make_stage_env(spec, seed_dir / "runtime_eval", scaler, reward, 1000 + seed, f"eval_seed{seed}")
    checkpoint_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    learn_calls = 0
    try:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=60,
            batch_size=30,
            n_epochs=5,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [32, 32]},
            seed=seed,
            device="cpu",
            verbose=0,
        )
        row, stages = evaluate(model, eval_env, 0, baselines)
        stem = seed_dir / "checkpoint_000000"
        model.save(stem)
        model_path = stem.with_suffix(".zip")
        digest = sha256(model_path)
        row.update({"seed": seed, "model_path": str(model_path.relative_to(ROOT)), "model_sha256": digest})
        checkpoint_rows.append(row)
        stage_rows.extend({"seed": seed, **stage} for stage in stages)
        hashes["0"] = digest
        for block_index, checkpoint in enumerate(CHECKPOINTS[1:]):
            model.learn(total_timesteps=60, reset_num_timesteps=(block_index == 0), progress_bar=False)
            learn_calls += 1
            if int(model.num_timesteps) != checkpoint:
                raise RuntimeError(f"{spec.code}/seed{seed}: expected checkpoint {checkpoint}, got {model.num_timesteps}")
            row, stages = evaluate(model, eval_env, checkpoint, baselines)
            stem = seed_dir / f"checkpoint_{checkpoint:06d}"
            model.save(stem)
            model_path = stem.with_suffix(".zip")
            digest = sha256(model_path)
            row.update({"seed": seed, "model_path": str(model_path.relative_to(ROOT)), "model_sha256": digest})
            checkpoint_rows.append(row)
            stage_rows.extend({"seed": seed, **stage} for stage in stages)
            hashes[str(checkpoint)] = digest
        selected = select_by_reward(checkpoint_rows)
        expected_training_seasons = 240 // len(spec.executable_stage_daps)
        metric_keys = ("final_yield", "final_biomass", "irrigation_total", "nitrogen_total", "episode_total_reward", "WP_ET_kg_m3")
        checks = {
            "versions_2p8p0": stable_baselines3.__version__ == "2.8.0" and sb3_contrib.__version__ == "2.8.0",
            "exact_checkpoints": [item["checkpoint"] for item in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_training_steps": int(model.num_timesteps) == 240,
            "expected_training_seasons": len(train_env.completed_episodes) == expected_training_seasons,
            "exact_four_learn_calls": learn_calls == 4,
            "five_evaluation_seasons": len(eval_env.completed_episodes) == 5,
            "all_evaluations_expected_stage_count": len(stage_rows) == 5 * len(spec.executable_stage_daps),
            "zero_masked_action_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "finite_required_metrics": all(math.isfinite(float(item[key])) for item in checkpoint_rows for key in metric_keys),
            "all_checkpoint_n_accounting_difference_le_2": all(
                abs(float(item["n_accounting_difference_vs_summary"])) <= 2.0
                for item in checkpoint_rows
            ),
            "all_checkpoint_i_accounting_difference_le_2": all(
                abs(float(item["i_accounting_difference_vs_summary"])) <= 2.0
                for item in checkpoint_rows
            ),
            "finite_model_parameters": model_finite(model),
            "reward_components_close": all(
                abs(float(item["episode_total_reward"]) - float(item["resource_reward_total"]) - float(item["terminal_reward_total"])) <= 1e-9
                for item in checkpoint_rows
            ),
            "five_unique_model_hashes": len(set(hashes.values())) == 5,
            "source_hashes_unchanged": source_hashes(spec) == hashes_before,
        }
        engineering_pass = all(checks.values())
        branch = (
            "A_seed_primary_signal"
            if engineering_pass and bool(selected["primary_pass"])
            else ("B_seed_no_primary" if engineering_pass else "C_execution_failed")
        )
        pd.DataFrame(checkpoint_rows).to_csv(seed_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(stage_rows).to_csv(seed_dir / "checkpoint_stage_actions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(train_env.completed_episodes).to_csv(seed_dir / "training_episode_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(train_env.all_stage_rows).to_csv(seed_dir / "training_stage_actions.csv", index=False, encoding="utf-8-sig")
        payload = {
            "status": "completed",
            "branch": branch,
            "site": spec.code,
            "year": spec.year,
            "seed": seed,
            "checks": checks,
            "engineering_pass": engineering_pass,
            "config": {
                "total_timesteps": 240,
                "n_steps": 60,
                "batch_size": 30,
                "n_epochs": 5,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "learning_rate": 3e-4,
                "net_arch": [32, 32],
                "checkpoints": list(CHECKPOINTS),
                "observation_dimension": spec.observation_dimension,
                "executable_stage_daps": list(spec.executable_stage_daps),
            },
            "selection_rule": "max deterministic episode_total_reward among checkpoints 60/120/180/240; exact ties choose earlier",
            "selected_checkpoint": selected,
            "checkpoint_results": checkpoint_rows,
            "training_seasons": len(train_env.completed_episodes),
            "evaluation_seasons": len(eval_env.completed_episodes),
            "learn_calls": learn_calls,
            "next_step_allowed": branch == "A_seed_primary_signal",
        }
        (seed_dir / "seed_result.json").write_text(
            json_text(payload, indent=2), encoding="utf-8"
        )
        return payload
    finally:
        train_env.close()
        eval_env.close()


def selected_summary(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_checkpoint"]
    return {
        "seed": int(payload["seed"]),
        "engineering_pass": bool(payload["engineering_pass"]),
        "selected_checkpoint": int(selected["checkpoint"]),
        "model_sha256": selected["model_sha256"],
        "yield": float(selected["final_yield"]),
        "irrigation": float(selected["irrigation_total"]),
        "nitrogen": float(selected["nitrogen_total"]),
        "WP_ET": float(selected["WP_ET_kg_m3"]),
        "PFP_N": float(selected["PFP_N_kg_kg"]) if pd.notna(selected["PFP_N_kg_kg"]) else math.nan,
        "reward": float(selected["episode_total_reward"]),
        "primary_pass": bool(selected["primary_pass"]),
        "advisor_any_metric_strict_winner": bool(selected["advisor_any_metric_strict_winner"]),
        "winning_yield": bool(selected["strict_yield_winner_all_four"]),
        "winning_WP_ET": bool(selected["strict_wp_winner_all_four"]),
        "winning_PFP_N": bool(selected["strict_pfp_winner_all_four_positive_n"]),
        "yield_gap_vs_all_four_max": float(selected["yield_gap_vs_all_four_max"]),
        "wp_gap_vs_all_four_max": float(selected["wp_gap_vs_all_four_max"]),
        "pfp_gap_vs_all_four_positive_n_max": float(selected["pfp_gap_vs_all_four_positive_n_max"])
        if pd.notna(selected["pfp_gap_vs_all_four_positive_n_max"])
        else math.nan,
    }


def run_site(spec: SiteSpec, out_root: Path) -> dict[str, Any]:
    site_root = out_root / spec.code
    if site_root.exists():
        raise FileExistsError(f"Refusing to overwrite {site_root}")
    site_root.mkdir(parents=True)
    readiness = run_readiness(spec, site_root)
    if not readiness["next_step_allowed"]:
        payload = {
            "site": spec.code,
            "year": spec.year,
            "status": "stopped",
            "branch": "B_readiness_failed",
            "readiness": readiness,
            "seeds_run": [],
            "next_step_allowed": False,
        }
    else:
        seed_results = [run_seed(spec, site_root, 0)]
        if seed_results[0]["next_step_allowed"]:
            seed_results.extend(run_seed(spec, site_root, seed) for seed in (1, 2))
        summaries = [selected_summary(result) for result in seed_results]
        primary_count = sum(bool(item["primary_pass"]) for item in summaries)
        advisor_winner_count = sum(bool(item["advisor_any_metric_strict_winner"]) for item in summaries)
        if len(seed_results) == 3:
            branch = "A_three_seed_primary_replicated" if primary_count >= 2 else "B_three_seed_primary_not_replicated"
        else:
            branch = "B_seed0_primary_failed_stop"
        pd.DataFrame(summaries).to_csv(site_root / "selected_seed_summary.csv", index=False, encoding="utf-8-sig")
        payload = {
            "site": spec.code,
            "year": spec.year,
            "status": "completed",
            "branch": branch,
            "readiness_branch": readiness["branch"],
            "seeds_run": [int(result["seed"]) for result in seed_results],
            "primary_pass_count": primary_count,
            "advisor_any_metric_strict_winner_count": advisor_winner_count,
            "selected_seed_summary": summaries,
            "next_step_allowed": branch == "A_three_seed_primary_replicated",
            "cross_year_started": False,
        }
    (site_root / "site_result.json").write_text(
        json_text(payload, indent=2), encoding="utf-8"
    )
    return payload


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据_"
    safe = df.copy().fillna("")
    columns = list(safe.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in safe.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def record_path(out_root: Path) -> Path:
    if out_root.name == DEFAULT_OUT.name:
        suffix = ""
    elif out_root.name.startswith(DEFAULT_OUT.name + "_"):
        suffix = "_" + out_root.name[len(DEFAULT_OUT.name) + 1 :]
    else:
        suffix = "_" + out_root.name
    return ROOT / "docs" / f"2026-07-17_027_07_yc_fq_lc_site_specific_stage_maskable_ppo{suffix}.md"


def write_record(results: list[dict[str, Any]], failures: list[dict[str, Any]], out_root: Path) -> Path:
    overview_rows = []
    seed_rows = []
    for result in results:
        overview_rows.append(
            {
                "site": result["site"],
                "year": result["year"],
                "branch": result["branch"],
                "seeds_run": ",".join(map(str, result.get("seeds_run", []))),
                "primary_count": result.get("primary_pass_count", 0),
                "advisor_any_metric_winner_count": result.get("advisor_any_metric_strict_winner_count", 0),
            }
        )
        for row in result.get("selected_seed_summary", []):
            seed_rows.append({"site": result["site"], **row})
    overview = pd.DataFrame(overview_rows)
    seeds = pd.DataFrame(seed_rows)
    overview.to_csv(out_root / "027_07_three_site_overview.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(out_root / "027_07_three_site_selected_seed_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 027_07 YC/FQ/LC 站点专属阶段型 MaskablePPO 记录",
        "",
        "## 执行边界",
        "",
        "- 三站点串行执行；未做联合训练、权重迁移或跨年验证。",
        "- PPO 核心参数、9动作、I120/N300预算与 reward 结构未修改。",
        "- YC使用官方六个可执行阶段7/30/45/60/80/100；FQ和LC因在DAP100前收获，仅使用真实可执行的7/30/45/60/80，未移动第六阶段。",
        "- 原027_00 primary用于训练扩展硬门槛；导师最新至少一项严格领先视图只做报告，不参与reward或选模。",
        "- 每个运行目录均保存派生MZX哈希、指针/日期/天气/土壤/品种/管理模式审计；Summary.OUT按Y/I/N三项匹配。",
        "",
        "## 站点分支",
        "",
        markdown_table(overview),
        "",
        "## 预注册选中模型",
        "",
        markdown_table(seeds.round(6) if not seeds.empty else seeds),
        "",
        "## 失败记录",
        "",
        markdown_table(pd.DataFrame(failures)),
        "",
        "## 结论边界",
        "",
        "本记录只回答三个训练锚点在冻结阶段型MaskablePPO下能否产生跨seed初步信号。任何未达到2/3原primary的站点均按预注册规则停止，不据此调参；任何达到的站点也尚未完成同站跨年泛化。",
    ]
    path = record_path(out_root)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment record {path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sites", nargs="+", choices=list(SPECS), default=["YC", "FQ", "LC"])
    args = parser.parse_args()
    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite {out_root}")
    out_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for site in args.sites:
        spec = SPECS[site]
        print(f"[027_07] START {site}{spec.year}", flush=True)
        try:
            result = run_site(spec, out_root)
            results.append(result)
            print(
                json_text(
                    {
                        "site": site,
                        "branch": result["branch"],
                        "seeds_run": result.get("seeds_run", []),
                        "primary_pass_count": result.get("primary_pass_count", 0),
                    },
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "site": site,
                "year": spec.year,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_dir = out_root / site
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(
                json_text(failure, indent=2), encoding="utf-8"
            )
            print(json_text(failure), flush=True)
    doc_path = write_record(results, failures, out_root)
    payload = {
        "status": "completed" if not failures else "partial",
        "sites_requested": args.sites,
        "sites_completed": [result["site"] for result in results],
        "failures": failures,
        "training_parallelism": 1,
        "results": results,
        "record_path": str(doc_path.relative_to(ROOT)),
    }
    (out_root / "027_07_result.json").write_text(
        json_text(payload, indent=2), encoding="utf-8"
    )
    print(json_text({"status": payload["status"], "sites_completed": payload["sites_completed"], "failure_count": len(failures)}, indent=2))


if __name__ == "__main__":
    main()

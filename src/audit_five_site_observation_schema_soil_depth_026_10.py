from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hla_frozen_sy_stage_ppo_cross_site_026_09 as hla_transfer
import run_lc_fixed_input_year_screening_017_11 as lc
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as sy_transfer
import run_yc_fq_frozen_nstep_cross_site_020_12 as cross


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmark_results" / "026_10"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
NON_SOIL_FEATURE_COUNT = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_soil_id(filex: Path) -> str:
    text = filex.read_text(encoding="latin-1", errors="ignore")
    matches = re.findall(r"\b(?:HL|YC|FQ|LC|SY)\d{8,10}\b", text, flags=re.IGNORECASE)
    if not matches:
        raise ValueError(f"No station soil id found in {filex}")
    return matches[0].upper()


def parse_profile(soil_path: Path, requested_id: str) -> tuple[str, list[float]]:
    lines = soil_path.read_text(encoding="latin-1", errors="ignore").splitlines()
    wanted = requested_id.upper()
    candidates: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        if not line.startswith("*"):
            continue
        token = line[1:].split()[0].upper() if line[1:].split() else ""
        if token == wanted or token.startswith(wanted) or wanted.startswith(token):
            candidates.append((token, index))
    if not candidates:
        raise ValueError(f"Soil profile {requested_id} not found in {soil_path}")
    candidates.sort(key=lambda item: (item[0] != wanted, abs(len(item[0]) - len(wanted))))
    profile_id, start = candidates[0]
    header_index = None
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("*"):
            break
        if re.match(r"^@\s*SLB\b", stripped, flags=re.IGNORECASE):
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"@SLB header not found for {profile_id}")
    depths: list[float] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if stripped.startswith(("*", "@")):
            break
        try:
            depths.append(float(stripped.split()[0]))
        except (ValueError, IndexError):
            break
    if not depths:
        raise ValueError(f"No soil layers parsed for {profile_id}")
    return profile_id, depths


def reset_observation(site: str, out: Path) -> tuple[np.ndarray, Path, Path, str, np.ndarray, np.ndarray]:
    if site == "HLA":
        run_dir = ROOT / "benchmark_results" / "026_09_smoke_hla2010_seed0" / "2010" / "seed0" / "frozen_sy_stage_ppo"
        env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
        filex = Path(env_args["fileX_template_path"])
        env = sy_transfer.sy.make_raw_env(env_args)
    elif site == "SY":
        source_run = ROOT / "benchmark_results" / "026_07_attempt2" / "2014" / "ppo_runs" / "2014" / "seed0" / "transfer_frozen_ppo_seed0"
        case = out / "runtime" / "SY"
        input_dir = case / "input"
        input_dir.mkdir(parents=True)
        for source in sorted((source_run / "input").iterdir()):
            if source.is_file():
                shutil.copyfile(source, input_dir / source.name)
        filex = input_dir / "CNSY1201.MZX"
        env_args = {
            "log_saving_path": str(case / "pdi_gym.log"), "mode": "all", "seed": 0,
            "random_weather": False, "evaluation": True,
            "fileX_template_path": str(filex), "experiment_number": 2,
            "auxiliary_file_paths": [str(path) for path in sorted(input_dir.iterdir()) if path.is_file() and path != filex],
            "run_dssat_location": "/opt/dssat_pdi/run_dssat",
        }
        env = sy_transfer.sy.make_raw_env(env_args)
    elif site in {"YC", "FQ"}:
        spec = cross.SITE_SPECS[site]
        case = out / "runtime" / site
        input_dir = case / "input"
        filex = cross.copy_site_inputs(spec, input_dir, cross.prepare_site_text(spec, "dqn"), f"{site}_026_10.MZX")
        env_args = cross.build_env_args(spec, case / "run", filex)
        env = cross.make_raw_env(env_args)
    elif site == "LC":
        lc.OUT_DIR = out / "runtime" / "LC"
        run_dir = lc.prepare_run(2010, "null")
        env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
        filex = Path(env_args["fileX_template_path"])
        env = sy_transfer.sy.make_raw_env(env_args)
    else:
        raise ValueError(site)
    try:
        observation, _ = env.reset()
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        state = dict(getattr(env.unwrapped, "_state", {}) or {})
        dlayr = np.asarray(state.get("dlayr", []), dtype=np.float64).reshape(-1)
        sw = np.asarray(state.get("sw", []), dtype=np.float64).reshape(-1)
    finally:
        env.close()
    soil_path = filex.parent / "SOIL.SOL"
    return vector, filex, soil_path, parse_soil_id(filex), dlayr, sw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    scaler = pd.read_csv(SCALER)
    scaler_labels = scaler["observation_label"].astype(str).tolist()
    if len(scaler_labels) != 25 or sum(label.startswith("sw_layer_") for label in scaler_labels) != 9:
        raise ValueError("SY scaler is not the expected 25-feature/9-soil-layer schema")

    schema_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    runtime_layer_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    year_map = {"SY": 2014, "HLA": 2010, "YC": 2014, "FQ": 2016, "LC": 2010}

    for site in ("SY", "HLA", "YC", "FQ", "LC"):
        try:
            vector, filex, soil_path, soil_id, dlayr, sw = reset_observation(site, out)
            observed_layers = int(vector.size - NON_SOIL_FEATURE_COUNT)
            if len(dlayr) != observed_layers or len(sw) != observed_layers:
                raise ValueError(
                    f"{site}: observation-implied layers={observed_layers}, dlayr={len(dlayr)}, sw={len(sw)}"
                )
            schema_row = {
                "site": site, "year": year_map[site], "source": "runtime_reset",
                "observation_dimension": int(vector.size),
                "soil_water_layer_count": observed_layers,
                "schema_matches_sy": bool(vector.size == 25 and observed_layers == 9),
                "reset_only": True,
                "filex": str(filex.relative_to(ROOT)),
                "soil_id_requested": soil_id,
                "soil_file_sha256": sha256(soil_path),
            }
            schema_rows.append(schema_row)
            profile_id, depths = parse_profile(soil_path, soil_id)
            schema_row["soil_profile_layer_count"] = len(depths)
            schema_row["soil_id_resolved"] = profile_id
            top = 0.0
            for number, bottom in enumerate(depths, start=1):
                layer_rows.append({
                    "site": site, "year": year_map[site], "soil_id_requested": soil_id,
                    "soil_id_resolved": profile_id, "layer": number,
                    "top_depth_cm": top, "bottom_depth_cm": bottom,
                    "thickness_cm": bottom - top,
                })
                top = bottom
            runtime_bottom = 0.0
            for number, (thickness, water) in enumerate(zip(dlayr, sw), start=1):
                runtime_layer_rows.append({
                    "site": site, "year": year_map[site], "runtime_layer": number,
                    "top_depth_cm": runtime_bottom,
                    "bottom_depth_cm": runtime_bottom + float(thickness),
                    "thickness_cm": float(thickness), "reset_sw_vwc": float(water),
                })
                runtime_bottom += float(thickness)
        except Exception as exc:  # preserve all provenance failures in one audit
            errors.append({"site": site, "error_type": type(exc).__name__, "error": str(exc)})

    schema = pd.DataFrame(schema_rows)
    layers = pd.DataFrame(layer_rows)
    runtime_layers = pd.DataFrame(runtime_layer_rows)
    schema.to_csv(out / "026_10_observation_schema.csv", index=False, encoding="utf-8-sig")
    layers.to_csv(out / "026_10_soil_profile_layers.csv", index=False, encoding="utf-8-sig")
    runtime_layers.to_csv(out / "026_10_runtime_soil_water_layers.csv", index=False, encoding="utf-8-sig")
    all_sites = set(schema["site"]) == {"SY", "HLA", "YC", "FQ", "LC"}
    all_depths = set(layers["site"]) == {"SY", "HLA", "YC", "FQ", "LC"} if not layers.empty else False
    all_runtime_depths = set(runtime_layers["site"]) == {"SY", "HLA", "YC", "FQ", "LC"} if not runtime_layers.empty else False
    all_match = bool(all_sites and schema.loc[schema.site.ne("SY"), "schema_matches_sy"].all())
    if errors or not all_sites or not all_depths or not all_runtime_depths:
        branch = "C_soil_depth_or_runtime_schema_provenance_incomplete"
    elif all_match:
        branch = "A_all_external_sites_directly_schema_compatible"
    else:
        branch = "B_depth_aware_observation_harmonization_required"
    result = {
        "status": "completed", "branch": branch, "training_steps": 0,
        "policy_actions_executed": 0, "dssat_resets": int(schema["reset_only"].sum()),
        "sy_reference_dimension": 25, "sy_reference_soil_layers": 9,
        "all_sites_audited": all_sites, "all_external_soil_depths_traced": all_depths,
        "all_runtime_layer_depths_traced": all_runtime_depths,
        "all_external_schemas_match_sy": all_match, "errors": errors,
        "next_step_allowed": branch == "A_all_external_sites_directly_schema_compatible",
        "next_step": (
            "Resume direct frozen-policy transfer" if branch.startswith("A_")
            else "Preregister a physical-depth-conserving observation harmonization adapter; do not resume formal cross-site transfer yet"
            if branch.startswith("B_") else "Repair unresolved input provenance before any policy transfer"
        ),
    }
    (out / "026_10_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

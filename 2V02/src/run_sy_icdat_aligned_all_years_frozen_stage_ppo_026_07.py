from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as base
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    set_management_for_treatment,
    set_treatment_pointers,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "026_07_attempt2"
SMOKE_OUT = ROOT / "benchmark_results" / "026_07_smoke_2015_aligned_null"
EXPERT_SMOKE_OUT = ROOT / "benchmark_results" / "026_07_smoke_2014_expert_render_fix_attempt3"
YEARS = (2012, 2014, 2015)
TARGET_ICDAT = {2012: 12099, 2014: 14099, 2015: 15099}
ORIGINAL_PREPARE_RUN_DIR = base.sy.prepare_run_dir
ORIGINAL_MZX_SHA256 = base.sha256(base.MZX)


def selected_ic_line(text: str, ic_pointer: int) -> str:
    pattern = re.compile(rf"(?m)^\s*{ic_pointer}\s+MZ\s+\d{{5}}\b.*$")
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one IC={ic_pointer} row, found {len(matches)}")
    return matches[0]


def align_runtime_icdat(filex: Path, year: int) -> dict[str, Any]:
    if year not in TARGET_ICDAT:
        raise ValueError(f"No approved ICDAT mapping for {year}")
    treatment, ic_pointer = base.EXPECTED_TREATMENTS[year]
    before_bytes = filex.read_bytes()
    before_text = before_bytes.decode("latin-1", errors="ignore")
    before_line = selected_ic_line(before_text, ic_pointer)
    pattern = re.compile(rf"(?m)^(\s*{ic_pointer}\s+MZ\s+)\d{{5}}(\b.*)$")
    after_text, count = pattern.subn(rf"\g<1>{TARGET_ICDAT[year]}\g<2>", before_text)
    if count != 1:
        raise ValueError(f"{year}: ICDAT replacement count was {count}, expected 1")
    after_line = selected_ic_line(after_text, ic_pointer)
    before_tokens = before_line.split()
    after_tokens = after_line.split()
    if before_tokens[:2] != after_tokens[:2] or before_tokens[3:] != after_tokens[3:]:
        raise ValueError(f"{year}: fields other than ICDAT changed")
    if int(after_tokens[2]) != TARGET_ICDAT[year]:
        raise ValueError(f"{year}: aligned ICDAT is {after_tokens[2]}, expected {TARGET_ICDAT[year]}")
    filex.write_text(after_text, encoding="latin-1", errors="ignore")
    manifest = {
        "year": year,
        "treatment": treatment,
        "ic_pointer": ic_pointer,
        "approved_target_icdat": TARGET_ICDAT[year],
        "original_authoritative_mzx": str(base.MZX.relative_to(ROOT)),
        "original_authoritative_mzx_sha256": ORIGINAL_MZX_SHA256,
        "runtime_mzx_sha256_before": base.hashlib.sha256(before_bytes).hexdigest(),
        "runtime_mzx_sha256_after": base.sha256(filex),
        "ic_row_before": before_line,
        "ic_row_after": after_line,
        "changed": before_line != after_line,
        "only_icdat_field_changed": True,
    }
    (filex.parent.parent / "icdat_alignment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def aligned_prepare_run_dir(year: int, scenario: str, seed: int = 0, root: Path | None = None) -> Path:
    run_dir = ORIGINAL_PREPARE_RUN_DIR(year, scenario, seed=seed, root=root)
    align_runtime_icdat(run_dir / "input" / base.sy.MZX_NAME, year)
    return run_dir


def reset_reported_application_tables(text: str, treatment: int, safe_date: int, year: int) -> str:
    """Keep one valid zero placeholder per table; external actions supply actual management."""
    output: list[str] = []
    skipping: str | None = None
    irrigation_header_seen = False
    fertilizer_header_seen = False
    for line in text.splitlines():
        if line.startswith("@I IDATE"):
            output.append(line)
            output.append(f" 1 {safe_date:05d} IR001     0")
            skipping = "irrigation"
            irrigation_header_seen = True
            continue
        if line.startswith("@F FDATE"):
            output.append(line)
            output.append(
                f" 1 {safe_date:05d} FE005 AP002     0     0   -99   -99   -99   -99   -99 runtime_zero_{year}"
            )
            skipping = "fertilizer"
            fertilizer_header_seen = True
            continue
        if skipping is not None:
            if line.startswith("*") or line.startswith("@"):
                output.append("")
                skipping = None
                output.append(line)
            continue
        output.append(line)
    if not irrigation_header_seen or not fertilizer_header_seen:
        raise ValueError("Could not safely redraw reported irrigation/fertilizer tables")
    return "\n".join(output) + "\n"


def prepare_linked_run_aligned(year: int, scenario: str, seed: int, root: Path) -> tuple[Path, dict[str, Any]]:
    preparation_scenario = scenario if scenario.startswith(("dqn", "transfer")) else f"transfer_{scenario}"
    run_dir = aligned_prepare_run_dir(year, preparation_scenario, seed=seed, root=root)
    filex = run_dir / "input" / base.sy.MZX_NAME
    treatment, ic_pointer = base.EXPECTED_TREATMENTS[year]
    text = filex.read_text(encoding="latin-1", errors="ignore")
    text = set_treatment_pointers(text, treatment, "1", "1")
    text = set_management_for_treatment(text, treatment, "L", "L")
    audit_row = base.treatment_audit().loc[lambda frame: frame["year"].eq(year)].iloc[0]
    safe_date = int(audit_row["sdate"])
    text = reset_reported_application_tables(text, treatment, safe_date, year)
    filex.write_text(text, encoding="latin-1", errors="ignore")
    line = selected_ic_line(text, ic_pointer)
    if int(line.split()[2]) != TARGET_ICDAT[year]:
        raise ValueError(f"{year}: management rewrite overwrote aligned ICDAT")
    manifest_path = run_dir / "icdat_alignment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_mzx_sha256_after_management_rewrite"] = base.sha256(filex)
    manifest["ic_row_after_management_rewrite"] = line
    manifest["alignment_survived_management_rewrite"] = True
    manifest["reported_management_tables_redrawn_for_external_actions"] = True
    manifest["reported_management_level_pointer"] = 1
    manifest["reported_zero_placeholder_date"] = safe_date
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    env_args_path = run_dir / "env_args.json"
    env_args = json.loads(env_args_path.read_text(encoding="utf-8"))
    env_args["fileX_template_path"] = str(filex)
    env_args_path.write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def install_runtime_patch(output: Path) -> None:
    base.OUT = output
    base.sy.prepare_run_dir = aligned_prepare_run_dir
    base.prepare_linked_run = prepare_linked_run_aligned


def verify_original_unchanged() -> None:
    current = base.sha256(base.MZX)
    if current != ORIGINAL_MZX_SHA256:
        raise RuntimeError("Authoritative CNSY1201.MZX changed during 026_07")


def runtime_manifest_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("icdat_alignment_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_path"] = str(path.relative_to(ROOT))
        rows.append(payload)
    return pd.DataFrame(rows)


def all_model_metrics_valid(models: pd.DataFrame) -> bool:
    always_finite = ["final_gwad", "irrigation_total", "fertilizer_total", "WP_ET_kg_m3"]
    if not np.isfinite(models[always_finite].to_numpy(dtype=float)).all():
        return False
    nitrogen = pd.to_numeric(models["fertilizer_total"], errors="coerce")
    pfp = pd.to_numeric(models["PFP_N_kg_kg"], errors="coerce")
    return bool(np.isfinite(pfp[nitrogen > 0].to_numpy(dtype=float)).all() and pfp[nitrogen <= 0].isna().all())


def run_smoke() -> None:
    if SMOKE_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {SMOKE_OUT}")
    install_runtime_patch(SMOKE_OUT)
    base.sy.OUT_DIR = SMOKE_OUT
    base.sy.configure_globals()
    _, _, summary = base.sy.run_zero_action(2015, "null")
    snapshot = ROOT / summary["run_dir"] / "pdi_tmp_snapshot_eval"
    metrics = base.metrics_from_snapshot(
        snapshot, summary["final_gwad"], summary["irrigation_total"], summary["fertilizer_total"]
    )
    manifests = runtime_manifest_rows(SMOKE_OUT)
    verify_original_unchanged()
    checks = {
        "one_runtime_manifest": len(manifests) == 1,
        "runtime_icdat_is_15099": bool(len(manifests) == 1 and int(manifests.iloc[0]["approved_target_icdat"]) == 15099),
        "runtime_copy_changed": bool(len(manifests) == 1 and manifests.iloc[0]["changed"]),
        "original_mzx_hash_unchanged": base.sha256(base.MZX) == ORIGINAL_MZX_SHA256,
        "finite_yield": bool(np.isfinite(float(summary["final_gwad"]))),
        "summary_match": float(metrics["summary_match_score"]) <= 4.0,
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "year": 2015,
        "scenario": "null",
        "checks": checks,
        "summary": summary,
        "metrics": metrics,
        "training_steps": 0,
    }
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    manifests.to_csv(SMOKE_OUT / "026_07_smoke_icdat_alignment_manifest.csv", index=False)
    (SMOKE_OUT / "026_07_smoke_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload["status"] != "passed":
        raise RuntimeError(f"026_07 smoke failed: {checks}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_expert_render_smoke() -> None:
    if EXPERT_SMOKE_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {EXPERT_SMOKE_OUT}")
    install_runtime_patch(EXPERT_SMOKE_OUT)
    case = {"site": "SY", "station": "Shenyang", "year": 2014, "region": "northeast_greatwall_spring_maize"}
    schedule = base.expert.build_region_schedule()
    schedule = schedule[schedule["region"].eq(case["region"])].copy()
    run_dir, env_args = prepare_linked_run_aligned(
        2014, "official_extension_expert", 0, EXPERT_SMOKE_OUT / "runs"
    )
    _, _, summary = base.expert.run_fixed_schedule(case, env_args, schedule, run_dir)
    metrics = base.metrics_from_snapshot(
        run_dir / "pdi_tmp_snapshot_eval",
        summary["final_gwad"],
        float(summary["action_irrigation_total"]),
        float(summary["action_fertilizer_total"]),
    )
    filex = run_dir / "input" / base.sy.MZX_NAME
    text = filex.read_text(encoding="latin-1", errors="ignore")
    invalid_irrigation_placeholders = re.findall(r"(?m)^\s*\d+\s+\d{5}\s+-99\s+-99\s*$", text)
    verify_original_unchanged()
    checks = {
        "no_invalid_irrigation_placeholder_rows": len(invalid_irrigation_placeholders) == 0,
        "finite_yield": bool(np.isfinite(float(summary["final_gwad"]))),
        "summary_match": float(metrics["summary_match_score"]) <= 4.0,
        "original_mzx_hash_unchanged": base.sha256(base.MZX) == ORIGINAL_MZX_SHA256,
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "year": 2014,
        "scenario": "official_extension_expert",
        "checks": checks,
        "summary": summary,
        "metrics": metrics,
        "training_steps": 0,
    }
    (EXPERT_SMOKE_OUT / "026_07_expert_render_smoke_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload["status"] != "passed":
        raise RuntimeError(f"026_07 expert render smoke failed: {checks}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    missing = [str(path) for path in [base.MZX, base.SCALER, *base.MODELS.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    OUT.mkdir(parents=True)
    install_runtime_patch(OUT)
    original_audit = base.treatment_audit()
    original_audit["approved_aligned_icdat"] = original_audit["year"].map(TARGET_ICDAT)
    original_audit["approved_runtime_date_chain"] = (
        original_audit["sdate"].le(original_audit["approved_aligned_icdat"])
        & original_audit["approved_aligned_icdat"].le(original_audit["pdate"])
        & original_audit["year"].mod(100).eq((original_audit["approved_aligned_icdat"] // 1000).astype(int))
    )
    original_audit.to_csv(OUT / "026_07_sy_original_and_approved_date_audit.csv", index=False)
    registered = original_audit[original_audit["registered_authoritative_year"]]
    audit_pass = (
        set(registered["year"].astype(int)) == set(YEARS)
        and bool(registered[["treatment_matches", "ic_pointer_matches", "weather_exists", "approved_runtime_date_chain"]].all().all())
    )
    if not audit_pass:
        raise ValueError("Approved runtime date-chain audit did not pass")

    model_hashes = {str(seed): base.sha256(path) for seed, path in base.MODELS.items()}
    all_baselines: list[pd.DataFrame] = []
    all_models: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    year_payloads: dict[str, Any] = {}

    for year in YEARS:
        baseline_frame = pd.DataFrame(base.run_baselines(year))
        expected = {"null", "recorded", "dssat_auto", "official_extension_expert"}
        if set(baseline_frame["scenario"]) != expected:
            raise ValueError(f"{year}: four-baseline set is incomplete")
        targets = base.local_targets(baseline_frame)
        model_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        for seed, model_path in base.MODELS.items():
            result, actions = base.run_frozen_model(year, seed, model_path, targets)
            model_rows.append(result)
            action_rows.extend(actions)
        model_frame = pd.DataFrame(model_rows)
        action_frame = pd.DataFrame(action_rows)
        baseline_frame.to_csv(OUT / f"026_07_sy{year}_four_baselines.csv", index=False)
        model_frame.to_csv(OUT / f"026_07_sy{year}_frozen_ppo_summary.csv", index=False)
        action_frame.to_csv(OUT / f"026_07_sy{year}_frozen_ppo_stage_actions.csv", index=False)
        all_baselines.append(baseline_frame)
        all_models.append(model_frame)
        all_actions.append(action_frame)
        year_payloads[str(year)] = {
            "source": "fresh_026_07_after_approved_icdat_alignment",
            "targets": targets,
            "pass_count": int(model_frame["local_primary_pass"].sum()),
        }

    baselines = pd.concat(all_baselines, ignore_index=True, sort=False)
    models = pd.concat(all_models, ignore_index=True, sort=False)
    actions = pd.concat(all_actions, ignore_index=True, sort=False)
    recorded = baselines.loc[baselines["scenario"].eq("recorded"), ["year", "final_gwad"]].rename(
        columns={"final_gwad": "recorded_yield"}
    )
    models = models.merge(recorded, on="year", how="left")
    models["yield_ge_recorded"] = models["final_gwad"] >= models["recorded_yield"]
    baselines.to_csv(OUT / "026_07_sy_all_years_four_baselines.csv", index=False)
    models.to_csv(OUT / "026_07_sy_all_years_frozen_ppo_summary.csv", index=False)
    actions.to_csv(OUT / "026_07_sy_all_years_frozen_ppo_stage_actions.csv", index=False)
    pd.concat([baselines.assign(group="baseline"), models.assign(group="frozen_ppo")], ignore_index=True, sort=False).to_csv(
        OUT / "026_07_sy_all_years_all_scenarios.csv", index=False
    )
    matrix = models.pivot(index="year", columns="seed", values="local_primary_pass").reset_index()
    matrix.columns = ["year", *[f"seed{int(col)}_local_primary" for col in matrix.columns[1:]]]
    matrix["pass_count"] = matrix.filter(like="_local_primary").sum(axis=1).astype(int)
    matrix["year_transfer_pass"] = matrix["pass_count"] >= 2
    matrix.to_csv(OUT / "026_07_sy_year_seed_pass_matrix.csv", index=False)
    manifests = runtime_manifest_rows(OUT)
    manifests.to_csv(OUT / "026_07_runtime_icdat_alignment_manifests.csv", index=False)
    verify_original_unchanged()

    engineering_checks = {
        "approved_date_audit_pass": audit_pass,
        "all_three_models_each_year": bool(models.groupby("year")["seed"].nunique().eq(3).all()),
        "all_model_hashes_unchanged": all(base.sha256(base.MODELS[int(seed)]) == digest for seed, digest in model_hashes.items()),
        "zero_training_steps": True,
        "zero_invalid_actions": int(pd.to_numeric(models["invalid_action_attempts"], errors="coerce").sum()) == 0,
        "all_model_metrics_valid_with_pfp_undefined_at_zero_n": all_model_metrics_valid(models),
        "all_runtime_manifests_present": len(manifests) == 21,
        "all_runtime_icdat_targets_correct": bool(set(manifests["approved_target_icdat"].astype(int)) == set(TARGET_ICDAT.values())),
        "original_mzx_hash_unchanged": base.sha256(base.MZX) == ORIGINAL_MZX_SHA256,
    }
    engineering_pass = all(engineering_checks.values())
    all_years_pass = bool(matrix["year_transfer_pass"].all())
    branch = (
        "A_SY_all_years_transfer_after_icdat_alignment"
        if engineering_pass and all_years_pass
        else ("B_partial_year_transfer_after_icdat_alignment" if engineering_pass else "C_input_or_execution_blocked")
    )
    payload = {
        "status": "completed",
        "branch": branch,
        "eligible_years": list(YEARS),
        "approved_icdat_alignment": TARGET_ICDAT,
        "original_mzx_sha256": ORIGINAL_MZX_SHA256,
        "model_hashes": model_hashes,
        "year_results": year_payloads,
        "engineering_checks": engineering_checks,
        "all_years_at_least_two_of_three": all_years_pass,
        "training_steps": 0,
        "scientific_success_claimed": False,
        "recorded_is_separate_comparator": True,
        "next_step_allowed": bool(engineering_pass),
    }
    (OUT / "026_07_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def finalize_existing() -> None:
    result_path = OUT / "026_07_result.json"
    models_path = OUT / "026_07_sy_all_years_frozen_ppo_summary.csv"
    matrix_path = OUT / "026_07_sy_year_seed_pass_matrix.csv"
    if not result_path.exists() or not models_path.exists() or not matrix_path.exists():
        raise FileNotFoundError("026_07 attempt2 completed outputs are missing")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    before_path = OUT / "026_07_result_before_pfp_engineering_fix.json"
    if not before_path.exists():
        before_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    models = pd.read_csv(models_path)
    matrix = pd.read_csv(matrix_path)
    manifests = runtime_manifest_rows(OUT)
    corrected_check = all_model_metrics_valid(models)
    payload["engineering_checks"].pop("all_model_metrics_finite", None)
    payload["engineering_checks"]["all_model_metrics_valid_with_pfp_undefined_at_zero_n"] = corrected_check
    engineering_pass = all(bool(value) for value in payload["engineering_checks"].values())
    all_years_pass = bool(matrix["year_transfer_pass"].astype(bool).all())
    payload["branch"] = (
        "A_SY_all_years_transfer_after_icdat_alignment"
        if engineering_pass and all_years_pass
        else ("B_partial_year_transfer_after_icdat_alignment" if engineering_pass else "C_input_or_execution_blocked")
    )
    payload["all_years_at_least_two_of_three"] = all_years_pass
    payload["next_step_allowed"] = engineering_pass
    payload["post_run_aggregation_correction"] = {
        "reason": "PFP_N is mathematically undefined for zero-N strategies and must not be required to be finite.",
        "dssat_rerun": False,
        "affected_scientific_values": False,
        "runtime_manifest_count": len(manifests),
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--expert-render-smoke-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.smoke_only, args.expert_render_smoke_only, args.finalize_existing)) > 1:
        raise ValueError("Choose only one special execution mode")
    if args.smoke_only:
        run_smoke()
    elif args.expert_render_smoke_only:
        run_expert_render_smoke()
    elif args.finalize_existing:
        finalize_existing()
    else:
        main()

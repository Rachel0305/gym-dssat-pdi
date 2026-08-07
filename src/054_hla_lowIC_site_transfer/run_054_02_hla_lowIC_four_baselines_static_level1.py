"""054_02: HLA lowIC four-baseline rebuild with static level-1 correction.

This intentionally has one input-profile switch shared with 046_02.  It does
not import the lowIC-specific 040_21 wrapper, so the recorded input family is
not silently relabelled as lowIC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_static_level1_four_baseline_rebuild_037_07 as baseline_audit


DEFAULT_CONFIG = ROOT / "configs" / "054_02_hla_lowIC_four_baselines_static_level1.json"
PROMPT = ROOT / "prompts" / "054_hla_lowIC_site_transfer_expanded_action_ppo_auto_static_level1_figures.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto", "official_extension_expert"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if str(cfg.get("station_code")) != "HLA":
        raise ValueError("054_02 is registered for station_code=HLA.")
    if str(cfg.get("site")) != "HLA":
        raise ValueError("054_02 is registered for DSSAT site=HLA.")
    if str(cfg.get("input_profile")) not in INPUT_PROFILES:
        raise ValueError("input_profile must be originIC or lowIC.")
    if not cfg.get("scope", {}).get("validation_years"):
        raise ValueError("config missing scope.validation_years.")
    return cfg


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def output_root(cfg: dict[str, Any], run_id: str = "") -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}{run_suffix(run_id)}"


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    mzx = input_root / "HL" / "CNHL0701_corrected_IC123.MZX"
    schedule = recorded_template_schedules_dedup().get("HLA", {})
    issues = []
    if not input_root.exists():
        issues.append("input root missing")
    if not mzx.exists():
        issues.append("CNHL0701_corrected_IC123.MZX missing")
    return {
        "station_code": "HLA",
        "input_profile": str(cfg["input_profile"]),
        "resolved_input_root": rel(input_root),
        "source_mzx": rel(mzx),
        "source_mzx_sha256": sha256(mzx) if mzx.exists() else "",
        "years": list(map(int, cfg["scope"]["validation_years"])),
        "recorded_template_source": rel(baseline.RECORDED_TEMPLATE_DAILY),
        "recorded_template_total_irrigation_mm": float(sum(float(v.get("amir", 0.0)) for v in schedule.values())),
        "recorded_template_total_nitrogen_kg_ha": float(sum(float(v.get("anfer", 0.0)) for v in schedule.values())),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def recorded_template_schedules_dedup() -> dict[str, dict[int, dict[str, float]]]:
    """Load the historical recorded-farmer template without double-counting.

    The old 027_05 daily table stores the same recorded HLA2014 line under both
    DQN and MaskablePPO report rows.  For a static recorded template we need the
    unique agronomic events, not the report duplication.
    """

    path = baseline.RECORDED_TEMPLATE_DAILY
    if not path.exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    if "scenario" in df.columns:
        df = df[df["scenario"].astype(str).eq("recorded_farmer")].copy()
    required = ["site", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"recorded template source missing columns: {missing}")
    for col in ["dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df[
        df["irrigation_executed_mm"].abs().gt(1e-9)
        | df["nitrogen_executed_kg_ha"].abs().gt(1e-9)
    ].copy()
    df["dap_key"] = df["dap"].round().astype(int).clip(lower=1)
    df = df.drop_duplicates(
        subset=["site", "dap_key", "irrigation_executed_mm", "nitrogen_executed_kg_ha"],
        keep="first",
    )
    out: dict[str, dict[int, dict[str, float]]] = {}
    for site, group in df.groupby("site", sort=True):
        schedule: dict[int, dict[str, float]] = {}
        for row in group.itertuples(index=False):
            dap = int(getattr(row, "dap_key"))
            schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
            schedule[dap]["amir"] += float(getattr(row, "irrigation_executed_mm"))
            schedule[dap]["anfer"] += float(getattr(row, "nitrogen_executed_kg_ha"))
        out[str(site)] = schedule
    return out


def selected_rows(years: list[int]) -> pd.DataFrame:
    split = pd.read_csv(baseline.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    out = split[split["station_code"].astype(str).eq("HLA") & split["year"].isin(years)].copy()
    if sorted(out["year"].tolist()) != sorted(years):
        raise RuntimeError("Configured validation years are missing from the split registry.")
    return out.sort_values("year").reset_index(drop=True)


def build_configs(out: Path, selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(baseline.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = 0
    run_config["paths"]["output_root"] = rel(out)
    run_config["runtime"]["max_steps"] = 260
    env_config = baseline.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = rel(out)
    env_config["runtime"]["max_steps"] = 260
    env_config["seed"] = 0
    return run_config, env_config


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_record(out: Path, cfg: dict[str, Any], pf: dict[str, Any], summary: pd.DataFrame, manifest: pd.DataFrame, audit: pd.DataFrame, elapsed: float) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    by_scenario = pd.DataFrame()
    if not summary.empty:
        by_scenario = summary.groupby("scenario", as_index=False).agg(
            years=("year", "nunique"),
            mean_grain_yield_kg_ha=("grain_yield_kg_ha", "mean"),
            mean_biomass_kg_ha=("biomass_kg_ha", "mean"),
            mean_irrigation_mm=("actual_irrigation_mm", "mean"),
            mean_nitrogen_kg_ha=("actual_nitrogen_kg_ha", "mean"),
            mean_WP_ET_kg_m3=("WP_ET_kg_m3", "mean"),
            mean_PFP_N_kg_kg=("PFP_N_kg_kg", "mean"),
        )
    non_ok = manifest[manifest["status"].ne("ok")].copy() if not manifest.empty else pd.DataFrame()
    audit_preview_cols = [
        col for col in [
            "year",
            "scenario",
            "planned_i_events",
            "planned_i_total",
            "mgmt_i_events",
            "mgmt_i_total",
            "planned_n_events",
            "planned_n_total",
            "mgmt_n_events",
            "mgmt_n_total",
            "status",
            "issues",
            "planned_vs_inp_notes",
        ] if col in audit.columns
    ]
    lines = [
        f"# {cfg['task_id']} HLA {cfg['input_profile']} four-baseline static level-1 record",
        "",
        f"- Input root: `{pf['resolved_input_root']}`.",
        f"- Source MZX: `{pf['source_mzx']}`.",
        f"- Source MZX SHA256: `{pf['source_mzx_sha256']}`.",
        f"- Years: `{pf['years']}`.",
        f"- Elapsed: `{elapsed:.1f}` s.",
        "- Static recorded/expert rows were rendered with DSSAT management level ID `1` using the 037_07 correction.",
        "- Recorded farmer template remains frozen/static; it is not tuned for PPO and does not represent each year's true farmer management.",
        f"- Recorded template totals: `{pf['recorded_template_total_irrigation_mm']}` mm irrigation / `{pf['recorded_template_total_nitrogen_kg_ha']}` kg/ha nitrogen.",
        f"- Summary CSV: `{rel(out / 'evaluation' / '054_02_baseline_summary.csv')}`.",
        f"- Event audit CSV: `{rel(out / 'evaluation' / '054_02_management_event_audit.csv')}`.",
        "",
        "## Scenario means",
        "",
        markdown_table(by_scenario),
        "",
        "## Non-ok manifest rows",
        "",
        markdown_table(non_ok),
        "",
        "## Event-chain audit preview",
        "",
        markdown_table(audit[audit_preview_cols] if audit_preview_cols else audit),
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc


def run(cfg_path: Path, dry_run: bool, years_override: list[int] | None = None, run_id: str = "") -> dict[str, Any]:
    cfg, pf = read_config(cfg_path), None
    if years_override:
        cfg = json.loads(json.dumps(cfg))
        cfg["scope"]["validation_years"] = years_override
    pf = preflight(cfg)
    out = output_root(cfg, run_id)
    if dry_run:
        return {"task": "054_02_hla_static_level1_four_baselines", "mode": "dry_run", **pf, "output_root": rel(out)}
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))

    for sub in ["configs", "evaluation", "snapshots", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    selected = selected_rows(pf["years"])
    selected.to_csv(out / "configs" / "054_02_selected_years.csv", index=False, encoding="utf-8-sig")
    run_config, env_config = build_configs(out, selected)
    direct_ppo.write_yaml(env_config, out / "configs" / "054_02_resolved_env_config.yaml")

    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    originals = (
        baseline.MULTISITE_INPUT_ROOT,
        baseline.OUT,
        baseline.TASK_ID,
        baseline.MAX_STEPS,
        ppo_safe_rendering.MULTISITE_INPUT_ROOT,
        baseline.replace_static_application_rows,
    )
    baseline.MULTISITE_INPUT_ROOT = input_root
    baseline.OUT = out
    baseline.TASK_ID = "054_02"
    baseline.MAX_STEPS = 260
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
    baseline.replace_static_application_rows = baseline_audit.replace_static_application_rows_level1  # type: ignore[assignment]
    schedules = recorded_template_schedules_dedup()
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for row in selected.itertuples(index=False):
            for scenario in SCENARIOS:
                print(f"[054_02] HLA{int(row.year)} {scenario}", flush=True)
                try:
                    daily, summary, checks = baseline.evaluate_scenario(run_config, env_config, "HLA", int(row.year), scenario, schedules)
                    summary["season_last_dap"] = int(pd.to_numeric(daily["dap"], errors="coerce").max())
                    summary["source_input_profile"] = cfg["input_profile"]
                    summary["source_status"] = "generated_054_02_static_level1_corrected_baseline"
                    summary["render_check_count"] = len(checks)
                    daily["source_input_profile"] = cfg["input_profile"]
                    daily_frames.append(daily)
                    summary_rows.append(summary)
                    schedule = {} if scenario == "dssat_auto" else baseline.schedule_for(scenario, "HLA", schedules)
                    audit = baseline_audit.audit_row(summary, schedule)
                    audit_rows.append(audit)
                    manifest_rows.append({"year": int(row.year), "scenario": scenario, "status": audit.get("status", "ok"), "details": audit.get("issues", "")})
                except Exception as exc:
                    manifest_rows.append({"year": int(row.year), "scenario": scenario, "status": "failed", "details": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
    finally:
        (
            baseline.MULTISITE_INPUT_ROOT,
            baseline.OUT,
            baseline.TASK_ID,
            baseline.MAX_STEPS,
            ppo_safe_rendering.MULTISITE_INPUT_ROOT,
            baseline.replace_static_application_rows,
        ) = originals

    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    audit = pd.DataFrame(audit_rows)
    summary.to_csv(out / "evaluation" / "054_02_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "evaluation" / "054_02_baseline_daily.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(out / "evaluation" / "054_02_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "evaluation" / "054_02_management_event_audit.csv", index=False, encoding="utf-8-sig")
    (out / "054_02_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False), encoding="utf-8")
    record = write_record(out, cfg, pf, summary, manifest, audit, time.time() - started)
    result = {"task": "054_02_hla_static_level1_four_baselines", "output_root": rel(out), "record_md": rel(record), "successful_runs": int(len(summary)), "non_ok_runs": int((manifest["status"] != "ok").sum()) if not manifest.empty else 0}
    (out / "054_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", type=str, default="", help="鍙€?smoke 骞翠唤锛屼緥濡?2014锛涚┖鍊艰〃绀哄叏閮ㄩ獙璇佸勾")
    parser.add_argument("--run-id", type=str, default="", help="Optional run batch id; pass the same timestamp to avoid overwriting earlier outputs")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or None
    print(json.dumps(run(path, args.dry_run, years, args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





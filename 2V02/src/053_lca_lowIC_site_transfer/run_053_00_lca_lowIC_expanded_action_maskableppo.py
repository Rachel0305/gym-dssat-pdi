"""053_00: LCA lowIC expanded-action MaskablePPO site transfer.

This runner keeps the SYA 046_10 PPO method fixed and changes only station and
input years. It is intentionally conservative: formal 100K training requires a
completed matching 2K smoke run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "053_00_lca_lowIC_expanded_action_maskableppo.json"
PROMPT = ROOT / "prompts" / "053_lca_lowIC_site_transfer_expanded_action_ppo_auto_static_level1_figures.md"
SMOKE_TIMESTEPS = 2_000
SMOKE_CHECKPOINTS = [1_000, 2_000]
EXPECTED_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
INPUT_PROFILES = base04602.INPUT_PROFILES


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("station_code") != "LCA" or cfg.get("site") != "LC":
        raise ValueError("053_00 is registered for station_code=LCA and site=LC.")
    if cfg.get("input_profile") not in INPUT_PROFILES:
        raise ValueError(f"input_profile must be one of {list(INPUT_PROFILES)}.")
    actions = cfg.get("actions", {})
    irrigation = list(map(float, actions.get("irrigation_levels_mm", [])))
    nitrogen = list(map(float, actions.get("nitrogen_levels_kg_ha", [])))
    if irrigation != EXPECTED_IRRIGATION_LEVELS or nitrogen != EXPECTED_NITROGEN_LEVELS:
        raise ValueError("053_00 keeps the 046_10 action grid: I [0,15,30,45], N [0,40,80,120].")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation":
        raise ValueError("053_00 must inherit the 046_02 raw observation contract.")
    if bool(obs.get("normalization_enabled", True)) or bool(obs.get("weather_forecast_enabled", True)):
        raise ValueError("053_00 forbids observation normalization and weather forecast features.")
    if cfg.get("reference_run") != "046_10_sya_originIC_expanded_action_maskableppo":
        raise ValueError("053_00 must reference the frozen SYA 046_10 method.")
    return cfg


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def task_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg["task_id"])


def prepare_config(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    planned = json.loads(json.dumps(cfg))
    if smoke:
        planned["task_name"] = f"{planned['task_name']}_smoke2k"
        planned["training"] = {"total_timesteps": SMOKE_TIMESTEPS, "checkpoint_steps": SMOKE_CHECKPOINTS}
    return planned


def split_for_station(station: str) -> pd.DataFrame:
    split = engine.base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split[split["station_code"].astype(str).eq(station)].sort_values("year").reset_index(drop=True)


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    station = str(cfg["station_code"])
    site = str(cfg["site"])
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    source_mzx = input_root / "LC" / "CNLC0801.MZX"
    split = split_for_station(station)
    train_actual = sorted(split.loc[split["split"].eq("train"), "year"].astype(int).tolist())
    validation_actual = sorted(split.loc[split["split"].eq("validation"), "year"].astype(int).tolist())
    expected_train = list(map(int, cfg["scope"]["train_years"]))
    expected_validation = list(map(int, cfg["scope"]["validation_years"]))
    issues: list[str] = []
    if not input_root.exists():
        issues.append(f"input root missing: {rel(input_root)}")
    if not source_mzx.exists():
        issues.append(f"LCA source MZX missing: {rel(source_mzx)}")
    if train_actual != expected_train:
        issues.append(f"train years mismatch: config={expected_train}, engine={train_actual}")
    if validation_actual != expected_validation:
        issues.append(f"validation years mismatch: config={expected_validation}, engine={validation_actual}")
    if not PROMPT.exists():
        issues.append(f"prompt missing: {rel(PROMPT)}")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "station_code": station,
        "site": site,
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(input_root),
        "source_mzx": rel(source_mzx),
        "config_train_years": expected_train,
        "config_validation_years": expected_validation,
        "engine_train_years": train_actual,
        "engine_validation_years": validation_actual,
        "reference_run": cfg["reference_run"],
        "observation_contract": cfg["observation_contract"],
        "irrigation_levels_mm": cfg["actions"]["irrigation_levels_mm"],
        "nitrogen_levels_kg_ha": cfg["actions"]["nitrogen_levels_kg_ha"],
        "combined_action_count": len(cfg["actions"]["irrigation_levels_mm"]) * len(cfg["actions"]["nitrogen_levels_kg_ha"]),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def write_manifest(out: Path, cfg_path: Path, cfg: dict[str, Any], pf: dict[str, Any], status: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(cfg_path, config_dir / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, config_dir / PROMPT.name)
    path = out / f"{task_prefix(cfg)}_run_manifest.json"
    path.write_text(json.dumps({"status": status, "config": cfg, "preflight": pf}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def copy_clean_names(out: Path, prefix: str) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": f"evaluation/{prefix}_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": f"evaluation/{prefix}_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": f"evaluation/{prefix}_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": f"{prefix}_engine_result.json",
    }
    copied: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[target_rel] = source_rel
    return copied


def _on_grid(values: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return values.map(lambda value: bool(np.isclose(float(value), allowed, atol=1e-6).any()))


def audit_actions(out: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = task_prefix(cfg)
    summary_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for record in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(record, "daily_csv_path"))
        if not daily_path.exists():
            rows.append({"checkpoint_step": int(record.checkpoint_step), "year": int(record.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        irrigation = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
        previous_i = pd.to_numeric(daily.get("previous_cumulative_irrigation"), errors="coerce")
        previous_n = pd.to_numeric(daily.get("previous_cumulative_n"), errors="coerce")
        cumulative_i = pd.to_numeric(daily.get("season_cumulative_irrigation"), errors="coerce")
        cumulative_n = pd.to_numeric(daily.get("season_cumulative_n"), errors="coerce")
        positive = (irrigation > 1e-9) | (nitrogen > 1e-9)
        delta_i = cumulative_i - previous_i
        delta_n = cumulative_n - previous_n
        transmitted = (~positive) | (np.isclose(delta_i, irrigation, atol=1e-6) & np.isclose(delta_n, nitrogen, atol=1e-6))
        nonzero_pairs = {(float(i), float(n)) for i, n in zip(irrigation[positive], nitrogen[positive])}
        novel = irrigation.isin([15.0, 30.0]) | nitrogen.isin([40.0, 120.0])
        daps = pd.to_numeric(daily.get("dap"), errors="coerce")
        rows.append(
            {
                "checkpoint_step": int(record.checkpoint_step),
                "year": int(record.year),
                "daily_exists": True,
                "row_count": int(len(daily)),
                "off_grid_irrigation_rows": int((~_on_grid(irrigation, EXPECTED_IRRIGATION_LEVELS)).sum()),
                "off_grid_nitrogen_rows": int((~_on_grid(nitrogen, EXPECTED_NITROGEN_LEVELS)).sum()),
                "positive_action_rows": int(positive.sum()),
                "positive_action_rows_after_dap1": int((positive & (daps > 1)).sum()),
                "transmission_mismatch_rows": int((positive & ~transmitted).sum()),
                "novel_level_event_rows": int((positive & novel).sum()),
                "unique_nonzero_action_pairs": len(nonzero_pairs),
                "nonzero_action_pairs": "; ".join(f"I{i:g}/N{n:g}" for i, n in sorted(nonzero_pairs)),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final_checkpoint = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_checkpoint)].copy()
    total_pairs: set[str] = set()
    for text in final.get("nonzero_action_pairs", pd.Series(dtype=str)).fillna(""):
        total_pairs.update(piece for piece in str(text).split("; ") if piece)
    gate = {
        "final_checkpoint": final_checkpoint,
        "expected_validation_rows": len(cfg["scope"]["validation_years"]),
        "observed_validation_rows": int(len(final)),
        "all_daily_files_exist": bool(len(final) and final["daily_exists"].all()),
        "all_actions_on_declared_grid": bool(len(final) and final["off_grid_irrigation_rows"].sum() == 0 and final["off_grid_nitrogen_rows"].sum() == 0),
        "positive_actions_present": bool(len(final) and final["positive_action_rows"].sum() > 0),
        "positive_actions_transmitted": bool(len(final) and final["transmission_mismatch_rows"].sum() == 0),
        "not_all_positive_actions_at_dap1": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "multiple_nonzero_action_pairs": len(total_pairs) >= 2,
        "new_action_levels_used": bool(len(final) and final["novel_level_event_rows"].sum() > 0),
    }
    gate["next_step_allowed"] = bool(
        gate["observed_validation_rows"] == gate["expected_validation_rows"]
        and all(value for key, value in gate.items() if key not in {"final_checkpoint", "expected_validation_rows", "observed_validation_rows", "next_step_allowed"})
    )
    return audit, gate


def verify_completed_smoke(base_cfg: dict[str, Any]) -> dict[str, Any]:
    smoke_cfg = prepare_config(base_cfg, smoke=True)
    smoke_out = output_root(smoke_cfg)
    result_path = smoke_out / f"{task_prefix(smoke_cfg)}_smoke_result.json"
    manifest_path = smoke_out / f"{task_prefix(smoke_cfg)}_run_manifest.json"
    checks: dict[str, Any] = {
        "smoke_result_exists": result_path.exists(),
        "smoke_manifest_exists": manifest_path.exists(),
    }
    if not all(checks.values()):
        checks["next_step_allowed"] = False
        return checks
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke_config = manifest.get("config", {})
    gate = result.get("smoke_gate", {})
    checks.update(
        {
            "smoke_status_completed": manifest.get("status") == "completed_smoke",
            "smoke_gate_passed": gate.get("next_step_allowed") is True,
            "smoke_timesteps_match": smoke_config.get("training", {}).get("total_timesteps") == SMOKE_TIMESTEPS,
            "smoke_checkpoints_match": smoke_config.get("training", {}).get("checkpoint_steps") == SMOKE_CHECKPOINTS,
            "smoke_input_profile_match": smoke_config.get("input_profile") == base_cfg.get("input_profile"),
            "smoke_station_match": smoke_config.get("station_code") == base_cfg.get("station_code"),
            "smoke_actions_match": smoke_config.get("actions") == base_cfg.get("actions"),
            "smoke_observation_contract_match": smoke_config.get("observation_contract") == base_cfg.get("observation_contract"),
            "smoke_output_root": rel(smoke_out),
        }
    )
    checks["next_step_allowed"] = all(value for key, value in checks.items() if key not in {"smoke_output_root", "next_step_allowed"})
    return checks


def write_record(
    cfg: dict[str, Any],
    pf: dict[str, Any],
    copied: dict[str, str],
    audit: pd.DataFrame,
    gate: dict[str, Any],
    phase: str,
    smoke_verification: dict[str, Any] | None = None,
) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} LCA/LC lowIC expanded-action MaskablePPO record",
        "",
        "## Design",
        "",
        f"- Reference method: `{cfg['reference_run']}`.",
        "- Controlled change: station/input years only; PPO observation, reward, safety, seed, and action grid are inherited.",
        f"- Station/site: `{cfg['station_code']}` / `{cfg['site']}`.",
        f"- Input root: `{pf['resolved_input_root']}`.",
        f"- Train years: `{cfg['scope']['train_years']}`.",
        f"- Validation years: `{cfg['scope']['validation_years']}`.",
        f"- Training: `{cfg['training']['total_timesteps']}` steps, checkpoints `{cfg['training']['checkpoint_steps']}`.",
        "",
        f"## {phase} Action Audit",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: `{value}`")
    if smoke_verification is not None:
        lines.extend(["", "## Formal Smoke Verification", ""])
        for key, value in smoke_verification.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Year-Level Action Audit", "", audit.to_markdown(index=False), "", "## Normalized Outputs", ""])
    for target, source in copied.items():
        lines.append(f"- `{target}` <- `{source}`")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run(cfg_path: Path, dry_run: bool, smoke: bool, formal: bool) -> dict[str, Any]:
    base_cfg = read_config(cfg_path)
    cfg = prepare_config(base_cfg, smoke)
    pf = preflight(cfg)
    out = output_root(cfg)
    smoke_verification: dict[str, Any] | None = None
    if formal:
        smoke_verification = verify_completed_smoke(base_cfg)
        if not smoke_verification["next_step_allowed"]:
            pf["issues"].append("053_00 completed smoke verification did not pass")
            pf["next_step_allowed"] = False
    if dry_run:
        result = {**pf, "mode": "dry_run", "output_root": rel(out)}
        if smoke_verification is not None:
            result["smoke_verification"] = smoke_verification
        return result
    if not smoke and not formal:
        raise ValueError("Use --smoke for 2K validation or --formal for authorized 100K training.")
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {rel(out)}")

    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_site_names = dict(engine.base03222.SITE_NAMES)
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [str(cfg["station_code"])]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.base03222.SITE_NAMES[str(cfg["station_code"])] = str(cfg["site"])
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
        engine.base03222.SITE_NAMES.clear()
        engine.base03222.SITE_NAMES.update(old_site_names)

    copied = copy_clean_names(out, task_prefix(cfg))
    audit, gate = audit_actions(out, cfg)
    phase = "smoke" if smoke else "formal"
    audit_path = out / "audits" / f"{task_prefix(cfg)}_{phase}_action_audit.csv"
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = write_manifest(out, cfg_path, cfg, pf, f"completed_{phase}")
    record = write_record(cfg, pf, copied, audit, gate, "2K Smoke" if smoke else "100K Formal", smoke_verification)
    result = {
        **pf,
        "mode": f"{phase}_train_and_validate",
        "output_root": rel(out),
        "manifest": rel(manifest),
        "record_md": rel(record),
        "action_audit_path": rel(audit_path),
        "action_gate": gate,
    }
    if smoke:
        result["smoke_gate"] = gate
        result["next_step_allowed"] = gate["next_step_allowed"]
        result_path = out / f"{task_prefix(cfg)}_smoke_result.json"
    else:
        result["smoke_verification"] = smoke_verification
        result_path = out / f"{task_prefix(cfg)}_formal_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(cfg_path, args.dry_run, args.smoke, args.formal), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




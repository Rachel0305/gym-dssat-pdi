"""102_00 isolated FQ ncost=2.5 + entropy=0.02 seed=1 2K smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as base

BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
CONFIG = ROOT / "configs/102_00_fqa_originIC_ncost25_entropy002_smoke2k.json"
PROMPT = ROOT / "prompts/102_00_fq_originIC_ncost25_entropy002_smoke2k.md"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not INPUT_ROOT.exists() or not (INPUT_ROOT / "FQ/CNFQ0801.MZX").exists():
        issues.append("originIC FQ input/MZX missing")
    ppo = dict(cfg["fixed_ppo"])
    ppo.update(cfg.get("ppo_override", {}))
    if cfg.get("ppo_override") != {"ent_coef": 0.02}:
        issues.append("FQ intervention must change ent_coef from 0.01 to 0.02")
    if cfg.get("reward_override") != {"nitrogen_cost": 2.5, "reward_type": "fqa_low_n_project_cost_single_factor"}:
        issues.append("FQ reward override must remain the registered ncost=2.5 branch")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    return {
        "task_id": cfg["task_id"],
        "station_code": "FQA",
        "site": "FQ",
        "seed": int(cfg["seed"]),
        "input_root": rel(INPUT_ROOT),
        "effective_ppo": ppo,
        "reward_override": cfg.get("reward_override", {}),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    pf = preflight(cfg)
    out = ROOT / cfg["output_root"]
    if args.dry_run:
        print(json.dumps({**pf, "mode": "dry_run", "smoke_only": True, "output_root": rel(out), "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return 0
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml = ROOT / "configs/102_00_effective_fqa_ncost25_entropy002_ppo_config.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(cfg["seed"])
    yaml_cfg["ppo"].update(pf["effective_ppo"])
    yaml_cfg["reward"].update(cfg["reward_override"])
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")

    old = base.state_snapshot()
    old_seed = batch.SEED
    start = time.perf_counter()
    rss0 = psutil.Process().memory_info().rss
    doc = ROOT / "docs/102_00_fqa_originIC_ncost25_entropy002_smoke2k_record.md"
    try:
        engine.TASK_ID = cfg["task_id"]
        engine.TASK_NAME = cfg["task_name"]
        engine.BASE_OUT = out
        engine.BASE_DOC = doc
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        engine.STATION = "FQA"
        engine.SITES = ["FQA"]
        engine.BINARY_IRRIGATION_LEVELS = list(LEVELS_I)
        engine.BINARY_NITROGEN_LEVELS = list(LEVELS_N)
        batch.SEED = int(cfg["seed"])
        batch.CONFIG = effective_yaml
        batch.OUT = out
        batch.DOC = doc
        batch.SITE_NAMES["FQA"] = "FQ"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        engine.run_training(2000, [1000, 2000], suffix="")
    finally:
        base.restore_state(old)
        batch.SEED = old_seed

    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    inventory_path = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not summary_path.exists() or not inventory_path.exists():
        raise FileNotFoundError("training did not produce summary/inventory")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    summary.to_csv(out / "evaluation/102_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "evaluation/102_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    audit, gate = base.audit_25k(summary, out, 2000)
    shutil.copy2(out / "audits/096_00_action_audit.csv", out / "audits/102_00_action_audit.csv")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(2000)].iloc[0]
    observed = base.observed_kwargs(ROOT / str(row["model_path"]))
    metrics = final[[c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict()
    result = {
        **pf,
        "mode": "completed_smoke",
        "output_root": rel(out),
        "effective_yaml": rel(effective_yaml),
        "effective_kwargs_from_loaded_model": observed,
        "validation_metrics_mean": metrics,
        "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
        "action_gate": gate,
        "elapsed_s": round(time.perf_counter() - start, 3),
        "rss_delta_bytes": psutil.Process().memory_info().rss - rss0,
        "next_step_allowed": bool(gate.get("mechanism_pass")),
        "next_step_note": "Only if true may a separate corrected-2018 5K/10K replay be proposed; no automatic long training.",
    }
    (out / "102_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "102_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# 102_00 FQ ncost=2.5 + entropy=0.02 seed=1 2K smoke", "", "- no external-N; no weather forecast; no full replay", "", "## Gate", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "- WP_ET 未从 2K training summary 推断。", "- 仅 gate 通过才允许另行设计 5K/10K replay。", ""])
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

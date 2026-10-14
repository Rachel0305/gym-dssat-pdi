"""099_00: independent seed=1 2K smoke for the FQ ncost=2.5 branch."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as base


CONFIG = ROOT / "configs/099_00_fqa_originIC_reward_ncost25_seed1_smoke2k.json"
PROMPT = ROOT / "prompts/099_00_fqa_originIC_reward_ncost25_seed1_smoke2k.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def patch_training_config() -> tuple[Path, dict[str, Any]]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["ppo"].update(cfg["fixed_ppo"])
    effective["reward"].update(cfg["reward_override"])
    effective_path = ROOT / "configs/099_00_effective_reward_ncost25_seed1_ppo_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    return effective_path, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "station": "FQA", "seed": 1, "training_steps": 2000, "checkpoint_steps": [1000, 2000], "reward_override": cfg["reward_override"], "external_n_allowed": False, "weather_forecast_allowed": False, "full_replay_allowed": False, "input_root_exists": INPUT_ROOT.exists(), "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "configs").mkdir(parents=True, exist_ok=False)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml, cfg = patch_training_config()
    old = base.state_snapshot()
    old_seed = base.batch.SEED
    try:
        base.engine.TASK_ID = cfg["task_id"]
        base.engine.TASK_NAME = cfg["task_name"]
        base.engine.BASE_OUT = out
        base.engine.BASE_DOC = ROOT / "docs/099_00_fqa_originIC_reward_ncost25_seed1_smoke2k_record.md"
        base.engine.PROMPT = PROMPT
        base.engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        base.engine.STATION = "FQA"
        base.engine.SITES = ["FQA"]
        base.engine.BINARY_IRRIGATION_LEVELS = list(base.LEVELS_I)
        base.engine.BINARY_NITROGEN_LEVELS = list(base.LEVELS_N)
        base.batch.SEED = 1
        base.batch.CONFIG = effective_yaml
        base.batch.OUT = out
        base.batch.DOC = base.engine.BASE_DOC
        base.batch.SITE_NAMES["FQA"] = "FQ"
        base.ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        base.engine.run_training(2000, [1000, 2000], suffix="")
    finally:
        base.restore_state(old)
        base.batch.SEED = old_seed

    source_summary = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    source_inventory = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not source_summary.exists() or not source_inventory.exists():
        raise FileNotFoundError("training did not produce checkpoint summary/inventory")
    summary = pd.read_csv(source_summary, keep_default_na=False)
    inventory = pd.read_csv(source_inventory, keep_default_na=False)
    summary.to_csv(out / "evaluation/099_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "evaluation/099_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    audit, gate = base.audit_25k(summary, out, 2000)
    shutil.copy2(out / "audits/096_00_action_audit.csv", out / "audits/099_00_action_audit.csv")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)]
    model_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(2000)]
    model_path = ROOT / str(model_row.iloc[0]["model_path"])
    observed = base.observed_kwargs(model_path)
    result = {
        "task_id": cfg["task_id"],
        "status": "completed_seed1_smoke",
        "training_run": True,
        "seed": 1,
        "external_n_included": False,
        "weather_forecast_included": False,
        "effective_yaml": rel(effective_yaml),
        "effective_kwargs_from_loaded_model": observed,
        "validation_rows": int(len(final)),
        "mean_final_grnwt_raw2018": float(pd.to_numeric(final["final_grnwt"], errors="coerce").mean()),
        "mean_total_irrigation": float(pd.to_numeric(final["total_irrigation"], errors="coerce").mean()),
        "mean_total_n": float(pd.to_numeric(final["total_n"], errors="coerce").mean()),
        "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
        "action_gate": gate,
        "output_root": rel(out),
        "next_step_allowed": bool(gate["mechanism_pass"]),
        "next_step_note": "If true, separately validate 5K/10K seed=1; no automatic long training.",
    }
    (out / "099_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "099_00_manifest.json").write_text(json.dumps({"config": cfg, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 099_00 FQ ncost=2.5 seed=1 2K smoke", "", "- 唯一变化：seed=1；其余条件沿用 seed=0 分支。", "- 无 external-N、无天气预报、无五情景 replay。", "", "## Gate", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "- WP_ET 未从 2K training summary 推断。", "- 只有 gate 通过才允许另行设计 seed=1 的 5K/10K replay。", ""])
    (ROOT / "docs/099_00_fqa_originIC_reward_ncost25_seed1_smoke2k_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

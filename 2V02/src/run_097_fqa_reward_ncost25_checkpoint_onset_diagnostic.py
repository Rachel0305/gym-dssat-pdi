"""097_00: FQ checkpoint-onset diagnostic for the ncost=2.5 branch."""

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


CONFIG = ROOT / "configs/097_00_fqa_reward_ncost25_checkpoint_onset_diagnostic.json"
PROMPT = ROOT / "prompts/097_00_fqa_reward_ncost25_checkpoint_onset_diagnostic.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def patch_training_config() -> tuple[Path, dict[str, Any]]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["ppo"].update(cfg["fixed_ppo"])
    effective["reward"].update(cfg["reward_override"])
    effective_path = ROOT / "configs/097_00_effective_reward_ncost25_checkpoint_onset_ppo_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    return effective_path, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    checkpoints = list(map(int, cfg["training"]["checkpoint_steps"]))
    if args.dry_run:
        print(json.dumps({
            "task_id": cfg["task_id"],
            "station": "FQA",
            "training_steps": cfg["training"]["total_timesteps"],
            "checkpoint_steps": checkpoints,
            "reward_override": cfg["reward_override"],
            "external_n_allowed": False,
            "weather_forecast_allowed": False,
            "full_replay_allowed": False,
            "input_root_exists": INPUT_ROOT.exists(),
            "prompt_exists": PROMPT.exists(),
        }, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "configs").mkdir(parents=True, exist_ok=False)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml, cfg = patch_training_config()
    old = base.state_snapshot()
    try:
        base.engine.TASK_ID = cfg["task_id"]
        base.engine.TASK_NAME = cfg["task_name"]
        base.engine.BASE_OUT = out
        base.engine.BASE_DOC = ROOT / "docs/097_00_fqa_reward_ncost25_checkpoint_onset_diagnostic_record.md"
        base.engine.PROMPT = PROMPT
        base.engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        base.engine.STATION = "FQA"
        base.engine.SITES = ["FQA"]
        base.engine.BINARY_IRRIGATION_LEVELS = list(base.LEVELS_I)
        base.engine.BINARY_NITROGEN_LEVELS = list(base.LEVELS_N)
        base.batch.CONFIG = effective_yaml
        base.batch.OUT = out
        base.batch.DOC = base.engine.BASE_DOC
        base.batch.SITE_NAMES["FQA"] = "FQ"
        base.ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        base.engine.run_training(int(cfg["training"]["total_timesteps"]), checkpoints, suffix="")
    finally:
        base.restore_state(old)

    source_summary = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    source_inventory = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not source_summary.exists() or not source_inventory.exists():
        raise FileNotFoundError("training did not produce checkpoint summary/inventory")
    summary = pd.read_csv(source_summary, keep_default_na=False)
    inventory = pd.read_csv(source_inventory, keep_default_na=False)
    summary.to_csv(out / "evaluation/097_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "evaluation/097_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, Any]] = []
    previous_pass = None
    collapse_onset = None
    for checkpoint in checkpoints:
        audit, gate = base.audit_25k(summary, out, checkpoint)
        audit_path = out / "audits" / f"097_00_action_audit_ckpt{checkpoint}.csv"
        shutil.copy2(out / "audits/096_00_action_audit.csv", audit_path)
        final = audit[audit["checkpoint_step"].eq(checkpoint)]
        inv_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
        rows.append({
            "checkpoint_step": checkpoint,
            "validation_rows": int(len(final)),
            "mean_final_grnwt_raw2018": float(pd.to_numeric(final["final_grnwt"], errors="coerce").mean()),
            "mean_total_irrigation": float(pd.to_numeric(final["total_irrigation"], errors="coerce").mean()),
            "mean_total_n": float(pd.to_numeric(final["total_n"], errors="coerce").mean()),
            "distinct_action_signatures": int(final["action_signature"].nunique()),
            "union_nonzero_action_pairs": int(len(set("; ".join(final["action_signature"]).split("; ")) - {""})),
            "non_2018_distinct_action_signatures": int(final[final["year"].ne(2018)]["action_signature"].nunique()),
            "mechanism_pass": bool(gate["mechanism_pass"]),
            "model_path": str(inv_row.iloc[0]["model_path"]) if len(inv_row) == 1 else "",
            "audit_path": rel(audit_path),
        })
        if previous_pass is True and not gate["mechanism_pass"] and collapse_onset is None:
            collapse_onset = checkpoint
        previous_pass = bool(gate["mechanism_pass"])

    checkpoint_df = pd.DataFrame(rows)
    checkpoint_df.to_csv(out / "097_00_checkpoint_onset_summary.csv", index=False, encoding="utf-8-sig")
    candidates = checkpoint_df[checkpoint_df["mechanism_pass"]].copy()
    result = {
        "task_id": cfg["task_id"],
        "status": "completed_checkpoint_onset_diagnostic",
        "training_run": True,
        "external_n_included": False,
        "weather_forecast_included": False,
        "full_replay_run": False,
        "effective_yaml": rel(effective_yaml),
        "checkpoints": rows,
        "mechanism_pass_checkpoints": [int(x) for x in candidates["checkpoint_step"].tolist()],
        "collapse_onset_after_pass_checkpoint": int(collapse_onset) if collapse_onset is not None else None,
        "next_step_allowed": bool(not candidates.empty),
        "next_step_note": "Any candidate requires a separate corrected-2018 five-scenario replay; no automatic long training.",
        "output_root": rel(out),
    }
    (out / "097_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "097_00_manifest.json").write_text(json.dumps({"config": cfg, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 097_00 FQ ncost=2.5 checkpoint-onset diagnostic",
        "",
        "- 复现 096 的 seed=0、reward、PPO、动作格和安全层；只增加 checkpoint 观测密度。",
        "- 本任务没有五情景 replay、corrected 2018 性能替换、50K/75K/100K 或 external-N。",
        "",
        "## Checkpoint summary",
        "",
        checkpoint_df.to_markdown(index=False),
        "",
        f"- `mechanism_pass_checkpoints`: `{result['mechanism_pass_checkpoints']}`",
        f"- `collapse_onset_after_pass_checkpoint`: `{result['collapse_onset_after_pass_checkpoint']}`",
        "- WP_ET 未从训练汇总推断。",
        "",
    ]
    (ROOT / "docs/097_00_fqa_reward_ncost25_checkpoint_onset_diagnostic_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

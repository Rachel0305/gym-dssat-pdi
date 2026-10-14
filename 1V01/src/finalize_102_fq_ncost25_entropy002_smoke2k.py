"""Finalize an already completed 102_00 FQ smoke without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_102_fq_ncost25_entropy002_smoke2k as run
import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as base


ROOT = run.ROOT


def main() -> int:
    cfg = json.loads(run.CONFIG.read_text(encoding="utf-8"))
    pf = run.preflight(cfg)
    out = ROOT / cfg["output_root"]
    if not out.exists():
        raise FileNotFoundError(out)
    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    inventory_path = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not summary_path.exists() or not inventory_path.exists():
        raise FileNotFoundError("completed 102_00 summary/inventory missing")
    (out / "audits").mkdir(exist_ok=True)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    audit, gate = base.audit_25k(summary, out, 2000)
    audit.to_csv(out / "audits/102_00_action_audit.csv", index=False, encoding="utf-8-sig")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(2000)].iloc[0]
    observed = base.observed_kwargs(ROOT / str(row["model_path"]))
    metrics = final[[c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict()
    effective_yaml = ROOT / "configs/102_00_effective_fqa_ncost25_entropy002_ppo_config.yaml"
    result = {
        **pf,
        "mode": "completed_smoke_finalized_without_retraining",
        "output_root": run.rel(out),
        "effective_yaml": run.rel(effective_yaml),
        "effective_kwargs_from_loaded_model": observed,
        "validation_metrics_mean": metrics,
        "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
        "action_gate": gate,
        "training_reused": True,
        "next_step_allowed": bool(gate.get("mechanism_pass")),
        "next_step_note": "Only if true may a separate corrected-2018 5K/10K replay be proposed; no automatic long training.",
    }
    (out / "102_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "102_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result, "finalized_without_retraining": True}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    doc = ROOT / "docs/102_00_fqa_originIC_ncost25_entropy002_smoke2k_record.md"
    lines = ["# 102_00 FQ ncost=2.5 + entropy=0.02 seed=1 2K smoke", "", "- training completed in the isolated runner; final audit was completed without retraining", "- no external-N; no weather forecast; no full replay", "", "## Gate", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "- WP_ET 未从 2K training summary 推断。", "- 仅 gate 通过才允许另行设计 5K/10K replay。", ""])
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

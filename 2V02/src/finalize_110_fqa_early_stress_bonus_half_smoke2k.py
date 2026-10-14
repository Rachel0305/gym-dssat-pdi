"""Read-only finalizer for the completed 110_00 FQ 2K smoke."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as audit_base
from run_110_fqa_early_stress_bonus_half_smoke2k import CONFIG, PROMPT, preflight, rel


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    summary.to_csv(out / "evaluation/110_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    audit, gate = audit_base.audit_25k(summary, out, 2000)
    audit.to_csv(out / "audits/110_00_action_audit_ckpt2000.csv", index=False, encoding="utf-8-sig")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    metric_cols = [c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]
    metrics = final[metric_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict()
    daily = []
    for path in sorted((out / "daily_outputs/FQA").glob("*seed1_ckpt2000_daily.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        action_col = "discrete_action_index"
        irrigation = pd.to_numeric(frame.get("safe_action_amir", 0), errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(frame.get("safe_action_anfer", 0), errors="coerce").fillna(0.0)
        nonzero = frame[irrigation.ne(0) | nitrogen.ne(0)].copy()
        post = nonzero[pd.to_numeric(nonzero.get("dap", 0), errors="coerce").gt(1)] if not nonzero.empty else nonzero
        daily.append({"file": path.name, "nonzero_rows": int(len(nonzero)), "post_dap1_rows": int(len(post)), "action_pairs": int(nonzero[action_col].nunique()) if not nonzero.empty and action_col in nonzero.columns else None})
    post_rows = sum(item["post_dap1_rows"] for item in daily)
    result = {"task_id": cfg["task_id"], "mode": "finalized_existing_2k_reward_ablation_smoke", "training_run": True, "preflight": preflight(cfg), "effective_yaml": "configs/110_00_effective_fqa_early_stress_bonus_half.yaml", "metrics_mean_ckpt2000": metrics, "action_gate": gate, "daily_action_audit": daily, "total_post_dap1_rows": post_rows, "elapsed_s": None, "rss_delta_bytes": None, "WP_ET_status": "not_available_in_2K_summary; not inferred", "next_step_allowed": bool(gate.get("mechanism_pass"))}
    (out / "110_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "110_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": result["preflight"], "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    doc = ROOT / "docs/110_00_fqa_early_stress_bonus_half_smoke2k_record.md"
    lines = ["# 110_00 FQ early water-stress bonus half 2K smoke", "", "- 训练已完成；本文件由只读 finalizer 补全，未重训。", "- 唯一干预：water_stress_relief_coef 10.0 -> 5.0。", "- 保持当前天气观测、PPO、动作格、DSSAT 自动灌溉和安全 mask 不变。", "", "## Result", "", f"- metrics_mean_ckpt2000: `{metrics}`", f"- action_gate: `{gate}`", f"- total_post_dap1_rows: `{post_rows}`", "- WP_ET 未从 2K summary 推断。", "- 不自动进入 5K/25K/100K。", ""]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only finalization for a completed 107_00 smoke.

The training/evaluation process already completed; this helper only creates the
missing audit directory, computes the preregistered gate, and writes the result.
It never retrains or overwrites the 107 output root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as audit_base
import run_107_fqa_current_weather_observation_smoke2k as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/107_00_fqa_current_weather_observation_smoke2k.json"
OUT = ROOT / "benchmark_results/107_00_fqa_originIC_current_weather_observation_smoke2k"
DOC = ROOT / "docs/107_00_fqa_current_weather_observation_smoke2k_record.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "evaluation/042_10_checkpoint_validation_summary.csv", keep_default_na=False)
    inventory = pd.read_csv(OUT / "evaluation/042_10_training_checkpoint_inventory.csv", keep_default_na=False)
    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    audit, gate = audit_base.audit_25k(summary, OUT, 2000)
    audit.to_csv(OUT / "audits/107_00_action_audit.csv", index=False, encoding="utf-8-sig")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    daily = pd.read_csv(ROOT / str(final.iloc[0]["daily_csv_path"]), keep_default_na=False)
    dim_values = pd.to_numeric(daily["current_weather_observation_dim_107"], errors="coerce").dropna().unique().tolist() if "current_weather_observation_dim_107" in daily.columns else []
    prompt_gate = {
        "validation_rows_10_of_10": int(len(final)) == 10,
        "observation_dim_27": dim_values == [27],
        "raw_to_safe_mismatch_zero": int(gate.get("raw_to_safe_mismatch_rows", 1)) == 0,
        "safe_to_dssat_mismatch_zero": int(gate.get("safe_to_dssat_mismatch_rows", 1)) == 0,
        "off_grid_zero": int(gate.get("off_grid_rows", 1)) == 0,
        "post_dap1_actions_present": int(gate.get("positive_action_rows_after_dap1", 0)) > 0,
        "cross_year_signatures_at_least_2": int(gate.get("cross_year_distinct_signatures", 0)) >= 2,
        "nonzero_action_pairs_at_least_3": int(gate.get("nonzero_action_pair_count", 0)) >= 3,
    }
    prompt_gate["mechanism_pass"] = all(prompt_gate.values())
    result = {
        "task_id": cfg["task_id"], "mode": "completed_smoke_finalized_read_only", "training_run": True,
        "total_timesteps": 2000, "checkpoint_steps": [1000, 2000], "validation_rows_ckpt2000": int(len(final)),
        "observation_dim_values": dim_values, "observation_append_dim": 4,
        "validation_metrics_mean": final[[c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict(),
        "action_gate": gate, "prompt_gate": prompt_gate, "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
        "output_root": rel(OUT), "next_step_allowed": bool(prompt_gate["mechanism_pass"]),
        "finalization_note": "No retraining; initial runner stopped only while writing audit because audits/ was absent.",
    }
    (OUT / "107_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "107_00_manifest.json").write_text(json.dumps({"config": cfg, "result": result, "inventory_rows": int(len(inventory))}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 107_00 FQ current-weather observation seed=1 2K smoke", "", "- 只追加当日 rain/srad/tmax/tmin 四个观测字段；无 future forecast、无 external-N。", "- 训练与验证已完成 2K；本文件是只读 finalization，初始 runner 在写审计时因缺少 `audits/` 目录停止，未发生重训。", "", "## Observation", "", f"- 观测维度记录：`{dim_values}`（原始维度 23，追加 4 后为 27）。", "", "## Registered prompt gate", "", *[f"- `{key}`: `{value}`" for key, value in prompt_gate.items()], "", "## Stronger legacy gate (reported separately)", "", *[f"- `{key}`: `{value}`" for key, value in gate.items()], "", "- 本 prompt gate 通过时，只允许另行申请 5K/10K 或五情景验证设计；不自动进入 25K/100K，也不能直接冻结。", "- WP_ET 未从 2K summary 推断。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""181: WP_ET replay for the paired FQ seed2 2K smoke checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import replay_178_multisite_wp_et as base

base.TASK = "181FQA_fqa_seed2_2k_wp_et_replay"
base.OUT = ROOT / "benchmark_results" / base.TASK
base.POLICIES = {
    "fqa_forecast_seed2_2k": {
        "station": "FQA", "site": "FQ", "seed": 2, "checkpoint_step": 2000,
        "kind": "forecast", "cfg": ROOT / "configs/179FQA_fqa_originIC_dual_branch_weather_reader_2k_seed2.json",
        "inventory": ROOT / "benchmark_results/179FQA_fqa_originIC_dual_branch_weather_reader_2k_seed2_smoke2k/evaluation/179FQA_checkpoint_validation_summary.csv",
    },
    "fqa_noforecast_seed2_2k": {
        "station": "FQA", "site": "FQ", "seed": 2, "checkpoint_step": 2000,
        "kind": "noforecast", "cfg": ROOT / "configs/180FQA_fqa_originIC_dual_branch_noforecast_control_2k_seed2.json",
        "inventory": ROOT / "benchmark_results/180FQA_fqa_originIC_dual_branch_noforecast_control_2k_seed2_smoke2k/evaluation/180FQA_checkpoint_validation_summary.csv",
    },
}


if __name__ == "__main__":
    import json
    import shutil
    import pandas as pd

    manifest = base.run()
    out = base.OUT
    # Give the seed2 replay a task-specific, unambiguous set of aliases.  The
    # imported generic runner intentionally keeps its 178 filenames internally.
    aliases = {
        "178_wp_et_replay_summary_by_policy.csv": "181FQA_seed2_wp_et_replay_summary_by_policy.csv",
        "178_wp_et_replay_by_policy_year.csv": "181FQA_seed2_wp_et_replay_by_policy_year.csv",
        "178_wp_et_paired_delta_by_year.csv": "181FQA_seed2_wp_et_paired_delta_by_year.csv",
        "178_wp_et_paired_summary.csv": "181FQA_seed2_wp_et_paired_summary.csv",
        "178_wp_et_replay_manifest.json": "181FQA_seed2_wp_et_replay_manifest.json",
    }
    for source, target in aliases.items():
        source_path, target_path = out / source, out / target
        if source_path.exists():
            shutil.copy2(source_path, target_path)

    # Compare the independent seed2 pair with the already completed seed1 FQ 2K
    # pair using the same WP_ET definition and ten validation years.
    s1 = pd.read_csv(ROOT / "benchmark_results/178_multisite_wp_et_replay/178_wp_et_paired_delta_by_year.csv")
    s1 = s1[(s1["station"].eq("FQA")) & (s1["checkpoint_step"].eq(2000))].copy()
    s2 = pd.read_csv(out / "178_wp_et_paired_delta_by_year.csv")
    for frame, seed in [(s1, 1), (s2, 2)]:
        frame["seed"] = seed
    stability = pd.concat([s1, s2], ignore_index=True)
    stability_summary = stability.groupby(["station", "checkpoint_step", "seed"], as_index=False).agg(
        years=("year", "nunique"),
        mean_yield_delta=("yield_delta_forecast_minus_noforecast", "mean"),
        mean_WP_ET_delta=("WP_ET_delta_forecast_minus_noforecast", "mean"),
        mean_PFP_N_delta=("PFP_N_delta_forecast_minus_noforecast", "mean"),
        WP_ET_win_years=("WP_ET_delta_forecast_minus_noforecast", lambda x: int((x > 0).sum())),
        yield_win_years=("yield_delta_forecast_minus_noforecast", lambda x: int((x > 0).sum())),
    )
    stability.to_csv(out / "181FQA_seed1_seed2_wp_et_delta_by_year.csv", index=False, encoding="utf-8-sig")
    stability_summary.to_csv(out / "181FQA_seed1_seed2_wp_et_stability_summary.csv", index=False, encoding="utf-8-sig")
    report = out / "181FQA_fqa_seed2_2k_wp_et_replay_record.md"
    report.write_text(
        "# 181 FQ seed2 2K WP_ET replay\n\n"
        "- 本轮只读串行回放 FQ seed2 forecast/no-forecast 的 2K checkpoint，共 20 个年度组合；没有重新训练。\n"
        "- 与 FQ seed1 2K 配对结果合并检查跨 seed 稳定性。\n"
        "- WP_ET = YPEM × 0.1；若 YPEM 不可用，则用 HWAM / (ETCP × 10)。\n\n"
        "## seed1/seed2 paired stability\n\n"
        + stability_summary.round(4).to_markdown(index=False)
        + "\n\n- 逐年 delta：`benchmark_results/181FQA_fqa_seed2_2k_wp_et_replay/181FQA_seed1_seed2_wp_et_delta_by_year.csv`。\n"
        + "- 汇总：`benchmark_results/181FQA_fqa_seed2_2k_wp_et_replay/181FQA_seed1_seed2_wp_et_stability_summary.csv`。\n",
        encoding="utf-8",
    )
    manifest["seed2_summary_csv"] = "benchmark_results/181FQA_fqa_seed2_2k_wp_et_replay/181FQA_seed2_wp_et_replay_summary_by_policy.csv"
    manifest["seed1_seed2_stability_csv"] = "benchmark_results/181FQA_fqa_seed2_2k_wp_et_replay/181FQA_seed1_seed2_wp_et_stability_summary.csv"
    manifest["report_md"] = "docs/181FQA_fqa_seed2_2k_wp_et_replay_record.md"
    (out / "181FQA_seed2_wp_et_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs/181FQA_fqa_seed2_2k_wp_et_replay_record.md").write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

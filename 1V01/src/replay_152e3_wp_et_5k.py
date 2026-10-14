"""Complete WP_ET for the frozen E3 5K seed0 checkpoint.

This is a serial, no-training replay.  It reuses the established DSSAT
snapshot/ETCP matching code while replacing the forecast factory with E3's
zero-weather wrapper.  Existing E2/no-forecast outputs are not modified.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import replay_151_e2_noforecast_wp_et_5k as replay
from run_152E3_sya_dual_branch_noforecast_control import make_zero_weather_factory


TASK = "152E3N_wp_et_5k_replay"
OUT = ROOT / "benchmark_results" / TASK
E3_KEY = "e3_s0"


def main() -> int:
    replay.OUT = OUT
    replay.POLICIES[E3_KEY] = {
        "kind": "forecast",
        "seed": 0,
        "cfg": ROOT / "configs/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0.json",
        "inventory": ROOT / "benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0/evaluation/152E3_checkpoint_validation_summary.csv",
    }

    old_factory = forecast.make_forecast_env_factory
    forecast.make_forecast_env_factory = lambda base_make_env, _cfg: make_zero_weather_factory(base_make_env)
    try:
        manifest = replay.run([E3_KEY], list(range(2014, 2024)), resume=True)
    finally:
        forecast.make_forecast_env_factory = old_factory

    summary_src = OUT / "151_wp_et_replay_summary_by_policy.csv"
    by_year_src = OUT / "151_wp_et_replay_by_policy_year.csv"
    check_src = OUT / "151_wp_et_replay_reproducibility_check.csv"
    summary = pd.read_csv(summary_src)
    summary.to_csv(OUT / "152E3_wp_et_replay_summary.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(by_year_src).to_csv(OUT / "152E3_wp_et_replay_by_year.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(check_src).to_csv(OUT / "152E3_wp_et_replay_reproducibility_check.csv", index=False, encoding="utf-8-sig")

    row = summary.loc[summary["policy"].eq(E3_KEY)].iloc[0]
    report = [
        "# E3 5K seed0 WP_ET replay",
        "",
        "- 范围：SYA originIC，验证年 2014–2023，冻结 5000-step checkpoint。",
        "- 本轮不训练；串行回放 10 个 DSSAT 年份，并从 Summary.OUT 读取 ETCP。",
        "- 天气分支使用 E3 零天气 wrapper；奖励、动作网格和 checkpoint 不变。",
        "",
        "## 汇总",
        "",
        f"- 平均产量：{float(row['mean_yield']):.4f} kg/ha",
        f"- 平均 ETCP：{float(row['mean_etcp_mm']):.4f} mm",
        f"- 平均 WP_ET：{float(row['mean_WP_ET']):.4f} kg/m³",
        f"- 加权 WP_ET：{float(row['weighted_WP_ET']):.4f} kg/m³",
        f"- 平均 PFP-N：{float(row['mean_PFP_N']):.4f} kg/kg",
        f"- 加权 PFP-N：{float(row['weighted_PFP_N']):.4f} kg/kg",
        "",
        "## 文件",
        "",
        "- `152E3_wp_et_replay_summary.csv`：E3 汇总",
        "- `152E3_wp_et_replay_by_year.csv`：逐年结果",
        "- `152E3_wp_et_replay_reproducibility_check.csv`：与原 daily CSV 的闭合核验",
    ]
    report_path = OUT / "2026-08-16_sya_E3_wp_et_5k_replay.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest.update({
        "task": TASK,
        "policy": E3_KEY,
        "summary_csv": str((OUT / "152E3_wp_et_replay_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "by_year_csv": str((OUT / "152E3_wp_et_replay_by_year.csv").relative_to(ROOT)).replace("\\", "/"),
        "reproducibility_csv": str((OUT / "152E3_wp_et_replay_reproducibility_check.csv").relative_to(ROOT)).replace("\\", "/"),
        "report_md": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "no_training": True,
    })
    (OUT / "152E3_wp_et_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

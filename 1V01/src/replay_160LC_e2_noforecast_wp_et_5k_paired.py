"""160LC: same-checkpoint LC-E2 versus matched zero-weather control replay."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import replay_151_e2_noforecast_wp_et_5k as replay
from run_152E3_sya_dual_branch_noforecast_control import make_zero_weather_factory
import run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k  # noqa: F401 - LC extractor pickle registration


TASK = "160LC_lca_lowIC_e2_noforecast_wp_et_5k_paired_replay"
OUT = ROOT / "benchmark_results" / TASK
EXISTING_E2 = ROOT / "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/replays/e2_s0"
YEARS = list(range(2014, 2024))
E2_CFG = ROOT / "configs/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0.json"
E2_INV = ROOT / "benchmark_results/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0/evaluation/156LC_checkpoint_validation_summary.csv"
NOF_CFG = ROOT / "configs/158LC_lca_lowIC_dual_branch_noforecast_control_5k_seed0.json"
NOF_INV = ROOT / "benchmark_results/158LC_lca_lowIC_dual_branch_noforecast_control_5k_seed0/evaluation/158LC_checkpoint_validation_summary.csv"
BASELINE = ROOT / "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/evaluation/053_02_baseline_summary.csv"


def copy_existing_e2() -> None:
    target = OUT / "replays/e2_s0"
    if target.exists():
        return
    if not EXISTING_E2.exists():
        raise FileNotFoundError(EXISTING_E2)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(EXISTING_E2, target)


def lc_runtime_config(cfg, policy_key, policy_out):
    runtime_cfg, env_cfg, engine = ORIGINAL_CONFIGURE(cfg, policy_key, policy_out)
    if policy_key == "nof_s0":
        old_make_env = engine.base03222.base.make_env
        engine.base03222.base.make_env = make_zero_weather_factory(old_make_env)
    return runtime_cfg, env_cfg, engine


def write_lc_report(rows: pd.DataFrame) -> dict[str, str]:
    summary = rows.groupby("policy", as_index=False).agg(
        years=("year", "nunique"), mean_yield=("final_grnwt", "mean"), mean_etcp_mm=("etcp_mm", "mean"),
        mean_WP_ET=("WP_ET_kg_m3", "mean"), mean_PFP_N=("PFP_N_kg_kg", "mean"),
        total_yield=("final_grnwt", "sum"), total_n=("total_n", "sum"), total_etcp_mm=("etcp_mm", "sum"),
    )
    summary["weighted_WP_ET"] = summary["total_yield"] / (summary["total_etcp_mm"] * 10.0)
    summary["weighted_PFP_N"] = summary["total_yield"] / summary["total_n"].replace(0, np.nan)
    pair = rows[rows["policy"].eq("e2_s0")][["year", "final_grnwt", "total_irrigation", "total_n", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]].merge(
        rows[rows["policy"].eq("nof_s0")][["year", "final_grnwt", "total_irrigation", "total_n", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]],
        on="year", suffixes=("_forecast", "_noforecast"), validate="one_to_one",
    )
    pair["yield_delta_forecast_minus_noforecast"] = pair["final_grnwt_forecast"] - pair["final_grnwt_noforecast"]
    pair["WP_ET_delta_forecast_minus_noforecast"] = pair["WP_ET_kg_m3_forecast"] - pair["WP_ET_kg_m3_noforecast"]
    pair["PFP_N_delta_forecast_minus_noforecast"] = pair["PFP_N_kg_kg_forecast"] - pair["PFP_N_kg_kg_noforecast"]
    pair_summary = pd.DataFrame([{
        "seed": 0, "years": len(pair), "mean_yield_delta": pair["yield_delta_forecast_minus_noforecast"].mean(),
        "mean_WP_ET_delta": pair["WP_ET_delta_forecast_minus_noforecast"].mean(), "mean_PFP_N_delta": pair["PFP_N_delta_forecast_minus_noforecast"].mean(),
        "yield_win_years": int((pair["yield_delta_forecast_minus_noforecast"] > 0).sum()),
        "WP_ET_win_years": int((pair["WP_ET_delta_forecast_minus_noforecast"] > 0).sum()),
        "weighted_forecast_WP_ET": pair["final_grnwt_forecast"].sum() / (pair["etcp_mm_forecast"].sum() * 10.0),
        "weighted_noforecast_WP_ET": pair["final_grnwt_noforecast"].sum() / (pair["etcp_mm_noforecast"].sum() * 10.0),
    }])
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    for col in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    baseline_mean = baseline.groupby("scenario", as_index=False)[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].mean()
    forecast = rows[rows["policy"].eq("e2_s0")].copy()
    forecast_for_compare = forecast[["year", "final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg"]].rename(columns={"final_grnwt": "grain_yield_kg_ha"})
    merged = forecast_for_compare.merge(
        baseline.groupby("year", as_index=False)[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].max(), on="year", suffixes=("_e2", "_baseline_max"), validate="one_to_one"
    )
    wins = pd.DataFrame([{
        "metric": metric, "years": len(merged), "strict_wins_vs_four_baseline_max": int((merged[f"{metric}_e2"] > merged[f"{metric}_baseline_max"]).sum()),
        "mean_gap": float((merged[f"{metric}_e2"] - merged[f"{metric}_baseline_max"]).mean()),
    } for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]])
    summary.to_csv(OUT / "160LC_replay_summary_by_policy.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(OUT / "160LC_paired_forecast_noforecast_by_year.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(OUT / "160LC_paired_forecast_noforecast_summary.csv", index=False, encoding="utf-8-sig")
    baseline_mean.to_csv(OUT / "160LC_vs_four_baselines_mean.csv", index=False, encoding="utf-8-sig")
    wins.to_csv(OUT / "160LC_four_baseline_max_win_summary.csv", index=False, encoding="utf-8-sig")
    report = [
        "# LC-E2 与 matched no-forecast 5K WP_ET 对照", "", "- 站点：LCA/LC，lowIC，验证年 2014–2023。",
        "- 两个策略均为 seed0、5000-step checkpoint；E2 使用 12 个 rolling-7d 天气特征，无预报 control 保留相同 dual-branch 结构但天气分支全为 0。",
        "- reward、动作网格、mask、安全约束和 DSSAT 输入保持一致。", "", "## 策略汇总", "",
        summary.round(4).to_markdown(index=False), "", "## 同 seed 配对差值（E2 - no-forecast）", "", pair_summary.round(4).to_markdown(index=False),
        "", "## 相对四基线逐年最高值", "", wins.round(4).to_markdown(index=False), "", "## 解释边界", "",
        "这是同 checkpoint 的 LC 配对对照，不等于多 seed 稳定性，也不等于真实预报误差评估。",
    ]
    report_path = ROOT / "docs/160LC_lca_lowIC_e2_noforecast_wp_et_5k_paired_replay_record.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"summary": str((OUT / "160LC_replay_summary_by_policy.csv").relative_to(ROOT).as_posix()), "paired": str((OUT / "160LC_paired_forecast_noforecast_summary.csv").relative_to(ROOT).as_posix()), "report": str(report_path.relative_to(ROOT).as_posix())}


def main() -> int:
    global ORIGINAL_CONFIGURE
    replay.TASK = TASK
    replay.OUT = OUT
    replay.STATION = "LCA"
    replay.YEARS = YEARS
    replay.POLICIES = {
        "e2_s0": {"kind": "forecast", "seed": 0, "cfg": E2_CFG, "inventory": E2_INV},
        "nof_s0": {"kind": "noforecast", "seed": 0, "cfg": NOF_CFG, "inventory": NOF_INV},
    }
    copy_existing_e2()
    ORIGINAL_CONFIGURE = replay.configure_runtime
    replay.configure_runtime = lc_runtime_config
    try:
        replay.run(["e2_s0", "nof_s0"], YEARS, resume=True)
    finally:
        replay.configure_runtime = ORIGINAL_CONFIGURE
    rows = pd.read_csv(OUT / "151_wp_et_replay_by_policy_year.csv")
    rows.to_csv(OUT / "160LC_replay_by_policy_year.csv", index=False, encoding="utf-8-sig")
    record = write_lc_report(rows)
    manifest = {"task": TASK, "policies": ["e2_s0", "nof_s0"], "years": YEARS, "rows_completed": int(len(rows)), "no_training": True, **record}
    (OUT / "160LC_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""169LC: read-only 2K checkpoint replay for LC seed1 E2/no-forecast."""

from __future__ import annotations

import json
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
import run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k  # noqa: F401

TASK = "169LC_lca_lowIC_e2_seed1_noforecast_wp_et_2k_paired_replay"
OUT = ROOT / "benchmark_results" / TASK
YEARS = list(range(2014, 2024))
E2_CFG = ROOT / "configs/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1.json"
E2_INV = ROOT / "benchmark_results/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1/evaluation/163LC_checkpoint_validation_summary.csv"
NOF_CFG = ROOT / "configs/167LC_lca_lowIC_dual_branch_noforecast_control_5k_seed1.json"
NOF_INV = ROOT / "benchmark_results/167LC_lca_lowIC_dual_branch_noforecast_control_5k_seed1/evaluation/167LC_checkpoint_validation_summary.csv"


def load_inventory_2k(policy_key: str) -> pd.DataFrame:
    path = Path(replay.POLICIES[policy_key]["inventory"])
    df = pd.read_csv(path, keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    out = df[df["checkpoint_step"].eq(2000) & df["year"].isin(YEARS)].copy()
    if len(out) != 10:
        raise RuntimeError(f"{policy_key}: expected 10 2K rows, got {len(out)}")
    return out.sort_values("year").reset_index(drop=True)


def lc_runtime_config(cfg, policy_key, policy_out):
    runtime_cfg, env_cfg, engine = ORIGINAL_CONFIGURE(cfg, policy_key, policy_out)
    if policy_key == "nof_s1":
        engine.base03222.base.make_env = make_zero_weather_factory(engine.base03222.base.make_env)
    return runtime_cfg, env_cfg, engine


def write_report(rows: pd.DataFrame) -> dict[str, str]:
    summary = rows.groupby("policy", as_index=False).agg(
        years=("year", "nunique"), mean_yield=("final_grnwt", "mean"), mean_irrigation=("total_irrigation", "mean"),
        mean_n=("total_n", "mean"), mean_etcp_mm=("etcp_mm", "mean"), mean_WP_ET=("WP_ET_kg_m3", "mean"),
        mean_PFP_N=("PFP_N_kg_kg", "mean"), total_yield=("final_grnwt", "sum"), total_etcp_mm=("etcp_mm", "sum"),
    )
    summary["weighted_WP_ET"] = summary["total_yield"] / (summary["total_etcp_mm"] * 10.0)
    summary["weighted_PFP_N"] = summary["total_yield"] / (rows.groupby("policy")["total_n"].sum().values)
    keep = ["year", "final_grnwt", "total_irrigation", "total_n", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]
    f = rows[rows.policy.eq("e2_s1")][keep].merge(rows[rows.policy.eq("nof_s1")][keep], on="year", suffixes=("_forecast", "_noforecast"), validate="one_to_one")
    f["yield_delta_forecast_minus_noforecast"] = f.final_grnwt_forecast - f.final_grnwt_noforecast
    f["WP_ET_delta_forecast_minus_noforecast"] = f.WP_ET_kg_m3_forecast - f.WP_ET_kg_m3_noforecast
    f["PFP_N_delta_forecast_minus_noforecast"] = f.PFP_N_kg_kg_forecast - f.PFP_N_kg_kg_noforecast
    pair = pd.DataFrame([{"checkpoint": 2000, "years": len(f), "mean_yield_delta": f.yield_delta_forecast_minus_noforecast.mean(), "mean_WP_ET_delta": f.WP_ET_delta_forecast_minus_noforecast.mean(), "mean_PFP_N_delta": f.PFP_N_delta_forecast_minus_noforecast.mean(), "yield_win_years": int((f.yield_delta_forecast_minus_noforecast > 0).sum()), "WP_ET_win_years": int((f.WP_ET_delta_forecast_minus_noforecast > 0).sum()), "PFP_N_win_years": int((f.PFP_N_delta_forecast_minus_noforecast > 0).sum())}])
    out_rows = ROOT / OUT.relative_to(ROOT)
    summary.to_csv(out_rows / "169LC_replay_summary_by_policy.csv", index=False, encoding="utf-8-sig")
    f.to_csv(out_rows / "169LC_paired_forecast_noforecast_by_year.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(out_rows / "169LC_paired_forecast_noforecast_summary.csv", index=False, encoding="utf-8-sig")
    report_path = ROOT / "docs/169LC_lca_lowIC_e2_seed1_noforecast_wp_et_2k_paired_replay_record.md"
    report_path.write_text("\n".join(["# LC-E2 seed1 与无预报 seed1 的 2K checkpoint 回放", "", "- 只读回放，未重新训练；验证年 2014–2023。", "- 判定重点：比较 2K 与此前 5K 的 E2/无预报差异，识别 5K 训练后的退化。", "", "## 2K 汇总", "", summary.round(4).to_markdown(index=False), "", "## 同 seed 配对（E2 - 无预报）", "", pair.round(4).to_markdown(index=False), ""]), encoding="utf-8")
    return {"summary": str((out_rows / "169LC_replay_summary_by_policy.csv").relative_to(ROOT).as_posix()), "paired": str((out_rows / "169LC_paired_forecast_noforecast_summary.csv").relative_to(ROOT).as_posix()), "report": str(report_path.relative_to(ROOT).as_posix())}


def main() -> int:
    global ORIGINAL_CONFIGURE
    replay.TASK = TASK
    replay.OUT = OUT
    replay.STATION = "LCA"
    replay.YEARS = YEARS
    replay.POLICIES = {"e2_s1": {"kind": "forecast", "seed": 1, "cfg": E2_CFG, "inventory": E2_INV}, "nof_s1": {"kind": "noforecast", "seed": 1, "cfg": NOF_CFG, "inventory": NOF_INV}}
    replay.load_inventory = load_inventory_2k
    ORIGINAL_CONFIGURE = replay.configure_runtime
    replay.configure_runtime = lc_runtime_config
    try:
        replay.run(["e2_s1", "nof_s1"], YEARS, resume=True)
    finally:
        replay.configure_runtime = ORIGINAL_CONFIGURE
    rows = pd.read_csv(OUT / "151_wp_et_replay_by_policy_year.csv")
    rows.to_csv(OUT / "169LC_replay_by_policy_year.csv", index=False, encoding="utf-8-sig")
    record = write_report(rows)
    manifest = {"task": TASK, "checkpoint_step": 2000, "policies": ["e2_s1", "nof_s1"], "years": YEARS, "rows_completed": int(len(rows)), "no_training": True, **record}
    (OUT / "169LC_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

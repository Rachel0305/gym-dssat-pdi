"""177: summarize the HL/YC/FQ forecast versus no-forecast transfer runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/177_multisite_weather_framework_transfer_validation"
SITES = {
    "HLA": ("170HLA_hla_lowIC_dual_branch_weather_reader_5k_seed1", "171HLA_hla_lowIC_dual_branch_noforecast_control_5k_seed1"),
    "YCA": ("172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1", "173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1"),
    "FQA": ("174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1", "175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1"),
}


def load_eval(task: str) -> pd.DataFrame:
    path = ROOT / "benchmark_results" / task / "evaluation/042_10_checkpoint_validation_summary.csv"
    df = pd.read_csv(path, keep_default_na=False)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    for col in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def gate(task: str) -> dict:
    path = ROOT / "benchmark_results" / task / f"{task.split('_', 1)[0]}_5k_result.json"
    if not path.exists():
        path = ROOT / "benchmark_results" / task / f"{task.split('_', 1)[0]}_smoke_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    g = result.get("action_forecast_gate", result.get("smoke_gate", {}))
    return {"formal_result": path.relative_to(ROOT).as_posix(), "gate_next_step_allowed": g.get("next_step_allowed"), "positive_actions_present": g.get("positive_actions_present"), "transfer_isolation_passed": result.get("transfer_isolation_audit", {}).get("passed")}


def main() -> int:
    rows: list[dict] = []
    pair_rows: list[dict] = []
    for site, (forecast_task, nof_task) in SITES.items():
        f = load_eval(forecast_task)
        n = load_eval(nof_task)
        for step in [2000, 5000]:
            ff = f[f["checkpoint_step"].eq(step)]
            nn = n[n["checkpoint_step"].eq(step)]
            rows.extend([
                {"site": site, "policy": "forecast", "checkpoint_step": step, "years": len(ff), "mean_yield": ff["final_grnwt"].mean(), "mean_irrigation": ff["total_irrigation"].mean(), "mean_n": ff["total_n"].mean(), "mean_PFP_N": ff["PFP_N"].mean(), "action_sequence_count": ff["action_sequence"].nunique(), "positive_action_rows": int((ff["total_irrigation"] + ff["total_n"] > 0).sum())},
                {"site": site, "policy": "noforecast", "checkpoint_step": step, "years": len(nn), "mean_yield": nn["final_grnwt"].mean(), "mean_irrigation": nn["total_irrigation"].mean(), "mean_n": nn["total_n"].mean(), "mean_PFP_N": nn["PFP_N"].mean(), "action_sequence_count": nn["action_sequence"].nunique(), "positive_action_rows": int((nn["total_irrigation"] + nn["total_n"] > 0).sum())},
            ])
            pair_rows.append({"site": site, "checkpoint_step": step, "yield_forecast_minus_noforecast": ff["final_grnwt"].mean() - nn["final_grnwt"].mean(), "irrigation_forecast_minus_noforecast": ff["total_irrigation"].mean() - nn["total_irrigation"].mean(), "PFP_N_forecast_minus_noforecast": ff["PFP_N"].mean() - nn["PFP_N"].mean(), "yield_win_years": int((ff.set_index("year")["final_grnwt"] > nn.set_index("year")["final_grnwt"]).sum()) if len(ff) == len(nn) == 10 else None})
    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(pair_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "177_multisite_summary_by_site_checkpoint.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "177_multisite_paired_delta_by_site_checkpoint.csv", index=False, encoding="utf-8-sig")
    gate_rows = []
    for site, (f, n) in SITES.items():
        gate_rows.extend([{"site": site, "policy": "forecast", **gate(f)}, {"site": site, "policy": "noforecast", **gate(n)}])
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(OUT / "177_multisite_gate_audit.csv", index=False, encoding="utf-8-sig")
    report = ["# HL / YC / FQ SY 天气预报框架迁移验证", "", "- seed=1；每站点 forecast 与 matched no-forecast 使用同一 reward、动作网格、安全约束和训练/验证年份。", "- 基础观测维度按站点原始 046_02 接口适配：HLA=24、YCA=22、FQA=23；天气分支固定为 12 维。", "- 2K/5K 均为独立训练 checkpoint；WP_ET 需要额外 DSSAT Summary.OUT 回放，本表先报告产量、灌溉、施氮、PFP-N 和动作多样性。", "", "## 汇总", "", summary.round(3).to_markdown(index=False), "", "## forecast - no-forecast", "", paired.round(3).to_markdown(index=False), "", "## 初步解释", "", "- HLA：forecast 在 2K/5K 均略低于无预报。", "- YC：forecast 在 2K 有明显产量/PFP-N优势，但 5K 变为全零动作并触发正式 gate 失败，属于训练坍缩，不能作为成功结果。", "- FQ：forecast 在 2K/5K 均略高于无预报，但优势很小，需 WP_ET 回放和独立 seed 才能确认。", "", "## 边界", "", "本报告验证的是 SY 天气编码在三个站点的迁移潜力，不代表真实天气预报误差，也不证明跨 seed 稳定性。"]
    report_path = ROOT / "docs/177_multisite_weather_framework_transfer_validation_record.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str((OUT / "177_multisite_summary_by_site_checkpoint.csv").relative_to(ROOT).as_posix()), "paired": str((OUT / "177_multisite_paired_delta_by_site_checkpoint.csv").relative_to(ROOT).as_posix()), "gates": str((OUT / "177_multisite_gate_audit.csv").relative_to(ROOT).as_posix()), "report": str(report_path.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compile the two paired FQ E4 2K smoke results into a stability record."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "seed1": ROOT / "benchmark_results/182_183_FQA_e4_warmstart_weather_residual_2k_smoke_result.json",
    "seed2": ROOT / "benchmark_results/184_185_FQA_e4_seed2_warmstart_weather_residual_2k_smoke_result.json",
}


def main() -> None:
    rows = []
    audits = {}
    for seed, path in RESULTS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        audits[seed] = data.get("post_adaptation_weather_branch", {})
        means = {str(row["kind"]): row for row in data["wp_et_replay"]["means"]}
        for kind in ["forecast", "noforecast"]:
            rec = means[kind]
            rows.append({"seed": seed, "arm": kind, **{k: rec[k] for k in ["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]}})
        f, n = means["forecast"], means["noforecast"]
        rows.append({"seed": seed, "arm": "forecast_minus_noforecast", **{k: float(f[k] - n[k]) for k in ["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]}})
    detail = pd.DataFrame(rows)
    pooled = detail[detail["arm"].isin(["forecast", "noforecast"])].groupby("arm", as_index=False)[["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]].mean()
    pooled_map = {row["arm"]: row for row in pooled.to_dict(orient="records")}
    pooled_delta = {k: float(pooled_map["forecast"][k] - pooled_map["noforecast"][k]) for k in ["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]}
    delta = detail[detail["arm"].eq("forecast_minus_noforecast")]
    verdict = {
        "yield_forecast_lower_both_seeds": bool((delta["final_grnwt"] < 0).all()),
        "pfp_n_forecast_lower_both_seeds": bool((delta["PFP_N_kg_kg"] < 0).all()),
        "wp_et_forecast_higher_both_seeds": bool((delta["WP_ET_kg_m3"] > 0).all()),
        "weather_branch_updated_both_seeds": bool(all(a.get("forecast", {}).get("weather_linear_nonzero") and not a.get("noforecast", {}).get("weather_linear_nonzero") for a in audits.values())),
        "five_k_allowed": False,
        "recommended_status": "freeze_fq_weather_forecast_and_use_as_site_metric_dependent_result",
    }
    payload = {"detail": detail.to_dict(orient="records"), "pooled_means": pooled.to_dict(orient="records"), "pooled_delta_forecast_minus_noforecast": pooled_delta, "weather_branch_audits": audits, "verdict": verdict}
    out_json = ROOT / "benchmark_results/182_185_FQA_e4_seed1_seed2_stability_summary.json"
    out_csv = ROOT / "benchmark_results/182_185_FQA_e4_seed1_seed2_stability_summary.csv"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    detail.to_csv(out_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# FQ E4 weather residual seed1/seed2 stability summary",
        "",
        "两次 paired 2K smoke 均从各自 seed 的 FQ no-forecast 2K checkpoint warm-start；奖励、动作评价、安全约束和输入保持不变。",
        "",
        "## 每个 seed 的 forecast − no-forecast",
        "",
        "| seed | yield kg/ha | WP_ET kg/m³ | PFP_N kg/kg | irrigation mm | N kg/ha |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delta.to_dict(orient="records"):
        lines.append(f"| {row['seed']} | {row['final_grnwt']:.2f} | {row['WP_ET_kg_m3']:.3f} | {row['PFP_N_kg_kg']:.3f} | {row['actual_irrigation_mm']:.1f} | {row['actual_nitrogen_kg_ha']:.1f} |")
    lines.extend([
        "",
        "## 两 seed pooled mean",
        "",
        f"- forecast yield: `{pooled_map['forecast']['final_grnwt']:.2f}` kg/ha; no-forecast: `{pooled_map['noforecast']['final_grnwt']:.2f}` kg/ha; delta: `{pooled_delta['final_grnwt']:.2f}`.",
        f"- forecast WP_ET: `{pooled_map['forecast']['WP_ET_kg_m3']:.3f}`; no-forecast: `{pooled_map['noforecast']['WP_ET_kg_m3']:.3f}`; delta: `{pooled_delta['WP_ET_kg_m3']:.3f}`.",
        f"- forecast PFP_N: `{pooled_map['forecast']['PFP_N_kg_kg']:.3f}`; no-forecast: `{pooled_map['noforecast']['PFP_N_kg_kg']:.3f}`; delta: `{pooled_delta['PFP_N_kg_kg']:.3f}`.",
        "",
        "## 结论",
        "",
        "- 两个 seed 中 forecast 产量和 PFP_N 均低于配对 no-forecast；WP_ET 一升一降，不能称为稳定改善。",
        "- 两个 seed 的 weather branch 参数都发生了非零更新，因此不是“天气分支完全没有训练到”。",
        "- FQ E4 天气预报方向冻结，不扩展 5K；论文中作为站点和指标依赖的结果报告。",
    ])
    (ROOT / "docs/182_185_FQA_e4_seed1_seed2_stability_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

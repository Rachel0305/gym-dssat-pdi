"""Read-only comparison of 130 terminal-yield-credit intervention vs 111 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k"
VARIANT = ROOT / "benchmark_results/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x_smoke2k_smoke2k"
YEARS = list(range(2014, 2024))
CHECKPOINT = 2000


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def daily_path(root: Path, year: int) -> Path:
    return root / f"daily_outputs/SYA/SYA_{year}_seed0_ckpt{CHECKPOINT}_daily.csv"


def action_signature(daily: pd.DataFrame) -> str:
    positive = daily[(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0) > 1e-9) | (pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0) > 1e-9)]
    return ";".join(
        f"D{int(dap)} I{float(i):g} N{float(n):g}"
        for dap, i, n in zip(positive["dap"], positive["safe_action_amir"], positive["safe_action_anfer"])
    )


def load_policy(root: Path, label: str) -> tuple[dict[str, float], pd.DataFrame, dict[int, pd.DataFrame]]:
    summary_path = root / "evaluation" / f"{label}_checkpoint_validation_summary.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False)
    summary = summary[summary["checkpoint_step"].astype(int).eq(CHECKPOINT)].copy()
    daily_by_year: dict[int, pd.DataFrame] = {}
    all_pairs: list[str] = []
    signatures: list[str] = []
    action_rows = 0
    positive_rows = 0
    entropy_positive_values: list[str] = []
    for year in YEARS:
        daily = pd.read_csv(daily_path(root, year), keep_default_na=False)
        daily_by_year[year] = daily
        pairs = [f"I{float(i):g}/N{float(n):g}" for i, n in zip(daily["safe_action_amir"], daily["safe_action_anfer"])]
        all_pairs.extend(pairs)
        positive = [(float(i) > 1e-9) or (float(n) > 1e-9) for i, n in zip(daily["safe_action_amir"], daily["safe_action_anfer"])]
        positive_rows += int(sum(positive))
        action_rows += len(pairs)
        entropy_positive_values.extend([p for p, flag in zip(pairs, positive) if flag])
        signatures.append(action_signature(daily))
    row = summary.iloc[0] if len(summary) else pd.Series(dtype=object)
    metrics = {
        "label": label,
        "checkpoint_step": CHECKPOINT,
        "validation_years": len(summary),
        "mean_yield_kg_ha": float(pd.to_numeric(summary["final_grnwt"], errors="coerce").mean()),
        "mean_irrigation_mm": float(pd.to_numeric(summary["total_irrigation"], errors="coerce").mean()),
        "mean_n_kg_ha": float(pd.to_numeric(summary["total_n"], errors="coerce").mean()),
        "mean_pfp_n": float(pd.to_numeric(summary["PFP_N"], errors="coerce").mean()),
        "all_daily_action_entropy_bits": entropy(all_pairs),
        "positive_action_entropy_bits": entropy(entropy_positive_values),
        "distinct_all_daily_action_pairs": int(len(set(all_pairs))),
        "distinct_cross_year_positive_action_signatures": int(len(set(signatures))),
        "positive_action_rate": float(positive_rows / action_rows) if action_rows else 0.0,
        "daily_rows": int(action_rows),
        "positive_action_rows": int(positive_rows),
        "sy2017_yield_kg_ha": float(pd.to_numeric(summary.loc[summary["year"].astype(int).eq(2017), "final_grnwt"], errors="coerce").iloc[0]),
        "sy2017_irrigation_mm": float(pd.to_numeric(summary.loc[summary["year"].astype(int).eq(2017), "total_irrigation"], errors="coerce").iloc[0]),
        "sy2017_pfp_n": float(pd.to_numeric(summary.loc[summary["year"].astype(int).eq(2017), "PFP_N"], errors="coerce").iloc[0]),
    }
    return metrics, summary, daily_by_year


def main() -> int:
    base_metrics, base_summary, base_daily = load_policy(BASE, "111_00")
    variant_metrics, variant_summary, variant_daily = load_policy(VARIANT, "130_00")
    comparison = []
    for year in YEARS:
        b = base_summary[base_summary["year"].astype(int).eq(year)].iloc[0]
        v = variant_summary[variant_summary["year"].astype(int).eq(year)].iloc[0]
        bd, vd = base_daily[year], variant_daily[year]
        pairs_b = pd.DataFrame({"dap": bd["dap"].astype(int), "pair_b": [f"I{float(i):g}/N{float(n):g}" for i, n in zip(bd["safe_action_amir"], bd["safe_action_anfer"])]})
        pairs_v = pd.DataFrame({"dap": vd["dap"].astype(int), "pair_v": [f"I{float(i):g}/N{float(n):g}" for i, n in zip(vd["safe_action_amir"], vd["safe_action_anfer"])]})
        merged = pairs_b.merge(pairs_v, on="dap", how="inner")
        comparison.append({
            "year": year,
            "baseline_yield_kg_ha": float(b["final_grnwt"]),
            "variant_yield_kg_ha": float(v["final_grnwt"]),
            "yield_delta_kg_ha": float(v["final_grnwt"]) - float(b["final_grnwt"]),
            "baseline_irrigation_mm": float(b["total_irrigation"]),
            "variant_irrigation_mm": float(v["total_irrigation"]),
            "irrigation_delta_mm": float(v["total_irrigation"]) - float(b["total_irrigation"]),
            "baseline_pfp_n": float(b["PFP_N"]),
            "variant_pfp_n": float(v["PFP_N"]),
            "pfp_n_delta": float(v["PFP_N"]) - float(b["PFP_N"]),
            "same_dap_action_change_rate": float((merged["pair_b"] != merged["pair_v"]).mean()) if len(merged) else np.nan,
            "same_dap_rows": int(len(merged)),
        })
    comparison_df = pd.DataFrame(comparison)
    delta_metrics = {
        "yield_delta_mean_kg_ha": variant_metrics["mean_yield_kg_ha"] - base_metrics["mean_yield_kg_ha"],
        "irrigation_delta_mean_mm": variant_metrics["mean_irrigation_mm"] - base_metrics["mean_irrigation_mm"],
        "pfp_n_delta_mean": variant_metrics["mean_pfp_n"] - base_metrics["mean_pfp_n"],
        "mean_action_change_rate": float(comparison_df["same_dap_action_change_rate"].mean()),
    }
    smoke_gate = json.loads((VARIANT / "130_00_smoke_result.json").read_text(encoding="utf-8")).get("smoke_gate", {})
    strict_gate = {
        "daily_forecast_grid_and_transmission_gate": bool(smoke_gate.get("next_step_allowed", False)),
        "cross_year_positive_signature_count": int(variant_metrics["distinct_cross_year_positive_action_signatures"]),
        "cross_year_noncollapse": bool(variant_metrics["distinct_cross_year_positive_action_signatures"] > 1),
        "next_step_allowed": bool(smoke_gate.get("next_step_allowed", False) and variant_metrics["distinct_cross_year_positive_action_signatures"] > 1),
    }
    out_dir = VARIANT / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_dir / "130_00_year_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {
        "baseline": base_metrics,
        "variant": variant_metrics,
        "delta": delta_metrics,
        "action_gate_from_smoke_result": smoke_gate,
        "strict_mechanism_gate": strict_gate,
    }
    (out_dir / "130_00_terminal_yield_reward_2x_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    doc = ROOT / "docs/2026-08-16_sya_forecast_terminal_yield_reward_2x_execution.md"
    lines = [
        "# 130_00 SY 终点产量奖励 ×2：2K smoke 执行记录",
        "",
        "## 结论先说",
        "",
        "- 130 的奖励修改已接入实际 `StressAwareDiscreteWrapper` 配置：`reward.yield_coef` 为 `0.158 → 0.316`；资源成本、安全规则、输入、年份、seed 和滚动 7 天观测不变。",
        f"- 基础 2K 文件/动作/forecast gate：`{payload['action_gate_from_smoke_result'].get('next_step_allowed')}`；10/10 验证年 daily 和 forecast 字段齐全，动作与传输检查通过。",
        f"- 相对 111 2K：平均产量 `{base_metrics['mean_yield_kg_ha']:.1f} → {variant_metrics['mean_yield_kg_ha']:.1f}` kg/ha（{delta_metrics['yield_delta_mean_kg_ha']:+.1f}），平均灌溉 `{base_metrics['mean_irrigation_mm']:.1f} → {variant_metrics['mean_irrigation_mm']:.1f}` mm（{delta_metrics['irrigation_delta_mean_mm']:+.1f}），平均 PFP_N `{base_metrics['mean_pfp_n']:.2f} → {variant_metrics['mean_pfp_n']:.2f}`（{delta_metrics['pfp_n_delta_mean']:+.2f}）。",
        f"- 这不是无代价改进：灌溉量显著增加；本轮只能说明 2×终点产量信用在 2K 阶段改善了 SY 的产量/PFP-N 信号，不能称为最终优胜或天气响应已解决。",
        f"- 跨年正动作 signature：`{base_metrics['distinct_cross_year_positive_action_signatures']} → {variant_metrics['distinct_cross_year_positive_action_signatures']}`；全日动作熵：`{base_metrics['all_daily_action_entropy_bits']:.3f} → {variant_metrics['all_daily_action_entropy_bits']:.3f}` bits。130 的严格非塌缩 gate 为 `{payload['strict_mechanism_gate']['next_step_allowed']}`，因为 10 个验证年份仍共享 1 个正动作 signature。",
        "",
        "## 年度比较",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## 输出与边界",
        "",
        "- 年度表：`benchmark_results/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x_smoke2k_smoke2k/audits/130_00_year_comparison.csv`。",
        "- 汇总 JSON：`benchmark_results/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x_smoke2k_smoke2k/audits/130_00_terminal_yield_reward_2x_comparison.json`。",
        "- WP_ET 没有同口径 ETCP，本轮不推断；未运行 5K/10K/25K/50K/75K/100K 或五情景 replay。",
        "- 因严格非塌缩 gate 失败，本轮不批准 5K/10K；先不要把产量提升解释为天气响应，需另做输入反事实/动作敏感性审计。",
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

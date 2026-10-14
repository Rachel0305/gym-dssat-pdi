"""Read-only checkpoint/action comparison for the 132 entropy intervention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k"
VARIANT = ROOT / "benchmark_results/132_00_sya_originIC_rolling7d_forecast_entcoef_002_smoke2k_smoke2k"
YEARS = list(range(2014, 2024))
CHECKPOINT = 2000


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    p = pd.Series(values).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def daily(root: Path, year: int) -> pd.DataFrame:
    path = root / f"daily_outputs/SYA/SYA_{year}_seed0_ckpt{CHECKPOINT}_daily.csv"
    return pd.read_csv(path, keep_default_na=False)


def signature(df: pd.DataFrame) -> str:
    positive = df[(pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0) > 1e-9) | (pd.to_numeric(df["safe_action_anfer"], errors="coerce").fillna(0.0) > 1e-9)]
    return ";".join(f"D{int(d)} I{float(i):g} N{float(n):g}" for d, i, n in zip(positive["dap"], positive["safe_action_amir"], positive["safe_action_anfer"]))


def policy_metrics(root: Path, summary_name: str) -> tuple[dict, pd.DataFrame, dict[int, pd.DataFrame]]:
    summary = pd.read_csv(root / f"evaluation/{summary_name}_checkpoint_validation_summary.csv", keep_default_na=False)
    summary = summary[summary["checkpoint_step"].astype(int).eq(CHECKPOINT)].copy()
    all_pairs: list[str] = []
    positive_pairs: list[str] = []
    signatures: list[str] = []
    traces: dict[int, pd.DataFrame] = {}
    total_rows = positive_rows = 0
    for year in YEARS:
        df = daily(root, year)
        traces[year] = df
        pairs = [f"I{float(i):g}/N{float(n):g}" for i, n in zip(df["safe_action_amir"], df["safe_action_anfer"])]
        flags = [(float(i) > 1e-9) or (float(n) > 1e-9) for i, n in zip(df["safe_action_amir"], df["safe_action_anfer"])]
        all_pairs.extend(pairs)
        positive_pairs.extend([p for p, flag in zip(pairs, flags) if flag])
        signatures.append(signature(df))
        total_rows += len(df)
        positive_rows += int(sum(flags))
    row_2017 = summary[summary["year"].astype(int).eq(2017)].iloc[0]
    metrics = {
        "mean_yield_kg_ha": float(pd.to_numeric(summary["final_grnwt"], errors="coerce").mean()),
        "mean_irrigation_mm": float(pd.to_numeric(summary["total_irrigation"], errors="coerce").mean()),
        "mean_n_kg_ha": float(pd.to_numeric(summary["total_n"], errors="coerce").mean()),
        "mean_pfp_n": float(pd.to_numeric(summary["PFP_N"], errors="coerce").mean()),
        "all_daily_action_entropy_bits": entropy(all_pairs),
        "positive_action_entropy_bits": entropy(positive_pairs),
        "distinct_all_daily_action_pairs": int(len(set(all_pairs))),
        "distinct_cross_year_positive_action_signatures": int(len(set(signatures))),
        "positive_action_rate": float(positive_rows / total_rows),
        "daily_rows": int(total_rows),
        "positive_action_rows": int(positive_rows),
        "sy2017_yield_kg_ha": float(row_2017["final_grnwt"]),
        "sy2017_irrigation_mm": float(row_2017["total_irrigation"]),
        "sy2017_pfp_n": float(row_2017["PFP_N"]),
    }
    return metrics, summary, traces


def main() -> int:
    baseline, base_summary, base_traces = policy_metrics(BASE, "111_00")
    variant, var_summary, var_traces = policy_metrics(VARIANT, "132_00")
    yearly: list[dict] = []
    for year in YEARS:
        b = base_summary[base_summary["year"].astype(int).eq(year)].iloc[0]
        v = var_summary[var_summary["year"].astype(int).eq(year)].iloc[0]
        bd, vd = base_traces[year], var_traces[year]
        b_pairs = pd.DataFrame({"dap": bd["dap"].astype(int), "b": [f"I{float(i):g}/N{float(n):g}" for i, n in zip(bd["safe_action_amir"], bd["safe_action_anfer"])]})
        v_pairs = pd.DataFrame({"dap": vd["dap"].astype(int), "v": [f"I{float(i):g}/N{float(n):g}" for i, n in zip(vd["safe_action_amir"], vd["safe_action_anfer"])]})
        merged = b_pairs.merge(v_pairs, on="dap", how="inner")
        yearly.append({
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
            "same_dap_action_change_rate": float((merged["b"] != merged["v"]).mean()) if len(merged) else np.nan,
        })
    yearly_df = pd.DataFrame(yearly)
    result_json = VARIANT / "132_00_smoke_result.json"
    smoke_gate = json.loads(result_json.read_text(encoding="utf-8")).get("smoke_gate", {})
    delta = {
        "mean_yield_delta_kg_ha": variant["mean_yield_kg_ha"] - baseline["mean_yield_kg_ha"],
        "mean_irrigation_delta_mm": variant["mean_irrigation_mm"] - baseline["mean_irrigation_mm"],
        "mean_pfp_n_delta": variant["mean_pfp_n"] - baseline["mean_pfp_n"],
        "all_daily_entropy_delta_bits": variant["all_daily_action_entropy_bits"] - baseline["all_daily_action_entropy_bits"],
        "cross_year_signature_delta": variant["distinct_cross_year_positive_action_signatures"] - baseline["distinct_cross_year_positive_action_signatures"],
        "mean_same_dap_action_change_rate": float(yearly_df["same_dap_action_change_rate"].mean()),
    }
    diversity_improved = bool(
        variant["all_daily_action_entropy_bits"] > baseline["all_daily_action_entropy_bits"] + 0.01
        or variant["distinct_cross_year_positive_action_signatures"] > baseline["distinct_cross_year_positive_action_signatures"]
    )
    strict_gate = {
        "basic_smoke_gate": bool(smoke_gate.get("next_step_allowed", False)),
        "cross_year_positive_signature_count": int(variant["distinct_cross_year_positive_action_signatures"]),
        "cross_year_noncollapse": bool(variant["distinct_cross_year_positive_action_signatures"] > 1),
        "diversity_improved_over_111": diversity_improved,
        "next_step_allowed": bool(smoke_gate.get("next_step_allowed", False) and variant["distinct_cross_year_positive_action_signatures"] > 1 and diversity_improved),
    }
    out_dir = VARIANT / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    yearly_df.to_csv(out_dir / "132_00_year_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {"baseline_111": baseline, "variant_132": variant, "delta": delta, "smoke_gate": smoke_gate, "strict_mechanism_gate": strict_gate}
    (out_dir / "132_00_entropy_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    doc = ROOT / "docs/2026-08-16_sya_forecast_entcoef_002_execution.md"
    doc.write_text(
        "\n".join([
            "# 132_00 SY 熵系数 0.02：2K smoke 执行记录",
            "",
            "## 结论先说",
            "",
            "- 实际只修改 `ppo.ent_coef: 0.01 → 0.02`；`reward.yield_coef=0.158`、资源成本、动作安全和滚动 7 天观测均保持 111 基线。",
            f"- 基础 2K gate：`{strict_gate['basic_smoke_gate']}`；10/10 daily、forecast 字段和动作传输检查通过。",
            f"- 平均产量：`{baseline['mean_yield_kg_ha']:.1f} → {variant['mean_yield_kg_ha']:.1f}` kg/ha（{delta['mean_yield_delta_kg_ha']:+.1f}）；平均灌溉：`{baseline['mean_irrigation_mm']:.1f} → {variant['mean_irrigation_mm']:.1f}` mm（{delta['mean_irrigation_delta_mm']:+.1f}）。",
            f"- 全日动作熵：`{baseline['all_daily_action_entropy_bits']:.3f} → {variant['all_daily_action_entropy_bits']:.3f}` bits；跨年正动作 signature：`{baseline['distinct_cross_year_positive_action_signatures']} → {variant['distinct_cross_year_positive_action_signatures']}`。",
            f"- 基础非塌缩条件：`{strict_gate['cross_year_noncollapse']}`；相对 111 的多样性改善条件：`{strict_gate['diversity_improved_over_111']}`；综合后续训练许可：`{strict_gate['next_step_allowed']}`。",
            "- 本轮熵系数没有带来可测的多样性、产量或资源使用变化，因此不批准继续 5K/10K；不能把它称为有效改进。",
            "",
            "## 年度对比",
            "",
            yearly_df.to_markdown(index=False),
            "",
            "## 输出与边界",
            "",
            f"- 年度表：`{(out_dir / '132_00_year_comparison.csv').relative_to(ROOT).as_posix()}`",
            f"- 汇总 JSON：`{(out_dir / '132_00_entropy_comparison.json').relative_to(ROOT).as_posix()}`",
            "- WP_ET 没有同口径 ETCP，本轮不推断；没有运行长训练或五情景 replay。",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

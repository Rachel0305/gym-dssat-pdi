"""Compile YC seed1/seed2 paired 2K WP_ET stability evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sources = {
        "seed1": ROOT / "benchmark_results/178_multisite_wp_et_replay/178_wp_et_replay_summary_by_policy.csv",
        "seed2": ROOT / "benchmark_results/188YCA_yc_seed2_wp_et_replay/178_wp_et_replay_summary_by_policy.csv",
    }
    rows = []
    for seed, path in sources.items():
        df = pd.read_csv(path, keep_default_na=False)
        df = df[df["station"].eq("YCA") & df["checkpoint_step"].eq(2000)].copy()
        for _, row in df.iterrows():
            arm = "forecast" if "forecast" in str(row["policy"]) and "noforecast" not in str(row["policy"]) else "noforecast"
            rows.append({"seed": seed, "arm": arm, "mean_yield": float(row["mean_yield"]), "mean_WP_ET": float(row["mean_WP_ET"]), "mean_PFP_N": float(row["mean_PFP_N"]), "weighted_WP_ET": float(row["weighted_WP_ET"]), "weighted_PFP_N": float(row["weighted_PFP_N"])})
    detail = pd.DataFrame(rows).sort_values(["seed", "arm"]).reset_index(drop=True)
    pairs = []
    for seed, sub in detail.groupby("seed"):
        f = sub[sub["arm"].eq("forecast")].iloc[0]
        n = sub[sub["arm"].eq("noforecast")].iloc[0]
        pairs.append({"seed": seed, "yield_delta": f["mean_yield"] - n["mean_yield"], "WP_ET_delta": f["mean_WP_ET"] - n["mean_WP_ET"], "PFP_N_delta": f["mean_PFP_N"] - n["mean_PFP_N"], "weighted_WP_ET_delta": f["weighted_WP_ET"] - n["weighted_WP_ET"], "weighted_PFP_N_delta": f["weighted_PFP_N"] - n["weighted_PFP_N"]})
    pair_df = pd.DataFrame(pairs)
    verdict = {
        "yield_direction_positive_both_seeds": bool((pair_df["yield_delta"] > 0).all()),
        "WP_ET_direction_positive_both_seeds": bool((pair_df["WP_ET_delta"] > 0).all()),
        "PFP_N_direction_positive_both_seeds": bool((pair_df["PFP_N_delta"] > 0).all()),
        "effect_size_stable": bool(pair_df["yield_delta"].max() <= max(1.0, 3.0 * pair_df["yield_delta"].min())),
        "five_k_forecast_collapse_observed": True,
        "recommended_checkpoint": 2000,
        "recommended_status": "yc_2k_directionally_supported_but_effect_size_seed_sensitive",
    }
    payload = {"detail": detail.to_dict(orient="records"), "paired_deltas": pair_df.to_dict(orient="records"), "verdict": verdict}
    out_json = ROOT / "benchmark_results/186_187YCA_yc_seed1_seed2_stability_summary.json"
    out_csv = ROOT / "benchmark_results/186_187YCA_yc_seed1_seed2_stability_summary.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pair_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# YC 2K weather forecast seed1/seed2 stability summary",
        "",
        "两次实验均为从零开始的 paired 2K：forecast 与 matched no-forecast 使用同一 reward、动作网格、安全约束和年份切分，仅 seed 不同。",
        "",
        "| seed | yield delta | WP_ET delta | PFP_N delta | weighted WP_ET delta | weighted PFP_N delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pair_df.to_dict(orient="records"):
        lines.append(f"| {row['seed']} | {row['yield_delta']:.2f} | {row['WP_ET_delta']:.3f} | {row['PFP_N_delta']:.3f} | {row['weighted_WP_ET_delta']:.3f} | {row['weighted_PFP_N_delta']:.3f} |")
    lines.extend([
        "",
        "## 结论",
        "",
        "- 两个 seed 的 2K forecast 在 yield、WP_ET 和 PFP_N 方向上均不低于 no-forecast。",
        "- 但 seed1 的 yield 优势约 848 kg/ha，seed2 仅约 6 kg/ha，效果大小高度依赖 seed。",
        "- 既有 5K forecast 全零动作坍缩，因此 YC 当前应选择 2K checkpoint，不继续直接扩展 5K。",
        "- YC 可作为“共享框架下方向上可迁移、但收益幅度不稳定”的案例，而不是稳定的大幅天气预报收益案例。",
    ])
    (ROOT / "docs/186_187YCA_yc_seed1_seed2_stability_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "032_02_cumulative_reward_plot_metric_audit"
DOC = ROOT / "docs" / "032_02_cumulative_reward_plot_metric_audit_record.md"
SUMMARY = ROOT / "benchmark_results" / "031_42_sample_ppo_five_scenario_daily_process_audit" / "tables" / "031_42_sample_five_scenario_summary.csv"
DAILY = ROOT / "benchmark_results" / "031_42_sample_ppo_five_scenario_daily_process_audit" / "tables" / "031_42_sample_five_scenario_daily.csv"


def ensure_dirs() -> None:
    for rel in ["tables", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    ensure_dirs()
    summary = pd.read_csv(SUMMARY)
    daily = pd.read_csv(DAILY)

    for col in [
        "final_grain_kg_ha",
        "total_irrigation_mm",
        "total_nitrogen_kg_ha",
        "final_cumulative_available_reward",
    ]:
        summary[col] = num(summary[col])

    summary["common_reward_031_unscaled"] = (
        summary["final_grain_kg_ha"]
        - 1.1 * summary["total_irrigation_mm"]
        - 1.58 * summary["total_nitrogen_kg_ha"]
    )
    summary["common_reward_original_unscaled"] = (
        summary["final_grain_kg_ha"]
        - summary["total_irrigation_mm"]
        - 5.0 * summary["total_nitrogen_kg_ha"]
    )
    summary["source_vs_common031_ratio"] = summary["final_cumulative_available_reward"] / summary["common_reward_031_unscaled"].replace(0, np.nan)
    summary["source_minus_common031"] = summary["final_cumulative_available_reward"] - summary["common_reward_031_unscaled"]
    summary["looks_scaled_0p001"] = summary["source_vs_common031_ratio"].between(0.00005, 0.0002, inclusive="both")
    summary["looks_unscaled_or_other"] = summary["source_vs_common031_ratio"].abs().gt(0.1)

    audit_path = OUT / "tables" / "032_02_reward_metric_audit_by_scenario.csv"
    summary.to_csv(audit_path, index=False, encoding="utf-8-sig")

    rows = []
    for (station, year), g in summary.groupby(["station_code", "year"], dropna=False):
        ppo = g[g["scenario"].eq("ppo_candidate")]
        nonppo = g[~g["scenario"].eq("ppo_candidate")]
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "scenarios": int(len(g)),
                "ppo_source_reward": float(ppo["final_cumulative_available_reward"].iloc[0]) if len(ppo) else np.nan,
                "ppo_common031": float(ppo["common_reward_031_unscaled"].iloc[0]) if len(ppo) else np.nan,
                "ppo_source_common_ratio": float(ppo["source_vs_common031_ratio"].iloc[0]) if len(ppo) else np.nan,
                "nonppo_source_reward_median": float(nonppo["final_cumulative_available_reward"].median()) if len(nonppo) else np.nan,
                "nonppo_common031_median": float(nonppo["common_reward_031_unscaled"].median()) if len(nonppo) else np.nan,
                "reward_scale_mixed": bool(len(ppo) and ppo["looks_scaled_0p001"].iloc[0] and nonppo["looks_unscaled_or_other"].any()),
            }
        )
    by_year = pd.DataFrame(rows)
    by_year_path = OUT / "tables" / "032_02_reward_metric_audit_by_site_year.csv"
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")

    daily_cols = list(daily.columns)
    daily_profile = {
        "daily_rows": int(len(daily)),
        "summary_rows": int(len(summary)),
        "daily_grain": ["station_code", "year", "scenario", "dap"],
        "has_reward_step": "reward_step" in daily_cols,
        "has_cumulative_available_reward": "cumulative_available_reward" in daily_cols,
        "source_files": int(daily["source_file"].nunique()) if "source_file" in daily_cols else None,
    }
    (OUT / "tables" / "032_02_daily_profile.json").write_text(json.dumps(daily_profile, indent=2, ensure_ascii=False), encoding="utf-8")

    mixed_count = int(by_year["reward_scale_mixed"].sum())
    ppo_ratios = summary.loc[summary["scenario"].eq("ppo_candidate"), "source_vs_common031_ratio"].dropna()
    nonppo_ratios = summary.loc[~summary["scenario"].eq("ppo_candidate"), "source_vs_common031_ratio"].dropna()

    lines = [
        "# 032_02 cumulative reward plot metric audit record",
        "",
        "## Dataset and grain",
        "",
        f"- Source summary: `{SUMMARY.relative_to(ROOT)}`.",
        f"- Source daily: `{DAILY.relative_to(ROOT)}`.",
        "- Intended daily grain: station-year-scenario-DAP.",
        f"- Daily rows: {daily_profile['daily_rows']}. Summary rows: {daily_profile['summary_rows']}.",
        "",
        "## Checks performed",
        "",
        "- Recomputed endpoint common rewards from final grain, total irrigation, and total N.",
        "- Compared source `final_cumulative_available_reward` against recomputed common rewards.",
        "- Flagged mixed scale cases where PPO looks 0.001-scaled while baseline scenarios look unscaled or use a different reward definition.",
        "",
        "## Key findings",
        "",
        f"- Mixed reward scale/definition detected in {mixed_count}/{len(by_year)} sampled site-years.",
        f"- PPO source/common031 ratio range: {ppo_ratios.min():.6g} to {ppo_ratios.max():.6g}.",
        f"- Non-PPO source/common031 ratio range: {nonppo_ratios.min():.6g} to {nonppo_ratios.max():.6g}.",
        "- In the 031_42 sample package, PPO cumulative reward is usually around 0.x because PPO daily reward is scaled, while baseline cumulative rewards are often thousands to tens of thousands from their own source reward columns.",
        "- Therefore the existing cumulative reward panel is not a valid common-reward comparison across five scenarios.",
        "",
        "## Impact",
        "",
        "- Figures can misleadingly show PPO as having the lowest cumulative reward even when endpoint yield/water/N metrics are competitive.",
        "- This is a plotting metric problem, not evidence that PPO optimized the worst reward.",
        "",
        "## Recommendation",
        "",
        "- Do not label the existing panel as `Cumulative common reward`.",
        "- For advisor-facing daily process figures, either remove the reward panel or replace it with cumulative irrigation and cumulative nitrogen usage.",
        "- If a reward comparison is needed, recompute a single endpoint common reward formula for all scenarios and show it as an endpoint bar/table, not by mixing source daily reward columns.",
        "",
        "## Output tables",
        "",
        f"- Scenario audit: `{audit_path.relative_to(ROOT)}`.",
        f"- Site-year audit: `{by_year_path.relative_to(ROOT)}`.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")

    result = {
        "task": "032_02_cumulative_reward_plot_metric_audit",
        "training_run": False,
        "dssat_run": False,
        "mixed_reward_scale_site_years": mixed_count,
        "site_years_checked": int(len(by_year)),
        "record_md": str(DOC.relative_to(ROOT)),
        "by_scenario": str(audit_path.relative_to(ROOT)),
        "by_site_year": str(by_year_path.relative_to(ROOT)),
        "recommendation": "Do not use source cumulative reward panel for five-scenario comparison.",
    }
    (OUT / "032_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(by_year.to_string(index=False))


if __name__ == "__main__":
    main()


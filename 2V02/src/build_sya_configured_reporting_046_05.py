"""046_05: configuration-driven SYA five-scenario metric report.

The auto row deliberately comes from 046_04: DSSAT native automatic irrigation
plus an auditable, external NSTRES-triggered nitrogen rule.  It therefore has a
valid PFP_N whenever DSSAT records actual nitrogen greater than zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sya_experiment_artifacts import resolve_validation_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "046_02_sya_originIC_binary_timing_ppo.json"
PROMPT = ROOT / "prompts" / "046_05_sya_configured_reporting.md"
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto irrigation + external N rule",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO",
}
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto_external_n": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "rl_candidate": "#2A9D55",
}
METRIC_COLUMNS = ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]
RESOURCE_COLUMNS = ["actual_irrigation_mm", "actual_nitrogen_kg_ha"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def external_auto_root(cfg: dict, run_id: str = "") -> Path:
    override = str(cfg.get("figure_package", {}).get("auto_output_root", "")).strip()
    if override:
        path = ROOT / override
        return path if path.is_absolute() else (ROOT / override)
    suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    name = f"046_04_sya_{cfg['input_profile']}_external_auto_n_rule"
    if suffix:
        name = f"{name}_{suffix}"
    name = f"{name}{run_suffix(run_id)}"
    return ROOT / "benchmark_results" / name


def daily_report_root(cfg: dict, checkpoint: int, run_id: str = "") -> Path:
    auto_suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    auto_part = f"_auto_{auto_suffix}" if auto_suffix else ""
    return ROOT / "benchmark_results" / f"046_06_sya_{cfg['input_profile']}_{cfg['task_id']}_{cfg['task_name']}{auto_part}_five_scenario_daily_ckpt{checkpoint}{run_suffix(run_id)}"


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def draw_grouped(combined: pd.DataFrame, years: list[int], metrics: list[str], labels: dict[str, str], colors: dict[str, str], path: Path) -> None:
    x = np.arange(len(years))
    width = 0.16
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for idx, scenario in enumerate(SCENARIOS):
            values = []
            for year in years:
                row = combined[(combined["year"].eq(year)) & (combined["scenario"].eq(scenario))]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (idx - 2) * width, values, width, label=labels[scenario], color=colors[scenario])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(years)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--ppo-run-dir", type=Path, default=None, help="Optional PPO output directory; default is derived from task_id/task_name in config")
    parser.add_argument("--run-id", type=str, default="", help="Optional run batch id; reads matching 046_03/046_04 outputs and writes a separate report")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    cfg = read_config(cfg_path)
    profile = str(cfg["input_profile"])
    checkpoint = int(args.checkpoint or cfg.get("report_checkpoint", 25_000))
    baseline_root = ROOT / "benchmark_results" / f"046_03_sya_{profile}_four_baselines{run_suffix(args.run_id)}"
    auto_root = external_auto_root(cfg, args.run_id)
    ppo_root = args.ppo_run_dir if args.ppo_run_dir else ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"
    if not ppo_root.is_absolute():
        ppo_root = (Path.cwd() / ppo_root).resolve()
    baseline_path = baseline_root / "evaluation" / "046_03_baseline_summary.csv"
    auto_path = auto_root / "evaluation" / "046_04_external_auto_n_summary.csv"
    absent = [rel(path) for path in [baseline_path, auto_path] if not path.exists()]
    if absent:
        raise FileNotFoundError("Required 046_02/03/04 outputs are missing: " + "; ".join(absent))

    # A report is derived from one concrete PPO run.  Include its task identity
    # so different parameter trials cannot overwrite each other's figures.
    auto_suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    auto_part = f"_auto_{auto_suffix}" if auto_suffix else ""
    out = ROOT / "benchmark_results" / f"046_05_sya_{profile}_{cfg['task_id']}_{cfg['task_name']}{auto_part}_reporting_ckpt{checkpoint}{run_suffix(args.run_id)}"
    fig_dir, tab_dir, cfg_dir = out / "figures", out / "tables", out / "configs"
    for directory in [fig_dir, tab_dir, cfg_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, cfg_dir / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, cfg_dir / PROMPT.name)

    ppo_path = resolve_validation_summary(ppo_root, checkpoint, list(map(int, cfg["scope"]["validation_years"])), preferred_prefix=str(cfg["task_id"]))
    baseline = pd.read_csv(baseline_path, keep_default_na=False)
    auto = pd.read_csv(auto_path, keep_default_na=False)
    ppo = pd.read_csv(ppo_path, keep_default_na=False)
    require_columns(baseline, ["year", "scenario", *METRIC_COLUMNS, *RESOURCE_COLUMNS], "046_03 baseline")
    require_columns(auto, ["station_code", "site", "year", *METRIC_COLUMNS, *RESOURCE_COLUMNS], "046_04 external auto")
    require_columns(ppo, ["station_code", "year", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n"], "046_02 PPO")

    validation_years = sorted(map(int, cfg["scope"]["validation_years"]))
    ppo = ppo[pd.to_numeric(ppo["checkpoint_step"], errors="coerce").eq(checkpoint)].copy()
    if sorted(pd.to_numeric(ppo["year"], errors="coerce").dropna().astype(int).tolist()) != validation_years:
        raise RuntimeError(f"PPO checkpoint {checkpoint} does not cover exactly the configured validation years")
    auto_years = sorted(pd.to_numeric(auto["year"], errors="coerce").dropna().astype(int).tolist())
    if auto_years != validation_years:
        raise RuntimeError(f"046_04 external-auto coverage mismatch: expected={validation_years}, got={auto_years}")

    base_rows = baseline[baseline["scenario"].isin(["null", "recorded_farmer_template", "official_extension_expert"])].copy()
    auto_rows = auto[["station_code", "site", "year", *METRIC_COLUMNS, *RESOURCE_COLUMNS]].copy()
    auto_rows["scenario"] = "dssat_auto_external_n"
    auto_n = pd.to_numeric(auto_rows["actual_nitrogen_kg_ha"], errors="coerce")
    auto_y = pd.to_numeric(auto_rows["grain_yield_kg_ha"], errors="coerce")
    auto_pfp = pd.to_numeric(auto_rows["PFP_N_kg_kg"], errors="coerce")
    auto_rows["PFP_N_kg_kg"] = auto_pfp.where(auto_pfp.notna(), auto_y / auto_n.where(auto_n.gt(0)))
    ppo_rows = pd.DataFrame({
        "station_code": ppo["station_code"],
        "site": ppo.get("site", "SY"),
        "year": pd.to_numeric(ppo["year"], errors="coerce").astype(int),
        "scenario": "rl_candidate",
        "grain_yield_kg_ha": pd.to_numeric(ppo["final_grnwt"], errors="coerce"),
        "actual_irrigation_mm": pd.to_numeric(ppo["total_irrigation"], errors="coerce"),
        "actual_nitrogen_kg_ha": pd.to_numeric(ppo["total_n"], errors="coerce"),
        "WP_ET_kg_m3": pd.to_numeric(ppo.get("WP_ET", ppo.get("WP_ET_kg_m3")), errors="coerce"),
        "PFP_N_kg_kg": pd.to_numeric(ppo.get("PFP_N", ppo.get("PFP_N_kg_kg")), errors="coerce"),
        "checkpoint_step": checkpoint,
    })
    ppo_metric_source = "046_02_validation_summary"
    replay_summary_path = daily_report_root(cfg, checkpoint, args.run_id) / "tables" / "046_06_sya_five_scenario_season_summary.csv"
    if replay_summary_path.exists():
        replay = pd.read_csv(replay_summary_path, keep_default_na=False)
        exact = replay[replay["scenario"].astype(str).eq("rl_candidate")].copy()
        if len(exact):
            exact["year"] = pd.to_numeric(exact["year"], errors="coerce").astype(int)
            exact_cols = ["year", "WP_ET_kg_m3", "PFP_N_kg_kg", "irrigation_mm", "nitrogen_kg_ha"]
            missing_exact = [column for column in exact_cols if column not in exact.columns]
            if missing_exact:
                ppo_metric_source = f"046_02_validation_summary; ignored older 046_06 summary missing {missing_exact}"
            else:
                exact = exact[exact_cols].rename(columns={
                    "WP_ET_kg_m3": "WP_ET_kg_m3_exact",
                    "PFP_N_kg_kg": "PFP_N_kg_kg_exact",
                    "irrigation_mm": "actual_irrigation_mm_exact",
                    "nitrogen_kg_ha": "actual_nitrogen_kg_ha_exact",
                })
                ppo_rows = ppo_rows.merge(exact, on="year", how="left", validate="one_to_one")
                for column in ["WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]:
                    ppo_rows[column] = pd.to_numeric(ppo_rows[f"{column}_exact"], errors="coerce").combine_first(pd.to_numeric(ppo_rows[column], errors="coerce"))
                ppo_rows = ppo_rows.drop(columns=[column for column in ppo_rows.columns if column.endswith("_exact")])
                ppo_metric_source = rel(replay_summary_path)
    combined = pd.concat([base_rows, auto_rows, ppo_rows], ignore_index=True, sort=False)
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype(int)
    for column in [*METRIC_COLUMNS, *RESOURCE_COLUMNS]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    # PFP_N is mathematically undefined when actual N is zero. Keep it NaN,
    # rather than plotting an artificial zero or comparing it against PPO.
    combined.loc[combined["actual_nitrogen_kg_ha"].fillna(0).le(0), "PFP_N_kg_kg"] = np.nan
    expected_rows = len(validation_years) * len(SCENARIOS)
    if len(combined) != expected_rows:
        raise RuntimeError(f"Five-scenario row count mismatch: expected {expected_rows}, got {len(combined)}")
    duplicate = combined.duplicated(["year", "scenario"], keep=False)
    if duplicate.any():
        raise RuntimeError("Duplicate year/scenario records detected in assembled report")

    gaps = []
    for year, subset in combined.groupby("year"):
        candidate = subset[subset["scenario"].eq("rl_candidate")].iloc[0]
        base_subset = subset[subset["scenario"].ne("rl_candidate")]
        for metric in METRIC_COLUMNS:
            maximum = pd.to_numeric(base_subset[metric], errors="coerce").max(skipna=True)
            gaps.append({"year": int(year), "metric": metric, "ppo_value": candidate[metric], "four_baseline_max": maximum, "ppo_minus_four_max": candidate[metric] - maximum if pd.notna(maximum) else np.nan})
    gaps_df = pd.DataFrame(gaps)
    combined.to_csv(tab_dir / "046_05_five_scenario_metric_summary.csv", index=False, encoding="utf-8-sig")
    gaps_df.to_csv(tab_dir / "046_05_metric_gaps_vs_four_baseline_max.csv", index=False, encoding="utf-8-sig")
    draw_grouped(combined, validation_years, METRIC_COLUMNS, LABELS, COLORS, fig_dir / "046_05_five_scenario_metrics.png")
    draw_grouped(combined, validation_years, RESOURCE_COLUMNS, LABELS, COLORS, fig_dir / "046_05_five_scenario_management.png")

    pfp_auto = combined[combined["scenario"].eq("dssat_auto_external_n")][["year", "actual_nitrogen_kg_ha", "PFP_N_kg_kg"]]
    doc = ROOT / "docs" / f"046_05_sya_{profile}_{cfg['task_id']}_{cfg['task_name']}_reporting_ckpt{checkpoint}_record.md"
    wins = gaps_df.groupby("metric")["ppo_minus_four_max"].apply(lambda values: int((values > 0).sum())).to_dict()
    doc.write_text("\n".join([
        f"# 046_05 SYA {profile} 五情景统一指标汇总",
        "",
        f"- PPO checkpoint: `{checkpoint}`。",
        "- auto 行替换为 046_04：DSSAT 原生自动灌溉 + 外部 NSTRES 触发施氮；不再使用施氮为 0 的 native-auto 行。",
        "- PFP_N 按实际施氮量计算；实际 N=0 时为未定义（N/A），不写成 0。",
        f"- PPO 超过四基线最高值的年份数：{wins}。",
        "- 产量为籽粒产量 GRNWT/HWAM；生物量不参与本指标比较。",
        "",
        "## external-auto PFP_N 审计",
        "",
        pfp_auto.to_markdown(index=False),
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "task": "046_05_sya_configured_reporting",
        "ppo_validation_source": rel(ppo_path),
        "ppo_metric_source": ppo_metric_source,
        "auto_source": rel(auto_path),
        "combined_summary": rel(tab_dir / "046_05_five_scenario_metric_summary.csv"),
        "gaps": rel(tab_dir / "046_05_metric_gaps_vs_four_baseline_max.csv"),
        "figures": [rel(fig_dir / "046_05_five_scenario_metrics.png"), rel(fig_dir / "046_05_five_scenario_management.png")],
        "record_md": rel(doc),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

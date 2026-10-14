from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "032_14_lc_75k_ppo_four_baseline_success_figures"
FIG = OUT / "figures"
TAB = OUT / "tables"
DOC = ROOT / "docs" / "032_14_lc_75k_ppo_four_baseline_success_figures_record.md"

BASELINE = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
TRAIN_EVAL = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "evaluation" / "032_11_train_year_checkpoint_eval_summary.csv"
TRANSFER_EVAL = ROOT / "benchmark_results" / "032_12_lc_multiyear_75k_future_year_transfer" / "evaluation" / "032_12_lc2011_2020_frozen_75k_vs_baselines.csv"

FOUR = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_baselines() -> pd.DataFrame:
    base = pd.read_csv(BASELINE)
    base = base[(base["station_code"].eq("LCA")) & (base["scenario"].isin(FOUR))].copy()
    for col in [
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    return base


def load_rl() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_EVAL)
    train = train[pd.to_numeric(train["checkpoint_step"], errors="coerce").eq(75000)].copy()
    train = train.rename(
        columns={
            "final_grnwt": "rl_yield_kg_ha",
            "total_irrigation": "rl_irrigation_mm",
            "total_n": "rl_nitrogen_kg_ha",
            "PFP_N": "rl_PFP_N_kg_kg",
            "max_swfac": "rl_max_swfac",
            "max_nstres": "rl_max_nstres",
        }
    )
    train["split"] = "train_2005_2010"

    val = pd.read_csv(TRANSFER_EVAL)
    val = val.rename(
        columns={
            "rl_action_sequence": "action_sequence",
            "rl_max_water_stress": "rl_max_swfac",
            "rl_max_nitrogen_stress": "rl_max_nstres",
        }
    )
    val["split"] = "transfer_2011_2020"

    keep = [
        "split",
        "year",
        "rl_yield_kg_ha",
        "rl_irrigation_mm",
        "rl_nitrogen_kg_ha",
        "rl_PFP_N_kg_kg",
        "rl_max_swfac",
        "rl_max_nstres",
        "action_sequence",
    ]
    out = pd.concat([train[keep], val[keep]], ignore_index=True)
    for col in ["year", "rl_yield_kg_ha", "rl_irrigation_mm", "rl_nitrogen_kg_ha", "rl_PFP_N_kg_kg", "rl_max_swfac", "rl_max_nstres"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["year"] = out["year"].astype(int)
    return out.sort_values(["year"]).reset_index(drop=True)


def compare(rl: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in rl.itertuples(index=False):
        b = base[base["year"].astype(int).eq(int(row.year))].copy()
        expert = b[b["scenario"].eq("official_extension_expert")]
        ex = expert.iloc[0] if not expert.empty else None
        four_max_yield = float(b["grain_yield_kg_ha"].max())
        four_max_pfp = float(b["PFP_N_kg_kg"].max(skipna=True))
        four_max_wp = float(b["WP_ET_kg_m3"].max(skipna=True)) if b["WP_ET_kg_m3"].notna().any() else np.nan
        rl_y = float(row.rl_yield_kg_ha)
        rl_pfp = float(row.rl_PFP_N_kg_kg)
        item = {
            "site": "LC",
            "station_code": "LCA",
            "split": row.split,
            "year": int(row.year),
            "rl_yield_kg_ha": rl_y,
            "four_max_yield_kg_ha": four_max_yield,
            "yield_gap_vs_four_max_kg_ha": rl_y - four_max_yield,
            "rl_PFP_N_kg_kg": rl_pfp,
            "four_max_PFP_N_kg_kg": four_max_pfp,
            "PFP_N_gap_vs_four_max": rl_pfp - four_max_pfp,
            "four_max_WP_ET_kg_m3": four_max_wp,
            "rl_WP_ET_kg_m3": np.nan,
            "WP_ET_gap_vs_four_max": np.nan,
            "rl_irrigation_mm": float(row.rl_irrigation_mm),
            "rl_nitrogen_kg_ha": float(row.rl_nitrogen_kg_ha),
            "rl_max_swfac": float(row.rl_max_swfac),
            "rl_max_nstres": float(row.rl_max_nstres),
            "action_sequence": row.action_sequence,
            "four_baseline_count": int(len(b)),
        }
        if ex is not None:
            item.update(
                {
                    "expert_yield_kg_ha": float(ex["grain_yield_kg_ha"]),
                    "expert_irrigation_mm": float(ex["actual_irrigation_mm"]),
                    "expert_nitrogen_kg_ha": float(ex["actual_nitrogen_kg_ha"]),
                    "water_saving_vs_expert_mm": float(ex["actual_irrigation_mm"]) - float(row.rl_irrigation_mm),
                    "n_saving_vs_expert_kg_ha": float(ex["actual_nitrogen_kg_ha"]) - float(row.rl_nitrogen_kg_ha),
                }
            )
        rows.append(item)
    out = pd.DataFrame(rows)
    out["yield_win_four"] = out["yield_gap_vs_four_max_kg_ha"] > 1e-9
    out["PFP_N_win_four"] = out["PFP_N_gap_vs_four_max"] > 1e-9
    out["WP_ET_available_for_rl"] = False
    out["any_available_metric_win_four"] = out["yield_win_four"] | out["PFP_N_win_four"]
    return out


def plot_yearly(comp: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    x = np.arange(len(comp))
    labels = [str(y) for y in comp["year"]]
    split_colors = comp["split"].map({"train_2005_2010": "#6B7280", "transfer_2011_2020": "#2563EB"}).fillna("#6B7280")

    ygap = comp["yield_gap_vs_four_max_kg_ha"].astype(float)
    pfp_gap = comp["PFP_N_gap_vs_four_max"].astype(float)
    axes[0].bar(x, ygap, color=["#2563EB" if v >= 0 else "#CBD5E1" for v in ygap], edgecolor="#1F2937", linewidth=0.5)
    axes[0].axhline(0, color="#111827", linewidth=1.0)
    axes[0].set_ylabel("Yield gap (kg/ha)")
    fig.suptitle("LC 75k MaskablePPO vs best of four baselines by year", x=0.08, y=0.99, ha="left", fontweight="bold")
    fig.text(
        0.08,
        0.955,
        "Positive bars exceed the four-scenario maximum. Four baselines: null, recorded farmer, DSSAT auto, official expert.",
        fontsize=9,
        color="#374151",
    )

    axes[1].bar(x, pfp_gap, color=["#059669" if v >= 0 else "#FCA5A5" for v in pfp_gap], edgecolor="#1F2937", linewidth=0.5)
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_ylabel("PFP_N gap (kg/kg N)")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_xlabel("Year")
    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Split markers
    for i, color in enumerate(split_colors):
        axes[1].plot(i, axes[1].get_ylim()[0], marker="s", color=color, markersize=4, clip_on=False)
    axes[1].text(0, -0.32, "Grey tick = train-year evaluation; blue tick = frozen transfer year.", transform=axes[1].transAxes, fontsize=9, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG / "032_14_lc_75k_ppo_yearly_vs_four_baseline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_std_summary(comp: pd.DataFrame) -> Path:
    rows = []
    metric_specs = [
        ("yield_gap_vs_four_max_kg_ha", "Yield gap vs four-max", "kg/ha"),
        ("PFP_N_gap_vs_four_max", "PFP_N gap vs four-max", "kg/kg N"),
        ("water_saving_vs_expert_mm", "Water saving vs expert", "mm"),
        ("n_saving_vs_expert_kg_ha", "N saving vs expert", "kg/ha"),
    ]
    for split in ["train_2005_2010", "transfer_2011_2020", "all_2005_2020"]:
        part = comp if split == "all_2005_2020" else comp[comp["split"].eq(split)]
        for col, label, unit in metric_specs:
            vals = pd.to_numeric(part[col], errors="coerce").dropna()
            rows.append(
                {
                    "split": split,
                    "metric": label,
                    "unit": unit,
                    "n_years": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "032_14_lc_75k_ppo_metric_mean_std.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.flatten()
    palette = {"train_2005_2010": "#6B7280", "transfer_2011_2020": "#2563EB", "all_2005_2020": "#059669"}
    for ax, (col, label, unit) in zip(axes, metric_specs):
        sub = summary[summary["metric"].eq(label)].copy()
        xpos = np.arange(len(sub))
        ax.bar(xpos, sub["mean"], yerr=sub["std"], color=[palette[x] for x in sub["split"]], edgecolor="#1F2937", linewidth=0.5, capsize=4)
        ax.axhline(0, color="#111827", linewidth=0.9)
        ax.set_xticks(xpos, ["train", "transfer", "all"], rotation=0)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("LC 75k MaskablePPO cross-year mean and standard deviation", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.94, "Error bars are across-year sample standard deviations. WP_ET is not plotted because the current RL outputs do not include an ET denominator.", fontsize=9, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG / "032_14_lc_75k_ppo_crossyear_std_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def write_record(comp: pd.DataFrame, yearly_path: Path, std_path: Path) -> None:
    counts = (
        comp.groupby("split", as_index=False)
        .agg(
            n_years=("year", "count"),
            yield_win_four=("yield_win_four", "sum"),
            PFP_N_win_four=("PFP_N_win_four", "sum"),
            any_available_metric_win_four=("any_available_metric_win_four", "sum"),
            mean_yield_gap_vs_four_max=("yield_gap_vs_four_max_kg_ha", "mean"),
            std_yield_gap_vs_four_max=("yield_gap_vs_four_max_kg_ha", "std"),
            mean_PFP_N_gap_vs_four_max=("PFP_N_gap_vs_four_max", "mean"),
            std_PFP_N_gap_vs_four_max=("PFP_N_gap_vs_four_max", "std"),
        )
    )
    all_row = {
        "split": "all_2005_2020",
        "n_years": int(len(comp)),
        "yield_win_four": int(comp["yield_win_four"].sum()),
        "PFP_N_win_four": int(comp["PFP_N_win_four"].sum()),
        "any_available_metric_win_four": int(comp["any_available_metric_win_four"].sum()),
        "mean_yield_gap_vs_four_max": float(comp["yield_gap_vs_four_max_kg_ha"].mean()),
        "std_yield_gap_vs_four_max": float(comp["yield_gap_vs_four_max_kg_ha"].std(ddof=1)),
        "mean_PFP_N_gap_vs_four_max": float(comp["PFP_N_gap_vs_four_max"].mean()),
        "std_PFP_N_gap_vs_four_max": float(comp["PFP_N_gap_vs_four_max"].std(ddof=1)),
    }
    counts = pd.concat([counts, pd.DataFrame([all_row])], ignore_index=True)
    counts.to_csv(TAB / "032_14_lc_75k_ppo_success_counts.csv", index=False, encoding="utf-8-sig")
    actual_rows = []
    for split in ["train_2005_2010", "transfer_2011_2020", "all_2005_2020"]:
        part = comp if split == "all_2005_2020" else comp[comp["split"].eq(split)]
        for col, label, unit in [
            ("rl_yield_kg_ha", "PPO grain yield", "kg/ha"),
            ("rl_PFP_N_kg_kg", "PPO PFP_N", "kg/kg N"),
            ("rl_irrigation_mm", "PPO irrigation", "mm"),
            ("rl_nitrogen_kg_ha", "PPO nitrogen", "kg/ha"),
        ]:
            vals = pd.to_numeric(part[col], errors="coerce").dropna()
            actual_rows.append(
                {
                    "split": split,
                    "metric": label,
                    "unit": unit,
                    "n_years": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                }
            )
    actual_stats = pd.DataFrame(actual_rows)
    actual_stats.to_csv(TAB / "032_14_lc_75k_ppo_actual_metric_mean_std.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 032_14 LC 75k PPO four-baseline success figures record",
        "",
        "## Status",
        "",
        "- Completed.",
        "- Training run: 0.",
        "- DSSAT run: 0.",
        "- Model reselection: none.",
        "",
        "## Figures",
        "",
        f"- Yearly comparison: `{yearly_path.relative_to(ROOT).as_posix()}`",
        f"- Mean/std summary: `{std_path.relative_to(ROOT).as_posix()}`",
        "",
        "## Success counts",
        "",
        md_table(counts),
        "",
        "## Actual PPO metric mean/std",
        "",
        md_table(actual_stats),
        "",
        "## Per-year comparison",
        "",
        md_table(
            comp[
                [
                    "split",
                    "year",
                    "yield_gap_vs_four_max_kg_ha",
                    "PFP_N_gap_vs_four_max",
                    "water_saving_vs_expert_mm",
                    "n_saving_vs_expert_kg_ha",
                    "yield_win_four",
                    "PFP_N_win_four",
                    "any_available_metric_win_four",
                ]
            ],
            max_rows=100,
        ),
        "",
        "## Metric definition boundary",
        "",
        "- PFP_N is computed as grain yield divided by fertilizer N applied; N=0 would be undefined and must not be coerced to infinity.",
        "- WP_ET is not computed for the PPO candidate in this figure package because the current 032_11/032_12 RL outputs do not include a defensible ET denominator.",
        "- Water saving and N saving are reported versus official expert, not versus the four-scenario maximum/minimum envelope.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_baselines()
    rl = load_rl()
    comp = compare(rl, base)
    comp.to_csv(TAB / "032_14_lc_75k_ppo_vs_four_baseline.csv", index=False, encoding="utf-8-sig")
    yearly = plot_yearly(comp)
    stdfig = plot_std_summary(comp)
    write_record(comp, yearly, stdfig)
    print(f"wrote {yearly}")
    print(f"wrote {stdfig}")
    print(comp[["split", "year", "yield_gap_vs_four_max_kg_ha", "PFP_N_gap_vs_four_max", "water_saving_vs_expert_mm", "n_saving_vs_expert_kg_ha", "any_available_metric_win_four"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

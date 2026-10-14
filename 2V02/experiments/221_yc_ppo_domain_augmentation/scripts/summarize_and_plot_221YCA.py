"""Create auditable summaries, sampling coverage and Python-only figures for 221YCA."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "221_yc_ppo_domain_augmentation"
RESULTS = EXP / "results"
FIGURES = RESULTS / "figures"

A_EVAL = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo" / "evaluation" / "055_00_checkpoint_validation_summary.csv"
B_FORMAL = ROOT / "benchmark_results" / "221YCA_yc_ppo_domain_augmentation" / "evaluation" / "221YCA_checkpoint_validation_summary.csv"
B_SMOKE = ROOT / "benchmark_results" / "221YCA_yc_ppo_domain_augmentation_smoke2k" / "evaluation" / "221YCA_checkpoint_validation_summary.csv"
BASELINES = ROOT / "benchmark_results" / "055_02_yca_lowIC_four_baselines_static_level1" / "evaluation" / "055_02_baseline_summary.csv"
EXTERNAL = ROOT / "benchmark_results" / "055_01_yca_lowIC_external_auto_n_rule_nstd050_minimal" / "evaluation" / "055_01_external_auto_n_summary.csv"
WEATHER_MANIFEST = ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width" / "217YCA_weather_scenario_manifest.csv"
A_YEAR_LOG = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo" / "logs" / "032_22_YCA_training_year_switch_log.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)


def ppo_rows(path: Path, method: str, group: str, source: str) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty or "checkpoint_step" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    for col in ["checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "PFP_N", "reward_stress_aware_sum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for step, group_df in df.groupby("checkpoint_step", sort=True):
        rows.append(
            {
                "group": group,
                "method": method,
                "checkpoint_step": int(step),
                "n": int(len(group_df)),
                "mean_yield_kg_ha": float(group_df["final_grnwt"].mean()),
                "std_yield_kg_ha": float(group_df["final_grnwt"].std(ddof=0)),
                "mean_total_irrigation_mm": float(group_df["total_irrigation"].mean()),
                "mean_total_n_kg_ha": float(group_df["total_n"].mean()),
                "mean_reward": float(group_df["reward_stress_aware_sum"].mean()) if "reward_stress_aware_sum" in group_df else np.nan,
                "mean_simple_profit": float(group_df["profit_simple"].mean()) if "profit_simple" in group_df else np.nan,
                "mean_PFP_N": float(group_df["PFP_N"].mean()) if "PFP_N" in group_df else np.nan,
                "mean_WP_ET_kg_m3": np.nan,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def baseline_rows(path: Path, method_col: str = "scenario", group: str = "baseline") -> pd.DataFrame:
    df = read_csv(path)
    if df.empty or method_col not in df.columns:
        return pd.DataFrame()
    cols = {
        "grain_yield_kg_ha": "mean_yield_kg_ha",
        "actual_irrigation_mm": "mean_total_irrigation_mm",
        "actual_nitrogen_kg_ha": "mean_total_n_kg_ha",
        "PFP_N_kg_kg": "mean_PFP_N",
        "WP_ET_kg_m3": "mean_WP_ET_kg_m3",
    }
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for method, sub in df.groupby(method_col, sort=True):
        rows.append(
            {
                "group": group,
                "method": str(method),
                "checkpoint_step": np.nan,
                "n": int(len(sub)),
                "mean_yield_kg_ha": float(sub["grain_yield_kg_ha"].mean()) if "grain_yield_kg_ha" in sub else np.nan,
                "std_yield_kg_ha": float(sub["grain_yield_kg_ha"].std(ddof=0)) if "grain_yield_kg_ha" in sub else np.nan,
                "mean_total_irrigation_mm": float(sub["actual_irrigation_mm"].mean()) if "actual_irrigation_mm" in sub else np.nan,
                "mean_total_n_kg_ha": float(sub["actual_nitrogen_kg_ha"].mean()) if "actual_nitrogen_kg_ha" in sub else np.nan,
                "mean_reward": np.nan,
                "mean_simple_profit": np.nan,
                "mean_PFP_N": float(sub["PFP_N_kg_kg"].mean()) if "PFP_N_kg_kg" in sub else np.nan,
                "mean_WP_ET_kg_m3": float(sub["WP_ET_kg_m3"].mean()) if "WP_ET_kg_m3" in sub else np.nan,
                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return pd.DataFrame(rows)


def write_sampling_coverage() -> pd.DataFrame:
    manifest = read_csv(WEATHER_MANIFEST)
    expected = manifest[manifest.get("split", pd.Series(dtype=str)).astype(str).eq("train")].copy() if not manifest.empty else pd.DataFrame()
    if expected.empty:
        return pd.DataFrame()
    expected["source_year"] = pd.to_numeric(expected["source_year"], errors="coerce").astype(int)
    expected["pseudo_year"] = pd.to_numeric(expected["pseudo_year"], errors="coerce").astype(int)
    expected = expected[["source_year", "pseudo_year", "variant"]].drop_duplicates().copy()

    b_episode = read_csv(B_FORMAL.parent.parent / "logs" / "221YCA_training_episode_metrics.csv")
    b_source = "formal" if not b_episode.empty else "smoke2k"
    if b_episode.empty:
        b_episode = read_csv(B_SMOKE.parent.parent / "logs" / "221YCA_training_episode_metrics.csv")
    b_counts = pd.DataFrame(columns=["source_year", "variant", "episode_count"])
    if not b_episode.empty:
        b_episode["source_year"] = pd.to_numeric(b_episode["source_year"], errors="coerce")
        b_counts = b_episode.dropna(subset=["source_year"]).groupby(["source_year", "weather_variant"], as_index=False).size()
        b_counts = b_counts.rename(columns={"weather_variant": "variant", "size": "episode_count"})
    b = expected.merge(b_counts, on=["source_year", "variant"], how="left")
    b["episode_count"] = b["episode_count"].fillna(0).astype(int)
    b["experiment_group"] = "B_augmented"
    b["observed_source"] = b_source
    b["covered"] = b["episode_count"].gt(0)

    a_expected = expected[expected["variant"].eq("original")].copy()
    a_log = read_csv(A_YEAR_LOG)
    a_counts = pd.DataFrame(columns=["source_year", "episode_count"])
    if not a_log.empty and "year" in a_log.columns:
        a_log["year"] = pd.to_numeric(a_log["year"], errors="coerce")
        a_counts = a_log.dropna(subset=["year"]).groupby("year", as_index=False).size().rename(columns={"year": "source_year", "size": "episode_count"})
    a = a_expected.merge(a_counts, on="source_year", how="left")
    a["episode_count"] = a["episode_count"].fillna(0).astype(int)
    a["experiment_group"] = "A_original"
    a["observed_source"] = "055_00_training_year_switch_log"
    a["covered"] = a["episode_count"].gt(0)
    out = pd.concat([a, b], ignore_index=True)
    out.to_csv(RESULTS / "yc_augmentation_sampling_coverage.csv", index=False, encoding="utf-8-sig")
    return out


def write_training_domain_manifest() -> pd.DataFrame:
    """Write the episode-level provenance required by the augmentation prompt."""
    rows: list[dict] = []
    a_log = read_csv(A_YEAR_LOG)
    if not a_log.empty:
        for item in a_log.itertuples(index=False):
            year = int(float(item.year))
            rows.append(
                {
                    "experiment_group": "A_original",
                    "episode_id": f"A-{int(item.episode_index):06d}",
                    "seed": 0,
                    "station_code": "YCA",
                    "site": "YC",
                    "split": "train",
                    "weather_year": year,
                    "pseudo_year": year,
                    "weather_variant": "original",
                    "IC_profile_id": "lowIC",
                    "cultivar_id": "ZD0985",
                    "weather_sequence_complete": True,
                    "source": str(A_YEAR_LOG.relative_to(ROOT)).replace("\\", "/"),
                }
            )
    b_episode = read_csv(B_FORMAL.parent.parent / "logs" / "221YCA_training_episode_metrics.csv")
    b_source = "formal"
    if b_episode.empty:
        b_episode = read_csv(B_SMOKE.parent.parent / "logs" / "221YCA_training_episode_metrics.csv")
        b_source = "smoke2k"
    if not b_episode.empty:
        for item in b_episode.itertuples(index=False):
            source_year = int(float(item.source_year))
            pseudo_year = int(float(item.pseudo_year))
            variant = str(item.weather_variant)
            rows.append(
                {
                    "experiment_group": "B_augmented",
                    "episode_id": f"B-{int(item.episode_index):06d}",
                    "seed": 0,
                    "station_code": "YCA",
                    "site": "YC",
                    "split": "train",
                    "weather_year": source_year,
                    "pseudo_year": pseudo_year,
                    "weather_variant": variant,
                    "IC_profile_id": "lowIC",
                    "cultivar_id": "ZD0985",
                    "weather_sequence_complete": True,
                    "source": f"benchmark_results/221YCA_yc_ppo_domain_augmentation{'_smoke2k' if b_source == 'smoke2k' else ''}/logs/221YCA_training_episode_metrics.csv",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "training_domain_manifest.csv", index=False, encoding="utf-8-sig")
    return out


def make_summary() -> pd.DataFrame:
    tables = [
        ppo_rows(A_EVAL, "Original_A_055_00", "A_original", str(A_EVAL.relative_to(ROOT)).replace("\\", "/")),
        ppo_rows(B_SMOKE, "Augmented_B_smoke2k", "B_augmented_smoke", str(B_SMOKE.relative_to(ROOT)).replace("\\", "/")),
        ppo_rows(B_FORMAL, "Augmented_B_221YCA", "B_augmented", str(B_FORMAL.relative_to(ROOT)).replace("\\", "/")),
        baseline_rows(BASELINES),
        baseline_rows(EXTERNAL, method_col="scenario", group="external_rule"),
    ]
    summary = pd.concat([table for table in tables if not table.empty], ignore_index=True)
    summary.to_csv(RESULTS / "yc_augmentation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def make_pairwise(summary: pd.DataFrame) -> pd.DataFrame:
    a = summary[summary["method"].eq("Original_A_055_00")].copy()
    b = summary[summary["method"].eq("Augmented_B_221YCA")].copy()
    if a.empty or b.empty:
        out = pd.DataFrame(columns=["checkpoint_step", "delta_mean_yield_kg_ha", "delta_mean_total_irrigation_mm", "delta_mean_total_n_kg_ha", "delta_mean_reward", "delta_mean_PFP_N"])
    else:
        a = a.set_index("checkpoint_step")
        b = b.set_index("checkpoint_step")
        out = pd.DataFrame(index=sorted(set(a.index) & set(b.index)))
        for metric in ["mean_yield_kg_ha", "mean_total_irrigation_mm", "mean_total_n_kg_ha", "mean_reward", "mean_PFP_N", "mean_WP_ET_kg_m3"]:
            out[f"delta_{metric}"] = b.loc[out.index, metric] - a.loc[out.index, metric]
        out.index.name = "checkpoint_step"
        out = out.reset_index()
    out.to_csv(RESULTS / "yc_augmentation_paired_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def make_figure(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    ax = axes[0]
    colors = {"A_original": "#2f5d7e", "B_augmented": "#d97745", "B_augmented_smoke": "#e6ad83"}
    for group, label in [("A_original", "Original A"), ("B_augmented", "Augmented B"), ("B_augmented_smoke", "Augmented B smoke")]:
        sub = summary[(summary["group"].eq(group)) & summary["checkpoint_step"].notna()].sort_values("checkpoint_step")
        if sub.empty:
            continue
        x = sub["checkpoint_step"].to_numpy(dtype=float)
        y = sub["mean_yield_kg_ha"].to_numpy(dtype=float)
        e = sub["std_yield_kg_ha"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.8, ms=4, color=colors[group], label=label)
        ax.fill_between(x, y - e, y + e, color=colors[group], alpha=0.12, linewidth=0)
    for method, color, linestyle in [("recorded_farmer_template", "#6b7280", "--"), ("official_extension_expert", "#3f8f6b", ":")]:
        sub = summary[summary["method"].eq(method)]
        if not sub.empty:
            ax.axhline(float(sub.iloc[0]["mean_yield_kg_ha"]), color=color, lw=1.0, linestyle=linestyle, label=method.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean grain yield (kg ha$^{-1}$)")
    ax.set_title("Yield trajectory and fixed baselines", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    for group, label in [("A_original", "Original A"), ("B_augmented", "Augmented B"), ("B_augmented_smoke", "Augmented B smoke")]:
        sub = summary[(summary["group"].eq(group)) & summary["checkpoint_step"].notna()].sort_values("checkpoint_step")
        if sub.empty:
            continue
        ax.plot(sub["checkpoint_step"], sub["mean_total_irrigation_mm"], marker="o", lw=1.6, color=colors[group], label=f"{label}: I")
        ax.plot(sub["checkpoint_step"], sub["mean_total_n_kg_ha"], marker="s", lw=1.2, linestyle="--", color=colors[group], alpha=0.85, label=f"{label}: N")
    ax.set_xscale("log")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean resource use (mm or kg ha$^{-1}$)")
    ax.set_title("Management-resource trajectory", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(fontsize=7, loc="best")
    fig.suptitle("YC/YCA PPO domain augmentation: single-factor audit", fontsize=9, fontweight="bold")
    base = FIGURES / "yc_ppo_domain_augmentation_metrics"
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    contract = """# Figure contract\n\n- Core conclusion: test whether YC training-domain diversity changes PPO stability or matched validation performance while all other contracts remain fixed.\n- Evidence chain: left panel shows yield trajectory against frozen farmer/expert baselines; right panel shows irrigation and nitrogen behavior needed to interpret any yield difference.\n- Archetype: quantitative grid with one performance hero and one management-process panel.\n- Backend: Python/matplotlib only.\n- Export: SVG/PDF with editable text plus 300 dpi PNG and 600 dpi TIFF; uncertainty ribbons are validation-year standard deviations, not confidence intervals.\n- Integrity note: WP_ET is not plotted because exact PPO Summary.OUT/ETCP replay is unavailable; it remains N/A in the summary.\n"""
    (RESULTS / "figure_contract.md").write_text(contract, encoding="utf-8")
    summary = make_summary()
    make_pairwise(summary)
    write_sampling_coverage()
    write_training_domain_manifest()
    make_figure(summary)
    print(json.dumps({"summary": str(RESULTS / "yc_augmentation_summary.csv"), "figure_dir": str(FIGURES)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

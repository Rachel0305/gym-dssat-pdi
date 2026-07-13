from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_04"
FORWARD = (
    ROOT
    / "DSSAT_auto_validation"
    / "sy2014_ic2_forward_validation_021_02"
    / "021_02_sy2014_ic2_forward_summary.csv"
)
EXTENSION = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_03_extension_expert_summary.csv"
)
RUNS = {
    0: ROOT
    / "benchmark_results"
    / "021_03"
    / "021_03_sy2014_ic2_dqn_seed0_50k__sy_2014_seed0",
    1: OUT / "021_04_sy2014_ic2_dqn_seed12_50k__sy_2014_seed1",
    2: OUT / "021_04_sy2014_ic2_dqn_seed12_50k__sy_2014_seed2",
}


def collect_checkpoints() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, run in RUNS.items():
        for directory in run.joinpath("checkpoints").glob("checkpoint_*"):
            summary_path = directory / "eval_summary.csv"
            audit_path = directory / "runtime_audit.json"
            if not summary_path.exists() or not audit_path.exists():
                continue
            row = pd.read_csv(summary_path).iloc[0]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "checkpoint": int(row["checkpoint_step"]),
                    "yield_kg_ha": float(row["final_grain_kg_ha"]),
                    "biomass_kg_ha": float(row["final_biomass_kg_ha"]),
                    "irrigation_mm": float(row["action_irrigation_total_mm"]),
                    "nitrogen_kg_ha": float(row["action_nitrogen_total_kg_ha"]),
                    "max_water_stress": float(row["max_water_stress"]),
                    "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                    "reward_total": float(row["total_reward"]),
                    "runtime_audit_passed": bool(audit["passed"]),
                    "source_run": run.relative_to(ROOT).as_posix(),
                }
            )
    result = pd.DataFrame(rows).sort_values(["seed", "checkpoint"]).reset_index(drop=True)
    if len(result) != 30:
        raise RuntimeError(f"Expected 30 checkpoint rows, found {len(result)}")
    if not result["runtime_audit_passed"].all():
        raise RuntimeError("At least one runtime audit failed")
    result.to_csv(
        OUT / "021_04_sy2014_multiseed_checkpoint_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def select_best(checkpoints: pd.DataFrame) -> pd.DataFrame:
    best = (
        checkpoints.sort_values(
            ["seed", "reward_total", "checkpoint"],
            ascending=[True, False, True],
        )
        .groupby("seed", as_index=False)
        .first()
    )
    expert_yield = 11077.0
    expert_irrigation = 266.1
    expert_nitrogen = 300.0
    best["yield_ge_official_expert"] = best["yield_kg_ha"] >= expert_yield
    best["irrigation_le_official_expert"] = best["irrigation_mm"] <= expert_irrigation
    best["nitrogen_le_official_expert"] = best["nitrogen_kg_ha"] <= expert_nitrogen
    best["strict_success_vs_official_expert"] = (
        best["yield_ge_official_expert"]
        & best["irrigation_le_official_expert"]
        & best["nitrogen_le_official_expert"]
        & best["runtime_audit_passed"]
    )
    best["yield_minus_official_expert_kg_ha"] = best["yield_kg_ha"] - expert_yield
    best.to_csv(
        OUT / "021_04_sy2014_multiseed_best_checkpoint_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return best


def baseline_comparison(best: pd.DataFrame) -> pd.DataFrame:
    forward = pd.read_csv(FORWARD, keep_default_na=False)
    base = forward[["scenario", "HWAM", "CWAM", "IRCM", "NICM"]].rename(
        columns={
            "HWAM": "yield_kg_ha",
            "CWAM": "biomass_kg_ha",
            "IRCM": "irrigation_mm",
            "NICM": "nitrogen_kg_ha",
        }
    )
    extension = pd.read_csv(EXTENSION)
    extension = extension[
        extension["site"].eq("SY") & extension["year"].eq(2014)
    ].iloc[0]
    base = pd.concat(
        [
            base,
            pd.DataFrame(
                [
                    {
                        "scenario": "official_extension_expert",
                        "yield_kg_ha": float(extension["grain_yield_kg_ha"]),
                        "biomass_kg_ha": float(extension["biomass_kg_ha"]),
                        "irrigation_mm": float(extension["irrigation_mm"]),
                        "nitrogen_kg_ha": float(extension["nitrogen_kg_ha"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    dqn = best[
        ["seed", "checkpoint", "yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha"]
    ].copy()
    dqn["scenario"] = dqn.apply(
        lambda row: f"dqn_seed{int(row['seed'])}_best_{int(row['checkpoint'])}", axis=1
    )
    dqn = dqn.drop(columns=["seed", "checkpoint"])
    result = pd.concat([base, dqn], ignore_index=True)
    null_yield = float(result.loc[result["scenario"].eq("null"), "yield_kg_ha"].iloc[0])
    result["reward_same_formula"] = (
        (result["yield_kg_ha"] - null_yield).clip(lower=0)
        - result["irrigation_mm"]
        - 5.0 * result["nitrogen_kg_ha"]
    )
    result["IWP_gross_kg_m3"] = result.apply(
        lambda row: 0.1 * row["yield_kg_ha"] / row["irrigation_mm"]
        if row["irrigation_mm"] > 0
        else pd.NA,
        axis=1,
    )
    result["PFP_N_kg_kg"] = result.apply(
        lambda row: row["yield_kg_ha"] / row["nitrogen_kg_ha"]
        if row["nitrogen_kg_ha"] > 0
        else pd.NA,
        axis=1,
    )
    result.to_csv(
        OUT / "021_04_sy2014_multiseed_baseline_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def plot_trajectory(checkpoints: pd.DataFrame, best: pd.DataFrame) -> None:
    colors = {0: "#222222", 1: "#B23A35", 2: "#2E5D9F"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [
        ("yield_kg_ha", "Yield (kg/ha)"),
        ("reward_total", "Reward"),
        ("irrigation_mm", "Irrigation (mm)"),
        ("nitrogen_kg_ha", "Nitrogen (kg/ha)"),
    ]
    for ax, (column, ylabel) in zip(axes.flat, panels):
        for seed, group in checkpoints.groupby("seed"):
            selected = best[best["seed"].eq(seed)].iloc[0]
            ax.plot(
                group["checkpoint"],
                group[column],
                marker="o",
                ms=4,
                lw=1.8,
                color=colors[int(seed)],
                label=f"seed {seed}",
            )
            ax.scatter(
                [selected["checkpoint"]],
                [selected[column]],
                s=70,
                facecolors="none",
                edgecolors=colors[int(seed)],
                linewidths=1.8,
                zorder=4,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E5E5E5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].axhline(11077, color="#777777", ls="--", lw=1.2, label="official expert")
    axes[0, 0].legend(frameon=False, fontsize=9, ncol=2)
    axes[1, 0].set_xlabel("Training steps")
    axes[1, 1].set_xlabel("Training steps")
    fig.suptitle("SY2014 IC=2 DQN: three-seed checkpoint trajectories")
    fig.tight_layout()
    fig.savefig(
        OUT / "021_04_sy2014_multiseed_checkpoint_trajectory.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = collect_checkpoints()
    best = select_best(checkpoints)
    comparison = baseline_comparison(best)
    plot_trajectory(checkpoints, best)
    print(best.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

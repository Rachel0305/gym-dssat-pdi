from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "benchmark_results" / "021_03" / "021_03_sy2014_ic2_dqn_seed0_50k__sy_2014_seed0"
OUT = ROOT / "benchmark_results" / "021_03"
FORWARD = ROOT / "DSSAT_auto_validation" / "sy2014_ic2_forward_validation_021_02" / "021_02_sy2014_ic2_forward_summary.csv"
EXTENSION = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_03_extension_expert_summary.csv"


def checkpoint_table() -> pd.DataFrame:
    rows = []
    for directory in RUN.joinpath("checkpoints").glob("checkpoint_*"):
        path = directory / "eval_summary.csv"
        if not path.exists():
            continue
        row = pd.read_csv(path).iloc[0]
        rows.append({
            "checkpoint": int(row["checkpoint_step"]),
            "yield_kg_ha": float(row["final_grain_kg_ha"]),
            "biomass_kg_ha": float(row["final_biomass_kg_ha"]),
            "irrigation_mm": float(row["action_irrigation_total_mm"]),
            "nitrogen_kg_ha": float(row["action_nitrogen_total_kg_ha"]),
            "reward_total": float(row["total_reward"]),
            "runtime_audit_passed": bool(json.loads((directory / "runtime_audit.json").read_text(encoding="utf-8"))["passed"]),
        })
    result = pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)
    result.to_csv(OUT / "021_03_sy2014_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    return result


def comparison_table(checkpoints: pd.DataFrame) -> pd.DataFrame:
    forward = pd.read_csv(FORWARD, keep_default_na=False)
    base = forward[["scenario", "HWAM", "CWAM", "IRCM", "NICM"]].rename(columns={
        "HWAM": "yield_kg_ha", "CWAM": "biomass_kg_ha", "IRCM": "irrigation_mm", "NICM": "nitrogen_kg_ha"
    })
    extension = pd.read_csv(EXTENSION)
    extension = extension[(extension["site"].eq("SY")) & (extension["year"].eq(2014))].iloc[0]
    base = pd.concat([base, pd.DataFrame([{
        "scenario": "official_extension_expert",
        "yield_kg_ha": float(extension["grain_yield_kg_ha"]),
        "biomass_kg_ha": float(extension["biomass_kg_ha"]),
        "irrigation_mm": float(extension["irrigation_mm"]),
        "nitrogen_kg_ha": float(extension["nitrogen_kg_ha"]),
    }])], ignore_index=True)
    best = checkpoints.sort_values(["reward_total", "checkpoint"], ascending=[False, True]).iloc[0]
    dqn = pd.DataFrame([{
        "scenario": f"dqn_best_{int(best['checkpoint'])}",
        "yield_kg_ha": best["yield_kg_ha"],
        "biomass_kg_ha": best["biomass_kg_ha"],
        "irrigation_mm": best["irrigation_mm"],
        "nitrogen_kg_ha": best["nitrogen_kg_ha"],
    }])
    result = pd.concat([base, dqn], ignore_index=True)
    null_yield = float(result.loc[result["scenario"].eq("null"), "yield_kg_ha"].iloc[0])
    result["reward_same_formula"] = (
        (result["yield_kg_ha"] - null_yield).clip(lower=0)
        - result["irrigation_mm"]
        - 5.0 * result["nitrogen_kg_ha"]
    )
    result["IWP_gross_kg_m3"] = result.apply(
        lambda r: 0.1 * r["yield_kg_ha"] / r["irrigation_mm"] if r["irrigation_mm"] > 0 else pd.NA, axis=1
    )
    result["PFP_N_kg_kg"] = result.apply(
        lambda r: r["yield_kg_ha"] / r["nitrogen_kg_ha"] if r["nitrogen_kg_ha"] > 0 else pd.NA, axis=1
    )
    result.to_csv(OUT / "021_03_sy2014_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    return result


def plot(checkpoints: pd.DataFrame) -> None:
    best = checkpoints.sort_values(["reward_total", "checkpoint"], ascending=[False, True]).iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [
        ("yield_kg_ha", "Yield (kg/ha)", "#222222"),
        ("reward_total", "Reward", "#B23A35"),
        ("irrigation_mm", "Irrigation (mm)", "#2E5D9F"),
        ("nitrogen_kg_ha", "Nitrogen (kg/ha)", "#4A7C3A"),
    ]
    for ax, (column, ylabel, color) in zip(axes.flat, panels):
        ax.plot(checkpoints["checkpoint"], checkpoints[column], marker="o", lw=2, color=color)
        ax.axvline(best["checkpoint"], color="#777777", ls="--", lw=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E5E5E5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Training steps")
    axes[1, 1].set_xlabel("Training steps")
    fig.suptitle(f"SY2014 IC=2 DQN seed0: checkpoint trajectory (best={int(best['checkpoint'])})")
    fig.tight_layout()
    fig.savefig(OUT / "021_03_sy2014_checkpoint_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_table()
    comparison = comparison_table(checkpoints)
    plot(checkpoints)
    print(checkpoints.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

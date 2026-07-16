"""整理021_09 seed0与021_10 seed1的延长探索跨seed对照。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    0: ROOT / "benchmark_results/021_09/021_09_sy2014_extended_exploration_seed0_25k__sy_2014_seed0",
    1: ROOT / "benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1",
}
OUT = ROOT / "benchmark_results/021_10"


def load(seed: int, run: Path) -> pd.DataFrame:
    season = pd.read_csv(run / "evaluations/season_summary_all_checkpoints.csv")
    log = pd.DataFrame(
        [json.loads(line) for line in (run / "logs/training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    )[["completed_steps", "exploration_rate"]].rename(columns={"completed_steps": "checkpoint"})
    result = season.merge(log, on="checkpoint", validate="one_to_one")
    result["seed"] = seed
    return result


def main() -> None:
    combined = pd.concat([load(seed, run) for seed, run in RUNS.items()], ignore_index=True)
    columns = [
        "seed", "checkpoint", "exploration_rate", "yield_kg_ha", "biomass_kg_ha",
        "irrigation_mm", "nitrogen_kg_ha", "reward_total",
        "number_of_irrigation_events", "number_of_nitrogen_events",
    ]
    combined[columns].to_csv(OUT / "021_10_seed0_seed1_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    colors = {0: "#4C78A8", 1: "#E45756"}
    for seed, group in combined.groupby("seed"):
        group = group.sort_values("checkpoint")
        axes[0, 0].plot(group.checkpoint, group.yield_kg_ha, "o-", label=f"seed{seed}", color=colors[seed])
        axes[0, 1].plot(group.checkpoint, group.nitrogen_kg_ha, "o-", label=f"seed{seed}", color=colors[seed])
        axes[1, 0].plot(group.checkpoint, group.irrigation_mm, "o-", label=f"seed{seed}", color=colors[seed])
        axes[1, 1].plot(group.checkpoint, group.reward_total, "o-", label=f"seed{seed}", color=colors[seed])
    for ax, title, ylabel in zip(
        axes.flat,
        ("Yield", "Season nitrogen", "Season irrigation", "Deterministic reward"),
        ("kg/ha", "kg N/ha", "mm", "reward"),
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel("Training checkpoint")
    fig.suptitle("SY2014 extended exploration (fraction=0.70): seed replication")
    fig.tight_layout()
    fig.savefig(OUT / "021_10_seed0_seed1_checkpoint_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    indexed = combined.set_index(["seed", "checkpoint"])
    summary = {
        "status": "completed_seed1_25k",
        "seed0_20k": indexed.loc[(0, 20000), ["yield_kg_ha", "nitrogen_kg_ha", "reward_total"]].to_dict(),
        "seed1_20k": indexed.loc[(1, 20000), ["yield_kg_ha", "nitrogen_kg_ha", "reward_total"]].to_dict(),
        "seed1_25k": indexed.loc[(1, 25000), ["yield_kg_ha", "nitrogen_kg_ha", "reward_total"]].to_dict(),
        "strict_cross_seed_stability_through_all_checkpoints": False,
        "permanent_n0_lock_in_avoided_through_25k_both_seeds": bool(
            all(indexed.loc[(seed, 25000), "nitrogen_kg_ha"] > 0 for seed in (0, 1))
        ),
        "interpretation": (
            "Extended exploration prevents permanent N0 lock-in through 25K, but seed1 shows "
            "a transient nitrogen/yield collapse at 15K-20K and recovery at 25K."
        ),
    }
    (OUT / "021_10_seed_replication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(combined[columns].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

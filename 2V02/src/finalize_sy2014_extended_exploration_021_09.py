"""整理021_09与021_05的同seed、同checkpoint单变量对照。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmark_results/021_05/021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"
NEW = ROOT / "benchmark_results/021_09/021_09_sy2014_extended_exploration_seed0_25k__sy_2014_seed0"
OUT = ROOT / "benchmark_results/021_09"


def load(run: Path, label: str) -> pd.DataFrame:
    season = pd.read_csv(run / "evaluations/season_summary_all_checkpoints.csv")
    logs = [json.loads(line) for line in (run / "logs/training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    epsilon = pd.DataFrame(logs)[["completed_steps", "exploration_rate"]].rename(
        columns={"completed_steps": "checkpoint"}
    )
    season = season.merge(epsilon, on="checkpoint", how="left", validate="one_to_one")
    season["protocol"] = label
    return season[season.checkpoint <= 25000].copy()


def main() -> None:
    old = load(OLD, "baseline exploration_fraction=0.35")
    new = load(NEW, "extended exploration_fraction=0.70")
    comparison = pd.concat([old, new], ignore_index=True)
    columns = [
        "protocol", "checkpoint", "exploration_rate", "yield_kg_ha", "biomass_kg_ha",
        "irrigation_mm", "nitrogen_kg_ha", "reward_total",
        "number_of_irrigation_events", "number_of_nitrogen_events",
    ]
    comparison[columns].to_csv(OUT / "021_09_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    colors = {old.protocol.iloc[0]: "#4C78A8", new.protocol.iloc[0]: "#E45756"}
    for protocol, group in comparison.groupby("protocol", sort=False):
        group = group.sort_values("checkpoint")
        label = "baseline 0.35" if "0.35" in protocol else "extended 0.70"
        axes[0, 0].plot(group.checkpoint, group.yield_kg_ha, "o-", label=label, color=colors[protocol])
        axes[0, 1].plot(group.checkpoint, group.nitrogen_kg_ha, "o-", label=label, color=colors[protocol])
        axes[1, 0].plot(group.checkpoint, group.exploration_rate, "o-", label=label, color=colors[protocol])
        axes[1, 1].plot(group.checkpoint, group.reward_total, "o-", label=label, color=colors[protocol])
    titles = ("Yield", "Season nitrogen", "Exploration rate", "Deterministic reward")
    ylabels = ("kg/ha", "kg N/ha", "epsilon", "reward")
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel("Training checkpoint")
    fig.suptitle("SY2014 seed0: single-variable exploration-schedule comparison")
    fig.tight_layout()
    fig.savefig(OUT / "021_09_exploration_schedule_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    old20 = old.set_index("checkpoint").loc[20000]
    new20 = new.set_index("checkpoint").loc[20000]
    summary = {
        "status": "completed_seed0_single_variable_25k",
        "only_changed_parameter": "exploration_fraction: 0.35 -> 0.70",
        "baseline_20k": {
            "epsilon": float(old20.exploration_rate),
            "yield_kg_ha": float(old20.yield_kg_ha),
            "nitrogen_kg_ha": float(old20.nitrogen_kg_ha),
        },
        "extended_20k": {
            "epsilon": float(new20.exploration_rate),
            "yield_kg_ha": float(new20.yield_kg_ha),
            "nitrogen_kg_ha": float(new20.nitrogen_kg_ha),
        },
        "collapse_prevented_through_25k_seed0": bool((new.nitrogen_kg_ha > 0).all() and (new.yield_kg_ha > 10000).all()),
        "causal_scope": "supports exploration schedule as a contributing mechanism; cross-seed replication pending",
    }
    (OUT / "021_09_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison[columns].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

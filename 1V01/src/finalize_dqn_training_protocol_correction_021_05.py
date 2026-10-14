from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_05"
OLD = ROOT / "benchmark_results" / "021_03" / "021_03_sy2014_ic2_dqn_seed0_50k__sy_2014_seed0"
NEW = OUT / "021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"


def collect(run: Path, protocol: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for checkpoint_dir in run.joinpath("checkpoints").glob("checkpoint_*"):
        summary = pd.read_csv(checkpoint_dir / "eval_summary.csv").iloc[0]
        model = DQN.load(str(checkpoint_dir / "model.zip"))
        rows.append(
            {
                "protocol": protocol,
                "checkpoint": int(summary["checkpoint_step"]),
                "exploration_rate": float(model.exploration_rate),
                "yield_kg_ha": float(summary["final_grain_kg_ha"]),
                "biomass_kg_ha": float(summary["final_biomass_kg_ha"]),
                "irrigation_mm": float(summary["action_irrigation_total_mm"]),
                "nitrogen_kg_ha": float(summary["action_nitrogen_total_kg_ha"]),
                "reward_total": float(summary["total_reward"]),
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint")


def plot(frame: pd.DataFrame) -> None:
    colors = {"old_segment_increment": "#777777", "corrected_global_horizon": "#B23A35"}
    labels = {"old_segment_increment": "Old segmented schedule", "corrected_global_horizon": "Corrected global horizon"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [
        ("exploration_rate", "Exploration rate"),
        ("yield_kg_ha", "Yield (kg/ha)"),
        ("irrigation_mm", "Irrigation (mm)"),
        ("nitrogen_kg_ha", "Nitrogen (kg/ha)"),
    ]
    for ax, (column, ylabel) in zip(axes.flat, panels):
        for protocol, group in frame.groupby("protocol", sort=False):
            ax.plot(
                group["checkpoint"],
                group[column],
                marker="o",
                lw=2,
                color=colors[protocol],
                label=labels[protocol],
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E5E5E5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Training steps")
    axes[1, 1].set_xlabel("Training steps")
    fig.suptitle("SY2014 seed0: old vs corrected exploration schedule")
    fig.tight_layout()
    fig.savefig(OUT / "021_05_old_vs_corrected_protocol.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def collect_q_replay_diagnostics(run: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in (5000, 10000, 15000, 20000, 25000):
        checkpoint_dir = run / "checkpoints" / f"checkpoint_{step}"
        model = DQN.load(str(checkpoint_dir / "model.zip"))
        model.load_replay_buffer(str(checkpoint_dir / "replay_buffer.pkl"))
        replay = model.replay_buffer
        assert replay is not None
        size = replay.size()
        sample_count = min(1000, size)
        indices = np.linspace(0, size - 1, sample_count, dtype=int)
        observations = replay.observations[indices, 0]
        actions = replay.actions[indices, 0].reshape(-1)
        rewards = replay.rewards[indices, 0].reshape(-1)
        obs_tensor = obs_as_tensor(observations, model.device)
        q_online = model.q_net(obs_tensor).detach().cpu().numpy()
        q_target = model.q_net_target(obs_tensor).detach().cpu().numpy()
        rows.append(
            {
                "checkpoint": step,
                "exploration_rate": float(model.exploration_rate),
                "replay_size": int(size),
                "sample_count": int(sample_count),
                "sample_action0_pct": float(np.mean(actions == 0) * 100.0),
                "q_online_max": float(np.max(q_online)),
                "q_target_max": float(np.max(q_target)),
                "q_online_abs_mean": float(np.mean(np.abs(q_online))),
                "sample_reward_max": float(np.max(rewards)),
            }
        )
    return pd.DataFrame(rows)


def write_impact_scope() -> None:
    rows = [
        ("HLA", "020_06/020_07/020_08 and downstream 020_11", "confirmed", "5K model exploration_rate=0.05", "DQN conclusions provisional; baseline and forward DSSAT results unaffected"),
        ("YC", "020_12/020_13 and 021_01 coefficient diagnostics", "confirmed", "5K model exploration_rate=0.05", "DQN conclusions and coefficient-insensitivity claim provisional"),
        ("FQ", "020_12/020_13 and 021_01 coefficient diagnostics", "confirmed", "5K model exploration_rate=0.05", "DQN conclusions and coefficient-insensitivity claim provisional"),
        ("LC", "021_01 three-seed stability", "confirmed", "all 5K models exploration_rate=0.05", "resource-use seed-sensitivity conclusion requires corrected-protocol recheck"),
        ("SY", "021_03/021_04", "confirmed", "all 5K models exploration_rate=0.05", "early high-yield checkpoints remain diagnostic only"),
        ("ALL_BASELINES", "null/recorded/expert/auto and deterministic forward scans", "unaffected", "do not use DQN learn()", "remain valid subject to their original input provenance"),
    ]
    pd.DataFrame(
        rows,
        columns=["scope", "affected_experiments", "status", "evidence", "scientific_handling"],
    ).to_csv(OUT / "021_05_training_protocol_impact_scope.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    frame = pd.concat(
        [collect(OLD, "old_segment_increment"), collect(NEW, "corrected_global_horizon")],
        ignore_index=True,
    )
    frame.to_csv(OUT / "021_05_old_vs_corrected_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    q_frame = collect_q_replay_diagnostics(NEW)
    q_frame.to_csv(OUT / "021_05_corrected_q_replay_diagnostics.csv", index=False, encoding="utf-8-sig")
    write_impact_scope()
    plot(frame)
    print(frame.to_string(index=False))
    print("\nCorrected Q/replay diagnostics:")
    print(q_frame.to_string(index=False))


if __name__ == "__main__":
    main()

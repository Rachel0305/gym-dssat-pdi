"""021_08：纯离线审计 SY2014 两个 target 冻结窗口的 replay 与探索差异。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "benchmark_results/021_05"
    / "021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"
)
Q_MAP = ROOT / "benchmark_results/021_06/021_06_q_values_long.csv"
OUT = ROOT / "benchmark_results/021_08"
WINDOWS = ((5000, 10000, "5K-10K"), (15000, 20000, "15K-20K"))


def checkpoint_dir(step: int) -> Path:
    return RUN / "checkpoints" / f"checkpoint_{step}"


def load_model_and_buffer(step: int) -> DQN:
    model = DQN.load(str(checkpoint_dir(step) / "model.zip"), device="cpu")
    model.load_replay_buffer(str(checkpoint_dir(step) / "replay_buffer.pkl"))
    if model.replay_buffer is None:
        raise RuntimeError(f"missing replay buffer: {step}")
    return model


def circular_indices(start: int, end: int, capacity: int) -> np.ndarray:
    if start == end:
        return np.arange(capacity, dtype=int)
    if end > start:
        return np.arange(start, end, dtype=int)
    return np.concatenate((np.arange(start, capacity, dtype=int), np.arange(0, end, dtype=int)))


def action_entropy(actions: np.ndarray) -> tuple[float, float]:
    counts = np.bincount(actions, minlength=9).astype(float)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return entropy, entropy / float(np.log(9.0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = (
        pd.read_csv(Q_MAP)[
            ["action_index", "requested_irrigation", "requested_nitrogen"]
        ]
        .drop_duplicates()
        .sort_values("action_index")
    )
    if mapping.action_index.tolist() != list(range(9)):
        raise RuntimeError("unexpected action mapping")
    action_map = mapping.set_index("action_index").to_dict("index")

    detail_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    buffer_metadata: list[dict[str, object]] = []

    for start_step, end_step, label in WINDOWS:
        start_model = load_model_and_buffer(start_step)
        start_buffer = start_model.replay_buffer
        assert start_buffer is not None
        start_pos = int(start_buffer.pos)
        start_full = bool(start_buffer.full)
        start_capacity = int(start_buffer.buffer_size)
        start_epsilon = float(start_model.exploration_rate)
        del start_model

        end_model = load_model_and_buffer(end_step)
        replay = end_model.replay_buffer
        assert replay is not None
        end_pos = int(replay.pos)
        end_full = bool(replay.full)
        capacity = int(replay.buffer_size)
        if capacity != start_capacity:
            raise RuntimeError(f"buffer capacity changed in {label}")
        indices = circular_indices(start_pos, end_pos, capacity)
        expected = int(end_model.num_timesteps - start_step)
        # NStepReplayBuffer 的开头/末尾可能因 n-step 排队少于环境步数，记录但不强制相等。

        actions = replay.actions[indices, 0, 0].astype(int)
        rewards = replay.rewards[indices, 0].astype(float)
        dones = replay.dones[indices, 0].astype(float)
        timeouts = replay.timeouts[indices, 0].astype(float)
        observations = replay.observations[indices, 0]
        with torch.no_grad():
            greedy_parts = []
            for begin in range(0, len(indices), 1024):
                obs = torch.as_tensor(observations[begin : begin + 1024], device=end_model.device)
                greedy_parts.append(end_model.q_net(obs).argmax(dim=1).cpu().numpy())
        endpoint_greedy = np.concatenate(greedy_parts).astype(int)

        requested_i = np.array([action_map[a]["requested_irrigation"] for a in actions], dtype=float)
        requested_n = np.array([action_map[a]["requested_nitrogen"] for a in actions], dtype=float)
        entropy, normalized_entropy = action_entropy(actions)
        effective_done = dones * (1.0 - timeouts)

        detail = pd.DataFrame(
            {
                "window": label,
                "replay_index": indices,
                "requested_action_index": actions,
                "requested_irrigation": requested_i,
                "requested_nitrogen": requested_n,
                "reward": rewards,
                "done_excluding_timeout": effective_done,
                "endpoint_greedy_action_index": endpoint_greedy,
                "matches_endpoint_greedy": actions == endpoint_greedy,
            }
        )
        detail_frames.append(detail)

        counts = np.bincount(actions, minlength=9)
        for action_index, count in enumerate(counts):
            distribution_rows.append(
                {
                    "window": label,
                    "action_index": action_index,
                    "requested_irrigation": action_map[action_index]["requested_irrigation"],
                    "requested_nitrogen": action_map[action_index]["requested_nitrogen"],
                    "count": int(count),
                    "fraction": float(count / len(actions)),
                }
            )

        summaries.append(
            {
                "window": label,
                "transition_count": int(len(actions)),
                "expected_environment_steps": expected,
                "start_exploration_rate": start_epsilon,
                "end_exploration_rate": float(end_model.exploration_rate),
                "noop_fraction": float(((requested_i == 0) & (requested_n == 0)).mean()),
                "positive_irrigation_fraction": float((requested_i > 0).mean()),
                "positive_nitrogen_fraction": float((requested_n > 0).mean()),
                "joint_positive_fraction": float(((requested_i > 0) & (requested_n > 0)).mean()),
                "requested_n0_fraction": float((requested_n == 0).mean()),
                "requested_n50_fraction": float((requested_n == 50).mean()),
                "requested_n100_fraction": float((requested_n == 100).mean()),
                "action_entropy": entropy,
                "normalized_action_entropy": normalized_entropy,
                "reward_mean": float(rewards.mean()),
                "reward_median": float(np.median(rewards)),
                "reward_positive_fraction": float((rewards > 0).mean()),
                "reward_negative_fraction": float((rewards < 0).mean()),
                "reward_zero_fraction": float((rewards == 0).mean()),
                "reward_p95": float(np.quantile(rewards, 0.95)),
                "reward_max": float(rewards.max()),
                "terminal_fraction": float(effective_done.mean()),
                "matches_endpoint_greedy_fraction": float((actions == endpoint_greedy).mean()),
                "endpoint_greedy_is_posthoc_only": True,
                "executed_action_available_in_replay": False,
                "exploration_origin_available_per_transition": False,
            }
        )
        buffer_metadata.append(
            {
                "window": label,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "capacity": capacity,
                "start_full": start_full,
                "end_full": end_full,
                "extracted_transition_count": int(len(indices)),
                "expected_environment_steps": expected,
            }
        )
        del end_model

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    distribution = pd.DataFrame(distribution_rows)
    metadata = pd.DataFrame(buffer_metadata)
    detail_all.to_csv(OUT / "021_08_window_transitions.csv", index=False)
    summary.to_csv(OUT / "021_08_window_summary.csv", index=False)
    distribution.to_csv(OUT / "021_08_action_distribution.csv", index=False)
    metadata.to_csv(OUT / "021_08_buffer_extraction_audit.csv", index=False)
    (OUT / "021_08_replay_audit_summary.json").write_text(
        json.dumps(
            {
                "status": "completed_offline",
                "training": False,
                "dssat_called": False,
                "limitations": [
                    "replay stores requested action index, not wrapper executed action",
                    "replay does not store whether each action came from epsilon exploration or greedy selection",
                    "scalar reward components cannot be reconstructed exactly from saved fields",
                    "endpoint greedy comparison is posthoc and not action-generation provenance",
                ],
                "buffer_extraction": buffer_metadata,
                "window_summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = summary.window.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    x = np.arange(len(labels))
    width = 0.25
    for offset, column, name, color in (
        (-width, "requested_n0_fraction", "N0", "#4C78A8"),
        (0, "requested_n50_fraction", "N50", "#F2CF5B"),
        (width, "requested_n100_fraction", "N100", "#E45756"),
    ):
        axes[0].bar(x + offset, summary[column], width, label=name, color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Requested-action fraction")
    axes[0].set_title("Nitrogen request distribution")
    axes[0].legend(frameon=False)

    axes[1].plot(labels, summary.start_exploration_rate, "o--", label="window start")
    axes[1].plot(labels, summary.end_exploration_rate, "s-", label="window end")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Epsilon")
    axes[1].set_title("Exploration-rate endpoints")
    axes[1].legend(frameon=False)

    axes[2].bar(x - width / 2, summary.reward_nonzero_fraction if "reward_nonzero_fraction" in summary else 1-summary.reward_zero_fraction, width, label="non-zero reward", color="#72B7B2")
    axes[2].bar(x + width / 2, summary.matches_endpoint_greedy_fraction, width, label="matches endpoint greedy*", color="#B279A2")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("Replay composition (*posthoc)")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("SY2014 replay/exploration audit: two target-freeze windows")
    fig.tight_layout()
    fig.savefig(OUT / "021_08_replay_exploration_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Buffer extraction audit:")
    print(metadata.to_string(index=False))
    print("\nWindow summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

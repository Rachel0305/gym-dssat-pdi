from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
import pandas as pd
import stable_baselines3
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "benchmark_results"
    / "021_05"
    / "021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"
)
Q_INPUT = ROOT / "benchmark_results" / "021_06" / "021_06_q_values_long.csv"
ALIAS_INPUT = ROOT / "benchmark_results" / "021_06" / "021_06_action_alias_summary.csv"
OUT = ROOT / "benchmark_results" / "021_07"
CHECKPOINTS = (5000, 10000, 15000, 20000, 25000)


def model_path(step: int) -> Path:
    return RUN / "checkpoints" / f"checkpoint_{step}" / "model.zip"


def replay_path(step: int) -> Path:
    return RUN / "checkpoints" / f"checkpoint_{step}" / "replay_buffer.pkl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def max_parameter_difference(left: torch.nn.Module, right: torch.nn.Module) -> float:
    differences = []
    for (left_name, left_value), (right_name, right_value) in zip(
        sorted(left.state_dict().items()), sorted(right.state_dict().items())
    ):
        if left_name != right_name:
            raise RuntimeError(f"Parameter mismatch: {left_name} != {right_name}")
        differences.append(float(torch.max(torch.abs(left_value - right_value)).cpu()))
    return max(differences, default=0.0)


def archive_metadata(path: Path) -> dict:
    with ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
        saved_version = archive.read("_stable_baselines3_version").decode("utf-8").strip()
    keys = (
        "num_timesteps",
        "_n_calls",
        "_n_updates",
        "target_update_interval",
        "exploration_rate",
        "exploration_fraction",
        "_total_timesteps",
    )
    return {key: data.get(key) for key in keys} | {"saved_sb3_version": saved_version}


def target_audit(models: dict[int, DQN]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for step, model in models.items():
        metadata = archive_metadata(model_path(step))
        n_calls = int(metadata["_n_calls"])
        interval = int(metadata["target_update_interval"])
        rows.append(
            {
                "checkpoint": step,
                **metadata,
                "runtime_sb3_version": stable_baselines3.__version__,
                "model_zip_sha256": file_sha256(model_path(step)),
                "online_parameter_sha256": parameter_sha256(model.q_net),
                "target_parameter_sha256": parameter_sha256(model.q_net_target),
                "completed_target_updates_by_n_calls": n_calls // interval,
                "calls_since_last_target_update": n_calls % interval,
                "calls_to_next_target_update": interval - (n_calls % interval),
            }
        )
    audit = pd.DataFrame(rows).sort_values("checkpoint")

    differences = []
    for left, right in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]):
        differences.append(
            {
                "from_checkpoint": left,
                "to_checkpoint": right,
                "online_max_parameter_abs_diff": max_parameter_difference(
                    models[left].q_net, models[right].q_net
                ),
                "target_max_parameter_abs_diff": max_parameter_difference(
                    models[left].q_net_target, models[right].q_net_target
                ),
                "target_hash_changed": parameter_sha256(models[left].q_net_target)
                != parameter_sha256(models[right].q_net_target),
                "target_update_count_change": int(
                    audit.set_index("checkpoint").loc[right, "completed_target_updates_by_n_calls"]
                    - audit.set_index("checkpoint").loc[left, "completed_target_updates_by_n_calls"]
                ),
            }
        )
    return audit, pd.DataFrame(differences)


def calculate_td_frame(buffer_step: int, model: DQN) -> tuple[pd.DataFrame, float]:
    model.load_replay_buffer(str(replay_path(buffer_step)))
    replay = model.replay_buffer
    if replay is None:
        raise RuntimeError(f"Replay buffer missing at {buffer_step}")
    size = replay.buffer_size if replay.full else replay.pos
    observations = replay.observations[:size, 0]
    next_observations = replay.next_observations[:size, 0]
    actions = replay.actions[:size, 0, 0].astype(int)
    rewards = replay.rewards[:size, 0]
    dones = replay.dones[:size, 0] * (1.0 - replay.timeouts[:size, 0])
    discount = float(replay.gamma ** replay.n_steps)

    current_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, size, batch_size):
            stop = min(size, start + batch_size)
            obs = torch.as_tensor(observations[start:stop], device=model.device)
            next_obs = torch.as_tensor(next_observations[start:stop], device=model.device)
            action = torch.as_tensor(actions[start:stop], device=model.device).long().reshape(-1, 1)
            reward = torch.as_tensor(rewards[start:stop], device=model.device).reshape(-1)
            done = torch.as_tensor(dones[start:stop], device=model.device).reshape(-1)
            current = model.q_net(obs).gather(1, action).reshape(-1)
            next_max = model.q_net_target(next_obs).max(dim=1).values
            target = reward + (1.0 - done) * discount * next_max
            current_values.append(current.cpu().numpy())
            target_values.append(target.cpu().numpy())

    current = np.concatenate(current_values)
    target = np.concatenate(target_values)
    error = target - current
    huber = F.smooth_l1_loss(
        torch.as_tensor(current), torch.as_tensor(target), reduction="none"
    ).numpy()
    frame = pd.DataFrame(
        {
            "action_index": actions,
            "reward": rewards,
            "done": dones,
            "q_online_selected": current,
            "td_target": target,
            "td_error": error,
            "abs_td_error": np.abs(error),
            "huber_loss": huber,
        }
    )
    return frame, discount


def summarize_td(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "count": len(group),
            "td_error_mean": group["td_error"].mean(),
            "abs_td_error_mean": group["abs_td_error"].mean(),
            "abs_td_error_median": group["abs_td_error"].median(),
            "abs_td_error_p95": group["abs_td_error"].quantile(0.95),
            "abs_td_error_max": group["abs_td_error"].max(),
            "huber_loss_mean": group["huber_loss"].mean(),
            "reward_nonzero_fraction": group["reward"].ne(0).mean(),
            "reward_max": group["reward"].max(),
        }
    )


def td_residuals(step: int, model: DQN) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, discount = calculate_td_frame(step, model)

    overall = summarize_td(frame).to_frame().T
    overall.insert(0, "checkpoint", step)
    overall.insert(1, "discount", discount)
    by_action = (
        frame.groupby("action_index", sort=True)
        .apply(summarize_td, include_groups=False)
        .reset_index()
    )
    by_action.insert(0, "checkpoint", step)
    return overall, by_action


def td_cross_evaluation(models: dict[int, DQN]) -> pd.DataFrame:
    rows = []
    for buffer_step in (15000, 20000):
        for model_step in (15000, 20000):
            frame, discount = calculate_td_frame(buffer_step, models[model_step])
            summary = summarize_td(frame).to_dict()
            rows.append(
                {
                    "buffer_checkpoint": buffer_step,
                    "evaluated_model_checkpoint": model_step,
                    "discount": discount,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def nitrogen_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_csv(Q_INPUT)
    alias = pd.read_csv(ALIAS_INPUT)[["state_id", "unique_executed_action_count"]]
    q = q[q["evaluated_checkpoint"].isin([15000, 20000])].copy()
    rows = []
    keys = ["state_id", "source_checkpoint", "target_dap", "actual_dap", "evaluated_checkpoint"]
    for key, group in q.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        for network in ("online", "target"):
            for irrigation, sub in group.groupby("requested_irrigation"):
                values = {
                    int(row.requested_nitrogen): float(getattr(row, f"{network}_q"))
                    for row in sub.itertuples()
                }
                order = sorted(values, key=lambda n: values[n], reverse=True)
                rows.append(
                    {
                        **base,
                        "network": network,
                        "irrigation_group": float(irrigation),
                        "q_n0": values[0],
                        "q_n50": values[50],
                        "q_n100": values[100],
                        "ranking": ">".join(f"N{n}" for n in order),
                        "n50_gt_n100": values[50] > values[100],
                        "n0_gt_n50": values[0] > values[50],
                        "n0_gt_n100": values[0] > values[100],
                    }
                )
    ranking = pd.DataFrame(rows).merge(alias, on="state_id", how="left")

    changes = []
    group_keys = [
        "state_id",
        "source_checkpoint",
        "target_dap",
        "actual_dap",
        "network",
        "irrigation_group",
        "unique_executed_action_count",
    ]
    for key, group in ranking.groupby(group_keys, sort=False):
        indexed = group.set_index("evaluated_checkpoint")
        if 15000 not in indexed.index or 20000 not in indexed.index:
            continue
        changes.append(
            {
                **dict(zip(group_keys, key)),
                "ranking_15k": indexed.loc[15000, "ranking"],
                "ranking_20k": indexed.loc[20000, "ranking"],
                "full_ranking_changed": indexed.loc[15000, "ranking"]
                != indexed.loc[20000, "ranking"],
                "n50_vs_n100_changed": bool(
                    indexed.loc[15000, "n50_gt_n100"]
                    != indexed.loc[20000, "n50_gt_n100"]
                ),
                "n0_vs_n50_changed": bool(
                    indexed.loc[15000, "n0_gt_n50"] != indexed.loc[20000, "n0_gt_n50"]
                ),
                "n0_vs_n100_changed": bool(
                    indexed.loc[15000, "n0_gt_n100"] != indexed.loc[20000, "n0_gt_n100"]
                ),
            }
        )
    return ranking, pd.DataFrame(changes)


def plot_summary(
    target_audit_frame: pd.DataFrame,
    parameter_differences: pd.DataFrame,
    td_summary: pd.DataFrame,
    ranking_changes: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].plot(
        target_audit_frame["checkpoint"],
        target_audit_frame["calls_since_last_target_update"],
        marker="o",
        color="#2B6CB0",
    )
    axes[0].set_title("Target update counter phase")
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("n_calls modulo 10000")

    axes[1].plot(
        td_summary["checkpoint"],
        td_summary["abs_td_error_mean"],
        marker="o",
        label="Mean |TD error|",
        color="#B23A35",
    )
    axes[1].plot(
        td_summary["checkpoint"],
        td_summary["abs_td_error_p95"],
        marker="s",
        label="P95 |TD error|",
        color="#D69E2E",
    )
    axes[1].set_title("Offline Bellman residual")
    axes[1].set_xlabel("Checkpoint")
    axes[1].legend(frameon=False)

    counts = (
        ranking_changes.groupby("network")[["full_ranking_changed", "n50_vs_n100_changed"]]
        .sum()
        .reindex(["online", "target"])
    )
    x = np.arange(len(counts.index))
    axes[2].bar(x - 0.18, counts["full_ranking_changed"], 0.36, label="Full ranking")
    axes[2].bar(x + 0.18, counts["n50_vs_n100_changed"], 0.36, label="N50 vs N100")
    axes[2].set_xticks(x, counts.index)
    axes[2].set_title("15K to 20K nitrogen ranking changes")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(True, color="#E5E5E5", lw=0.7, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "021_07_target_td_nitrogen_ranking_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {step: DQN.load(str(model_path(step))) for step in CHECKPOINTS}
    target_frame, differences = target_audit(models)
    overall_frames = []
    action_frames = []
    for step, model in models.items():
        overall, by_action = td_residuals(step, model)
        overall_frames.append(overall)
        action_frames.append(by_action)
    td_summary = pd.concat(overall_frames, ignore_index=True)
    td_by_action = pd.concat(action_frames, ignore_index=True)
    td_cross = td_cross_evaluation(models)
    ranking, ranking_changes = nitrogen_ranking()

    target_frame.to_csv(OUT / "021_07_target_update_audit.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(
        OUT / "021_07_network_parameter_differences.csv", index=False, encoding="utf-8-sig"
    )
    td_summary.to_csv(OUT / "021_07_td_residual_summary.csv", index=False, encoding="utf-8-sig")
    td_by_action.to_csv(OUT / "021_07_td_residual_by_action.csv", index=False, encoding="utf-8-sig")
    td_cross.to_csv(OUT / "021_07_td_cross_evaluation.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT / "021_07_full_nitrogen_ranking.csv", index=False, encoding="utf-8-sig")
    ranking_changes.to_csv(
        OUT / "021_07_full_nitrogen_ranking_changes.csv", index=False, encoding="utf-8-sig"
    )
    plot_summary(target_frame, differences, td_summary, ranking_changes)

    summary = {
        "checkpoint_archives_all_distinct": bool(target_frame["model_zip_sha256"].nunique() == 5),
        "saved_and_runtime_sb3_versions_match": bool(
            (target_frame["saved_sb3_version"] == target_frame["runtime_sb3_version"]).all()
        ),
        "target_unchanged_5k_to_10k": bool(
            not differences.set_index("to_checkpoint").loc[10000, "target_hash_changed"]
        ),
        "target_changed_10k_to_15k": bool(
            differences.set_index("to_checkpoint").loc[15000, "target_hash_changed"]
        ),
        "target_unchanged_15k_to_20k": bool(
            not differences.set_index("to_checkpoint").loc[20000, "target_hash_changed"]
        ),
        "target_changed_20k_to_25k": bool(
            differences.set_index("to_checkpoint").loc[25000, "target_hash_changed"]
        ),
        "online_full_n_ranking_changes": int(
            ranking_changes.query("network == 'online'")["full_ranking_changed"].sum()
        ),
        "online_n50_vs_n100_changes": int(
            ranking_changes.query("network == 'online'")["n50_vs_n100_changed"].sum()
        ),
        "target_full_n_ranking_changes": int(
            ranking_changes.query("network == 'target'")["full_ranking_changed"].sum()
        ),
        "historical_training_loss_available": False,
        "td_metric_scope": "posthoc checkpoint Bellman residual on each saved replay buffer",
        "td_cross_evaluation_completed": True,
    }
    (OUT / "021_07_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTarget audit:")
    print(target_frame.to_string(index=False))
    print("\nTD summary:")
    print(td_summary.to_string(index=False))


if __name__ == "__main__":
    main()

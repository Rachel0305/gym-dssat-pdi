"""021_11：SY2014 seed1 下降-恢复轨迹的纯离线Q/replay/TD审计。"""

from __future__ import annotations

import hashlib
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
RUN = ROOT / "benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1"
FIXED = ROOT / "benchmark_results/021_06/021_06_fixed_state_manifest.csv"
Q_MAP = ROOT / "benchmark_results/021_06/021_06_q_values_long.csv"
OUT = ROOT / "benchmark_results/021_11"
CHECKPOINTS = (10000, 15000, 20000, 25000)
WINDOWS = ((10000, 15000, "10K-15K"), (15000, 20000, "15K-20K"), (20000, 25000, "20K-25K"))


def checkpoint_dir(step: int) -> Path:
    return RUN / "checkpoints" / f"checkpoint_{step}"


def load_model(step: int) -> DQN:
    return DQN.load(str(checkpoint_dir(step) / "model.zip"), device="cpu")


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def circular_indices(start: int, end: int, capacity: int) -> np.ndarray:
    if start == end:
        return np.arange(capacity, dtype=int)
    if end > start:
        return np.arange(start, end, dtype=int)
    return np.concatenate((np.arange(start, capacity), np.arange(0, end))).astype(int)


def load_buffer(model: DQN, step: int):
    model.load_replay_buffer(str(checkpoint_dir(step) / "replay_buffer.pkl"))
    if model.replay_buffer is None:
        raise RuntimeError(f"missing replay buffer {step}")
    return model.replay_buffer


def entropy(actions: np.ndarray) -> float:
    p = np.bincount(actions, minlength=9).astype(float)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(9.0))


def td_summary(model: DQN, buffer_step: int) -> dict[str, float]:
    replay = load_buffer(model, buffer_step)
    size = replay.buffer_size if replay.full else replay.pos
    obs = replay.observations[:size, 0]
    next_obs = replay.next_observations[:size, 0]
    actions = replay.actions[:size, 0, 0].astype(int)
    rewards = replay.rewards[:size, 0]
    dones = replay.dones[:size, 0] * (1.0 - replay.timeouts[:size, 0])
    discount = float(replay.gamma**replay.n_steps)
    current_parts, target_parts = [], []
    with torch.no_grad():
        for start in range(0, size, 1024):
            stop = min(size, start + 1024)
            o = torch.as_tensor(obs[start:stop], device=model.device)
            no = torch.as_tensor(next_obs[start:stop], device=model.device)
            a = torch.as_tensor(actions[start:stop], device=model.device).long().reshape(-1, 1)
            current_parts.append(model.q_net(o).gather(1, a).reshape(-1).cpu().numpy())
            target_parts.append(model.q_net_target(no).max(dim=1).values.cpu().numpy())
    current = np.concatenate(current_parts)
    next_max = np.concatenate(target_parts)
    target = rewards + (1.0 - dones) * discount * next_max
    error = target - current
    return {
        "count": int(size),
        "td_error_mean": float(error.mean()),
        "abs_td_error_mean": float(np.abs(error).mean()),
        "abs_td_error_median": float(np.median(np.abs(error))),
        "abs_td_error_p95": float(np.quantile(np.abs(error), 0.95)),
        "abs_td_error_max": float(np.abs(error).max()),
    }


def ranking(values: dict[int, float]) -> str:
    return ">".join(f"N{n}" for n, _ in sorted(values.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED)
    fixed["observation"] = fixed.observation_vector_json.map(lambda text: np.asarray(json.loads(text), dtype=np.float32))
    observations = np.stack(fixed.observation.to_list())
    mapping = pd.read_csv(Q_MAP)[["action_index", "requested_irrigation", "requested_nitrogen"]].drop_duplicates()
    action_map = mapping.set_index("action_index").to_dict("index")

    q_rows, network_rows = [], []
    for step in CHECKPOINTS:
        model = load_model(step)
        with torch.no_grad():
            tensor = torch.as_tensor(observations, device=model.device)
            online = model.q_net(tensor).cpu().numpy()
            target = model.q_net_target(tensor).cpu().numpy()
        network_rows.append(
            {
                "checkpoint": step,
                "num_timesteps": int(model.num_timesteps),
                "n_calls": int(model._n_calls),
                "exploration_rate": float(model.exploration_rate),
                "online_parameter_sha256": parameter_hash(model.q_net),
                "target_parameter_sha256": parameter_hash(model.q_net_target),
            }
        )
        for state_index, state in fixed.iterrows():
            for action in range(9):
                q_rows.append(
                    {
                        "checkpoint": step,
                        "state_id": state.state_id,
                        "source_checkpoint": int(state.source_checkpoint),
                        "actual_dap": int(state.dap),
                        "action_index": action,
                        "requested_irrigation": action_map[action]["requested_irrigation"],
                        "requested_nitrogen": action_map[action]["requested_nitrogen"],
                        "online_q": float(online[state_index, action]),
                        "target_q": float(target[state_index, action]),
                    }
                )
    q_long = pd.DataFrame(q_rows)
    networks = pd.DataFrame(network_rows)
    q_long.to_csv(OUT / "021_11_fixed_state_q_values.csv", index=False)
    networks.to_csv(OUT / "021_11_network_audit.csv", index=False)

    ranking_rows = []
    argmax_rows = []
    for start_step, end_step, window in WINDOWS:
        for network in ("online", "target"):
            qcol = f"{network}_q"
            for (state_id, irrigation), group in q_long.groupby(["state_id", "requested_irrigation"], sort=False):
                a = group[group.checkpoint == start_step]
                b = group[group.checkpoint == end_step]
                va = {int(row.requested_nitrogen): float(getattr(row, qcol)) for row in a.itertuples()}
                vb = {int(row.requested_nitrogen): float(getattr(row, qcol)) for row in b.itertuples()}
                ranking_rows.append(
                    {
                        "window": window,
                        "network": network,
                        "state_id": state_id,
                        "irrigation_group": irrigation,
                        "ranking_start": ranking(va),
                        "ranking_end": ranking(vb),
                        "full_ranking_changed": ranking(va) != ranking(vb),
                        "n50_vs_n100_changed": (va[50] > va[100]) != (vb[50] > vb[100]),
                        "n0_vs_n50_changed": (va[0] > va[50]) != (vb[0] > vb[50]),
                        "n0_vs_n100_changed": (va[0] > va[100]) != (vb[0] > vb[100]),
                    }
                )
            for state_id, group in q_long.groupby("state_id", sort=False):
                a = group[group.checkpoint == start_step]
                b = group[group.checkpoint == end_step]
                argmax_rows.append(
                    {
                        "window": window,
                        "network": network,
                        "state_id": state_id,
                        "argmax_start": int(a.loc[a[qcol].idxmax(), "action_index"]),
                        "argmax_end": int(b.loc[b[qcol].idxmax(), "action_index"]),
                        "argmax_changed": int(a.loc[a[qcol].idxmax(), "action_index"])
                        != int(b.loc[b[qcol].idxmax(), "action_index"]),
                    }
                )
    rankings = pd.DataFrame(ranking_rows)
    argmax = pd.DataFrame(argmax_rows)
    rankings.to_csv(OUT / "021_11_nitrogen_ranking_changes.csv", index=False)
    argmax.to_csv(OUT / "021_11_argmax_changes.csv", index=False)

    replay_rows, replay_detail = [], []
    for start_step, end_step, window in WINDOWS:
        start_model = load_model(start_step)
        start_buffer = load_buffer(start_model, start_step)
        start_pos = int(start_buffer.pos)
        del start_model
        end_model = load_model(end_step)
        replay = load_buffer(end_model, end_step)
        indices = circular_indices(start_pos, int(replay.pos), int(replay.buffer_size))
        actions = replay.actions[indices, 0, 0].astype(int)
        rewards = replay.rewards[indices, 0].astype(float)
        requested_i = np.array([action_map[a]["requested_irrigation"] for a in actions])
        requested_n = np.array([action_map[a]["requested_nitrogen"] for a in actions])
        for action_index, count in enumerate(np.bincount(actions, minlength=9)):
            replay_detail.append(
                {
                    "window": window,
                    "action_index": action_index,
                    "requested_irrigation": action_map[action_index]["requested_irrigation"],
                    "requested_nitrogen": action_map[action_index]["requested_nitrogen"],
                    "count": int(count),
                    "fraction": float(count / len(actions)),
                }
            )
        replay_rows.append(
            {
                "window": window,
                "transition_count": len(actions),
                "positive_irrigation_fraction": float((requested_i > 0).mean()),
                "positive_nitrogen_fraction": float((requested_n > 0).mean()),
                "n0_fraction": float((requested_n == 0).mean()),
                "n50_fraction": float((requested_n == 50).mean()),
                "n100_fraction": float((requested_n == 100).mean()),
                "joint_positive_fraction": float(((requested_i > 0) & (requested_n > 0)).mean()),
                "normalized_action_entropy": entropy(actions),
                "reward_mean": float(rewards.mean()),
                "reward_negative_fraction": float((rewards < 0).mean()),
                "reward_positive_fraction": float((rewards > 0).mean()),
            }
        )
        del end_model
    replay_summary = pd.DataFrame(replay_rows)
    replay_distribution = pd.DataFrame(replay_detail)
    replay_summary.to_csv(OUT / "021_11_replay_window_summary.csv", index=False)
    replay_distribution.to_csv(OUT / "021_11_replay_action_distribution.csv", index=False)

    td_rows = []
    for start_step, end_step, window in WINDOWS:
        for model_step in (start_step, end_step):
            model = load_model(model_step)
            td_rows.append(
                {
                    "window": window,
                    "buffer_checkpoint": end_step,
                    "evaluated_model_checkpoint": model_step,
                    **td_summary(model, end_step),
                }
            )
            del model
    td = pd.DataFrame(td_rows)
    td.to_csv(OUT / "021_11_fixed_buffer_td_comparison.csv", index=False)

    rank_summary = (
        rankings.groupby(["window", "network"], sort=False)
        .agg(
            comparisons=("state_id", "size"),
            full_ranking_changes=("full_ranking_changed", "sum"),
            n50_vs_n100_changes=("n50_vs_n100_changed", "sum"),
        )
        .reset_index()
    )
    arg_summary = (
        argmax.groupby(["window", "network"], sort=False)
        .argmax_changed.sum().rename("global_argmax_changes").reset_index()
    )
    summary = rank_summary.merge(arg_summary, on=["window", "network"])
    summary.to_csv(OUT / "021_11_q_change_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    online = summary[summary.network == "online"].set_index("window")
    labels = [item[2] for item in WINDOWS]
    axes[0].bar(labels, online.loc[labels, "full_ranking_changes"], color=["#4C78A8", "#F2CF5B", "#59A14F"])
    axes[0].set_title("Online N-ranking changes")
    axes[0].set_ylabel("Count / 54")
    axes[1].plot(labels, replay_summary.set_index("window").loc[labels, "positive_nitrogen_fraction"], "o-", color="#E45756")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Positive-N requests in new replay")
    axes[1].set_ylabel("Fraction")
    td_end = td[td.evaluated_model_checkpoint == td.buffer_checkpoint].set_index("window")
    axes[2].plot(labels, td_end.loc[labels, "abs_td_error_mean"], "o-", color="#B279A2")
    axes[2].set_title("Endpoint model Bellman residual")
    axes[2].set_ylabel("Mean |TD error|")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("SY2014 seed1: offline audit of decline and recovery")
    fig.tight_layout()
    fig.savefig(OUT / "021_11_seed1_recovery_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed_offline",
        "training": False,
        "dssat_called": False,
        "fixed_state_origin": "021_06 seed0 reference trajectories; used only for same-state network comparison",
        "q_change_summary": summary.to_dict("records"),
        "replay_summary": replay_summary.to_dict("records"),
        "td_summary": td.to_dict("records"),
    }
    (OUT / "021_11_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Q change summary:\n", summary.to_string(index=False))
    print("\nReplay summary:\n", replay_summary.to_string(index=False))
    print("\nFixed-buffer TD:\n", td.to_string(index=False))


if __name__ == "__main__":
    main()

"""Offline process comparison: unscaled 021_10 vs reward-scaled 021_14."""

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
OUT = ROOT / "benchmark_results/021_14"
FIXED = ROOT / "benchmark_results/021_06/021_06_fixed_state_manifest.csv"
ACTION_SOURCE = ROOT / "benchmark_results/021_06/021_06_q_values_long.csv"
RUNS = {
    "unscaled_021_10": ROOT
    / "benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1",
    "scaled_0p1_021_14": ROOT
    / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1",
}
CHECKPOINTS = (5000, 10000, 15000, 20000, 25000)
WINDOWS = tuple(zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]))


def load_model(protocol: str, step: int) -> DQN:
    path = RUNS[protocol] / "checkpoints" / f"checkpoint_{step}" / "model.zip"
    return DQN.load(str(path), device="cpu")


def module_vector(module: torch.nn.Module) -> np.ndarray:
    return np.concatenate(
        [
            value.detach().cpu().contiguous().numpy().reshape(-1).astype(np.float64)
            for _, value in sorted(module.state_dict().items())
        ]
    )


def ranking(values: dict[int, float]) -> str:
    return ">".join(f"N{key}" for key, _ in sorted(values.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    fixed = pd.read_csv(FIXED)
    fixed["observation"] = fixed.observation_vector_json.map(
        lambda text: np.asarray(json.loads(text), dtype=np.float32)
    )
    observations = np.stack(fixed.observation.to_list())
    action_map = (
        pd.read_csv(ACTION_SOURCE)[
            ["action_index", "requested_irrigation", "requested_nitrogen"]
        ]
        .drop_duplicates()
        .sort_values("action_index")
    )

    models: dict[tuple[str, int], DQN] = {}
    q_rows: list[dict] = []
    magnitude_rows: list[dict] = []
    trajectory_frames: list[pd.DataFrame] = []
    for protocol, run in RUNS.items():
        trajectory = pd.read_csv(run / "evaluations/season_summary_all_checkpoints.csv")
        trajectory.insert(0, "protocol", protocol)
        trajectory_frames.append(trajectory)
        for step in CHECKPOINTS:
            model = load_model(protocol, step)
            models[(protocol, step)] = model
            with torch.no_grad():
                tensor = torch.as_tensor(observations, device=model.device)
                online = model.q_net(tensor).cpu().numpy()
                target = model.q_net_target(tensor).cpu().numpy()
            for network, values in (("online", online), ("target", target)):
                magnitude_rows.append(
                    {
                        "protocol": protocol,
                        "checkpoint": step,
                        "network": network,
                        "mean_abs_q": float(np.abs(values).mean()),
                        "max_abs_q": float(np.abs(values).max()),
                        "q_std": float(values.std()),
                    }
                )
            for state_index, state in fixed.iterrows():
                for action in action_map.itertuples(index=False):
                    action_index = int(action.action_index)
                    q_rows.append(
                        {
                            "protocol": protocol,
                            "checkpoint": step,
                            "state_id": state.state_id,
                            "dap": int(state.dap),
                            "action_index": action_index,
                            "requested_irrigation": float(action.requested_irrigation),
                            "requested_nitrogen": float(action.requested_nitrogen),
                            "online_q": float(online[state_index, action_index]),
                            "target_q": float(target[state_index, action_index]),
                        }
                    )
    q_long = pd.DataFrame(q_rows)
    magnitude = pd.DataFrame(magnitude_rows)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)

    event_rows: list[dict] = []
    q_change_rows: list[dict] = []
    for protocol in RUNS:
        for start, end in WINDOWS:
            row: dict[str, object] = {
                "protocol": protocol,
                "window": f"{start // 1000}K-{end // 1000}K",
                "start_checkpoint": start,
                "end_checkpoint": end,
            }
            for network, attr in (("online", "q_net"), ("target", "q_net_target")):
                before = module_vector(getattr(models[(protocol, start)], attr))
                after = module_vector(getattr(models[(protocol, end)], attr))
                delta = float(np.linalg.norm(after - before))
                row[f"{network}_parameter_l2_delta"] = delta
                row[f"{network}_parameter_relative_l2_delta"] = delta / float(np.linalg.norm(before))
            row["target_updated"] = bool(row["target_parameter_l2_delta"] > 0.0)
            event_rows.append(row)

            for network in ("online", "target"):
                qcol = f"{network}_q"
                a = q_long[(q_long.protocol == protocol) & (q_long.checkpoint == start)]
                b = q_long[(q_long.protocol == protocol) & (q_long.checkpoint == end)]
                merged = a.merge(
                    b,
                    on=[
                        "protocol",
                        "state_id",
                        "dap",
                        "action_index",
                        "requested_irrigation",
                        "requested_nitrogen",
                    ],
                    suffixes=("_start", "_end"),
                )
                changes = (merged[f"{qcol}_end"] - merged[f"{qcol}_start"]).abs()
                rank_changes = 0
                argmax_changes = 0
                for (_, irrigation), group in merged.groupby(["state_id", "requested_irrigation"]):
                    va = {
                        int(r.requested_nitrogen): float(getattr(r, f"{qcol}_start"))
                        for r in group.itertuples()
                    }
                    vb = {
                        int(r.requested_nitrogen): float(getattr(r, f"{qcol}_end"))
                        for r in group.itertuples()
                    }
                    rank_changes += int(ranking(va) != ranking(vb))
                for _, group in merged.groupby("state_id"):
                    argmax_start = int(group.loc[group[f"{qcol}_start"].idxmax(), "action_index"])
                    argmax_end = int(group.loc[group[f"{qcol}_end"].idxmax(), "action_index"])
                    argmax_changes += int(argmax_start != argmax_end)
                q_change_rows.append(
                    {
                        "protocol": protocol,
                        "window": row["window"],
                        "target_updated": row["target_updated"],
                        "network": network,
                        "mean_abs_q_change": float(changes.mean()),
                        "max_abs_q_change": float(changes.max()),
                        "full_n_ranking_changes_out_of_54": rank_changes,
                        "argmax_changes_out_of_18": argmax_changes,
                    }
                )

    events = pd.DataFrame(event_rows)
    q_changes = pd.DataFrame(q_change_rows)
    trajectories.to_csv(OUT / "021_14_scaled_vs_unscaled_trajectory.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT / "021_14_scaled_vs_unscaled_parameter_changes.csv", index=False, encoding="utf-8-sig")
    magnitude.to_csv(OUT / "021_14_scaled_vs_unscaled_q_magnitude.csv", index=False, encoding="utf-8-sig")
    q_changes.to_csv(OUT / "021_14_scaled_vs_unscaled_q_changes.csv", index=False, encoding="utf-8-sig")
    q_long.to_csv(OUT / "021_14_scaled_vs_unscaled_fixed_state_q_long.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"unscaled_021_10": "#4C78A8", "scaled_0p1_021_14": "#D55E00"}
    for protocol, group in trajectories.groupby("protocol"):
        group = group.sort_values("checkpoint")
        axes[0, 0].plot(group.checkpoint, group.yield_kg_ha, "o-", label=protocol, color=colors[protocol])
        axes[0, 1].plot(group.checkpoint, group.nitrogen_kg_ha, "o-", label=protocol, color=colors[protocol])
    axes[0, 0].set_title("Deterministic checkpoint yield")
    axes[0, 0].set_ylabel("kg/ha")
    axes[0, 1].set_title("Season nitrogen")
    axes[0, 1].set_ylabel("kg N/ha")

    online_mag = magnitude[magnitude.network == "online"]
    for protocol, group in online_mag.groupby("protocol"):
        axes[1, 0].plot(group.checkpoint, group.mean_abs_q, "o-", label=protocol, color=colors[protocol])
    axes[1, 0].set_title("Fixed-state online mean |Q|")
    online_change = q_changes[q_changes.network == "online"]
    labels = list(dict.fromkeys(online_change.window))
    x = np.arange(len(labels))
    for offset, protocol in zip((-0.18, 0.18), RUNS):
        group = online_change[online_change.protocol == protocol].set_index("window").loc[labels]
        axes[1, 1].bar(
            x + offset,
            group.full_n_ranking_changes_out_of_54,
            0.36,
            label=protocol,
            color=colors[protocol],
        )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_title("Online N-ranking changes / 54")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "021_14_scaled_vs_unscaled_process_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    scaled = trajectories[trajectories.protocol == "scaled_0p1_021_14"].sort_values("checkpoint")
    unscaled = trajectories[trajectories.protocol == "unscaled_021_10"].sort_values("checkpoint")
    summary = {
        "status": "completed",
        "comparison_is_single_variable_within_exploration_fraction_0p70": True,
        "scaled_yield_range_kg_ha": [float(scaled.yield_kg_ha.min()), float(scaled.yield_kg_ha.max())],
        "unscaled_yield_range_kg_ha": [float(unscaled.yield_kg_ha.min()), float(unscaled.yield_kg_ha.max())],
        "scaled_has_two_large_intermediate_declines": bool(
            (scaled.set_index("checkpoint").loc[10000, "yield_kg_ha"] < 9000)
            and (scaled.set_index("checkpoint").loc[20000, "yield_kg_ha"] < 9000)
        ),
        "scaled_final_yield_kg_ha": float(scaled.iloc[-1].yield_kg_ha),
        "unscaled_final_yield_kg_ha": float(unscaled.iloc[-1].yield_kg_ha),
        "interpretation": (
            "Reward scaling changes Q magnitude and training trajectory but does not eliminate "
            "checkpoint instability for seed1; it is not a sufficient standalone fix."
        ),
    }
    (OUT / "021_14_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


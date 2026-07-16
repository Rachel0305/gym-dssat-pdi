"""021_12：纯离线 target 更新幅度、固定状态 Q 变化和 reward 量级审计。"""

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
OUT = ROOT / "benchmark_results" / "021_12"
FIXED = ROOT / "benchmark_results/021_06/021_06_fixed_state_manifest.csv"
ACTION_SOURCE = ROOT / "benchmark_results/021_06/021_06_q_values_long.csv"
RUNS = {
    0: ROOT / "benchmark_results/021_09/021_09_sy2014_extended_exploration_seed0_25k__sy_2014_seed0",
    1: ROOT / "benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1",
}
CHECKPOINTS = (10000, 15000, 20000, 25000)
WINDOWS = tuple(zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]))


def checkpoint_model(seed: int, step: int) -> Path:
    return RUNS[seed] / "checkpoints" / f"checkpoint_{step}" / "model.zip"


def load_model(seed: int, step: int) -> DQN:
    path = checkpoint_model(seed, step)
    if not path.exists():
        raise FileNotFoundError(path)
    return DQN.load(str(path), device="cpu")


def module_vector(module: torch.nn.Module) -> np.ndarray:
    parts = [
        value.detach().cpu().contiguous().numpy().reshape(-1).astype(np.float64)
        for _, value in sorted(module.state_dict().items())
    ]
    return np.concatenate(parts)


def ranking(values: dict[int, float]) -> str:
    return ">".join(f"N{key}" for key, _ in sorted(values.items(), key=lambda item: (-item[1], item[0])))


def network_and_q_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    models: dict[tuple[int, int], DQN] = {}
    q_rows: list[dict] = []
    for seed in RUNS:
        for step in CHECKPOINTS:
            model = load_model(seed, step)
            models[(seed, step)] = model
            with torch.no_grad():
                tensor = torch.as_tensor(observations, device=model.device)
                online = model.q_net(tensor).cpu().numpy()
                target = model.q_net_target(tensor).cpu().numpy()
            for state_index, state in fixed.iterrows():
                for action in action_map.itertuples(index=False):
                    q_rows.append(
                        {
                            "seed": seed,
                            "checkpoint": step,
                            "state_id": state.state_id,
                            "dap": int(state.dap),
                            "action_index": int(action.action_index),
                            "requested_irrigation": float(action.requested_irrigation),
                            "requested_nitrogen": float(action.requested_nitrogen),
                            "online_q": float(online[state_index, int(action.action_index)]),
                            "target_q": float(target[state_index, int(action.action_index)]),
                        }
                    )
    q_long = pd.DataFrame(q_rows)

    event_rows: list[dict] = []
    q_summary_rows: list[dict] = []
    for seed in RUNS:
        for start, end in WINDOWS:
            start_model, end_model = models[(seed, start)], models[(seed, end)]
            event: dict[str, object] = {
                "seed": seed,
                "window": f"{start // 1000}K-{end // 1000}K",
                "start_checkpoint": start,
                "end_checkpoint": end,
                "start_n_calls": int(start_model._n_calls),
                "end_n_calls": int(end_model._n_calls),
                "start_epsilon": float(start_model.exploration_rate),
                "end_epsilon": float(end_model.exploration_rate),
            }
            for network, module_name in (("online", "q_net"), ("target", "q_net_target")):
                before = module_vector(getattr(start_model, module_name))
                after = module_vector(getattr(end_model, module_name))
                delta = float(np.linalg.norm(after - before))
                norm = float(np.linalg.norm(before))
                event[f"{network}_parameter_l2_delta"] = delta
                event[f"{network}_parameter_relative_l2_delta"] = delta / norm if norm else np.nan
            event["target_updated"] = bool(event["target_parameter_l2_delta"] > 0.0)
            event_rows.append(event)

            for network in ("online", "target"):
                qcol = f"{network}_q"
                a = q_long[(q_long.seed == seed) & (q_long.checkpoint == start)].copy()
                b = q_long[(q_long.seed == seed) & (q_long.checkpoint == end)].copy()
                merged = a.merge(
                    b,
                    on=[
                        "seed",
                        "state_id",
                        "dap",
                        "action_index",
                        "requested_irrigation",
                        "requested_nitrogen",
                    ],
                    suffixes=("_start", "_end"),
                )
                abs_change = (merged[f"{qcol}_end"] - merged[f"{qcol}_start"]).abs()
                rank_changes = 0
                argmax_changes = 0
                for (_, irrigation), group in merged.groupby(["state_id", "requested_irrigation"]):
                    start_values = {
                        int(row.requested_nitrogen): float(getattr(row, f"{qcol}_start"))
                        for row in group.itertuples()
                    }
                    end_values = {
                        int(row.requested_nitrogen): float(getattr(row, f"{qcol}_end"))
                        for row in group.itertuples()
                    }
                    rank_changes += int(ranking(start_values) != ranking(end_values))
                for _, group in merged.groupby("state_id"):
                    argmax_start = int(group.loc[group[f"{qcol}_start"].idxmax(), "action_index"])
                    argmax_end = int(group.loc[group[f"{qcol}_end"].idxmax(), "action_index"])
                    argmax_changes += int(argmax_start != argmax_end)
                q_summary_rows.append(
                    {
                        "seed": seed,
                        "window": event["window"],
                        "target_updated": event["target_updated"],
                        "network": network,
                        "mean_abs_q_change": float(abs_change.mean()),
                        "max_abs_q_change": float(abs_change.max()),
                        "full_n_ranking_changes_out_of_54": rank_changes,
                        "argmax_changes_out_of_18": argmax_changes,
                    }
                )
    return pd.DataFrame(event_rows), pd.DataFrame(q_summary_rows), q_long


def reward_scale_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    allowed_stages = {"021_01", "021_09", "021_10"}
    for path in (ROOT / "benchmark_results").rglob("selected_checkpoint.json"):
        relative = path.relative_to(ROOT / "benchmark_results")
        stage = relative.parts[0] if relative.parts else ""
        experiment = relative.parts[1] if len(relative.parts) > 1 else ""
        if stage not in allowed_stages or "smoke" in experiment.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            station = str(summary.get("station_code", "")).upper()
            reward = float(summary["reward_total"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "stage": stage,
                "experiment_id": summary.get("experiment_id", experiment),
                "station_code": station,
                "year": summary.get("year"),
                "seed": summary.get("seed"),
                "checkpoint": summary.get("checkpoint", payload.get("selected_checkpoint")),
                "reward_total": reward,
                "yield_kg_ha": summary.get("yield_kg_ha"),
                "irrigation_mm": summary.get("irrigation_mm"),
                "nitrogen_kg_ha": summary.get("nitrogen_kg_ha"),
                "source_path": str(relative).replace("\\", "/"),
                "comparison_status": (
                    "same_project_wrapper_formula_but_training_protocol_provisional"
                    if station in {"YC", "FQ", "LC"}
                    else "exploration_fixed_sy_candidate"
                ),
            }
        )
    detail = pd.DataFrame(rows).drop_duplicates(subset=["experiment_id"])
    station_rows: list[dict] = []
    for station in ("HLA", "YC", "FQ", "LC", "SY"):
        subset = detail[detail.station_code == station] if not detail.empty else pd.DataFrame()
        station_rows.append(
            {
                "station_code": station,
                "n_selected_results": int(len(subset)),
                "reward_min": float(subset.reward_total.min()) if len(subset) else np.nan,
                "reward_median": float(subset.reward_total.median()) if len(subset) else np.nan,
                "reward_max": float(subset.reward_total.max()) if len(subset) else np.nan,
                "evidence_status": (
                    "missing_standardized_selected_result_in_audited_scopes"
                    if not len(subset)
                    else "provisional_not_formal_cross_site_comparison"
                ),
            }
        )
    station_summary = pd.DataFrame(station_rows)
    sy_median = station_summary.loc[station_summary.station_code == "SY", "reward_median"].iloc[0]
    station_summary["sy_to_station_median_ratio"] = station_summary.reward_median.map(
        lambda value: sy_median / value if pd.notna(value) and value > 0 else np.nan
    )
    return detail, station_summary


def plot_summary(events: pd.DataFrame, q_summary: pd.DataFrame, rewards: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = [f"s{row.seed} {row.window}" for row in events.itertuples()]
    colors = ["#D55E00" if row.target_updated else "#999999" for row in events.itertuples()]
    axes[0].bar(labels, events.target_parameter_relative_l2_delta, color=colors)
    axes[0].set_title("Target parameter relative L2 change")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set_ylabel("relative L2")

    online = q_summary[q_summary.network == "online"]
    target = q_summary[q_summary.network == "target"]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, online.full_n_ranking_changes_out_of_54, 0.36, label="online")
    axes[1].bar(x + 0.18, target.full_n_ranking_changes_out_of_54, 0.36, label="target")
    axes[1].set_xticks(x, labels, rotation=55)
    axes[1].set_title("N-ranking changes on 18 fixed states")
    axes[1].set_ylabel("changes out of 54")
    axes[1].legend(frameon=False)

    valid = rewards.dropna(subset=["reward_median"])
    axes[2].bar(valid.station_code, valid.reward_median, color="#4C78A8")
    axes[2].set_title("Selected reward median (provisional)")
    axes[2].set_ylabel("reward_total")
    axes[2].axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "021_12_target_q_reward_scale_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, q_summary, q_long = network_and_q_audit()
    reward_detail, reward_summary = reward_scale_evidence()
    events.to_csv(OUT / "021_12_network_change_events.csv", index=False, encoding="utf-8-sig")
    q_summary.to_csv(OUT / "021_12_fixed_state_q_change_summary.csv", index=False, encoding="utf-8-sig")
    q_long.to_csv(OUT / "021_12_fixed_state_q_values_long.csv", index=False, encoding="utf-8-sig")
    reward_detail.to_csv(OUT / "021_12_reward_scale_evidence_detail.csv", index=False, encoding="utf-8-sig")
    reward_summary.to_csv(OUT / "021_12_reward_scale_evidence.csv", index=False, encoding="utf-8-sig")
    plot_summary(events, q_summary, reward_summary)

    update = events[events.target_updated]
    frozen = events[~events.target_updated]
    sy_row = reward_summary[reward_summary.station_code == "SY"].iloc[0]
    comparison = reward_summary[
        reward_summary.station_code.isin(["YC", "FQ", "LC"])
        & reward_summary.reward_median.notna()
        & (reward_summary.reward_median > 0)
    ]
    ratios = comparison.sy_to_station_median_ratio.to_numpy(dtype=float)
    summary = {
        "status": "completed_offline",
        "training_or_dssat_calls": 0,
        "target_update_events": int(len(update)),
        "target_frozen_events": int(len(frozen)),
        "target_update_relative_l2_range": [
            float(update.target_parameter_relative_l2_delta.min()),
            float(update.target_parameter_relative_l2_delta.max()),
        ],
        "target_frozen_relative_l2_max": float(frozen.target_parameter_relative_l2_delta.max()),
        "sy_selected_reward_median": float(sy_row.reward_median),
        "sy_to_yc_fq_lc_median_ratio_range": (
            [float(np.min(ratios)), float(np.max(ratios))] if len(ratios) else None
        ),
        "hla_reward_evidence_status": reward_summary.loc[
            reward_summary.station_code == "HLA", "evidence_status"
        ].iloc[0],
        "interpretation_limit": (
            "descriptive only; few update events and provisional cross-site reward scopes; "
            "do not infer causality or call reward scale a confirmed root cause"
        ),
    }
    (OUT / "021_12_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


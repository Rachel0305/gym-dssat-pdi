from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
from frozen_nstep_dqn_config_020_11 import (
    ACTION_TABLE_9,
    DAILY_IRRIGATION_CAP,
    DAILY_NITROGEN_CAP,
    IRRIGATION_BUDGET,
    IRRIGATION_WINDOWS,
    MIN_INTERVAL_DAYS,
    NITROGEN_BUDGET,
    NITROGEN_WINDOWS,
    apply_environment_constants,
)


RUN = (
    ROOT
    / "benchmark_results"
    / "021_05"
    / "021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"
)
OUT = ROOT / "benchmark_results" / "021_06"
CHECKPOINTS = (5000, 10000, 15000, 20000, 25000)
REFERENCE_CHECKPOINTS = (15000, 20000)
TARGET_DAPS = (1, 8, 15, 22, 29, 36, 50, 80, 121)
IRRIGATION_GROUPS = {
    0.0: (0, 3, 6),
    15.0: (1, 4, 7),
    30.0: (2, 5, 8),
}


def scalar(value) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else np.nan


def in_window(dap: int, windows: list[tuple[int, int]]) -> bool:
    return any(left <= dap <= right for left, right in windows)


def project_action(
    action_index: int,
    *,
    dap: int,
    used_irrigation: float,
    used_nitrogen: float,
    last_operation_dap: int | None,
) -> tuple[float, float, str]:
    raw = ACTION_TABLE_9[action_index]
    raw_i = float(raw["amir"])
    raw_n = float(raw["anfer"])
    reasons: list[str] = []
    if dap < 1:
        return 0.0, 0.0, "dap_lt_1"
    can_operate = last_operation_dap is None or dap - last_operation_dap >= MIN_INTERVAL_DAYS
    if not can_operate:
        return 0.0, 0.0, "shared_min_interval"
    remaining_i = max(0.0, IRRIGATION_BUDGET - used_irrigation)
    remaining_n = max(0.0, NITROGEN_BUDGET - used_nitrogen)
    if not in_window(dap, list(IRRIGATION_WINDOWS)):
        safe_i = 0.0
        if raw_i > 0:
            reasons.append("irrigation_window")
    else:
        safe_i = min(max(0.0, raw_i), DAILY_IRRIGATION_CAP, remaining_i)
        if raw_i > 0 and remaining_i <= 0:
            reasons.append("irrigation_budget_exhausted")
        elif safe_i < raw_i:
            reasons.append("irrigation_budget_clip")
    if not in_window(dap, list(NITROGEN_WINDOWS)):
        safe_n = 0.0
        if raw_n > 0:
            reasons.append("nitrogen_window")
    else:
        safe_n = min(max(0.0, raw_n), DAILY_NITROGEN_CAP, remaining_n)
        if raw_n > 0 and remaining_n <= 0:
            reasons.append("nitrogen_budget_exhausted")
        elif safe_n < raw_n:
            reasons.append("nitrogen_budget_clip")
    if not reasons:
        reasons.append("requested_zero" if raw_i == 0 and raw_n == 0 else "none")
    return float(safe_i), float(safe_n), ";".join(reasons)


def checkpoint_path(step: int) -> Path:
    return RUN / "checkpoints" / f"checkpoint_{step}" / "model.zip"


def capture_reference_trajectory(checkpoint: int, env_args: dict, null_yield: float) -> list[dict]:
    model = DQN.load(str(checkpoint_path(checkpoint)))
    env = legacy.make_train_env(env_args, null_yield)
    linked = env.env
    rows: list[dict] = []
    try:
        obs, _info = env.reset()
        for step in range(380):
            dap = int(round(linked._current_dap()))
            used_i = float(linked.used_irrigation)
            used_n = float(linked.used_nitrogen)
            last_dap = linked.last_operation_dap
            action, _ = model.predict(obs, deterministic=True)
            action_index = int(np.asarray(action).item())
            expected_i, expected_n, _reason = project_action(
                action_index,
                dap=dap,
                used_irrigation=used_i,
                used_nitrogen=used_n,
                last_operation_dap=last_dap,
            )
            pre_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
            next_obs, _reward, terminated, truncated, _info = env.step(action)
            actual = dict(linked.last_safe_real_action)
            if not np.isclose(expected_i, float(actual["amir"])) or not np.isclose(
                expected_n, float(actual["anfer"])
            ):
                raise RuntimeError(
                    f"Projection mismatch at checkpoint={checkpoint}, step={step}, dap={dap}: "
                    f"expected=({expected_i},{expected_n}), actual={actual}"
                )
            rows.append(
                {
                    "source_checkpoint": checkpoint,
                    "trajectory_step": step,
                    "dap": dap,
                    "used_irrigation": used_i,
                    "used_nitrogen": used_n,
                    "remaining_irrigation": IRRIGATION_BUDGET - used_i,
                    "remaining_nitrogen": NITROGEN_BUDGET - used_n,
                    "last_operation_dap": np.nan if last_dap is None else int(last_dap),
                    "can_operate": bool(last_dap is None or dap - last_dap >= MIN_INTERVAL_DAYS),
                    "reference_action_index": action_index,
                    "reference_executed_irrigation": expected_i,
                    "reference_executed_nitrogen": expected_n,
                    "observation": pre_obs,
                }
            )
            obs = next_obs
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows


def select_fixed_states(trajectory: list[dict]) -> list[dict]:
    selected: list[dict] = []
    used_steps: set[int] = set()
    for target in TARGET_DAPS:
        candidates = sorted(
            trajectory,
            key=lambda row: (abs(int(row["dap"]) - target), int(row["trajectory_step"])),
        )
        choice = next((row for row in candidates if int(row["trajectory_step"]) not in used_steps), candidates[0])
        used_steps.add(int(choice["trajectory_step"]))
        item = dict(choice)
        item["target_dap"] = target
        item["dap_offset"] = int(choice["dap"]) - target
        item["state_id"] = f"ref{choice['source_checkpoint']}_target{target}_dap{choice['dap']}"
        selected.append(item)
    return selected


def calculate_q_rows(states: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    models = {step: DQN.load(str(checkpoint_path(step))) for step in CHECKPOINTS}
    with torch.no_grad():
        for state in states:
            observation = np.asarray(state["observation"], dtype=np.float32).reshape(1, -1)
            for checkpoint, model in models.items():
                tensor = obs_as_tensor(observation, model.device)
                online = model.q_net(tensor).cpu().numpy()[0]
                target = model.q_net_target(tensor).cpu().numpy()[0]
                online_rank = (-online).argsort().argsort() + 1
                target_rank = (-target).argsort().argsort() + 1
                for action_index in range(9):
                    action = ACTION_TABLE_9[action_index]
                    rows.append(
                        {
                            "state_id": state["state_id"],
                            "source_checkpoint": state["source_checkpoint"],
                            "target_dap": state["target_dap"],
                            "actual_dap": state["dap"],
                            "evaluated_checkpoint": checkpoint,
                            "action_index": action_index,
                            "requested_irrigation": float(action["amir"]),
                            "requested_nitrogen": float(action["anfer"]),
                            "online_q": float(online[action_index]),
                            "target_q": float(target[action_index]),
                            "online_rank": int(online_rank[action_index]),
                            "target_rank": int(target_rank[action_index]),
                            "online_q_std_9": float(np.std(online)),
                            "target_q_std_9": float(np.std(target)),
                        }
                    )
    return pd.DataFrame(rows)


def calculate_margins(q_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["state_id", "source_checkpoint", "target_dap", "actual_dap", "evaluated_checkpoint"]
    for key, group in q_values.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        for irrigation, action_indices in IRRIGATION_GROUPS.items():
            sub = group.set_index("action_index").loc[list(action_indices)]
            for network in ("online", "target"):
                q0 = float(sub.loc[action_indices[0], f"{network}_q"])
                q50 = float(sub.loc[action_indices[1], f"{network}_q"])
                q100 = float(sub.loc[action_indices[2], f"{network}_q"])
                std = float(sub[f"{network}_q_std_9"].iloc[0])
                margin = max(q50, q100) - q0
                rows.append(
                    {
                        **base,
                        "network": network,
                        "irrigation_group": irrigation,
                        "q_n0": q0,
                        "q_n50": q50,
                        "q_n100": q100,
                        "best_high_n": 50 if q50 >= q100 else 100,
                        "nitrogen_preference_margin": margin,
                        "q_std_9": std,
                        "normalized_margin": margin / std if std > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def calculate_flips(margins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["state_id", "source_checkpoint", "target_dap", "actual_dap", "network", "irrigation_group"]
    for key, group in margins.groupby(keys, sort=False):
        indexed = group.set_index("evaluated_checkpoint")
        if 15000 not in indexed.index or 20000 not in indexed.index:
            continue
        m15 = float(indexed.loc[15000, "nitrogen_preference_margin"])
        m20 = float(indexed.loc[20000, "nitrogen_preference_margin"])
        s15 = float(indexed.loc[15000, "q_std_9"])
        s20 = float(indexed.loc[20000, "q_std_9"])
        rank_flip = bool((m15 > 0 > m20) or (m15 < 0 < m20))
        robust = bool(rank_flip and abs(m15) >= 0.25 * s15 and abs(m20) >= 0.25 * s20)
        rows.append(
            {
                **dict(zip(keys, key)),
                "margin_15k": m15,
                "margin_20k": m20,
                "margin_change_20k_minus_15k": m20 - m15,
                "rank_flip": rank_flip,
                "robust_rank_flip": robust,
            }
        )
    return pd.DataFrame(rows)


def calculate_alias(states: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    summaries: list[dict] = []
    for state in states:
        executed: list[str] = []
        for action_index in range(9):
            action = ACTION_TABLE_9[action_index]
            safe_i, safe_n, reason = project_action(
                action_index,
                dap=int(state["dap"]),
                used_irrigation=float(state["used_irrigation"]),
                used_nitrogen=float(state["used_nitrogen"]),
                last_operation_dap=(
                    None if not np.isfinite(state["last_operation_dap"]) else int(state["last_operation_dap"])
                ),
            )
            executed_key = f"I{safe_i:g}_N{safe_n:g}"
            executed.append(executed_key)
            rows.append(
                {
                    "state_id": state["state_id"],
                    "source_checkpoint": state["source_checkpoint"],
                    "target_dap": state["target_dap"],
                    "actual_dap": state["dap"],
                    "used_irrigation": state["used_irrigation"],
                    "used_nitrogen": state["used_nitrogen"],
                    "remaining_irrigation": state["remaining_irrigation"],
                    "remaining_nitrogen": state["remaining_nitrogen"],
                    "last_operation_dap": state["last_operation_dap"],
                    "can_operate": state["can_operate"],
                    "requested_action_index": action_index,
                    "requested_irrigation": float(action["amir"]),
                    "requested_nitrogen": float(action["anfer"]),
                    "executed_irrigation": safe_i,
                    "executed_nitrogen": safe_n,
                    "executed_action_key": executed_key,
                    "clip_reason": reason,
                }
            )
        unique = sorted(set(executed))
        summaries.append(
            {
                "state_id": state["state_id"],
                "source_checkpoint": state["source_checkpoint"],
                "target_dap": state["target_dap"],
                "actual_dap": state["dap"],
                "used_irrigation": state["used_irrigation"],
                "used_nitrogen": state["used_nitrogen"],
                "remaining_irrigation": state["remaining_irrigation"],
                "remaining_nitrogen": state["remaining_nitrogen"],
                "can_operate": state["can_operate"],
                "unique_executed_action_count": len(unique),
                "alias_fraction": 1.0 - len(unique) / 9.0,
                "executed_action_keys": "|".join(unique),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def state_manifest(states: list[dict], observation_names: list[str]) -> pd.DataFrame:
    rows = []
    for state in states:
        rows.append(
            {
                key: value
                for key, value in state.items()
                if key != "observation"
            }
            | {
                "observation_dimension": len(state["observation"]),
                "observation_vector_json": json.dumps(state["observation"].tolist()),
                "configured_observations_json": json.dumps(observation_names, ensure_ascii=False),
                "explicit_used_irrigation_observed": "used_irrigation" in observation_names,
                "explicit_used_nitrogen_observed": "used_nitrogen" in observation_names,
                "explicit_last_operation_dap_observed": "last_operation_dap" in observation_names,
                "totir_available_as_irrigation_proxy": "totir" in observation_names,
            }
        )
    return pd.DataFrame(rows)


def make_plot(flips: pd.DataFrame, alias_summary: pd.DataFrame) -> None:
    online = flips[flips["network"].eq("online")].copy()
    online["label"] = online.apply(
        lambda row: f"ref{int(row.source_checkpoint/1000)}K DAP{int(row.actual_dap)} I{int(row.irrigation_group)}",
        axis=1,
    )
    pivot = online.pivot(index="label", columns="irrigation_group", values="margin_change_20k_minus_15k")
    alias_plot = alias_summary.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    image = axes[0].imshow(pivot.values, aspect="auto", cmap="coolwarm")
    axes[0].set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    axes[0].set_xticks(range(len(pivot.columns)), [f"I={value:g}" for value in pivot.columns])
    axes[0].set_title("Online Q nitrogen-margin change: 20K - 15K")
    fig.colorbar(image, ax=axes[0], shrink=0.8, label="Q margin change")
    for checkpoint, group in alias_plot.groupby("source_checkpoint"):
        axes[1].plot(
            group["actual_dap"],
            group["unique_executed_action_count"],
            marker="o",
            label=f"Reference {int(checkpoint/1000)}K",
        )
    axes[1].set_xlabel("DAP")
    axes[1].set_ylabel("Unique executed actions from 9 requests")
    axes[1].set_ylim(0, 9.5)
    axes[1].set_title("State-dependent action aliasing")
    axes[1].legend(frameon=False)
    axes[1].grid(True, color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(OUT / "021_06_q_margin_and_action_alias.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_environment_constants(shared)
    env_args = json.loads((RUN / "dqn_env_args.json").read_text(encoding="utf-8"))
    null_yield = float(pd.read_csv(RUN / "null_evaluation" / "null_summary.csv").iloc[0]["final_grain_kg_ha"])
    config = yaml.safe_load((ROOT / "configs" / "benchmark_defaults.yaml").read_text(encoding="utf-8"))
    observation_names = list(config["observations"]["include"])

    states: list[dict] = []
    for checkpoint in REFERENCE_CHECKPOINTS:
        trajectory = capture_reference_trajectory(checkpoint, env_args, null_yield)
        states.extend(select_fixed_states(trajectory))

    manifest = state_manifest(states, observation_names)
    q_values = calculate_q_rows(states)
    margins = calculate_margins(q_values)
    flips = calculate_flips(margins)
    alias_long, alias_summary = calculate_alias(states)

    manifest.to_csv(OUT / "021_06_fixed_state_manifest.csv", index=False, encoding="utf-8-sig")
    q_values.to_csv(OUT / "021_06_q_values_long.csv", index=False, encoding="utf-8-sig")
    margins.to_csv(OUT / "021_06_nitrogen_preference_margins.csv", index=False, encoding="utf-8-sig")
    flips.to_csv(OUT / "021_06_rank_flip_summary.csv", index=False, encoding="utf-8-sig")
    alias_long.to_csv(OUT / "021_06_action_alias_long.csv", index=False, encoding="utf-8-sig")
    alias_summary.to_csv(OUT / "021_06_action_alias_summary.csv", index=False, encoding="utf-8-sig")
    make_plot(flips, alias_summary)

    result = {
        "fixed_state_count": len(manifest),
        "rank_flip_count_online": int(flips.query("network == 'online'")["rank_flip"].sum()),
        "robust_rank_flip_count_online": int(flips.query("network == 'online'")["robust_rank_flip"].sum()),
        "rank_flip_count_target": int(flips.query("network == 'target'")["rank_flip"].sum()),
        "robust_rank_flip_count_target": int(flips.query("network == 'target'")["robust_rank_flip"].sum()),
        "states_with_aliasing": int((alias_summary["unique_executed_action_count"] < 9).sum()),
        "states_all_requests_alias_to_one": int((alias_summary["unique_executed_action_count"] == 1).sum()),
        "observation_explicitly_contains_budget_or_last_operation": False,
        "totir_is_available_as_irrigation_proxy": "totir" in observation_names,
    }
    (OUT / "021_06_audit_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nRobust online flips:")
    print(flips.query("network == 'online' and robust_rank_flip").to_string(index=False))
    print("\nAlias summary:")
    print(alias_summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork
from stage_based_dqn_core_022 import ExecutedAction, terminal_complete_returns


OUT = ROOT / "benchmark_results" / "022_12"
NPZ_PATH = ROOT / "benchmark_results" / "022_03" / "022_03_fixed_grid_replay_dataset.npz"
MANIFEST_PATH = ROOT / "benchmark_results" / "022_03" / "022_03_fixed_grid_transition_manifest.csv"
SPLIT_PATH = ROOT / "benchmark_results" / "022_08" / "022_08_scenario_split.csv"
SUPPORT_PATH = ROOT / "benchmark_results" / "022_08" / "022_08_stage_action_identifiability.csv"
CHECKPOINT_DIR = ROOT / "benchmark_results" / "022_08" / "checkpoints"
LEARNING_RATE = 1e-4
PAIR_DAP = 65
ACTION_CONTROL = 1
ACTION_TREATMENT = 7


PAIR_SOURCES = (
    {
        "prefix": "W60_critical__N200_early",
        "actions": ROOT / "benchmark_results" / "022_10" / "022_10_controlled_stage_actions.csv",
        "summary": ROOT / "benchmark_results" / "022_10" / "022_10_controlled_summary.csv",
        "control": "control_action1_I15_N0",
        "treatment": "treatment_action7_I15_N100",
        "label_column": "arm",
    },
    {
        "prefix": "W75_uniform_pre90__N200_early",
        "actions": ROOT / "benchmark_results" / "022_11" / "022_11_new_prefix_stage_actions.csv",
        "summary": ROOT / "benchmark_results" / "022_11" / "022_11_new_prefix_summary.csv",
        "control": "prefix_b_w75_uniform_pre90__control",
        "treatment": "prefix_b_w75_uniform_pre90__treatment",
        "label_column": "label",
    },
    {
        "prefix": "W120_critical__N200_early",
        "actions": ROOT / "benchmark_results" / "022_11" / "022_11_new_prefix_stage_actions.csv",
        "summary": ROOT / "benchmark_results" / "022_11" / "022_11_new_prefix_summary.csv",
        "control": "prefix_c_w120_critical_context__control",
        "treatment": "prefix_c_w120_critical_context__treatment",
        "label_column": "label",
    },
)


def executed_actions(table: pd.DataFrame) -> list[ExecutedAction]:
    return [
        ExecutedAction(
            action_index=int(row.action_index),
            requested_irrigation=float(row.requested_irrigation),
            requested_nitrogen=float(row.requested_nitrogen),
            executed_irrigation=float(row.executed_irrigation),
            executed_nitrogen=float(row.executed_nitrogen),
        )
        for row in table.sort_values("stage_index").itertuples(index=False)
    ]


def build_pair_targets(
    observations: np.ndarray, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    records: list[dict[str, float | str]] = []
    states: list[np.ndarray] = []
    targets: list[float] = []
    for source in PAIR_SOURCES:
        actions = pd.read_csv(source["actions"])
        summaries = pd.read_csv(source["summary"])
        label_col = str(source["label_column"])
        arm_returns: dict[str, list[float]] = {}
        arm_yields: dict[str, float] = {}
        for arm_key in ("control", "treatment"):
            label = str(source[arm_key])
            action_rows = actions.loc[actions[label_col].astype(str) == label].copy()
            if len(action_rows) != 6:
                raise ValueError(f"{label}: expected 6 stage actions, got {len(action_rows)}")
            summary_label_col = "label" if "label" in summaries.columns else "arm"
            summary_row = summaries.loc[summaries[summary_label_col].astype(str) == label]
            if len(summary_row) != 1:
                raise ValueError(f"{label}: expected one summary row")
            final_yield = float(summary_row.iloc[0]["final_yield"])
            arm_yields[arm_key] = final_yield
            arm_returns[arm_key] = terminal_complete_returns(executed_actions(action_rows), final_yield)

        state_rows = manifest.loc[
            (manifest["scenario"].astype(str) == str(source["prefix"]))
            & (manifest["dap"].astype(int) == PAIR_DAP)
        ]
        if len(state_rows) != 1:
            raise ValueError(f"{source['prefix']}: DAP65 state row count={len(state_rows)}")
        state_index = int(state_rows.index[0])
        control_return = float(arm_returns["control"][3])
        treatment_return = float(arm_returns["treatment"][3])
        difference_scaled = (control_return - treatment_return) / 1000.0
        if difference_scaled <= 0:
            raise ValueError(f"{source['prefix']}: non-positive causal target {difference_scaled}")
        states.append(observations[state_index])
        targets.append(difference_scaled)
        records.append(
            {
                "prefix": str(source["prefix"]),
                "state_row_index": state_index,
                "dap": PAIR_DAP,
                "control_action": ACTION_CONTROL,
                "treatment_action": ACTION_TREATMENT,
                "control_final_yield": arm_yields["control"],
                "treatment_final_yield": arm_yields["treatment"],
                "control_return_raw_at_dap65": control_return,
                "treatment_return_raw_at_dap65": treatment_return,
                "causal_return_difference_raw": control_return - treatment_return,
                "causal_return_difference_scaled": difference_scaled,
            }
        )
    return (
        pd.DataFrame(records),
        torch.tensor(np.asarray(states), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
    )


def gradient_list(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(param) if grad is None else grad.detach().clone() for param, grad in zip(params, grads)]


def flatten(grads: list[torch.Tensor], mask: list[bool] | None = None) -> torch.Tensor:
    chosen = grads if mask is None else [grad for grad, keep in zip(grads, mask) if keep]
    return torch.cat([grad.reshape(-1) for grad in chosen])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(torch.dot(a, b) / denom) if denom > 0 else math.nan


def opposite_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    valid = (a.abs() > 1e-12) & (b.abs() > 1e-12)
    if int(valid.sum()) == 0:
        return math.nan
    return float(((a[valid] * b[valid]) < 0).float().mean())


def support_map() -> dict[int, list[int]]:
    table = pd.read_csv(SUPPORT_PATH)
    table = table.loc[table["primary_evaluable"].astype(str).str.lower().isin(("true", "1"))]
    return {
        int(dap): sorted(group["action_index"].astype(int).tolist())
        for dap, group in table.groupby("dap")
    }


def evaluate_losses(
    model: QNetwork,
    train_obs: torch.Tensor,
    train_actions: torch.Tensor,
    train_targets: torch.Tensor,
    pair_obs: torch.Tensor,
    pair_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = model(train_obs).gather(1, train_actions[:, None]).squeeze(1)
    mc_loss = nn.functional.smooth_l1_loss(selected, train_targets)
    pair_q = model(pair_obs)
    margins = pair_q[:, ACTION_CONTROL] - pair_q[:, ACTION_TREATMENT]
    pair_loss = nn.functional.smooth_l1_loss(margins, pair_targets)
    return mc_loss, pair_loss


def support_argmax(model: QNetwork, obs: torch.Tensor, daps: np.ndarray, support: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        q = model(obs).cpu().numpy()
    choices = []
    for row, dap in zip(q, daps):
        allowed = support[int(dap)]
        choices.append(allowed[int(np.argmax(row[allowed]))])
    return np.asarray(choices, dtype=int), q


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = np.load(NPZ_PATH)
    observations = dataset["observations"]
    actions = dataset["actions"]
    targets = dataset["targets"]
    scenarios = dataset["scenarios"].astype(str)
    stage_indices = dataset["stage_indices"]
    manifest = pd.read_csv(MANIFEST_PATH)
    if len(manifest) != len(observations):
        raise ValueError("manifest and NPZ row counts differ")
    if not np.array_equal(manifest["scenario"].astype(str).to_numpy(), scenarios):
        raise ValueError("manifest and NPZ scenario order differ")

    split = pd.read_csv(SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    test_scenarios = set(split.loc[split["split"] == "test", "scenario"].astype(str))
    train_mask = np.asarray([scenario in train_scenarios for scenario in scenarios])
    test_mask = np.asarray([scenario in test_scenarios for scenario in scenarios])
    if int(train_mask.sum()) != 216 or int(test_mask.sum()) != 72:
        raise ValueError(f"unexpected split sizes train={train_mask.sum()} test={test_mask.sum()}")

    pair_table, pair_obs, pair_targets = build_pair_targets(observations, manifest)
    pair_table.to_csv(OUT / "022_12_controlled_pair_targets.csv", index=False)

    train_obs = torch.tensor(observations[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    test_obs = torch.tensor(observations[test_mask], dtype=torch.float32)
    test_daps = np.asarray([int((1, 30, 50, 65, 85, 110)[i]) for i in stage_indices[test_mask]])
    support = support_map()

    seed_rows: list[dict[str, float | int | bool]] = []
    stage_rows: list[dict[str, float | int]] = []
    margin_rows: list[dict[str, float | int | str]] = []
    for seed in range(3):
        payload = torch.load(CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = QNetwork()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        named_params = list(model.named_parameters())
        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        shared_mask = [name.startswith("net.0") or name.startswith("net.2") for name in names]

        mc_loss, pair_loss = evaluate_losses(model, train_obs, train_actions, train_targets, pair_obs, pair_targets)
        grads_mc = gradient_list(mc_loss, params)
        grads_pair = gradient_list(pair_loss, params)
        all_mc, all_pair = flatten(grads_mc), flatten(grads_pair)
        shared_mc, shared_pair = flatten(grads_mc, shared_mask), flatten(grads_pair, shared_mask)
        shared_mc_norm = float(torch.linalg.vector_norm(shared_mc))
        shared_pair_norm = float(torch.linalg.vector_norm(shared_pair))
        lambda_norm = shared_mc_norm / shared_pair_norm if shared_pair_norm > 0 else math.nan

        pre_choices, pre_q = support_argmax(model, test_obs, test_daps, support)
        with torch.no_grad():
            pre_pair_q = model(pair_obs).numpy()
        virtual = copy.deepcopy(model)
        virtual_params = [param for _, param in virtual.named_parameters()]
        if not math.isfinite(lambda_norm):
            raise ValueError(f"seed{seed}: non-finite lambda")
        with torch.no_grad():
            for param, grad_mc, grad_pair in zip(virtual_params, grads_mc, grads_pair):
                param.add_(-LEARNING_RATE * (grad_mc + lambda_norm * grad_pair))
        post_mc, post_pair = evaluate_losses(virtual, train_obs, train_actions, train_targets, pair_obs, pair_targets)
        post_choices, post_q = support_argmax(virtual, test_obs, test_daps, support)
        with torch.no_grad():
            post_pair_q = virtual(pair_obs).numpy()

        for dap in sorted(support):
            mask = test_daps == dap
            q_delta = np.abs(post_q[mask] - pre_q[mask])
            stage_rows.append(
                {
                    "seed": seed,
                    "dap": dap,
                    "test_state_count": int(mask.sum()),
                    "support_argmax_changes": int(np.sum(pre_choices[mask] != post_choices[mask])),
                    "q_abs_change_mean_all_actions": float(q_delta.mean()),
                    "q_abs_change_max_all_actions": float(q_delta.max()),
                }
            )
        for index, row in pair_table.iterrows():
            margin_rows.append(
                {
                    "seed": seed,
                    "prefix": str(row["prefix"]),
                    "target_margin_scaled": float(pair_targets[index]),
                    "q_action1_before": float(pre_pair_q[index, ACTION_CONTROL]),
                    "q_action7_before": float(pre_pair_q[index, ACTION_TREATMENT]),
                    "predicted_margin_before": float(pre_pair_q[index, ACTION_CONTROL] - pre_pair_q[index, ACTION_TREATMENT]),
                    "predicted_margin_after": float(post_pair_q[index, ACTION_CONTROL] - post_pair_q[index, ACTION_TREATMENT]),
                }
            )

        mc_before_value = float(mc_loss.detach())
        pair_before_value = float(pair_loss.detach())
        mc_after_value = float(post_mc.detach())
        pair_after_value = float(post_pair.detach())
        mc_relative_increase = (mc_after_value - mc_before_value) / max(abs(mc_before_value), 1e-12)
        dap1_changes = next(row["support_argmax_changes"] for row in stage_rows if row["seed"] == seed and row["dap"] == 1)
        dap110_changes = next(row["support_argmax_changes"] for row in stage_rows if row["seed"] == seed and row["dap"] == 110)
        finite = all(
            math.isfinite(value)
            for value in (
                mc_before_value, pair_before_value, shared_mc_norm, shared_pair_norm, lambda_norm,
                cosine(shared_mc, shared_pair), mc_after_value, pair_after_value, mc_relative_increase,
            )
        )
        compatible = bool(
            cosine(shared_mc, shared_pair) >= -0.20
            and pair_after_value < pair_before_value
            and mc_relative_increase <= 0.01
            and dap1_changes == 0
            and dap110_changes == 0
            and finite
        )
        seed_rows.append(
            {
                "seed": seed,
                "mc_loss_before": mc_before_value,
                "pair_loss_before": pair_before_value,
                "all_mc_grad_norm": float(torch.linalg.vector_norm(all_mc)),
                "all_pair_grad_norm": float(torch.linalg.vector_norm(all_pair)),
                "shared_mc_grad_norm": shared_mc_norm,
                "shared_pair_grad_norm": shared_pair_norm,
                "shared_gradient_cosine": cosine(shared_mc, shared_pair),
                "all_gradient_cosine": cosine(all_mc, all_pair),
                "shared_opposite_sign_fraction": opposite_fraction(shared_mc, shared_pair),
                "all_opposite_sign_fraction": opposite_fraction(all_mc, all_pair),
                "lambda_norm": lambda_norm,
                "mc_loss_after_virtual_step": mc_after_value,
                "pair_loss_after_virtual_step": pair_after_value,
                "mc_loss_relative_increase": mc_relative_increase,
                "dap1_argmax_changes": int(dap1_changes),
                "dap110_argmax_changes": int(dap110_changes),
                "finite": finite,
                "compatible": compatible,
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    stage_df = pd.DataFrame(stage_rows)
    margin_df = pd.DataFrame(margin_rows)
    seed_df.to_csv(OUT / "022_12_seed_gradient_summary.csv", index=False)
    stage_df.to_csv(OUT / "022_12_stage_virtual_step_perturbation.csv", index=False)
    margin_df.to_csv(OUT / "022_12_pair_margin_before_after.csv", index=False)

    compatible_count = int(seed_df["compatible"].sum())
    branch = "A_compatible" if compatible_count >= 2 else ("B_local_conflict" if compatible_count == 1 else "C_severe_conflict")
    result = {
        "status": "completed",
        "branch": branch,
        "compatible_seed_count": compatible_count,
        "total_seed_count": 3,
        "dqn_training_steps": 0,
        "dssat_calls": 0,
        "virtual_learning_rate": LEARNING_RATE,
        "pair_loss_formula": "SmoothL1(Q(s,a1)-Q(s,a7), (G1-G7)/1000)",
        "lambda_formula": "norm(grad_shared_L_MC)/norm(grad_shared_L_pair)",
        "shared_parameters": [name for name, keep in zip(names, shared_mask) if keep],
        "output_parameters": [name for name, keep in zip(names, shared_mask) if not keep],
        "next_step_allowed": branch == "A_compatible",
    }
    (OUT / "022_12_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = seed_df["seed"].to_numpy()
    axes[0].bar(x, seed_df["shared_gradient_cosine"], color="#4472C4")
    axes[0].axhline(-0.20, color="#C00000", linestyle="--", linewidth=1.2)
    axes[0].set(title="Shared-gradient cosine", xlabel="Seed", ylabel="Cosine")
    axes[1].bar(x - 0.18, seed_df["mc_loss_before"], width=0.36, label="MC", color="#4472C4")
    axes[1].bar(x + 0.18, seed_df["pair_loss_before"], width=0.36, label="Pair", color="#C55A11")
    axes[1].set(title="Loss before virtual step", xlabel="Seed", ylabel="SmoothL1")
    axes[1].legend(frameon=False)
    axes[2].bar(x - 0.18, seed_df["mc_loss_relative_increase"] * 100, width=0.36, label="MC relative change", color="#70AD47")
    pair_change = (seed_df["pair_loss_after_virtual_step"] - seed_df["pair_loss_before"]) / seed_df["pair_loss_before"] * 100
    axes[2].bar(x + 0.18, pair_change, width=0.36, label="Pair relative change", color="#ED7D31")
    axes[2].axhline(1.0, color="#C00000", linestyle="--", linewidth=1.2)
    axes[2].set(title="Virtual-step loss change", xlabel="Seed", ylabel="Percent")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SY2014 controlled pairwise–MC gradient audit (022_12)")
    fig.tight_layout()
    fig.savefig(OUT / "022_12_gradient_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_12_gradient_audit.svg", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

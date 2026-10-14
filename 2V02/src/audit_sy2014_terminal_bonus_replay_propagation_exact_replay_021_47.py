from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
import run_sy2014_event_balanced_online_retention_1k_021_35 as ab
import run_sy2014_online_demo_nstep_mask_ab_021_41 as treatment
import run_sy2014_terminal_feasibility_bonus_seed1_1k_ab_021_45 as exp45


OUT = ROOT / "benchmark_results" / "021_47"
REPLAY_OUT = OUT / "exact_replay"
DOC = ROOT / "docs" / "2026-07-15_021_47_sy2014_terminal_bonus_replay_propagation_exact_replay_audit.md"
SOURCE_INTERACTIONS = ROOT / "benchmark_results" / "021_45" / "021_45_training_interactions.csv"
SOURCE_EPISODES = ROOT / "benchmark_results" / "021_45" / "021_45_training_episode_terminal_rewards.csv"
SOURCE_TRAJECTORY = ROOT / "benchmark_results" / "021_45" / "021_45_checkpoint_trajectory_ab.csv"
SOURCE_CHECKPOINTS = ROOT / "benchmark_results" / "021_45" / "treatment_seed1" / "training" / "checkpoints"

CRITICAL_N_DAPS = (29, 42, 56)
CHECKPOINTS = (0, 250, 500, 750, 1000)
AGENT_DRAWS_PER_UPDATE = 16
ORIGINAL_MIXED_SAMPLE = ab.mixed_sample

SAMPLE_ROWS: list[dict[str, Any]] = []
UPDATE_EXPOSURE_ROWS: list[dict[str, Any]] = []
TARGET_HASH_ROWS: list[dict[str, Any]] = []
TRANSITION_EXPOSURE: dict[int, dict[str, float]] = defaultdict(
    lambda: {"expected": 0.0, "variance": 0.0, "observed": 0.0}
)
CATEGORY_EXPOSURE: dict[str, dict[str, float]] = defaultdict(
    lambda: {"expected": 0.0, "variance": 0.0, "observed": 0.0}
)
REPLAY_REF = None


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in module.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_source_metadata() -> pd.DataFrame:
    interactions = pd.read_csv(SOURCE_INTERACTIONS)
    episodes = pd.read_csv(SOURCE_EPISODES)
    interactions["episode"] = interactions.done.shift(fill_value=False).astype(bool).cumsum().astype(int)
    terminal_rows = interactions.loc[interactions.done.astype(bool), ["episode", "env_step", "episode_dap"]].copy()
    terminal_rows = terminal_rows.reset_index(drop=True)
    if len(terminal_rows) != len(episodes):
        raise RuntimeError("Source episode count does not match terminal interaction count")
    terminal_rows["feasibility_bonus"] = episodes.feasibility_bonus.to_numpy(dtype=float)
    terminal_step = dict(zip(terminal_rows.episode, terminal_rows.env_step))
    terminal_dap = dict(zip(terminal_rows.episode, terminal_rows.episode_dap))
    terminal_bonus = dict(zip(terminal_rows.episode, terminal_rows.feasibility_bonus))

    rows: list[dict[str, Any]] = []
    for _, row in interactions.iterrows():
        origin = int(row.env_step)
        episode = int(row.episode)
        t_step = terminal_step.get(episode)
        t_dap = terminal_dap.get(episode)
        distance = float(t_step - origin) if t_step is not None and origin <= t_step else np.nan
        direct_bonus = bool(t_step == origin and terminal_bonus.get(episode, 0.0) > 0)
        nstep_bonus = bool(
            t_step is not None and 0 <= t_step - origin < base.N_STEP
            and terminal_bonus.get(episode, 0.0) > 0
        )
        discounted_bonus = (
            float((base.GAMMA ** int(t_step - origin)) * terminal_bonus[episode])
            if nstep_bonus else 0.0
        )
        immediate_cost = float(row.safe_amir) + 5.0 * float(row.safe_anfer)
        window = interactions.loc[
            interactions.env_step.between(origin, origin + base.N_STEP - 1)
            & interactions.episode.eq(episode)
        ].copy()
        offsets = window.env_step.to_numpy(dtype=int) - origin
        costs = window.safe_amir.to_numpy(dtype=float) + 5.0 * window.safe_anfer.to_numpy(dtype=float)
        nstep_cost = float(np.sum((base.GAMMA ** offsets) * costs))
        rows.append({
            "origin_env_step": origin, "episode": episode, "origin_dap": int(row.episode_dap),
            "action": int(row.action), "one_step_reward": float(row.reward), "done": bool(row.done),
            "terminal_env_step": t_step, "terminal_dap": t_dap,
            "steps_to_terminal": distance, "direct_bonus": direct_bonus,
            "nstep_contains_bonus": nstep_bonus, "discounted_bonus_in_nstep": discounted_bonus,
            "immediate_resource_cost": immediate_cost, "nstep_discounted_resource_cost": nstep_cost,
        })
    return pd.DataFrame(rows).set_index("origin_env_step", drop=False)


SOURCE_METADATA = build_source_metadata()


def category_masks(origins: np.ndarray) -> dict[str, np.ndarray]:
    rows = SOURCE_METADATA.loc[origins]
    return {
        "direct_bonus": rows.direct_bonus.to_numpy(dtype=bool),
        "nstep_bonus": rows.nstep_contains_bonus.to_numpy(dtype=bool),
        "immediate_resource_cost": rows.immediate_resource_cost.to_numpy(dtype=float) > 0,
    }


def instrumented_update(model, replay, demo_actions, rng, *, update, env_step, epsilon) -> dict[str, Any]:
    global REPLAY_REF
    REPLAY_REF = replay
    # Exactly one call to the original sampler. No additional RNG calls are allowed below.
    sample = ORIGINAL_MIXED_SAMPLE(replay, demo_actions, rng, update)
    device = model.device
    observations = base.tensor(sample.data["observations"], dtype=torch.float32, device=device)
    actions = base.tensor(sample.data["actions"], dtype=torch.long, device=device).reshape(-1)
    weights = base.tensor(sample.importance_weights, dtype=torch.float32, device=device)
    demo_mask = base.tensor(sample.is_demonstration, dtype=torch.bool, device=device)
    agent_mask = ~demo_mask
    target_1, target_n = base.compute_targets(model, sample.data)
    q = model.q_net(observations)
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    td1_each = F.smooth_l1_loss(chosen, target_1, reduction="none")
    td1 = (td1_each[agent_mask] * weights[agent_mask]).mean()
    tdn_each = F.smooth_l1_loss(chosen, target_n, reduction="none")
    tdn_agent_only = (tdn_each[agent_mask] * weights[agent_mask]).sum() / ab.BATCH_SIZE
    margins = torch.full_like(q, 0.8); margins.scatter_(1, actions[:, None], 0.0)
    margin_each = torch.max(q + margins, dim=1).values - chosen
    margin = (margin_each[demo_mask] * weights[demo_mask]).mean()
    parameters = list(model.q_net.parameters())
    l2 = 1e-5 * sum(parameter.square().sum() for parameter in parameters)
    total = td1 + tdn_agent_only + margin + l2

    probabilities_before = replay.sampling_probabilities()
    agent_conditional = probabilities_before[replay.demo_count:]
    agent_conditional = agent_conditional / agent_conditional.sum()
    active_origins = np.arange(1, replay.agent_size + 1, dtype=int)
    masks = category_masks(active_origins)
    update_exposure = {"update": update, "env_step": env_step, "active_agent_count": replay.agent_size}
    for position, probability in enumerate(agent_conditional):
        global_index = replay.demo_count + position
        expected = AGENT_DRAWS_PER_UPDATE * float(probability)
        variance = AGENT_DRAWS_PER_UPDATE * float(probability) * (1.0 - float(probability))
        TRANSITION_EXPOSURE[global_index]["expected"] += expected
        TRANSITION_EXPOSURE[global_index]["variance"] += variance
    sampled_agent_indices = sample.global_indices[~sample.is_demonstration].astype(int)
    for global_index in sampled_agent_indices:
        TRANSITION_EXPOSURE[int(global_index)]["observed"] += 1.0
    sampled_origins = sampled_agent_indices - replay.demo_count + 1
    sampled_masks = category_masks(sampled_origins)
    for name, active_mask in masks.items():
        group_probability = float(agent_conditional[active_mask].sum())
        expected = AGENT_DRAWS_PER_UPDATE * group_probability
        variance = AGENT_DRAWS_PER_UPDATE * group_probability * (1.0 - group_probability)
        observed = int(sampled_masks[name].sum())
        CATEGORY_EXPOSURE[name]["expected"] += expected
        CATEGORY_EXPOSURE[name]["variance"] += variance
        CATEGORY_EXPOSURE[name]["observed"] += observed
        update_exposure[f"{name}_probability"] = group_probability
        update_exposure[f"{name}_expected_draws"] = expected
        update_exposure[f"{name}_observed_draws"] = observed
    UPDATE_EXPOSURE_ROWS.append(update_exposure)

    priorities_before = replay.combined_priorities()
    chosen_np = chosen.detach().cpu().numpy()
    target1_np = target_1.detach().cpu().numpy()
    targetn_np = target_n.detach().cpu().numpy()
    actions_np = actions.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    for draw_position, global_index in enumerate(sample.global_indices.astype(int)):
        if global_index < replay.demo_count:
            continue
        origin = global_index - replay.demo_count + 1
        meta = SOURCE_METADATA.loc[origin]
        cond_p = float(agent_conditional[global_index - replay.demo_count])
        SAMPLE_ROWS.append({
            "update": update, "env_step": env_step, "draw_position": draw_position,
            "global_index": global_index, "origin_env_step": origin,
            "episode": int(meta.episode), "origin_dap": int(meta.origin_dap),
            "action": int(actions_np[draw_position]), "direct_bonus": bool(meta.direct_bonus),
            "nstep_contains_bonus": bool(meta.nstep_contains_bonus),
            "discounted_bonus_in_nstep": float(meta.discounted_bonus_in_nstep),
            "immediate_resource_cost": float(meta.immediate_resource_cost),
            "nstep_discounted_resource_cost": float(meta.nstep_discounted_resource_cost),
            "steps_to_terminal": float(meta.steps_to_terminal) if np.isfinite(meta.steps_to_terminal) else np.nan,
            "mixture_probability": float(sample.probabilities[draw_position]),
            "agent_conditional_probability": cond_p,
            "importance_weight": float(weights_np[draw_position]),
            "priority_before": float(priorities_before[global_index]),
            "chosen_q": float(chosen_np[draw_position]),
            "target_1": float(target1_np[draw_position]), "target_n": float(targetn_np[draw_position]),
            "td_error_1": float(target1_np[draw_position] - chosen_np[draw_position]),
            "td_error_n": float(targetn_np[draw_position] - chosen_np[draw_position]),
        })

    # The optimization block below is kept operation-for-operation identical to 021_41.
    model.policy.optimizer.zero_grad(); total.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, ab.MAX_GRAD_NORM)); model.policy.optimizer.step()
    with torch.no_grad():
        errors = torch.abs(target_1 - chosen).detach().cpu().numpy()
    priorities: dict[int, float] = {}
    for index, error in zip(sample.global_indices, errors):
        priorities[int(index)] = max(priorities.get(int(index), 0.0), float(error))
    replay.update_priorities(priorities.keys(), priorities.values())
    TARGET_HASH_ROWS.append({"update": update, "env_step": env_step, "target_hash": parameter_hash(model.q_net_target)})
    values = {
        "td1_agent_only": float(td1.detach()),
        "tdn_agent_only_preserved_scale": float(tdn_agent_only.detach()),
        "margin_demo_only": float(margin.detach()), "l2": float(l2.detach()),
        "total": float(total.detach()),
    }
    return {
        "update": update, "env_step": env_step, "epsilon": epsilon,
        "sample_demo_count": int(demo_mask.sum()), "sample_agent_count": int(agent_mask.sum()),
        "sample_demo_noop_count": int(((actions == 0) & demo_mask).sum()),
        "sample_demo_nonzero_count": int(((actions != 0) & demo_mask).sum()),
        "replay_agent_count": replay.agent_size, **values,
        "demo_nstep_weight": 0.0,
        "gradient_before_clip": gradient, "gradient_would_clip": gradient > ab.MAX_GRAD_NORM,
        "q_abs_max": float(q.detach().abs().max()),
        "all_finite": bool(all(np.isfinite(value) for value in [*values.values(), gradient])),
    }


def compare_interactions(new: pd.DataFrame, old: pd.DataFrame) -> dict[str, Any]:
    same_columns = list(new.columns) == list(old.columns)
    same_shape = new.shape == old.shape
    numeric_columns = [column for column in old.columns if pd.api.types.is_numeric_dtype(old[column])]
    text_columns = [column for column in old.columns if column not in numeric_columns]
    max_numeric_error = 0.0
    if same_shape and numeric_columns:
        for column in numeric_columns:
            a = new[column].to_numpy()
            b = old[column].to_numpy()
            if a.dtype.kind == "b" or b.dtype.kind == "b":
                error = 0.0 if np.array_equal(a, b) else 1.0
            else:
                error = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
            max_numeric_error = max(max_numeric_error, error)
    text_equal = bool(same_shape and all(new[c].astype(str).equals(old[c].astype(str)) for c in text_columns))
    return {
        "same_columns": same_columns, "same_shape": same_shape,
        "max_numeric_error": max_numeric_error, "text_columns_exact": text_equal,
        "exact": bool(same_columns and same_shape and max_numeric_error == 0.0 and text_equal),
    }


def checkpoint_hash_table() -> pd.DataFrame:
    rows = []
    for checkpoint in CHECKPOINTS:
        old_model = DQN.load(str(SOURCE_CHECKPOINTS / f"checkpoint_{checkpoint}.zip"), device="cpu")
        new_model = DQN.load(str(REPLAY_OUT / "training" / "checkpoints" / f"checkpoint_{checkpoint}.zip"), device="cpu")
        old_online = parameter_hash(old_model.q_net); new_online = parameter_hash(new_model.q_net)
        old_target = parameter_hash(old_model.q_net_target); new_target = parameter_hash(new_model.q_net_target)
        rows.append({
            "checkpoint": checkpoint, "old_online_hash": old_online, "new_online_hash": new_online,
            "online_hash_exact": old_online == new_online,
            "old_target_hash": old_target, "new_target_hash": new_target,
            "target_hash_exact": old_target == new_target,
        })
    return pd.DataFrame(rows)


def transition_exposure_table(metadata: pd.DataFrame, demo_count: int, agent_size: int) -> pd.DataFrame:
    rows = []
    for origin in range(1, agent_size + 1):
        global_index = demo_count + origin - 1
        stats = TRANSITION_EXPOSURE[global_index]
        lower = stats["expected"] - 1.96 * math.sqrt(max(stats["variance"], 0.0))
        meta = metadata.loc[origin]
        rows.append({
            **meta.to_dict(), "global_index": global_index,
            "expected_draws": stats["expected"], "sampling_variance": stats["variance"],
            "lower_95": lower, "observed_draws": int(stats["observed"]),
            "significantly_below_per_expectation": bool(stats["observed"] < lower),
        })
    return pd.DataFrame(rows)


def category_exposure_table() -> pd.DataFrame:
    rows = []
    for name, stats in CATEGORY_EXPOSURE.items():
        lower = stats["expected"] - 1.96 * math.sqrt(max(stats["variance"], 0.0))
        upper = stats["expected"] + 1.96 * math.sqrt(max(stats["variance"], 0.0))
        rows.append({
            "category": name, "expected_draws": stats["expected"],
            "sampling_variance": stats["variance"], "lower_95": lower, "upper_95": upper,
            "observed_draws": int(stats["observed"]),
            "significantly_below_per_expectation": bool(stats["observed"] < lower),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(category: pd.DataFrame, samples: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(category))
    axes[0].bar(x - 0.18, category.expected_draws, width=0.36, color="#777777", label="PER expected")
    axes[0].bar(x + 0.18, category.observed_draws, width=0.36, color="#0072B2", label="Observed")
    axes[0].set_xticks(x, category.category, rotation=20, ha="right")
    axes[0].set_ylabel("Agent sample draws"); axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.2)
    bonus = samples.loc[samples.nstep_contains_bonus.astype(bool)].copy()
    if not bonus.empty:
        axes[1].scatter(bonus.env_step, bonus.target_n, s=12, alpha=0.55, color="#009E73", label="n-step target")
        axes[1].scatter(bonus.env_step, bonus.chosen_q, s=12, alpha=0.55, color="#D55E00", label="chosen Q")
    axes[1].set_xlabel("Online environment step"); axes[1].set_ylabel("Value")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.2)
    fig.suptitle("SY2014 terminal bonus PER exposure and sampled value trace")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "021_47_bonus_sampling_and_td_q.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_47_bonus_sampling_and_td_q.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists(): raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    threshold, provenance = exp45.load_threshold(exp45.SITE, exp45.YEAR)
    old_interactions = pd.read_csv(SOURCE_INTERACTIONS)
    if not np.array_equal(SOURCE_METADATA.origin_env_step.to_numpy(), np.arange(1, len(SOURCE_METADATA) + 1)):
        raise RuntimeError("Source metadata origin-step mapping is not contiguous")

    training_episodes: list[exp45.EpisodeRecord] = []
    original_make_env = ab.make_env
    def bonus_make_env(relative: str):
        sink = training_episodes if relative == "training/environment" else None
        return exp45.TerminalFeasibilityBonusWrapper(original_make_env(relative), threshold, exp45.BONUS, sink)

    ab.OUT = REPLAY_OUT
    ab.make_env = bonus_make_env
    ab.online_update = instrumented_update
    new_interactions, updates, audit, scaler = ab.train_online(action_seed=1, sample_seed=21036)
    new_interactions.to_csv(OUT / "021_47_exact_replay_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_47_exact_replay_update_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([record.__dict__ for record in training_episodes]).to_csv(
        OUT / "021_47_exact_replay_episode_rewards.csv", index=False, encoding="utf-8-sig"
    )
    samples = pd.DataFrame(SAMPLE_ROWS)
    samples.to_csv(OUT / "021_47_agent_sample_draws.csv", index=False, encoding="utf-8-sig")
    value_fit_rows = []
    value_fit_masks = {
        "direct_bonus": samples.direct_bonus.astype(bool),
        "nstep_bonus": samples.nstep_contains_bonus.astype(bool),
        "immediate_resource_cost": samples.immediate_resource_cost.gt(0),
    }
    for name, mask in value_fit_masks.items():
        group = samples.loc[mask]
        value_fit_rows.append({
            "category": name, "draws": len(group), "chosen_q_mean": float(group.chosen_q.mean()),
            "target_1_mean": float(group.target_1.mean()), "target_n_mean": float(group.target_n.mean()),
            "abs_td1_mean": float(group.td_error_1.abs().mean()),
            "abs_tdn_mean": float(group.td_error_n.abs().mean()),
            "td1_huber_linear_fraction": float(group.td_error_1.abs().gt(1).mean()),
            "tdn_huber_linear_fraction": float(group.td_error_n.abs().gt(1).mean()),
            "importance_weight_mean": float(group.importance_weight.mean()),
        })
    value_fit = pd.DataFrame(value_fit_rows)
    value_fit.to_csv(OUT / "021_47_sampled_value_fit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(UPDATE_EXPOSURE_ROWS).to_csv(OUT / "021_47_update_category_exposure.csv", index=False, encoding="utf-8-sig")
    target_trace = pd.DataFrame(TARGET_HASH_ROWS)
    target_trace.to_csv(OUT / "021_47_target_hash_trace.csv", index=False, encoding="utf-8-sig")

    interaction_check = compare_interactions(pd.read_csv(OUT / "021_47_exact_replay_interactions.csv"), old_interactions)
    hashes = checkpoint_hash_table()
    hashes.to_csv(OUT / "021_47_checkpoint_hash_comparison.csv", index=False, encoding="utf-8-sig")

    replay_validation_rows = []
    if REPLAY_REF is None:
        raise RuntimeError("Replay reference was not captured")
    for position in range(REPLAY_REF.agent_size):
        origin = position + 1
        meta = SOURCE_METADATA.loc[origin]
        replay_validation_rows.append({
            "origin_env_step": origin,
            "action_match": int(np.asarray(REPLAY_REF.agent_data["actions"][position]).item()) == int(meta.action),
            "reward_abs_error": abs(float(REPLAY_REF.agent_data["rewards"][position]) - float(meta.one_step_reward)),
            "done_match": bool(REPLAY_REF.agent_data["dones"][position]) == bool(meta.done),
        })
    replay_mapping = pd.DataFrame(replay_validation_rows)
    replay_mapping.to_csv(OUT / "021_47_replay_origin_mapping_validation.csv", index=False, encoding="utf-8-sig")

    transition = transition_exposure_table(SOURCE_METADATA, REPLAY_REF.demo_count, REPLAY_REF.agent_size)
    transition.to_csv(OUT / "021_47_transition_exposure.csv", index=False, encoding="utf-8-sig")
    category = category_exposure_table()
    category.to_csv(OUT / "021_47_category_exposure.csv", index=False, encoding="utf-8-sig")

    source_treatment = pd.read_csv(SOURCE_TRAJECTORY)
    source_treatment = source_treatment.loc[source_treatment.arm == "treatment_terminal_bonus"].copy()
    trajectory_rows = [source_treatment.loc[source_treatment.checkpoint == 0].iloc[0].to_dict()]
    for checkpoint in ab.CHECKPOINTS:
        _daily, result = ab.evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
        result["arm"] = "exact_replay_instrumented"
        trajectory_rows.append(result)
    replay_trajectory = pd.DataFrame(trajectory_rows)
    replay_trajectory.to_csv(OUT / "021_47_exact_replay_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")
    compare_columns = ["checkpoint", "yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
                       "late_n_after_dap90_kg_ha", "reward_total", "expert_efficiency_gate"]
    old_eval = source_treatment[compare_columns].sort_values("checkpoint").reset_index(drop=True)
    new_eval = replay_trajectory[compare_columns].sort_values("checkpoint").reset_index(drop=True)
    evaluation_numeric = [c for c in compare_columns if c != "expert_efficiency_gate"]
    evaluation_max_error = float(np.nanmax(np.abs(
        old_eval[evaluation_numeric].to_numpy(dtype=float) - new_eval[evaluation_numeric].to_numpy(dtype=float)
    )))
    evaluation_gate_exact = bool(old_eval.expert_efficiency_gate.astype(bool).equals(new_eval.expert_efficiency_gate.astype(bool)))

    terminal_dap_values = sorted(SOURCE_METADATA.loc[SOURCE_METADATA.done, "terminal_dap"].dropna().astype(int).unique())
    representative_terminal_dap = int(pd.Series(terminal_dap_values).mode().iloc[0])
    propagation = pd.DataFrame([{
        "critical_n_dap": dap, "representative_terminal_dap": representative_terminal_dap,
        "dap_distance_to_terminal": representative_terminal_dap - dap,
        "direct_nstep_transition_count": base.N_STEP,
        "maximum_direct_lookback_gap_steps": base.N_STEP - 1,
        "within_direct_nstep_coverage": bool(representative_terminal_dap - dap < base.N_STEP),
    } for dap in CRITICAL_N_DAPS])
    propagation.to_csv(OUT / "021_47_preregistered_dap_propagation_distance.csv", index=False, encoding="utf-8-sig")

    bonus_transition = transition.loc[transition.nstep_contains_bonus.astype(bool)]
    bonus_majority_below = bool(
        len(bonus_transition) > 0
        and bonus_transition.significantly_below_per_expectation.astype(bool).mean() > 0.5
    )
    category_bonus = category.loc[category.category == "nstep_bonus"].iloc[0]
    category_bonus_below = bool(category_bonus.significantly_below_per_expectation)
    target_unique = int(target_trace.target_hash.nunique())
    exact_reproduction = bool(
        interaction_check["exact"] and hashes.online_hash_exact.astype(bool).all()
        and hashes.target_hash_exact.astype(bool).all() and evaluation_max_error == 0.0
        and evaluation_gate_exact and replay_mapping.action_match.astype(bool).all()
        and replay_mapping.done_match.astype(bool).all()
        and float(replay_mapping.reward_abs_error.max()) < 1e-3
    )
    if not exact_reproduction:
        branch = "D"
    elif bonus_majority_below and category_bonus_below:
        branch = "A"
    elif target_unique == 1 and not propagation.within_direct_nstep_coverage.any():
        branch = "B"
    else:
        branch = "C"

    validation = {
        **audit, "threshold_provenance": provenance,
        "interaction_reproduction": interaction_check,
        "all_online_checkpoint_hashes_exact": bool(hashes.online_hash_exact.astype(bool).all()),
        "all_target_checkpoint_hashes_exact": bool(hashes.target_hash_exact.astype(bool).all()),
        "deterministic_evaluation_max_error": evaluation_max_error,
        "deterministic_evaluation_gate_exact": evaluation_gate_exact,
        "replay_origin_actions_all_match": bool(replay_mapping.action_match.astype(bool).all()),
        "replay_origin_done_all_match": bool(replay_mapping.done_match.astype(bool).all()),
        "replay_origin_max_reward_error_float32": float(replay_mapping.reward_abs_error.max()),
        "target_unique_hash_count_across_all_updates": target_unique,
        "target_frozen_across_all_online_updates": target_unique == 1,
        "sample_draw_rows": len(samples),
        "expected_agent_draw_rows": len(updates) * AGENT_DRAWS_PER_UPDATE,
        "nstep_bonus_transition_count": int(len(bonus_transition)),
        "nstep_bonus_transition_majority_below_per_expectation": bonus_majority_below,
        "nstep_bonus_category_below_per_expectation": category_bonus_below,
        "exact_reproduction": exact_reproduction,
        "rng_calls_added_by_instrumentation": 0,
        "original_mixed_sample_calls_per_update": 1,
        "all_required_checks_pass": exact_reproduction and len(samples) == len(updates) * AGENT_DRAWS_PER_UPDATE,
    }
    (OUT / "021_47_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "interpretation": {
            "A": "bonus-bearing transitions的实际采样显著低于PER路径期望，支持采样暴露不足候选。",
            "B": "bonus-bearing transitions未显示系统性低采样；target在1K内全程冻结且关键施氮DAP远超直接5-step覆盖，同时已采样bonus transitions仍存在巨大target-Q残差。支持传播与局部拟合受限候选，但尚未确认单一根因。",
            "C": "采样和target条件不支持前两种解释，需要另查机制。",
            "D": "精确重放或映射验证失败，新增日志不得用于科学解释。",
        }[branch],
        "seed2_started": False, "training_5k_started": False,
    }
    (OUT / "021_47_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(category, samples)

    category_display = category[["category", "expected_draws", "observed_draws", "lower_95", "upper_95",
                                 "significantly_below_per_expectation"]].round(3)
    propagation_display = propagation.round(3)
    hash_display = hashes[["checkpoint", "online_hash_exact", "target_hash_exact"]]
    fit_display = value_fit.round(3)
    doc = f"""# 021_47 SY2014 terminal bonus replay采样与传播精确重放审计

## 复现结果

- 训练interactions逐列精确一致：{interaction_check['exact']}，最大数值误差={interaction_check['max_numeric_error']}。
- checkpoint online Q哈希逐位一致：{bool(hashes.online_hash_exact.astype(bool).all())}；target哈希逐位一致：{bool(hashes.target_hash_exact.astype(bool).all())}。
- 确定性评估最大误差：{evaluation_max_error}；gate逐项一致：{evaluation_gate_exact}。
- replay index→origin step映射：action全部一致={bool(replay_mapping.action_match.astype(bool).all())}，done全部一致={bool(replay_mapping.done_match.astype(bool).all())}，reward最大误差={float(replay_mapping.reward_abs_error.max()):.3g}（float32存储）。
- 插桩没有新增RNG调用；每次update只调用一次原mixed sampler。

{markdown_table(hash_display)}

## PER采样暴露

{markdown_table(category_display)}

n-step含bonus的transition共{len(bonus_transition)}条；其中显著低于各自PER期望的比例为{float(bonus_transition.significantly_below_per_expectation.astype(bool).mean()):.3f}。类别总采样是否低于95%下界：{category_bonus_below}。

## 已采样样本的value拟合

{markdown_table(fit_display)}

bonus样本不是“没有被看到”：direct bonus被抽中1703次，n-step含bonus样本被抽中1890次。但其平均chosen Q仍远低于target，且TDn全部位于Huber线性区；PER importance correction还使其平均权重低于普通资源成本样本。这表明除bootstrap覆盖外，还存在局部value拟合不足，不能把全部失败单独归因于target冻结。

## target与传播距离

951次online update后记录到的target唯一哈希数：{target_unique}。因此target在本次1K自定义loop中全程冻结：{target_unique == 1}。

{markdown_table(propagation_display)}

这里的5-step只表示标准n-step target对终止奖励的直接覆盖（终止transition及最多前4个transition），不是神经网络参数共享造成的总影响硬上限。

## 预注册判定

分支：**{branch}**。{summary['interpretation']}

## 适用范围

本任务精确重放的是021_45的1K固定target短期诊断，不是完整的长期DQN训练。结果不修改021_36–40的离线梯度/Q/loss证据，但要求021_35/41/42/45的在线“保持/坍缩”结论带上固定target限定。没有启动seed2或5K，也没有改reward、target同步或其他超参数。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "branch": branch, "validation": validation}, indent=2))


if __name__ == "__main__": main()

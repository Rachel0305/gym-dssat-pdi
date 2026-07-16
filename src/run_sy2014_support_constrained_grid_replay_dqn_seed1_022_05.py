from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

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

import run_sy2014_stage_mc_dqn_seed1_short_022_02 as base
import run_sy2014_fixed_grid_replay_prefill_dqn_seed1_022_03 as exp03
from stage_based_dqn_core_022 import EXECUTABLE_STAGE_DAPS, valid_action_indices


OUT_ROOT = ROOT / "benchmark_results" / "022_05"
MANIFEST = ROOT / "benchmark_results" / "022_03" / "022_03_fixed_grid_transition_manifest.csv"
CONTROL_SUMMARY = ROOT / "benchmark_results" / "022_03" / "022_03_checkpoint_evaluation_summary.csv"
SEED = 1
TRAINING_SEASONS = 60
CHECKPOINT_SEASONS = (15, 30, 45, 60)
EXPECTED_SUPPORT = {
    1: (0, 1, 3, 4),
    30: (4, 5, 7, 8),
    50: (4, 5, 7, 8),
    65: (1, 2, 4, 5, 7, 8),
    85: (1, 2, 4, 5),
    110: (0, 1),
}

# Imported environment helpers must use this task's isolated output directory.
base.OUT_ROOT = OUT_ROOT


def derive_support_and_precheck() -> tuple[dict[int, tuple[int, ...]], dict[str, Any]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    manifest = pd.read_csv(MANIFEST)
    required = {"dap", "action_index", "primary_scientific_pass_022_01", "scenario"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    manifest["primary_scientific_pass_022_01"] = (
        manifest["primary_scientific_pass_022_01"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if manifest["primary_scientific_pass_022_01"].isna().any():
        raise RuntimeError("Manifest contains invalid success labels")

    support = {
        int(dap): tuple(sorted(group["action_index"].astype(int).unique().tolist()))
        for dap, group in manifest.groupby("dap", sort=True)
    }
    checks: dict[str, bool] = {
        "dataset_validation_passed": False,
        "prefill_size_288": False,
        "six_stage_support_exact": support == EXPECTED_SUPPORT,
        "each_stage_has_success_and_failure_sources": True,
        "all_support_actions_originally_legal": True,
        "random_selection_never_leaves_support": True,
        "greedy_selection_never_leaves_support": True,
        "dap110_has_no_nitrogen_actions": support.get(110) == (0, 1),
    }

    validation = json.loads(exp03.DATASET_VALIDATION.read_text(encoding="utf-8"))
    checks["dataset_validation_passed"] = bool(
        validation.get("status") == "passed" and all(validation.get("checks", {}).values())
    )
    replay = exp03.load_prefill()
    checks["prefill_size_288"] = len(replay) == 288

    support_rows: list[dict[str, Any]] = []
    test_rng = np.random.default_rng(20260715)
    for dap in EXECUTABLE_STAGE_DAPS:
        dap = int(dap)
        stage = manifest.loc[manifest["dap"].eq(dap)]
        labels = set(stage["primary_scientific_pass_022_01"].tolist())
        if labels != {False, True}:
            checks["each_stage_has_success_and_failure_sources"] = False
        original_valid = set(valid_action_indices(dap))
        if not set(support[dap]).issubset(original_valid):
            checks["all_support_actions_originally_legal"] = False
        random_draws = test_rng.choice(support[dap], size=1000)
        if not set(random_draws.tolist()).issubset(set(support[dap])):
            checks["random_selection_never_leaves_support"] = False
        for preferred in range(9):
            q_values = np.zeros(9, dtype=np.float64)
            q_values[preferred] = 100.0
            chosen = base.masked_argmax(q_values, support[dap])
            if chosen not in support[dap]:
                checks["greedy_selection_never_leaves_support"] = False
        for action in support[dap]:
            action_rows = stage.loc[stage["action_index"].eq(action)]
            support_rows.append(
                {
                    "dap": dap,
                    "action_index": int(action),
                    "source_scenarios": int(action_rows["scenario"].nunique()),
                    "has_success_source": bool(action_rows["primary_scientific_pass_022_01"].any()),
                    "has_failure_source": bool((~action_rows["primary_scientific_pass_022_01"]).any()),
                    "originally_legal": int(action) in original_valid,
                }
            )

    status = "passed" if all(checks.values()) else "failed"
    payload = {
        "status": status,
        "checks": checks,
        "derived_support": {str(k): list(v) for k, v in support.items()},
        "expected_support": {str(k): list(v) for k, v in EXPECTED_SUPPORT.items()},
        "manifest_rows": int(len(manifest)),
        "replay_prefill_size": int(len(replay)),
        "scientific_change": "behavior_support_mask_applied_during_exploration_greedy_training_and_evaluation",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(support_rows).to_csv(OUT_ROOT / "022_05_behavior_support_sources.csv", index=False)
    (OUT_ROOT / "022_05_pretraining_checks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if status != "passed":
        raise RuntimeError(f"022_05 pretraining checks failed: {checks}")
    return support, payload


def evaluate_checkpoint(
    model: base.QNetwork,
    season: int,
    scaler: base.FixedObservationScaler,
    thresholds: dict[str, float],
    support: dict[int, tuple[int, ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = base.make_env(OUT_ROOT / "evaluation_runtime" / f"season_{season:03d}", seed=SEED)
    runner = base.StageSeasonRunner(env, scaler)
    model.eval()

    def choose(_index: int, dap: int, obs: np.ndarray, _valid: tuple[int, ...]) -> int:
        with torch.no_grad():
            q_values = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
        return base.masked_argmax(q_values, support[int(dap)])

    try:
        result = runner.run(choose, f"eval_s{season}", keep_daily=True)
        metrics = base.summary_metrics(env, result, thresholds)
    finally:
        env.close()
    row = {
        "checkpoint_season": season,
        "final_yield": result["final_yield"],
        "final_biomass": result["final_biomass"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "g0_raw": result["returns_raw"][0],
        "g0_scaled": result["returns_scaled"][0],
        "action_sequence": json.dumps(result["selected_actions"]),
        **metrics,
    }
    return row, result["stage_rows"], result["daily_rows"]


def train() -> dict[str, Any]:
    if (OUT_ROOT / "022_05_checkpoint_evaluation_summary.csv").exists():
        raise FileExistsError("022_05 outputs already exist; refusing to overwrite")
    support, prechecks = derive_support_and_precheck()
    replay: deque[tuple[np.ndarray, int, float]] = exp03.load_prefill()
    prefill_size = len(replay)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    scaler = base.FixedObservationScaler(base.SCALER_PATH)
    thresholds = json.loads(base.THRESHOLDS_PATH.read_text(encoding="utf-8"))
    model = base.QNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    env = base.make_env(OUT_ROOT / "training_runtime", seed=SEED)
    runner = base.StageSeasonRunner(env, scaler)

    episode_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_stage_rows: list[dict[str, Any]] = []
    evaluation_daily_rows: list[dict[str, Any]] = []
    update_index = 0
    try:
        for season in range(1, TRAINING_SEASONS + 1):
            epsilon = base.epsilon_for_season(season)
            model.eval()

            def choose(_idx: int, dap: int, obs: np.ndarray, _valid: tuple[int, ...]) -> int:
                effective_valid = support[int(dap)]
                if rng.random() < epsilon:
                    return int(rng.choice(effective_valid))
                with torch.no_grad():
                    q_values = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
                return base.masked_argmax(q_values, effective_valid)

            result = runner.run(choose, f"train_s{season}", keep_daily=False)
            for state, action, target in zip(result["states"], result["selected_actions"], result["returns_scaled"]):
                replay.append((state.copy(), int(action), float(target)))
            for row in result["stage_rows"]:
                if int(row["action_index"]) not in support[int(row["dap"])]:
                    raise RuntimeError(f"Training selected unsupported action: {row}")
            stage_rows.extend({"training_season": season, "epsilon": epsilon, **row} for row in result["stage_rows"])
            episode_rows.append(
                {
                    "training_season": season,
                    "epsilon": epsilon,
                    "final_yield": result["final_yield"],
                    "final_biomass": result["final_biomass"],
                    "irrigation_total": result["irrigation_total"],
                    "nitrogen_total": result["nitrogen_total"],
                    "g0_raw": result["returns_raw"][0],
                    "g0_scaled": result["returns_scaled"][0],
                    "action_sequence": json.dumps(result["selected_actions"]),
                    "replay_size": len(replay),
                    "prefill_size": prefill_size,
                }
            )
            if season >= 5:
                model.train()
                for _ in range(6):
                    update_index += 1
                    sample_indices = rng.integers(0, len(replay), size=32)
                    batch = [replay[int(index)] for index in sample_indices]
                    states = torch.from_numpy(np.stack([item[0] for item in batch])).float()
                    actions_t = torch.tensor([item[1] for item in batch], dtype=torch.long)
                    targets_t = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                    q_all = model(states)
                    q_selected = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(q_selected, targets_t)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_sq = sum(
                        float(parameter.grad.detach().pow(2).sum().item())
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                    grad_preclip = math.sqrt(grad_sq)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    update_rows.append(
                        {
                            "update": update_index,
                            "training_season": season,
                            "loss": float(loss.item()),
                            "grad_norm_preclip": grad_preclip,
                            "q_selected_mean": float(q_selected.detach().mean().item()),
                            "q_abs_max": float(q_all.detach().abs().max().item()),
                            "target_mean": float(targets_t.mean().item()),
                            "target_min": float(targets_t.min().item()),
                            "target_max": float(targets_t.max().item()),
                            "replay_size": len(replay),
                        }
                    )
            if season in CHECKPOINT_SEASONS:
                checkpoint = OUT_ROOT / "checkpoints" / f"season_{season:03d}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "season": season,
                        "seed": SEED,
                        "return_scale": base.RETURN_SCALE,
                        "fixed_grid_prefill_size": prefill_size,
                        "behavior_support": support,
                    },
                    checkpoint,
                )
                eval_row, eval_stages, eval_daily = evaluate_checkpoint(model, season, scaler, thresholds, support)
                evaluation_rows.append(eval_row)
                evaluation_stage_rows.extend(eval_stages)
                evaluation_daily_rows.extend(eval_daily)
                print(
                    f"checkpoint={season} yield={eval_row['final_yield']:.0f} "
                    f"I={eval_row['irrigation_total']:.0f} N={eval_row['nitrogen_total']:.0f} "
                    f"primary={eval_row['primary_pass']} strict={eval_row['strict_pass']}",
                    flush=True,
                )
    finally:
        env.close()

    pd.DataFrame(episode_rows).to_csv(OUT_ROOT / "022_05_training_seasons.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_ROOT / "022_05_training_stage_actions.csv", index=False)
    pd.DataFrame(update_rows).to_csv(OUT_ROOT / "022_05_update_log.csv", index=False)
    eval_df = pd.DataFrame(evaluation_rows)
    eval_df.to_csv(OUT_ROOT / "022_05_checkpoint_evaluation_summary.csv", index=False)
    pd.DataFrame(evaluation_stage_rows).to_csv(OUT_ROOT / "022_05_checkpoint_stage_actions.csv", index=False)
    pd.DataFrame(evaluation_daily_rows).to_csv(OUT_ROOT / "022_05_checkpoint_daily_values.csv", index=False)

    primary_count = int(eval_df["primary_pass"].sum())
    strict_count = int(eval_df["strict_pass"].sum())
    season60_primary = bool(eval_df.loc[eval_df["checkpoint_season"].eq(60), "primary_pass"].iloc[0])
    if primary_count >= 3 and season60_primary and strict_count >= 1:
        branch, next_step = "A_initial_success", "allow_one_independent_seed"
    elif primary_count >= 1:
        branch, next_step = "B_signal_but_unstable", "report_trajectory_no_automatic_expansion"
    else:
        branch, next_step = "C_failed", "stop_support_constrained_training_branch"

    control = pd.read_csv(CONTROL_SUMMARY)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(control.checkpoint_season, control.final_yield, "o--", color="#777777", label="022_03 no support mask")
    axes[0].plot(eval_df.checkpoint_season, eval_df.final_yield, "o-", color="#1f4e79", label="022_05 support-constrained")
    axes[0].axhline(thresholds["yield_min"], color="#b22222", linestyle=":", label="yield threshold")
    axes[0].set_ylabel("HWAM (kg/ha)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(control.checkpoint_season, control.irrigation_total, "o--", color="#56B4E9", label="022_03 I")
    axes[1].plot(control.checkpoint_season, control.nitrogen_total, "s--", color="#E69F00", label="022_03 N")
    axes[1].plot(eval_df.checkpoint_season, eval_df.irrigation_total, "o-", color="#0072B2", label="022_05 I")
    axes[1].plot(eval_df.checkpoint_season, eval_df.nitrogen_total, "s-", color="#D55E00", label="022_05 N")
    axes[1].set_xlabel("Online training season")
    axes[1].set_ylabel("Seasonal input")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.2)
    fig.suptitle("SY2014 behavior-support-constrained grid-replay DQN seed1")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_05_vs_022_03_training_diagnostics.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_05_vs_022_03_training_diagnostics.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "only_scientific_change": "fixed_grid_behavior_support_mask_during_training_and_evaluation",
        "pretraining_checks": prechecks,
        "behavior_support": {str(k): list(v) for k, v in support.items()},
        "prefill_transitions": prefill_size,
        "seed": SEED,
        "online_training_seasons": TRAINING_SEASONS,
        "online_stage_interactions": TRAINING_SEASONS * 6,
        "gradient_updates": update_index,
        "primary_checkpoint_count": primary_count,
        "strict_checkpoint_count": strict_count,
        "season60_primary": season60_primary,
        "checkpoints": evaluation_rows,
    }
    (OUT_ROOT / "022_05_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    result = train()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

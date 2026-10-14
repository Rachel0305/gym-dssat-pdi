from __future__ import annotations

import argparse
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
from stage_based_dqn_core_022 import ACTION_TABLE_9, EXECUTABLE_STAGE_DAPS


OUT_ROOT = ROOT / "benchmark_results" / "022_03"
SUMMARY_022_01 = ROOT / "benchmark_results" / "022_01" / "022_01_candidate_summary.csv"
RUNS_022_01 = ROOT / "benchmark_results" / "022_01" / "runs"
DATASET_NPZ = OUT_ROOT / "022_03_fixed_grid_replay_dataset.npz"
DATASET_VALIDATION = OUT_ROOT / "022_03_dataset_validation.json"
SEED = 1
TRAINING_SEASONS = 60
CHECKPOINT_SEASONS = (15, 30, 45, 60)

# Imported helpers must write into this task's isolated output directory.
base.OUT_ROOT = OUT_ROOT


def action_index_for(real: dict[str, float]) -> int:
    i = float(real.get("amir", 0.0))
    n = float(real.get("anfer", 0.0))
    matches = [
        index
        for index, action in ACTION_TABLE_9.items()
        if action["amir"] == i and action["anfer"] == n
    ]
    if len(matches) != 1:
        raise ValueError(f"Schedule action I{i}/N{n} has {len(matches)} action-table matches")
    return matches[0]


def schedule_actions(scenario: str) -> list[int]:
    schedule_path = RUNS_022_01 / scenario / "schedule.json"
    if not schedule_path.exists():
        raise FileNotFoundError(schedule_path)
    raw = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = {int(dap): value for dap, value in raw.items()}
    extra = sorted(set(schedule) - set(EXECUTABLE_STAGE_DAPS))
    if extra:
        raise ValueError(f"{scenario} contains non-stage scheduled DAPs: {extra}")
    return [action_index_for(schedule.get(dap, {"amir": 0.0, "anfer": 0.0})) for dap in EXECUTABLE_STAGE_DAPS]


def build_dataset() -> dict[str, Any]:
    if DATASET_NPZ.exists() or DATASET_VALIDATION.exists():
        raise FileExistsError("022_03 dataset outputs already exist; refusing to overwrite")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_022_01)
    grid = summary[summary["phase"].eq("fixed_official_stage_grid")].copy()
    if len(grid) != 48 or grid["scenario"].nunique() != 48:
        raise RuntimeError(f"Expected exactly 48 unique fixed-grid scenarios, got {len(grid)}")
    grid = grid.sort_values("scenario").reset_index(drop=True)
    scaler = base.FixedObservationScaler(base.SCALER_PATH)
    env = base.make_env(OUT_ROOT / "dataset_runtime", seed=SEED)
    runner = base.StageSeasonRunner(env, scaler)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    targets: list[float] = []
    raw_targets: list[float] = []
    transition_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    try:
        for _, reference in grid.iterrows():
            scenario = str(reference["scenario"])
            sequence = schedule_actions(scenario)
            result = runner.run(
                lambda index, _dap, _obs, _valid, seq=sequence: seq[index],
                f"grid_{scenario}",
                keep_daily=False,
            )
            yield_error = float(result["final_yield"] - float(reference["final_gwad"]))
            i_error = float(result["irrigation_total"] - float(reference["requested_irrigation_total"]))
            n_error = float(result["nitrogen_total"] - float(reference["requested_nitrogen_total"]))
            scenario_pass = abs(yield_error) <= 2.0 and abs(i_error) <= 1e-9 and abs(n_error) <= 1e-9
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "reference_yield": float(reference["final_gwad"]),
                    "replayed_yield": result["final_yield"],
                    "yield_error": yield_error,
                    "reference_irrigation": float(reference["requested_irrigation_total"]),
                    "replayed_irrigation": result["irrigation_total"],
                    "irrigation_error": i_error,
                    "reference_nitrogen": float(reference["requested_nitrogen_total"]),
                    "replayed_nitrogen": result["nitrogen_total"],
                    "nitrogen_error": n_error,
                    "primary_scientific_pass_022_01": bool(reference["primary_scientific_pass"]),
                    "strict_resource_pass_022_01": bool(reference["strict_resource_pass"]),
                    "action_sequence": json.dumps(sequence),
                    "replay_match": scenario_pass,
                }
            )
            for stage_index, (state, action, target, raw_target) in enumerate(
                zip(result["states"], result["selected_actions"], result["returns_scaled"], result["returns_raw"])
            ):
                if state.shape != (25,) or not np.isfinite(state).all():
                    raise RuntimeError(f"{scenario} stage {stage_index} has invalid standardized observation")
                observations.append(state.astype(np.float32, copy=True))
                actions.append(int(action))
                targets.append(float(target))
                raw_targets.append(float(raw_target))
                transition_rows.append(
                    {
                        "scenario": scenario,
                        "stage_index": stage_index,
                        "dap": EXECUTABLE_STAGE_DAPS[stage_index],
                        "action_index": int(action),
                        "target_raw": float(raw_target),
                        "target_scaled": float(target),
                        "final_yield": result["final_yield"],
                        "irrigation_total": result["irrigation_total"],
                        "nitrogen_total": result["nitrogen_total"],
                        "primary_scientific_pass_022_01": bool(reference["primary_scientific_pass"]),
                    }
                )
            print(f"dataset {len(scenario_rows):02d}/48 {scenario}", flush=True)
    finally:
        env.close()

    scenario_df = pd.DataFrame(scenario_rows)
    transition_df = pd.DataFrame(transition_rows)
    checks = {
        "exactly_48_unique_grid_scenarios": len(scenario_df) == 48 and scenario_df.scenario.nunique() == 48,
        "all_scenario_replays_match": bool(scenario_df["replay_match"].all()),
        "maximum_yield_error_le_2": float(scenario_df["yield_error"].abs().max()) <= 2.0,
        "resource_totals_exact": float(scenario_df["irrigation_error"].abs().max()) <= 1e-9
        and float(scenario_df["nitrogen_error"].abs().max()) <= 1e-9,
        "exactly_288_transitions": len(observations) == 288,
        "all_observations_25d_finite": bool(
            len(observations) == 288
            and np.asarray(observations).shape == (288, 25)
            and np.isfinite(np.asarray(observations)).all()
        ),
        "contains_primary_success_and_failure": bool(
            scenario_df["primary_scientific_pass_022_01"].any()
            and (~scenario_df["primary_scientific_pass_022_01"]).any()
        ),
        "six_transitions_per_scenario": bool((transition_df.groupby("scenario").size() == 6).all()),
    }
    passed = all(checks.values())
    scenario_df.to_csv(OUT_ROOT / "022_03_scenario_replay_validation.csv", index=False)
    transition_df.to_csv(OUT_ROOT / "022_03_fixed_grid_transition_manifest.csv", index=False)
    np.savez_compressed(
        DATASET_NPZ,
        observations=np.stack(observations).astype(np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float32),
        raw_targets=np.asarray(raw_targets, dtype=np.float32),
        scenarios=transition_df["scenario"].to_numpy(dtype=str),
        stage_indices=transition_df["stage_index"].to_numpy(dtype=np.int64),
    )
    payload = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "scenario_count": len(scenario_df),
        "transition_count": len(observations),
        "primary_success_scenarios": int(scenario_df["primary_scientific_pass_022_01"].sum()),
        "primary_failure_scenarios": int((~scenario_df["primary_scientific_pass_022_01"]).sum()),
        "strict_success_scenarios": int(scenario_df["strict_resource_pass_022_01"].sum()),
        "max_abs_yield_error": float(scenario_df["yield_error"].abs().max()),
        "max_abs_irrigation_error": float(scenario_df["irrigation_error"].abs().max()),
        "max_abs_nitrogen_error": float(scenario_df["nitrogen_error"].abs().max()),
        "dataset_path": str(DATASET_NPZ.relative_to(ROOT)),
    }
    DATASET_VALIDATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"022_03 dataset validation failed: {checks}")
    return payload


def load_prefill() -> deque[tuple[np.ndarray, int, float]]:
    validation = json.loads(DATASET_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not all(validation.get("checks", {}).values()):
        raise RuntimeError("Dataset validation is missing or failed")
    data = np.load(DATASET_NPZ)
    observations = data["observations"]
    actions = data["actions"]
    targets = data["targets"]
    if observations.shape != (288, 25) or actions.shape != (288,) or targets.shape != (288,):
        raise RuntimeError("Unexpected replay-prefill array shapes")
    replay: deque[tuple[np.ndarray, int, float]] = deque(maxlen=2000)
    for obs, action, target in zip(observations, actions, targets):
        replay.append((obs.astype(np.float32, copy=True), int(action), float(target)))
    if len(replay) != 288:
        raise RuntimeError(f"Expected replay prefill size 288, got {len(replay)}")
    return replay


def train() -> dict[str, Any]:
    if (OUT_ROOT / "022_03_checkpoint_evaluation_summary.csv").exists():
        raise FileExistsError("022_03 training outputs already exist; refusing to overwrite")
    replay = load_prefill()
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

            def choose(_idx: int, _dap: int, obs: np.ndarray, valid: tuple[int, ...]) -> int:
                if rng.random() < epsilon:
                    return int(rng.choice(valid))
                with torch.no_grad():
                    q = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
                return base.masked_argmax(q, valid)

            result = runner.run(choose, f"train_s{season}", keep_daily=False)
            for state, action, target in zip(result["states"], result["selected_actions"], result["returns_scaled"]):
                replay.append((state.copy(), int(action), float(target)))
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
                    },
                    checkpoint,
                )
                eval_row, eval_stages, eval_daily = base.evaluate_checkpoint(model, season, scaler, thresholds)
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

    pd.DataFrame(episode_rows).to_csv(OUT_ROOT / "022_03_training_seasons.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_ROOT / "022_03_training_stage_actions.csv", index=False)
    pd.DataFrame(update_rows).to_csv(OUT_ROOT / "022_03_update_log.csv", index=False)
    eval_df = pd.DataFrame(evaluation_rows)
    eval_df.to_csv(OUT_ROOT / "022_03_checkpoint_evaluation_summary.csv", index=False)
    pd.DataFrame(evaluation_stage_rows).to_csv(OUT_ROOT / "022_03_checkpoint_stage_actions.csv", index=False)
    pd.DataFrame(evaluation_daily_rows).to_csv(OUT_ROOT / "022_03_checkpoint_daily_values.csv", index=False)

    primary_count = int(eval_df["primary_pass"].sum())
    strict_count = int(eval_df["strict_pass"].sum())
    season60_primary = bool(eval_df.loc[eval_df["checkpoint_season"].eq(60), "primary_pass"].iloc[0])
    if primary_count >= 3 and season60_primary and strict_count >= 1:
        branch, next_step = "A_initial_success", "allow_one_independent_seed"
    elif primary_count >= 1:
        branch, next_step = "B_signal_but_unstable", "report_trajectory_no_automatic_expansion"
    else:
        branch, next_step = "C_failed", "stop_branch_no_onsite_tuning"

    comparison = pd.read_csv(ROOT / "benchmark_results" / "022_02" / "022_02_checkpoint_evaluation_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(comparison.checkpoint_season, comparison.final_yield, "o--", color="#777777", label="022_02 online only")
    axes[0].plot(eval_df.checkpoint_season, eval_df.final_yield, "o-", color="#1f4e79", label="022_03 grid replay prefill")
    axes[0].axhline(thresholds["yield_min"], color="#b22222", linestyle=":", label="yield threshold")
    axes[0].set_ylabel("HWAM (kg/ha)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(eval_df.checkpoint_season, eval_df.irrigation_total, "o-", label="I (mm)", color="#0072B2")
    axes[1].plot(eval_df.checkpoint_season, eval_df.nitrogen_total, "s-", label="N (kg/ha)", color="#D55E00")
    axes[1].set_xlabel("Online training season")
    axes[1].set_ylabel("Seasonal input")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle("SY2014 fixed-grid replay prefill DQN seed1")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_03_training_diagnostics.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_03_training_diagnostics.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "only_scientific_change": "uniform_replay_prefilled_with_all_48_fixed_grid_scenarios",
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
    (OUT_ROOT / "022_03_result.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-dataset", action="store_true")
    group.add_argument("--train", action="store_true")
    args = parser.parse_args()
    result = build_dataset() if args.build_dataset else train()
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from dataclasses import asdict
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

from calculate_five_site_wue_nue_from_summary_019_10 import (
    num,
    parse_summary_out,
    select_matching_row,
)
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
import run_sy2014_deterministic_oracle_search_021_18 as oracle
from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    EXECUTABLE_STAGE_DAPS,
    EXPERT_GATE_YIELD,
    FEASIBILITY_BONUS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
    NULL_YIELD,
    ExecutedAction,
    execute_stage_action,
    terminal_complete_returns,
    valid_action_indices,
)


OUT_ROOT = ROOT / "benchmark_results" / "022_02"
SCALER_PATH = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS_PATH = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
VALIDATION_ACTIONS = (3, 4, 7, 1, 1, 0)
RETURN_SCALE = 1000.0
SEED = 1
TRAINING_SEASONS = 60
CHECKPOINT_SEASONS = (15, 30, 45, 60)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 25, output_dim: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedObservationScaler:
    def __init__(self, path: Path) -> None:
        table = pd.read_csv(path).sort_values("observation_index")
        if table["observation_index"].tolist() != list(range(25)):
            raise ValueError("021_24 scaler does not contain exactly 25 ordered observations")
        self.labels = table["observation_label"].astype(str).tolist()
        self.mean = table["mean"].to_numpy(dtype=np.float32)
        self.scale = table["scale_std_or_one"].to_numpy(dtype=np.float32)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Invalid fixed observation scaler")

    def raw_vector(self, state: dict[str, Any]) -> np.ndarray:
        values = []
        missing = []
        for label in self.labels:
            value = scalar(state.get(label))
            if value is None:
                missing.append(label)
                values.append(np.nan)
            else:
                values.append(float(value))
        if missing:
            raise KeyError(f"Missing observation fields: {missing}")
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            bad = {
                label: repr(value)
                for label, value in zip(self.labels, values)
                if not np.isfinite(value)
            }
            raise ValueError(f"Non-finite raw observation fields: {bad}")
        return array

    def environment_vector(self, observation: Any, state: dict[str, Any]) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if array.size != 25:
            raise ValueError(f"Expected wrapper observation dimension 25, got {array.shape}")
        if not np.isfinite(array).all():
            bad = np.flatnonzero(~np.isfinite(array)).tolist()
            raise ValueError(f"Non-finite values in wrapper observation at indices {bad}")
        # Check that the wrapper's flattened order still matches the frozen
        # scaler wherever the underlying DSSAT dictionary is already finite.
        comparable = []
        for index, label in enumerate(self.labels):
            value = scalar(state.get(label))
            if value is not None and np.isfinite(value):
                comparable.append(abs(float(array[index]) - float(value)))
        if not comparable or max(comparable) > 2e-3:
            raise ValueError(
                f"Wrapper observation order mismatch; max finite-field error={max(comparable) if comparable else None}"
            )
        return array

    def transform(self, observation: Any, state: dict[str, Any]) -> np.ndarray:
        return (self.environment_vector(observation, state) - self.mean) / self.scale


def make_env(run_dir: Path, seed: int = SEED) -> Any:
    run_dir.mkdir(parents=True, exist_ok=True)
    args = oracle.make_env_args(run_dir)
    args["seed"] = int(seed)
    args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    (run_dir / "env_args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    return oracle.shared.make_raw_env(args)


class StageSeasonRunner:
    def __init__(self, env: Any, scaler: FixedObservationScaler) -> None:
        self.env = env
        self.scaler = scaler

    @staticmethod
    def dap_of(state: dict[str, Any]) -> int:
        value = scalar(state.get("dap"))
        if value is None:
            raise RuntimeError("DSSAT state has no DAP")
        return int(round(float(value)))

    def _noop(self) -> np.ndarray:
        return normalize_action(
            self.env.formator.action_names,
            self.env.formator.action_space_dict,
            {"amir": 0.0, "anfer": 0.0},
        )

    def _action(self, action: ExecutedAction) -> np.ndarray:
        return normalize_action(
            self.env.formator.action_names,
            self.env.formator.action_space_dict,
            {"amir": action.executed_irrigation, "anfer": action.executed_nitrogen},
        )

    def run(
        self,
        action_selector: Any,
        episode_label: str,
        keep_daily: bool = False,
    ) -> dict[str, Any]:
        daily_rows: list[dict[str, Any]] = []
        obs, info = self.env.reset()
        state = latest_observation_dict(self.env, obs, info)
        initial_dap = self.dap_of(state)
        # gym-DSSAT exposes a DAP0 initialization state.  The frozen scientific
        # schedule starts at executable DAP1, so DAP0 must be an explicit no-op
        # initialization transition rather than an agent decision.
        initialization_steps = 0
        while initial_dap == 0:
            initialization_steps += 1
            if initialization_steps > 60:
                raise RuntimeError("DAP0 initialization exceeded 60 no-op transitions")
            obs, raw_reward, terminated, truncated, info = self.env.step(self._noop())
            state = latest_observation_dict(self.env, obs, info)
            if keep_daily:
                daily_rows.append(
                    {
                        "episode": episode_label,
                        "dap_action": 0,
                        "is_stage_action": False,
                        "action_index": 0,
                        "irrigation_mm": 0.0,
                        "nitrogen_kg_ha": 0.0,
                        "dap_after": self.dap_of(state),
                        "grnwt_after": scalar(state.get("grnwt")),
                        "topwt_after": scalar(state.get("topwt")),
                        "swfac_after": scalar(state.get("swfac")),
                        "nstres_after": scalar(state.get("nstres")),
                        "raw_env_reward": repr(raw_reward),
                    }
                )
            if terminated or truncated:
                raise RuntimeError("Season ended during DAP0 initialization transition")
            initial_dap = self.dap_of(state)
        initial_dap = self.dap_of(state)
        if initial_dap != EXECUTABLE_STAGE_DAPS[0]:
            raise RuntimeError(f"Expected executable initial DAP 1, got {initial_dap}")

        used_i = 0.0
        used_n = 0.0
        selected: list[int] = []
        executed: list[ExecutedAction] = []
        stage_rows: list[dict[str, Any]] = []
        states: list[np.ndarray] = []
        terminated = truncated = False
        final_state = state

        for stage_index, stage_dap in enumerate(EXECUTABLE_STAGE_DAPS):
            current_dap = self.dap_of(state)
            if current_dap != stage_dap:
                raise RuntimeError(f"Stage {stage_index}: expected DAP {stage_dap}, got {current_dap}")
            obs_vector = self.scaler.transform(obs, state)
            valid = valid_action_indices(stage_dap)
            action_index = int(action_selector(stage_index, stage_dap, obs_vector.copy(), valid))
            action = execute_stage_action(action_index, stage_dap, used_i, used_n)
            states.append(obs_vector)
            selected.append(action_index)
            executed.append(action)
            used_i += action.executed_irrigation
            used_n += action.executed_nitrogen
            stage_rows.append(
                {
                    "episode": episode_label,
                    "stage_index": stage_index,
                    "dap": stage_dap,
                    "action_index": action_index,
                    **asdict(action),
                    "cumulative_irrigation": used_i,
                    "cumulative_nitrogen": used_n,
                }
            )

            obs, raw_reward, terminated, truncated, info = self.env.step(self._action(action))
            final_state = latest_observation_dict(self.env, obs, info)
            if keep_daily:
                daily_rows.append(
                    {
                        "episode": episode_label,
                        "dap_action": stage_dap,
                        "is_stage_action": True,
                        "action_index": action_index,
                        "irrigation_mm": action.executed_irrigation,
                        "nitrogen_kg_ha": action.executed_nitrogen,
                        "dap_after": self.dap_of(final_state),
                        "grnwt_after": scalar(final_state.get("grnwt")),
                        "topwt_after": scalar(final_state.get("topwt")),
                        "swfac_after": scalar(final_state.get("swfac")),
                        "nstres_after": scalar(final_state.get("nstres")),
                        "raw_env_reward": repr(raw_reward),
                    }
                )
            if terminated or truncated:
                if stage_index != len(EXECUTABLE_STAGE_DAPS) - 1:
                    raise RuntimeError(f"Season ended before stage DAP {EXECUTABLE_STAGE_DAPS[stage_index + 1]}")
                state = final_state
                break

            next_stage = (
                EXECUTABLE_STAGE_DAPS[stage_index + 1]
                if stage_index + 1 < len(EXECUTABLE_STAGE_DAPS)
                else None
            )
            state = final_state
            while not (terminated or truncated):
                current_dap = self.dap_of(state)
                if next_stage is not None and current_dap == next_stage:
                    break
                if next_stage is not None and current_dap > next_stage:
                    raise RuntimeError(f"DSSAT skipped stage DAP {next_stage}; current DAP={current_dap}")
                obs, raw_reward, terminated, truncated, info = self.env.step(self._noop())
                final_state = latest_observation_dict(self.env, obs, info)
                if keep_daily:
                    daily_rows.append(
                        {
                            "episode": episode_label,
                            "dap_action": current_dap,
                            "is_stage_action": False,
                            "action_index": 0,
                            "irrigation_mm": 0.0,
                            "nitrogen_kg_ha": 0.0,
                            "dap_after": self.dap_of(final_state),
                            "grnwt_after": scalar(final_state.get("grnwt")),
                            "topwt_after": scalar(final_state.get("topwt")),
                            "swfac_after": scalar(final_state.get("swfac")),
                            "nstres_after": scalar(final_state.get("nstres")),
                            "raw_env_reward": repr(raw_reward),
                        }
                    )
                state = final_state
            if terminated or truncated:
                break

        if not (terminated or truncated):
            raise RuntimeError("Season did not terminate after final stage")
        if len(executed) != len(EXECUTABLE_STAGE_DAPS):
            raise RuntimeError(f"Expected six stage decisions, got {len(executed)}")
        final_yield = float(scalar(final_state.get("grnwt")) or 0.0)
        final_biomass = float(scalar(final_state.get("topwt")) or 0.0)
        returns_raw = terminal_complete_returns(executed, final_yield)
        returns_scaled = [value / RETURN_SCALE for value in returns_raw]
        if not np.isfinite(returns_scaled).all():
            raise RuntimeError("Non-finite terminal-complete return")
        return {
            "states": states,
            "selected_actions": selected,
            "executed_actions": executed,
            "stage_rows": stage_rows,
            "daily_rows": daily_rows,
            "final_yield": final_yield,
            "final_biomass": final_biomass,
            "irrigation_total": used_i,
            "nitrogen_total": used_n,
            "returns_raw": returns_raw,
            "returns_scaled": returns_scaled,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }


def summary_metrics(env: Any, result: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    tmp = Path(getattr(env.unwrapped, "_tmp_folder"))
    rows = parse_summary_out(tmp / "Summary.OUT")
    row, match_score, row_index = select_matching_row(
        rows,
        result["final_yield"],
        result["irrigation_total"],
        result["nitrogen_total"],
    )
    ircm, nicm = num(row, "IRCM"), num(row, "NICM")
    etcp, ypem, ypnam = num(row, "ETCP"), num(row, "YPEM"), num(row, "YPNAM")
    wp = ypem * 0.1 if ypem is not None and ypem >= 0 else result["final_yield"] / etcp / 10
    pfp = ypnam if nicm and nicm > 0 and ypnam is not None and ypnam >= 0 else math.inf
    no_late_n = all(
        action.executed_nitrogen == 0.0
        for action, dap in zip(result["executed_actions"], EXECUTABLE_STAGE_DAPS)
        if dap >= 90
    )
    primary = (
        result["final_yield"] >= thresholds["yield_min"]
        and wp >= thresholds["wp_et_min"]
        and pfp >= thresholds["pfp_n_min"]
        and result["irrigation_total"] <= thresholds["budget_i_max"]
        and result["nitrogen_total"] <= thresholds["budget_n_max"]
        and no_late_n
    )
    strict = (
        primary
        and result["irrigation_total"] <= thresholds["strict_i_max"]
        and result["nitrogen_total"] <= thresholds["strict_n_max"]
    )
    return {
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": etcp,
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": match_score,
        "summary_row_index": row_index,
        "no_late_n": no_late_n,
        "primary_pass": bool(primary),
        "strict_pass": bool(strict),
    }


def validate_stage_environment() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    scaler = FixedObservationScaler(SCALER_PATH)
    env = make_env(OUT_ROOT / "validation_runtime")
    runner = StageSeasonRunner(env, scaler)
    try:
        result = runner.run(
            lambda index, _dap, _obs, _valid: VALIDATION_ACTIONS[index],
            "exact_replay",
            keep_daily=True,
        )
        thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        metrics = summary_metrics(env, result, thresholds)
    finally:
        env.close()

    invalid_110_raised = False
    try:
        execute_stage_action(3, 110, 0.0, 0.0)
    except ValueError:
        invalid_110_raised = True
    nonstage = [row for row in result["daily_rows"] if not row["is_stage_action"]]
    checks = {
        "six_exact_stage_daps": [row["dap"] for row in result["stage_rows"]] == list(EXECUTABLE_STAGE_DAPS),
        "yield_11202_pm2": abs(result["final_yield"] - 11202.0) <= 2.0,
        "irrigation_total_60": abs(result["irrigation_total"] - 60.0) <= 1e-9,
        "nitrogen_total_200": abs(result["nitrogen_total"] - 200.0) <= 1e-9,
        "g0_6354": abs(result["returns_raw"][0] - 6354.0) <= 2.0,
        "all_nonstage_noop": all(row["irrigation_mm"] == 0 and row["nitrogen_kg_ha"] == 0 for row in nonstage),
        "dap110_n_action_raises": invalid_110_raised,
        "summary_resources_match": abs(metrics["summary_irrigation_total"] - 60.0) <= 1e-9
        and abs(metrics["summary_nitrogen_total"] - 200.0) <= 1e-9,
        "finite_scaled_returns": bool(np.isfinite(result["returns_scaled"]).all()),
    }
    passed = all(checks.values())
    pd.DataFrame(result["stage_rows"]).to_csv(OUT_ROOT / "022_02_validation_stage_actions.csv", index=False)
    pd.DataFrame(result["daily_rows"]).to_csv(OUT_ROOT / "022_02_validation_daily_values.csv", index=False)
    payload = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "action_sequence": list(VALIDATION_ACTIONS),
        "final_yield": result["final_yield"],
        "final_biomass": result["final_biomass"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "returns_raw": result["returns_raw"],
        "returns_scaled": result["returns_scaled"],
        "summary_metrics": metrics,
        "return_formula": f"{result['final_yield']}-5408+1620-60-5*200={result['returns_raw'][0]}",
    }
    (OUT_ROOT / "022_02_stage_environment_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Stage environment validation failed: {checks}")
    return payload


def masked_argmax(q_values: np.ndarray, valid: tuple[int, ...]) -> int:
    return max(valid, key=lambda index: (float(q_values[index]), -index))


def epsilon_for_season(season: int) -> float:
    if TRAINING_SEASONS == 1:
        return 0.2
    return 1.0 + (0.2 - 1.0) * ((season - 1) / (TRAINING_SEASONS - 1))


def evaluate_checkpoint(
    model: QNetwork,
    season: int,
    scaler: FixedObservationScaler,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = make_env(OUT_ROOT / "evaluation_runtime" / f"season_{season:03d}")
    runner = StageSeasonRunner(env, scaler)
    model.eval()

    def choose(_index: int, _dap: int, obs: np.ndarray, valid: tuple[int, ...]) -> int:
        with torch.no_grad():
            q = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
        return masked_argmax(q, valid)

    try:
        result = runner.run(choose, f"eval_s{season}", keep_daily=True)
        metrics = summary_metrics(env, result, thresholds)
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
    validation_path = OUT_ROOT / "022_02_stage_environment_validation.json"
    if not validation_path.exists():
        raise RuntimeError("Validation JSON is missing; run --validate-only first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not all(validation.get("checks", {}).values()):
        raise RuntimeError("Stage validation did not pass; training refused")
    if (OUT_ROOT / "022_02_checkpoint_evaluation_summary.csv").exists():
        raise FileExistsError("022_02 training outputs already exist; refusing to overwrite")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    scaler = FixedObservationScaler(SCALER_PATH)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    model = QNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay: deque[tuple[np.ndarray, int, float]] = deque(maxlen=2000)
    env = make_env(OUT_ROOT / "training_runtime")
    runner = StageSeasonRunner(env, scaler)
    episode_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_stage_rows: list[dict[str, Any]] = []
    evaluation_daily_rows: list[dict[str, Any]] = []
    update_index = 0

    try:
        for season in range(1, TRAINING_SEASONS + 1):
            epsilon = epsilon_for_season(season)
            model.eval()

            def choose(_index: int, _dap: int, obs: np.ndarray, valid: tuple[int, ...]) -> int:
                if rng.random() < epsilon:
                    return int(rng.choice(valid))
                with torch.no_grad():
                    q = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
                return masked_argmax(q, valid)

            result = runner.run(choose, f"train_s{season}", keep_daily=False)
            for state, action, target in zip(
                result["states"], result["selected_actions"], result["returns_scaled"]
            ):
                replay.append((state.copy(), int(action), float(target)))
            for row in result["stage_rows"]:
                stage_rows.append({"training_season": season, "epsilon": epsilon, **row})
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
                }
            )

            if season >= 5:
                model.train()
                for _ in range(6):
                    update_index += 1
                    sample_indices = rng.integers(0, len(replay), size=32)
                    batch = [replay[int(index)] for index in sample_indices]
                    states = torch.from_numpy(np.stack([item[0] for item in batch])).float()
                    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
                    targets = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                    q_all = model(states)
                    q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(q_selected, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_sq = 0.0
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            grad_sq += float(parameter.grad.detach().pow(2).sum().item())
                    grad_preclip = math.sqrt(grad_sq)
                    grad_returned = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                    optimizer.step()
                    update_rows.append(
                        {
                            "update": update_index,
                            "training_season": season,
                            "loss": float(loss.item()),
                            "grad_norm_preclip": grad_preclip,
                            "clip_grad_norm_return": grad_returned,
                            "q_selected_mean": float(q_selected.detach().mean().item()),
                            "q_abs_max": float(q_all.detach().abs().max().item()),
                            "target_mean": float(targets.mean().item()),
                            "target_min": float(targets.min().item()),
                            "target_max": float(targets.max().item()),
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
                        "return_scale": RETURN_SCALE,
                    },
                    checkpoint,
                )
                eval_row, eval_stages, eval_daily = evaluate_checkpoint(model, season, scaler, thresholds)
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

    pd.DataFrame(episode_rows).to_csv(OUT_ROOT / "022_02_training_seasons.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_ROOT / "022_02_training_stage_actions.csv", index=False)
    pd.DataFrame(update_rows).to_csv(OUT_ROOT / "022_02_update_log.csv", index=False)
    eval_df = pd.DataFrame(evaluation_rows)
    eval_df.to_csv(OUT_ROOT / "022_02_checkpoint_evaluation_summary.csv", index=False)
    pd.DataFrame(evaluation_stage_rows).to_csv(OUT_ROOT / "022_02_checkpoint_stage_actions.csv", index=False)
    pd.DataFrame(evaluation_daily_rows).to_csv(OUT_ROOT / "022_02_checkpoint_daily_values.csv", index=False)

    primary_count = int(eval_df["primary_pass"].sum())
    strict_count = int(eval_df["strict_pass"].sum())
    season60_primary = bool(eval_df.loc[eval_df["checkpoint_season"].eq(60), "primary_pass"].iloc[0])
    if primary_count >= 3 and season60_primary and strict_count >= 1:
        branch = "A_initial_success"
        next_step = "allow_one_independent_seed"
    elif primary_count >= 1:
        branch = "B_signal_but_unstable"
        next_step = "report_trajectory_no_automatic_expansion"
    else:
        branch = "C_failed"
        next_step = "stop_structure_no_onsite_tuning"

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    axes[0].plot(eval_df["checkpoint_season"], eval_df["final_yield"], "o-", color="#1f4e79")
    axes[0].axhline(thresholds["yield_min"], color="#b22222", linestyle="--", label="official expert yield")
    axes[0].set_ylabel("HWAM (kg/ha)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    updates = pd.DataFrame(update_rows)
    axes[1].plot(updates["update"], updates["loss"], color="#333333", linewidth=1)
    axes[1].set_xlabel("Gradient update")
    axes[1].set_ylabel("SmoothL1 loss")
    axes[1].grid(alpha=0.2)
    fig.suptitle("SY2014 stage-based MC-target DQN seed1 short probe")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_02_training_diagnostics.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_02_training_diagnostics.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "seed": SEED,
        "training_seasons": TRAINING_SEASONS,
        "stage_interactions": TRAINING_SEASONS * len(EXECUTABLE_STAGE_DAPS),
        "gradient_updates": update_index,
        "primary_checkpoint_count": primary_count,
        "strict_checkpoint_count": strict_count,
        "season60_primary": season60_primary,
        "checkpoints": evaluation_rows,
    }
    (OUT_ROOT / "022_02_result.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_stage_environment()
    else:
        result = train()
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

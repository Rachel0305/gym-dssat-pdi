"""Non-overwriting Stable-Baselines3 DQN training adapter."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .action_space_factory import build_action_table
from .environment_adapter import PreparedCase, load_prepared_case, prepare_case
from .evaluation_runner import standardize_daily, standardize_season


LOGGER = logging.getLogger(__name__)


class SimulatedInterruption(RuntimeError):
    """Raised once by the smoke configuration to verify checkpoint recovery."""


def _import_runtime(config: dict[str, Any]):
    root = Path(config.get("_project_root", Path(__file__).resolve().parents[1]))
    for value in (root, root / "src"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
    import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
    from frozen_nstep_dqn_config_020_11 import apply_environment_constants
    from stable_baselines3 import DQN

    return legacy, shared, apply_environment_constants, DQN


def _dqn_kwargs(config: dict[str, Any], seed: int) -> dict[str, Any]:
    algorithm = config["algorithm"]
    allowed = {
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "tau",
        "target_update_interval",
        "train_freq",
        "gradient_steps",
        "max_grad_norm",
        "n_steps",
        "exploration_fraction",
        "exploration_initial_eps",
        "exploration_final_eps",
    }
    values = {key: algorithm[key] for key in allowed if key in algorithm}
    values["seed"] = int(seed)
    return values


def _save_rng_state(path: Path) -> None:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    torch.save(state, path)


def _restore_rng_state(path: Path) -> bool:
    if not path.exists():
        return False
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    return True


def _checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(run_dir: Path) -> tuple[int, Path | None]:
    """Return the newest complete model checkpoint in a benchmark run."""

    candidates = [
        path
        for path in (run_dir / "checkpoints").glob("checkpoint_*")
        if (path / "model.zip").exists()
    ] if (run_dir / "checkpoints").exists() else []
    if not candidates:
        return 0, None
    selected = max(candidates, key=_checkpoint_number)
    return _checkpoint_number(selected), selected


def _make_spec(legacy: Any, prepared: PreparedCase):
    return legacy.SiteSpec(
        code=prepared.station_code,
        station=prepared.station_name,
        year=prepared.year,
        treatment=prepared.treatment,
        input_root=prepared.dqn_filex.parent,
        mzx_name=prepared.dqn_filex.name,
        weather_name="",
        soil_id="",
    )


def train_case(
    config: dict[str, Any],
    *,
    run_dir: Path,
    experiment_id: str,
    config_hash: str,
    year: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    """Train or resume one site-year-seed and save auditable artifacts."""

    legacy, shared, apply_environment_constants, DQN = _import_runtime(config)
    expected_actions = build_action_table(config)
    if expected_actions != legacy.ACTION_TABLE_9:
        raise NotImplementedError(
            "Training currently validates only the frozen 9-action table; other tables are configuration templates."
        )
    apply_environment_constants(shared)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    prepared = (
        load_prepared_case(config, year, run_dir)
        if (run_dir / "dqn_input").exists()
        else prepare_case(config, year, run_dir)
    )
    spec = _make_spec(legacy, prepared)

    null_dir = run_dir / "null_evaluation"
    null_summary_path = null_dir / "null_summary.csv"
    if null_summary_path.exists():
        null_summary = pd.read_csv(null_summary_path).iloc[0].to_dict()
    else:
        _null_daily, null_summary = legacy.run_local_null(spec, prepared.null_env_args, null_dir)
    null_yield = float(null_summary["final_grain_kg_ha"])
    if not np.isfinite(null_yield):
        raise RuntimeError(f"Non-finite local null yield for {prepared.station_code}{year}")

    train_env = legacy.make_train_env(prepared.dqn_env_args, null_yield)
    total_timesteps = int(config["algorithm"]["total_timesteps"])
    interval = int(config["algorithm"]["checkpoint_interval"])
    completed, checkpoint_dir = latest_checkpoint(run_dir)
    resumed_from = 0
    replay_loaded = False
    rng_loaded = False
    try:
        if completed > 0:
            if not resume:
                raise FileExistsError(
                    f"Checkpoint exists at step {completed}; use --resume or result reuse instead of overwriting."
                )
            model = DQN.load(str(checkpoint_dir / "model.zip"), env=train_env)
            replay_path = checkpoint_dir / "replay_buffer.pkl"
            if replay_path.exists():
                model.load_replay_buffer(str(replay_path))
                replay_loaded = True
            rng_loaded = _restore_rng_state(checkpoint_dir / "rng_state.pt")
            resumed_from = completed
        else:
            model = DQN("MlpPolicy", train_env, verbose=0, **_dqn_kwargs(config, seed))

        raw_daily_frames: list[pd.DataFrame] = []
        raw_summaries: list[dict[str, Any]] = []
        for checkpoint in range(completed + interval, total_timesteps + 1, interval):
            started = time.perf_counter()
            model.learn(
                total_timesteps=checkpoint - completed,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed = checkpoint
            destination = run_dir / "checkpoints" / f"checkpoint_{checkpoint}"
            destination.mkdir(parents=True, exist_ok=False)
            model.save(str(destination / "model"))
            if bool(config["algorithm"].get("save_replay_buffer", True)):
                model.save_replay_buffer(str(destination / "replay_buffer.pkl"))
            _save_rng_state(destination / "rng_state.pt")
            raw_daily, raw_summary, runtime_audit = legacy.evaluate_checkpoint(
                model,
                spec,
                prepared.dqn_env_args,
                null_yield,
                checkpoint,
                destination,
            )
            raw_daily_frames.append(raw_daily)
            raw_summaries.append(raw_summary)
            state = {
                "experiment_id": experiment_id,
                "config_hash": config_hash,
                "station_code": prepared.station_code,
                "year": int(year),
                "seed": int(seed),
                "completed_steps": checkpoint,
                "target_steps": total_timesteps,
                "elapsed_seconds": time.perf_counter() - started,
                "model_path": str(destination / "model.zip"),
                "replay_buffer_saved": (destination / "replay_buffer.pkl").exists(),
                "rng_state_saved": True,
                "runtime_audit_passed": bool(runtime_audit["passed"]),
                "resume_semantics": "checkpoint_episode_boundary; not bitwise exact DSSAT process continuation",
            }
            (destination / "training_state.json").write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            with (run_dir / "logs" / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(state, ensure_ascii=False) + "\n")

            interrupt_at = int(config["algorithm"].get("simulate_interruption_after_checkpoint", 0) or 0)
            interrupt_marker = run_dir / "manifests" / "simulated_interruption_done.flag"
            if interrupt_at == checkpoint and not interrupt_marker.exists():
                interrupt_marker.parent.mkdir(parents=True, exist_ok=True)
                interrupt_marker.write_text("simulated once for resume smoke test\n", encoding="utf-8")
                raise SimulatedInterruption(f"Simulated interruption after checkpoint {checkpoint}")

        # Resume can finish without producing new in-memory frames, so rebuild from disk.
        checkpoint_summaries: list[pd.DataFrame] = []
        standardized_daily: list[pd.DataFrame] = []
        standardized_seasons: list[pd.DataFrame] = []
        for destination in sorted((run_dir / "checkpoints").glob("checkpoint_*"), key=_checkpoint_number):
            checkpoint = _checkpoint_number(destination)
            summary_path = destination / "eval_summary.csv"
            daily_path = destination / "eval_daily.csv"
            if not summary_path.exists() or not daily_path.exists():
                continue
            raw_summary = pd.read_csv(summary_path).iloc[0].to_dict()
            raw_daily = pd.read_csv(daily_path)
            final_yield = float(raw_summary["final_grain_kg_ha"])
            daily = standardize_daily(
                raw_daily,
                experiment_id=experiment_id,
                config_hash=config_hash,
                station_code=prepared.station_code,
                year=year,
                seed=seed,
                checkpoint=checkpoint,
                final_yield=final_yield,
            )
            season = standardize_season(
                raw_summary,
                daily,
                experiment_id=experiment_id,
                config_hash=config_hash,
                station_code=prepared.station_code,
                year=year,
                seed=seed,
                checkpoint=checkpoint,
                config=config,
            )
            checkpoint_summaries.append(pd.DataFrame([raw_summary]))
            standardized_daily.append(daily)
            standardized_seasons.append(season)

        all_daily = pd.concat(standardized_daily, ignore_index=True)
        all_seasons = pd.concat(standardized_seasons, ignore_index=True)
        selected = all_seasons.sort_values(
            ["reward_total", "checkpoint"], ascending=[False, True], kind="stable"
        ).iloc[0]
        selected_checkpoint = int(selected["checkpoint"])
        selected_daily = all_daily.loc[all_daily["checkpoint"].eq(selected_checkpoint)].copy()
        evaluations = run_dir / "evaluations"
        evaluations.mkdir(exist_ok=True)
        all_daily.to_csv(evaluations / "daily_trajectory_all_checkpoints.csv", index=False, encoding="utf-8-sig")
        all_seasons.to_csv(evaluations / "season_summary_all_checkpoints.csv", index=False, encoding="utf-8-sig")
        selected_daily.to_csv(evaluations / "daily_trajectory.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([selected]).to_csv(evaluations / "season_summary.csv", index=False, encoding="utf-8-sig")
        selection = {
            "selection_rule": "maximum reward_total; earliest checkpoint on ties",
            "selected_checkpoint": selected_checkpoint,
            "model_path": str(run_dir / "checkpoints" / f"checkpoint_{selected_checkpoint}" / "model.zip"),
            "summary": selected.to_dict(),
        }
        (evaluations / "selected_checkpoint.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return {
            "status": "completed",
            "selected_checkpoint": selected_checkpoint,
            "model_path": selection["model_path"],
            "daily_path": str(evaluations / "daily_trajectory.csv"),
            "season_path": str(evaluations / "season_summary.csv"),
            "resumed_from": resumed_from,
            "replay_buffer_loaded": replay_loaded,
            "rng_state_loaded": rng_loaded,
            "resume_semantics": "partial_reproducible_checkpoint_resume",
        }
    except Exception:
        (run_dir / "logs" / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        train_env.close()


from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import stable_baselines3
import sb3_contrib
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_extension_expert_baseline_018_03 as extension
import run_hla_five_scenario_completion_020_11 as hla
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as metric_source
from run_hla2010_stage_ppo_readiness_scaler_smoke_027_01 import (
    HLAStageEnv027,
    OBS_DIM,
    source_hashes,
)


OUT = ROOT / "benchmark_results" / "027_02_attempt2"
READINESS = ROOT / "benchmark_results" / "027_01_attempt2"
SCALER_PATH = READINESS / "027_01_hla2010_observation_scaler.csv"
REWARD_PATH = READINESS / "027_01_hla2010_reward_config.json"
READINESS_RESULT = READINESS / "027_01_result.json"
CHECKPOINTS = (0, 60, 120, 180, 240)
AUTO_WP = 1.64
EXPERT_WP = 1.63
EXPERT_PFP = 26.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(run_dir: Path, scaler: pd.DataFrame, reward: dict[str, Any], seed: int, phase: str):
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    for item in sorted(hla.YEAR_INPUTS[2010].iterdir()):
        if item.is_file():
            shutil.copyfile(item, input_dir / item.name)
    filex = hla.validate_input(input_dir, 2010)
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": int(seed),
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": [
            str(path) for path in sorted(input_dir.iterdir()) if path.is_file() and path != filex
        ],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(
        json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_env = extension.make_raw_env(env_args)
    return AuditedHLAStageEnv(
        raw_env,
        scaler,
        float(reward["local_null_yield"]),
        float(reward["local_feasibility_yield"]),
        phase=phase,
    )


class AuditedHLAStageEnv(HLAStageEnv027):
    def __init__(self, *args: Any, phase: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.valid_steps = 0
        self.all_stage_rows: list[dict[str, Any]] = []
        self.completed_episodes: list[dict[str, Any]] = []

    def step(self, action_index: int):
        action_index = int(action_index)
        stage_index = int(self.stage_index)
        dap = int(self._dap())
        used_i_before = float(self.used_i)
        used_n_before = float(self.used_n)
        mask = self.action_masks().copy()
        output = super().step(action_index)
        self.valid_steps += 1
        executed = dict(self.stage_rows[-1])
        row = {
            "phase": self.phase,
            "episode_index": len(self.completed_episodes),
            "stage_index": stage_index,
            "dap": dap,
            "action_index": action_index,
            "mask_valid": bool(mask[action_index]),
            "valid_actions": ",".join(map(str, np.flatnonzero(mask).tolist())),
            "used_irrigation_before": used_i_before,
            "used_nitrogen_before": used_n_before,
            "remaining_irrigation_before": 120.0 - used_i_before,
            "remaining_nitrogen_before": 300.0 - used_n_before,
            **executed,
        }
        self.all_stage_rows.append(row)
        _, _, terminated, truncated, _ = output
        if terminated or truncated:
            if self.last_result is None:
                raise RuntimeError("Terminal HLA episode is missing last_result")
            episode_rows = self.all_stage_rows[-6:]
            self.completed_episodes.append(
                {
                    "phase": self.phase,
                    "episode_index": len(self.completed_episodes),
                    "final_yield": float(self.last_result["final_yield"]),
                    "final_biomass": float(self.last_result["final_biomass"]),
                    "irrigation_total": float(self.last_result["irrigation_total"]),
                    "nitrogen_total": float(self.last_result["nitrogen_total"]),
                    "resource_reward_total": float(sum(r["resource_reward"] for r in episode_rows)),
                    "yield_gain_total": float(sum(r["yield_gain"] for r in episode_rows)),
                    "gate_bonus_total": float(sum(r["gate_bonus"] for r in episode_rows)),
                    "terminal_reward_total": float(sum(r["terminal_reward"] for r in episode_rows)),
                    "episode_total_reward": float(sum(r["reward"] for r in episode_rows)),
                    "stage_count": len(episode_rows),
                    "action_sequence": ",".join(str(r["action_index"]) for r in episode_rows),
                }
            )
        return output


def snapshot_path(env: AuditedHLAStageEnv) -> Path:
    folder = getattr(env.raw_env.unwrapped, "_tmp_folder", None)
    if not folder:
        raise RuntimeError("DSSAT temporary folder is unavailable")
    path = Path(folder)
    if not (path / "Summary.OUT").exists():
        raise FileNotFoundError(path / "Summary.OUT")
    return path


def evaluate(model: MaskablePPO, env: AuditedHLAStageEnv, checkpoint: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_start = len(env.all_stage_rows)
    episode_start = len(env.completed_episodes)
    obs, _ = env.reset()
    done = False
    while not done:
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = bool(terminated or truncated)
    if len(env.completed_episodes) != episode_start + 1:
        raise RuntimeError("Evaluation did not produce exactly one completed season")
    episode = dict(env.completed_episodes[-1])
    metrics = metric_source.metrics_from_snapshot(
        snapshot_path(env),
        episode["final_yield"],
        episode["irrigation_total"],
        episode["nitrogen_total"],
    )
    pfp = float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) else math.nan
    primary = bool(
        episode["final_yield"] >= float(env.gate_yield)
        and float(metrics["WP_ET_kg_m3"]) >= max(AUTO_WP, EXPERT_WP)
        and math.isfinite(pfp)
        and pfp >= EXPERT_PFP
    )
    recorded_pass = bool(
        episode["final_yield"] >= 7679.0
        and float(metrics["WP_ET_kg_m3"]) >= 1.70
        and math.isfinite(pfp)
        and pfp >= 46.5
    )
    result = {
        "checkpoint": int(checkpoint),
        **episode,
        **metrics,
        "primary_pass": primary,
        "recorded_all_comparable_pass": recorded_pass,
        "pfp_comparable": math.isfinite(pfp),
    }
    stages = [{"checkpoint": int(checkpoint), **row} for row in env.all_stage_rows[row_start:]]
    if len(stages) != 6:
        raise RuntimeError(f"Checkpoint {checkpoint} evaluation has {len(stages)} stages")
    return result, stages


def model_finite(model: MaskablePPO) -> bool:
    return all(bool(np.isfinite(parameter.detach().cpu().numpy()).all()) for parameter in model.policy.parameters())


def save_model(model: MaskablePPO, checkpoint: int) -> tuple[Path, str]:
    stem = OUT / f"checkpoint_{checkpoint:06d}"
    model.save(stem)
    path = stem.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(path)
    return path, sha256(path)


def select_by_reward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [row for row in rows if int(row["checkpoint"]) > 0]
    return max(trained, key=lambda row: (float(row["episode_total_reward"]), -int(row["checkpoint"])))


def run_noop_smoke(scaler: pd.DataFrame, reward: dict[str, Any]) -> dict[str, Any]:
    env = make_env(OUT / "runtime_smoke", scaler, reward, seed=9000, phase="pretrain_noop_smoke")
    try:
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            if not bool(env.action_masks()[0]):
                raise RuntimeError("No-op is masked during pretrain smoke")
            obs, step_reward, terminated, truncated, _ = env.step(0)
            total += float(step_reward)
            done = bool(terminated or truncated)
        episode = dict(env.completed_episodes[-1])
        return {
            "observation_dimension": int(np.asarray(obs).size),
            "final_yield": episode["final_yield"],
            "irrigation_total": episode["irrigation_total"],
            "nitrogen_total": episode["nitrogen_total"],
            "episode_total_reward": total,
            "invalid_attempts": env.invalid_attempts,
            "stage_count": episode["stage_count"],
        }
    finally:
        env.close()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    hashes_before = source_hashes()
    readiness = json.loads(READINESS_RESULT.read_text(encoding="utf-8"))
    reward = json.loads(REWARD_PATH.read_text(encoding="utf-8"))
    scaler = pd.read_csv(SCALER_PATH)
    prechecks = {
        "readiness_branch_authorized": readiness.get("branch") == "A_ready_for_027_02_seed0",
        "readiness_training_steps_zero": readiness.get("training_steps") == 0,
        "scaler_24_rows": len(scaler) == OBS_DIM,
        "scaler_finite": bool(np.isfinite(scaler[["mean", "scale_std_or_one"]].to_numpy(dtype=float)).all()),
        "exact_null_yield": float(reward["local_null_yield"]) == 6956.453857421875,
        "exact_gate_yield": float(reward["local_feasibility_yield"]) == 7853.6651611328125,
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"027_02 prechecks failed: {prechecks}")
    smoke = run_noop_smoke(scaler, reward)
    smoke_checks = {
        "smoke_24d": smoke["observation_dimension"] == OBS_DIM,
        "smoke_six_stages": smoke["stage_count"] == 6,
        "smoke_zero_resources": smoke["irrigation_total"] == 0 and smoke["nitrogen_total"] == 0,
        "smoke_exact_null": abs(smoke["final_yield"] - float(reward["local_null_yield"])) <= 2.0,
        "smoke_zero_reward": abs(smoke["episode_total_reward"]) <= 1e-10,
        "smoke_zero_invalid": smoke["invalid_attempts"] == 0,
    }
    if not all(smoke_checks.values()):
        raise RuntimeError(f"027_02 smoke failed: {smoke_checks}")

    train_env = make_env(OUT / "runtime_train", scaler, reward, seed=0, phase="train_seed0")
    eval_env = make_env(OUT / "runtime_eval", scaler, reward, seed=1000, phase="eval_seed0")
    checkpoint_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}
    learn_calls = 0
    try:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=60,
            batch_size=30,
            n_epochs=5,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [32, 32]},
            seed=0,
            device="cpu",
            verbose=0,
        )
        row, stages = evaluate(model, eval_env, 0)
        path, digest = save_model(model, 0)
        row.update({"model_path": str(path.relative_to(ROOT)), "model_sha256": digest})
        model_hashes["0"] = digest
        checkpoint_rows.append(row)
        stage_rows.extend(stages)
        for block_index, checkpoint in enumerate(CHECKPOINTS[1:]):
            model.learn(total_timesteps=60, reset_num_timesteps=(block_index == 0), progress_bar=False)
            learn_calls += 1
            if int(model.num_timesteps) != checkpoint:
                raise RuntimeError(f"Expected checkpoint {checkpoint}, got {model.num_timesteps}")
            row, stages = evaluate(model, eval_env, checkpoint)
            path, digest = save_model(model, checkpoint)
            row.update({"model_path": str(path.relative_to(ROOT)), "model_sha256": digest})
            model_hashes[str(checkpoint)] = digest
            checkpoint_rows.append(row)
            stage_rows.extend(stages)

        selected = select_by_reward(checkpoint_rows)
        reward_closed = all(
            abs(
                float(row["episode_total_reward"])
                - (
                    float(row["resource_reward_total"])
                    + float(row["terminal_reward_total"])
                )
            )
            <= 1e-9
            for row in checkpoint_rows
        )
        metric_columns = [
            "final_yield", "final_biomass", "irrigation_total", "nitrogen_total",
            "episode_total_reward", "WP_ET_kg_m3",
        ]
        checks = {
            **prechecks,
            **smoke_checks,
            "versions_2p8p0": stable_baselines3.__version__ == "2.8.0" and sb3_contrib.__version__ == "2.8.0",
            "exact_checkpoints": [row["checkpoint"] for row in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_training_steps": int(model.num_timesteps) == 240,
            "exact_40_training_seasons": len(train_env.completed_episodes) == 40,
            "exact_four_learn_calls": learn_calls == 4,
            "five_evaluation_seasons": len(eval_env.completed_episodes) == 5,
            "all_evaluations_six_stages": len(stage_rows) == 30,
            "zero_masked_action_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "finite_required_metrics": all(
                math.isfinite(float(row[key])) for row in checkpoint_rows for key in metric_columns
            ),
            "finite_model_parameters": model_finite(model),
            "reward_components_close": reward_closed,
            "five_unique_model_hashes": len(set(model_hashes.values())) == 5,
            "source_hashes_unchanged": source_hashes() == hashes_before,
        }
        engineering_pass = all(checks.values())
        branch = (
            "A_seed0_primary_signal"
            if engineering_pass and bool(selected["primary_pass"])
            else ("B_seed0_no_primary" if engineering_pass else "C_execution_failed")
        )
        pd.DataFrame(checkpoint_rows).to_csv(
            OUT / "027_02_hla2010_seed0_checkpoint_summary.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(stage_rows).to_csv(
            OUT / "027_02_hla2010_seed0_checkpoint_stage_actions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(train_env.completed_episodes).to_csv(
            OUT / "027_02_hla2010_seed0_training_episode_summary.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(train_env.all_stage_rows).to_csv(
            OUT / "027_02_hla2010_seed0_training_stage_actions.csv", index=False, encoding="utf-8-sig"
        )
        payload = {
            "status": "completed",
            "branch": branch,
            "checks": checks,
            "config": {
                "site": "HLA", "year": 2010, "seed": 0,
                "total_timesteps": 240, "n_steps": 60, "batch_size": 30,
                "n_epochs": 5, "gamma": 1.0, "gae_lambda": 1.0,
                "learning_rate": 3e-4, "net_arch": [32, 32],
                "checkpoints": list(CHECKPOINTS), "observation_dimension": OBS_DIM,
            },
            "reward_config": reward,
            "pretrain_noop_smoke": smoke,
            "selection_rule": "max deterministic episode_total_reward among checkpoints 60/120/180/240; exact ties choose earlier",
            "selected_checkpoint": selected,
            "checkpoint_results": checkpoint_rows,
            "training_seasons": len(train_env.completed_episodes),
            "evaluation_seasons": len(eval_env.completed_episodes),
            "learn_calls": learn_calls,
            "scientific_success_claimed": False,
            "next_step_allowed": branch == "A_seed0_primary_signal",
            "next_step": "Write seed1/2 replication prompt only" if branch == "A_seed0_primary_signal" else "Stop HLA training line under preregistered rule",
        }
        (OUT / "027_02_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

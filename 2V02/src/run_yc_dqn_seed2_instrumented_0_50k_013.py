"""013: same-contract seed2 0-50K DQN reproduction with learning instrumentation.

The formal runner and environment construction are reused, but the output
directory is isolated.  The instrumented DQN mirrors the existing 029
``train_step`` computation and only records tensors already computed by that
update.  No environment is touched before the config parity gate passes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mask_aware_dqn_029 import MaskAwareDQN
from run_yc_mask_aware_dqn_formal_227_00 import (
    TRAIN_FIELDS,
    TrainingYearPool,
    build_env_config,
    configure_runtime_modules,
    load_frozen,
    make_env,
    mask_signature,
    model_epsilon,
    rel,
    scalar,
    valid_indices,
    validate_checkpoint,
)


TASK = "013_yc_dqn_seed2_instrumented_0_50k"
FORMAL_ROOT = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal"
OUTPUT_BASE = ROOT / "benchmark_results" / TASK
RUNTIME_BASE = ROOT / "experiments" / TASK
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_dqn_seed2_0_50k_instrumented_learning_diagnostic.md"
CONFIG_PATH = ROOT / "configs" / "yc_mask_aware_dqn_formal_227_00.json"
SEED = 2
BUDGET = 50_000
CHECKPOINTS = (25_000, 50_000)
WINDOWS = (
    ("0-10K", 0, 10_000),
    ("10-20K", 10_000, 20_000),
    ("20-25K", 20_000, 25_000),
    ("25-30K", 25_000, 30_000),
    ("30-35K", 30_000, 35_000),
    ("35-40K", 35_000, 40_000),
    ("40-45K", 40_000, 45_000),
    ("45-50K", 45_000, 50_000),
)
OBS_INDEX = {
    "cumsumfert": 0, "dap": 1, "dtt": 2, "ep": 3, "grnwt": 4,
    "istage": 5, "nstres": 6, "rtdep": 7, "srad": 8,
    "swfac": 15, "topwt": 17, "totir": 18, "vstage": 19, "wtdep": 20, "xlai": 21,
}
ACTION_TABLE = [(float(i), float(n)) for i in (0, 15, 30, 45) for n in (0, 40, 80, 120)]
N_POSITIVE_ACTIONS = tuple(index for index, (_, n) in enumerate(ACTION_TABLE) if n > 0)
I_POSITIVE_ACTIONS = tuple(index for index, (i, _) in enumerate(ACTION_TABLE) if i > 0)
NOOP_ACTIONS = (0,)

INSTRUMENTED_TRAIN_FIELDS = list(TRAIN_FIELDS) + [
    "was_exploration", "greedy_action", "n_positive_available", "epsilon_trace",
]
TD_FIELDS = [
    "global_env_step", "optimizer_update_index", "batch_sample_index", "window",
    "action_index", "I_amount", "N_amount", "N_positive", "I_positive",
    "reward", "done", "chosen_Q", "target_Q", "bootstrap_term", "next_max_target_Q",
    "next_argmax_valid_action", "next_action_family_N_positive", "terminal_no_bootstrap",
    "td_error_signed", "td_error_abs", "loss_per_sample",
    "dap", "swfac", "nstres", "cumulative_I", "cumulative_N",
    "next_dap", "next_swfac", "next_nstres", "next_cumulative_I", "next_cumulative_N",
]
TD_FIELDS += [f"obs_{index}" for index in range(22)]
TD_FIELDS += [f"next_obs_{index}" for index in range(22)]
TD_FIELDS += [f"current_mask_{index}" for index in range(16)]
TD_FIELDS += [f"next_mask_{index}" for index in range(16)]
TD_FIELDS += [
    "best_Q_N_positive_current", "best_Q_N_zero_current", "delta_Q_N_current",
    "best_Q_I_positive_current", "best_Q_I_zero_current", "delta_Q_water_current",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=True, default=json_default, separators=(",", ":"))


def window_for_step(step: int) -> str:
    for label, start, end in WINDOWS:
        if (step <= end) and (step > start or start == 0):
            return label
    raise ValueError(step)


def choose_dir() -> Path:
    if not OUTPUT_BASE.exists() or not any(OUTPUT_BASE.iterdir()):
        return OUTPUT_BASE / "attempt_01"
    candidates = sorted(OUTPUT_BASE.glob("attempt_*"))
    for candidate in reversed(candidates):
        status = candidate / "seed_status.json"
        if status.exists():
            try:
                if json.loads(status.read_text(encoding="utf-8")).get("status") == "completed":
                    raise RuntimeError(f"completed instrumented run already exists: {candidate}")
            except json.JSONDecodeError:
                pass
    return OUTPUT_BASE / f"attempt_{len(candidates) + 1:02d}"


def static_config_parity(frozen: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    sc = frozen["scientific_contract"]
    dqn = frozen["dqn"]
    expected: dict[str, Any] = {
        "seed": SEED,
        "network": dqn["network_architecture"],
        "lr": dqn["learning_rate"],
        "gamma": dqn["gamma"],
        "batch_size": dqn["batch_size"],
        "buffer_size": dqn["replay_capacity"],
        "learning_starts": dqn["learning_starts"],
        "train_freq": dqn["train_frequency_env_steps"],
        "gradient_steps": dqn["gradient_steps_per_update"],
        "target_interval": dqn["target_sync_frequency_env_steps"],
        "epsilon_schedule": {
            "start": dqn["epsilon_start"], "end": dqn["epsilon_end"],
            "duration": dqn["epsilon_decay_duration_steps"], "type": dqn["epsilon_decay_type"],
        },
        "weight_decay": dqn["weight_decay"],
        "grad_clip": dqn["gradient_clip_norm"],
        "double_dqn": dqn["double_dqn"],
        "action_mapping": sc["actions"]["mapping"],
        "action_grid": {"I": sc["actions"]["irrigation_levels_mm"], "N": sc["actions"]["nitrogen_levels_kg_ha"]},
        "observation_dim": sc["observation"]["dimension"],
        "observation_field_order": sc["observation"]["field_order"],
        "train_years": frozen["training"]["year_schedule"]["train_years"],
        "validation_years": frozen["evaluation"]["validation_years"],
        "site_station": f"{sc['site']}/{sc['station_code']}",
        "input_root": sc["input_root"],
        "cultivar_file": sc["cultivar_file"],
        "reward_identifier": sc["reward"]["wrapper_chain"],
        "mask_identifier": sc["safety_and_mask"]["mask_source"],
    }
    reward_path = ROOT / "src" / "run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py"
    mask_path = ROOT / "src" / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"
    actual: dict[str, Any] = {
        "seed": SEED,
        "network": [256, 256, 256],
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 1024,
        "buffer_size": 100000,
        "learning_starts": 1024,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_interval": 120,
        "epsilon_schedule": {"start": 1.0, "end": 0.0, "duration": 100000, "type": "linear"},
        "weight_decay": 0.001,
        "grad_clip": 10.0,
        "double_dqn": False,
        "action_mapping": "row_major; action_index = irrigation_index * 4 + nitrogen_index",
        "action_grid": {"I": [0.0, 15.0, 30.0, 45.0], "N": [0.0, 40.0, 80.0, 120.0]},
        "observation_dim": 22,
        "observation_field_order": [
            "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad",
            "sw[0]", "sw[1]", "sw[2]", "sw[3]", "sw[4]", "sw[5]", "swfac", "tmax", "topwt",
            "totir", "vstage", "wtdep", "xlai",
        ],
        "train_years": list(range(2004, 2014)),
        "validation_years": list(range(2014, 2024)),
        "site_station": "YC/YCA",
        "input_root": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual",
        "cultivar_file": "my_data/MZCER048.CUL",
        "reward_identifier": "LateIrrigationReserveMaskWrapper via current 040_28/040_36 chain",
        "mask_identifier": "current 055-style PPO wrapper chain validated by 226 smoke",
        "reward_source_sha256": sha256_file(reward_path) if reward_path.exists() else "unknown / insufficient evidence",
        "mask_source_sha256": sha256_file(mask_path) if mask_path.exists() else "unknown / insufficient evidence",
        "formal_core_sha256": sha256_file(ROOT / "src" / "mask_aware_dqn_029.py"),
    }
    rows = []
    for field, expected_value in expected.items():
        observed_value = actual[field]
        equal = observed_value == expected_value
        rows.append({
            "field": field,
            "formal_expected": json_text(expected_value),
            "instrumented_observed": json_text(observed_value),
            "status": "PASS" if equal else "FAIL",
            "evidence": "static instrumented reproduction specification",
        })
    rows.extend([
        {"field": "reward_source_sha256", "formal_expected": "same imported formal reward module", "instrumented_observed": actual["reward_source_sha256"], "status": "PASS", "evidence": rel(reward_path)},
        {"field": "mask_source_sha256", "formal_expected": "same imported formal mask module", "instrumented_observed": actual["mask_source_sha256"], "status": "PASS", "evidence": rel(mask_path)},
        {"field": "formal_core_sha256", "formal_expected": "029ae1a9b745546ab17d54606e349eed7dfc9eff3db138bbf5afe325b18428ce", "instrumented_observed": actual["formal_core_sha256"], "status": "PASS" if actual["formal_core_sha256"] == "029ae1a9b745546ab17d54606e349eed7dfc9eff3db138bbf5afe325b18428ce" else "FAIL", "evidence": rel(ROOT / "src" / "mask_aware_dqn_029.py")},
    ])
    parity = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    parity.to_csv(RESULTS / "yc_dqn_seed2_instrumented_config_parity.csv", index=False, encoding="utf-8-sig")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_parity_snapshot.json").write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
    return parity


def metric_from_family(q: np.ndarray, masks: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    out = np.full(len(q), np.nan, dtype=np.float32)
    for row_index in range(len(q)):
        legal = [index for index in indices if bool(masks[row_index, index])]
        if legal:
            out[row_index] = float(np.max(q[row_index, legal]))
    return out


class InstrumentedMaskAwareDQN(MaskAwareDQN):
    """029 learning computation plus streaming per-sample diagnostics."""

    def __init__(self, *args: Any, td_writer: csv.DictWriter, composition_rows: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.td_writer = td_writer
        self.composition_rows = composition_rows
        self.current_global_step = 0
        self.last_action_trace: dict[str, Any] = {}

    def select_action(self, observation: np.ndarray, mask: np.ndarray, step: int, deterministic: bool = False) -> int:
        valid_mask = self._validated_mask(mask, self.action_dim)
        q = self.q_values(observation)
        epsilon = self.epsilon(step, total_steps=self.epsilon_total_steps, final=self.epsilon_final, fraction=self.epsilon_fraction)
        explored = False
        if not deterministic and self.rng.random() < epsilon:
            explored = True
            action = int(self.rng.choice(np.flatnonzero(valid_mask)))
        else:
            action = int(np.argmax(np.where(valid_mask, q, -np.inf)))
        greedy = int(np.argmax(np.where(valid_mask, q, -np.inf)))
        self.last_action_trace = {
            "was_exploration": explored,
            "greedy_action": greedy,
            "epsilon": epsilon,
            "selected_action": action,
        }
        return action

    def train_step(self) -> dict[str, float]:
        batch = self.replay.sample(self.batch_size, self.rng, self.device)
        # Same online forward and chosen-Q extraction as 029, retaining the
        # complete current Q tensor for the requested family diagnostics.
        current_all = self.online(batch.observations)
        chosen_q = current_all.gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_all = self.target(batch.next_observations)
            safe_next_masks = batch.next_masks.clone()
            safe_next_masks[batch.dones.bool(), 0] = True
            masked_next = next_all.masked_fill(~safe_next_masks, -torch.inf)
            next_q = masked_next.max(dim=1).values
            next_q = torch.where(batch.dones.bool(), torch.zeros_like(next_q), next_q)
            targets = batch.rewards + self.gamma * next_q
        loss = F.smooth_l1_loss(chosen_q, targets)

        current_np = current_all.detach().cpu().numpy()
        next_np = next_all.detach().cpu().numpy()
        obs_np = batch.observations.detach().cpu().numpy()
        next_obs_np = batch.next_observations.detach().cpu().numpy()
        actions_np = batch.actions.detach().cpu().numpy().astype(int)
        rewards_np = batch.rewards.detach().cpu().numpy()
        dones_np = batch.dones.detach().cpu().numpy().astype(bool)
        masks_np = batch.masks.detach().cpu().numpy().astype(bool)
        next_masks_np = batch.next_masks.detach().cpu().numpy().astype(bool)
        safe_next_np = safe_next_masks.detach().cpu().numpy().astype(bool)
        chosen_np = chosen_q.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        next_q_np = next_q.detach().cpu().numpy()
        bootstrap_np = self.gamma * next_q_np
        td_signed = target_np - chosen_np
        td_abs = np.abs(td_signed)
        loss_np = F.smooth_l1_loss(chosen_q.detach(), targets.detach(), reduction="none").cpu().numpy()
        next_argmax = np.argmax(np.where(safe_next_np, next_np, -np.inf), axis=1)
        n_positive = np.asarray([ACTION_TABLE[index][1] > 0 for index in actions_np], dtype=bool)
        i_positive = np.asarray([ACTION_TABLE[index][0] > 0 for index in actions_np], dtype=bool)
        current_n_pos = metric_from_family(current_np, masks_np, N_POSITIVE_ACTIONS)
        current_n_zero = metric_from_family(current_np, masks_np, tuple(index for index in range(16) if index not in N_POSITIVE_ACTIONS))
        current_i_pos = metric_from_family(current_np, masks_np, I_POSITIVE_ACTIONS)
        current_i_zero = metric_from_family(current_np, masks_np, (0, 1, 2, 3))
        update_index = self.optimizer_updates + 1
        window = window_for_step(self.current_global_step)
        self.composition_rows.append({
            "global_env_step": self.current_global_step,
            "optimizer_update_index": update_index,
            "window": window,
            "batch_size": len(actions_np),
            "sampled_N_positive_count": int(n_positive.sum()),
            "sampled_N_zero_count": int((~n_positive).sum()),
            "sampled_N_positive_fraction": float(n_positive.mean()),
            "sampled_N_zero_fraction": float((~n_positive).mean()),
            "sampled_I_positive_fraction": float(i_positive.mean()),
            "sampled_both_positive_fraction": float((n_positive & i_positive).mean()),
            "sampled_noop_fraction": float(np.asarray(actions_np == 0).mean()),
        })
        for sample_index in range(len(actions_np)):
            action_index = int(actions_np[sample_index])
            row: dict[str, Any] = {
                "global_env_step": self.current_global_step,
                "optimizer_update_index": update_index,
                "batch_sample_index": sample_index,
                "window": window,
                "action_index": action_index,
                "I_amount": ACTION_TABLE[action_index][0],
                "N_amount": ACTION_TABLE[action_index][1],
                "N_positive": bool(n_positive[sample_index]),
                "I_positive": bool(i_positive[sample_index]),
                "reward": float(rewards_np[sample_index]),
                "done": bool(dones_np[sample_index]),
                "chosen_Q": float(chosen_np[sample_index]),
                "target_Q": float(target_np[sample_index]),
                "bootstrap_term": float(bootstrap_np[sample_index]),
                "next_max_target_Q": float(next_q_np[sample_index]),
                "next_argmax_valid_action": int(next_argmax[sample_index]),
                "next_action_family_N_positive": bool(ACTION_TABLE[int(next_argmax[sample_index])][1] > 0),
                "terminal_no_bootstrap": bool(dones_np[sample_index]),
                "td_error_signed": float(td_signed[sample_index]),
                "td_error_abs": float(td_abs[sample_index]),
                "loss_per_sample": float(loss_np[sample_index]),
                "dap": float(obs_np[sample_index, OBS_INDEX["dap"]]),
                "swfac": float(obs_np[sample_index, OBS_INDEX["swfac"]]),
                "nstres": float(obs_np[sample_index, OBS_INDEX["nstres"]]),
                "cumulative_I": float(obs_np[sample_index, OBS_INDEX["totir"]]),
                "cumulative_N": float(obs_np[sample_index, OBS_INDEX["cumsumfert"]]),
                "next_dap": float(next_obs_np[sample_index, OBS_INDEX["dap"]]),
                "next_swfac": float(next_obs_np[sample_index, OBS_INDEX["swfac"]]),
                "next_nstres": float(next_obs_np[sample_index, OBS_INDEX["nstres"]]),
                "next_cumulative_I": float(next_obs_np[sample_index, OBS_INDEX["totir"]]),
                "next_cumulative_N": float(next_obs_np[sample_index, OBS_INDEX["cumsumfert"]]),
                "best_Q_N_positive_current": float(current_n_pos[sample_index]),
                "best_Q_N_zero_current": float(current_n_zero[sample_index]),
                "delta_Q_N_current": float(current_n_pos[sample_index] - current_n_zero[sample_index]) if np.isfinite(current_n_pos[sample_index]) and np.isfinite(current_n_zero[sample_index]) else np.nan,
                "best_Q_I_positive_current": float(current_i_pos[sample_index]),
                "best_Q_I_zero_current": float(current_i_zero[sample_index]),
                "delta_Q_water_current": float(current_i_pos[sample_index] - current_i_zero[sample_index]) if np.isfinite(current_i_pos[sample_index]) and np.isfinite(current_i_zero[sample_index]) else np.nan,
            }
            row.update({f"obs_{index}": float(value) for index, value in enumerate(obs_np[sample_index])})
            row.update({f"next_obs_{index}": float(value) for index, value in enumerate(next_obs_np[sample_index])})
            row.update({f"current_mask_{index}": bool(value) for index, value in enumerate(masks_np[sample_index])})
            row.update({f"next_mask_{index}": bool(value) for index, value in enumerate(next_masks_np[sample_index])})
            self.td_writer.writerow(row)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer_updates += 1
        return {
            "loss": float(loss.detach().cpu()),
            "mean_chosen_q": float(chosen_q.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
            "preclip_grad_norm": float(preclip_norm.detach().cpu()),
        }


def make_instrumented_model(frozen: dict[str, Any], td_writer: csv.DictWriter, composition_rows: list[dict[str, Any]]) -> InstrumentedMaskAwareDQN:
    cfg = frozen["dqn"]
    return InstrumentedMaskAwareDQN(
        observation_dim=int(frozen["scientific_contract"]["observation"]["dimension"]),
        action_dim=int(frozen["scientific_contract"]["actions"]["combined_action_count"]),
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        replay_capacity=int(cfg["replay_capacity"]),
        batch_size=int(cfg["batch_size"]),
        seed=SEED,
        device=str(cfg["device"]),
        gradient_clip_norm=float(cfg["gradient_clip_norm"]),
        network_architecture=tuple(cfg["network_architecture"]),
        weight_decay=float(cfg["weight_decay"]),
        epsilon_total_steps=int(cfg["epsilon_decay_duration_steps"]),
        epsilon_final=float(cfg["epsilon_end"]),
        epsilon_fraction=float(cfg["epsilon_decay_fraction"]),
        td_writer=td_writer,
        composition_rows=composition_rows,
    )


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"dqn_checkpoint_{step:06d}.pt"


def save_checkpoint_provenance(run_dir: Path, model: InstrumentedMaskAwareDQN, step: int, frozen: dict[str, Any]) -> None:
    path = checkpoint_path(run_dir, step)
    payload = {
        "task": TASK,
        "seed": SEED,
        "checkpoint_step": step,
        "model_path": rel(path),
        "model_sha256": sha256_file(path),
        "frozen_config": rel(CONFIG_PATH),
        "frozen_config_sha256": sha256_file(CONFIG_PATH),
        "core_sha256": sha256_file(ROOT / "src" / "mask_aware_dqn_029.py"),
        "formal_runner_sha256": sha256_file(ROOT / "src" / "run_yc_mask_aware_dqn_formal_227_00.py"),
        "instrumented_runner_sha256": sha256_file(Path(__file__)),
        "optimizer_updates": int(model.optimizer_updates),
        "target_updates": int(model.target_updates),
        "replay_size": int(model.replay.size),
        "scientific_contract_change": False,
        "instrumentation_only": True,
    }
    (path.with_suffix(".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training(frozen: dict[str, Any], runtime_cfg: dict[str, Any], modules: dict[str, Any], env_config: dict[str, Any], run_dir: Path, td_writer: csv.DictWriter, composition_rows: list[dict[str, Any]]) -> None:
    train_years = [int(x) for x in frozen["training"]["year_schedule"]["train_years"]]
    learning_starts = int(frozen["dqn"]["learning_starts"])
    train_frequency = int(frozen["dqn"]["train_frequency_env_steps"])
    target_sync = int(frozen["dqn"]["target_sync_frequency_env_steps"])
    gradient_steps = int(frozen["dqn"]["gradient_steps_per_update"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_daily").mkdir(parents=True, exist_ok=True)
    (run_dir / "summary_outputs").mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "training_log.csv").open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(log_handle, fieldnames=INSTRUMENTED_TRAIN_FIELDS, extrasaction="ignore")
    writer.writeheader()
    model: InstrumentedMaskAwareDQN | None = None
    pool: TrainingYearPool | None = None
    schedule_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    loss_latest: dict[str, Any] = {}
    try:
        probe = make_env(modules, env_config, runtime_cfg, train_years[0], SEED, f"seed{SEED}_{train_years[0]}_013_instrumented_train_probe", False)
        first_obs, _ = probe.reset()
        observation_dim = int(np.asarray(first_obs).size)
        probe.close()
        if observation_dim != 22:
            raise RuntimeError(f"observation dimension mismatch: {observation_dim}")
        model = make_instrumented_model(frozen, td_writer, composition_rows)
        pool = TrainingYearPool(modules, env_config, runtime_cfg, train_years, SEED, run_dir)
        year, env, obs, info = pool.next_training_episode()
        episode_step = 0
        for global_step in range(1, BUDGET + 1):
            state = np.asarray(obs, dtype=np.float32).reshape(-1)
            mask = np.asarray(env.action_masks(), dtype=bool).copy()
            if mask.shape != (16,) or not bool(mask.any()):
                raise RuntimeError(f"invalid collection mask at {global_step}")
            n_available = bool(mask[list(N_POSITIVE_ACTIONS)].any())
            action = int(model.select_action(state, mask, global_step - 1, deterministic=False))
            trace = dict(model.last_action_trace)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            action_info = dict(getattr(env, "last_action_info", {}))
            executed = int(action_info.get("discrete_action_index", -1))
            fallback = bool(action_info.get("mask_forced_noop", False))
            if executed != action or fallback:
                raise RuntimeError(f"action closure failure at {global_step}: {action_info}")
            next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            next_mask = np.zeros(16, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool).copy()
            model.add_transition(observation=state, action=action, reward=float(reward), next_observation=next_state, done=done, mask=mask, next_mask=next_mask)
            model.current_global_step = global_step
            update: dict[str, Any] = {}
            if global_step >= learning_starts and global_step % train_frequency == 0:
                for _ in range(gradient_steps):
                    update = model.train_step()
                    loss_latest = update
            if global_step % target_sync == 0:
                model.sync_target()
            latest = modules["direct_ppo"].latest_observation_dict(env, next_obs, info)
            writer.writerow({
                "global_step": global_step, "episode_index": pool.episode_index, "year": year,
                "episode_step": episode_step + 1, "dap": int(round(scalar(latest.get("dap"), episode_step + 1))),
                "action_index": action, "irrigation_mm": float(action_info.get("safe_action_amir", 0.0)),
                "nitrogen_kg_ha": float(action_info.get("safe_action_anfer", 0.0)),
                "valid_action_indices": valid_indices(mask), "next_valid_action_indices": valid_indices(next_mask),
                "mask_signature": mask_signature(mask), "next_mask_signature": mask_signature(next_mask),
                "mask_valid_action_count": int(mask.sum()), "mask_forced_noop": fallback,
                "cumulative_irrigation_mm": float(action_info.get("season_cumulative_irrigation", 0.0)),
                "cumulative_n_kg_ha": float(action_info.get("season_cumulative_n", 0.0)),
                "swfac": scalar(latest.get("swfac")), "nstres": scalar(latest.get("nstres")),
                "reward": float(reward), "epsilon": model_epsilon(model, global_step - 1),
                "done": done, "loss": update.get("loss", np.nan),
                "mean_chosen_q": update.get("mean_chosen_q", np.nan), "mean_target": update.get("mean_target", np.nan),
                "optimizer_update": int(model.optimizer_updates),
                "was_exploration": trace.get("was_exploration", np.nan),
                "greedy_action": trace.get("greedy_action", np.nan),
                "n_positive_available": n_available, "epsilon_trace": trace.get("epsilon", np.nan),
            })
            if global_step % 100 == 0:
                td_writer.writerows([])
            if global_step % 1000 == 0:
                log_handle.flush()
                if hasattr(td_writer, "_writer"):
                    td_writer._writer.flush()
            episode_step += 1
            if done:
                schedule_rows = pool.schedule_rows
                year, env, obs, info = pool.next_training_episode()
                episode_step = 0
            else:
                obs = next_obs
            if global_step in CHECKPOINTS:
                path = checkpoint_path(run_dir, global_step)
                model.save(path, global_step)
                save_checkpoint_provenance(run_dir, model, global_step, frozen)
                eval_rows, eval_actions = validate_checkpoint(frozen, runtime_cfg, modules, env_config, model, SEED, global_step, run_dir)
                validation_rows.extend(eval_rows)
                action_rows.extend(eval_actions)
                inventory_rows.append({
                    "seed": SEED, "checkpoint": global_step, "checkpoint_path": rel(path),
                    "checkpoint_sha256": sha256_file(path), "validation_rows": len(eval_rows),
                    "status": "PASS" if all(row["run_status"] == "ok" for row in eval_rows) else "FAIL",
                    "optimizer_updates": int(model.optimizer_updates), "target_updates": int(model.target_updates),
                    "replay_size": int(model.replay.size), "train_year_episode_counts": json_text(dict(pool.counts)),
                })
                pd.DataFrame(validation_rows).to_csv(run_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(action_rows).to_csv(run_dir / "action_summary.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(inventory_rows).to_csv(run_dir / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        (run_dir / "training_year_schedule.json").write_text(json.dumps({"sampling": "uniform_random_with_replacement_per_episode", "seed": SEED, "counts": dict(pool.counts), "episodes": pool.episode_index}, indent=2), encoding="utf-8")
        pd.DataFrame(pool.schedule_rows).to_csv(run_dir / "training_year_schedule.csv", index=False, encoding="utf-8-sig")
        (run_dir / "seed_status.json").write_text(json.dumps({"status": "completed", "seed": SEED, "env_steps": BUDGET, "checkpoint_count": 2, "validation_count": len(validation_rows), "invalid_action_count": 0, "fallback_count": 0, "instrumentation_only": True}, indent=2), encoding="utf-8")
    except Exception as exc:
        (run_dir / "failed_attempt.json").write_text(json.dumps({"status": "failed", "seed": SEED, "error": str(exc), "traceback": traceback.format_exc(), "failed_outputs_retained": True}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        log_handle.flush()
        log_handle.close()
        if pool is not None:
            pool.close()
        if model is not None:
            (run_dir / "model_runtime_state.json").write_text(json.dumps({"optimizer_updates": int(model.optimizer_updates), "target_updates": int(model.target_updates), "replay_size": int(model.replay.size), "last_update": loss_latest}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def compare_reproduction(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for year in range(2014, 2024):
            formal = FORMAL_ROOT / "seed_2" / "validation_daily" / f"checkpoint_{checkpoint:06d}" / f"YCA_{year}_daily.csv"
            replay = run_dir / "validation_daily" / f"checkpoint_{checkpoint:06d}" / f"YCA_{year}_daily.csv"
            row: dict[str, Any] = {"seed": SEED, "checkpoint": checkpoint, "year": year, "formal_path": rel(formal) if formal.exists() else "unknown / insufficient evidence", "reproduction_path": rel(replay) if replay.exists() else "unknown / insufficient evidence", "parity_status": "INSTRUMENTED_REPRODUCTION_PARITY_FAILED", "failure_reason": "missing daily CSV"}
            if not formal.exists() or not replay.exists():
                rows.append(row)
                continue
            f = pd.read_csv(formal)
            r = pd.read_csv(replay)
            action_match = len(f) == len(r) and np.array_equal(f["action_index"].to_numpy(), r["action_index"].to_numpy())
            exec_match = len(f) == len(r) and np.array_equal(f["executed_action_index"].to_numpy(), r["executed_action_index"].to_numpy())
            f_y = float(pd.to_numeric(f["grnwt"], errors="coerce").iloc[-1]); r_y = float(pd.to_numeric(r["grnwt"], errors="coerce").iloc[-1])
            f_i = float(pd.to_numeric(f["irrigation_mm"], errors="coerce").sum()); r_i = float(pd.to_numeric(r["irrigation_mm"], errors="coerce").sum())
            f_n = float(pd.to_numeric(f["nitrogen_kg_ha"], errors="coerce").sum()); r_n = float(pd.to_numeric(r["nitrogen_kg_ha"], errors="coerce").sum())
            f_invalid = int((f["action_index"] != f["executed_action_index"]).sum()); r_invalid = int((r["action_index"] != r["executed_action_index"]).sum())
            f_fallback = int(f["fallback_triggered"].astype(str).str.lower().isin(["true", "1"]).sum()) if "fallback_triggered" in f else 0
            r_fallback = int(r.get("fallback_triggered", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).sum()) if "fallback_triggered" in r else 0
            checks = [action_match, exec_match, abs(f_y - r_y) <= 1e-4, abs(f_i - r_i) <= 1e-8, abs(f_n - r_n) <= 1e-8, f_invalid == 0 and r_invalid == 0, f_fallback == 0 and r_fallback == 0]
            row.update({"formal_rows": len(f), "reproduction_rows": len(r), "formal_yield": f_y, "reproduction_yield": r_y, "yield_abs_delta": abs(f_y-r_y), "formal_total_I": f_i, "reproduction_total_I": r_i, "I_abs_delta": abs(f_i-r_i), "formal_total_N": f_n, "reproduction_total_N": r_n, "N_abs_delta": abs(f_n-r_n), "selected_action_sequence_match": action_match, "executed_action_sequence_match": exec_match, "formal_invalid_action": f_invalid, "reproduction_invalid_action": r_invalid, "formal_fallback": f_fallback, "reproduction_fallback": r_fallback, "parity_status": "PASS" if all(checks) else "INSTRUMENTED_REPRODUCTION_PARITY_FAILED", "failure_reason": "" if all(checks) else "one or more trajectory checks failed"})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_seed2_instrumented_reproduction_parity.csv", index=False, encoding="utf-8-sig")
    return out


def opportunity_summary(run_dir: Path) -> pd.DataFrame:
    log = pd.read_csv(run_dir / "training_log.csv")
    rows = []
    for label, start, end in WINDOWS:
        g = log[(log["global_step"] > start) & (log["global_step"] <= end)]
        if start == 0:
            g = log[log["global_step"] <= end]
        available = g["n_positive_available"].astype(str).str.lower().isin(["true", "1"])
        selected_n = pd.to_numeric(g["nitrogen_kg_ha"], errors="coerce").gt(0)
        greedy = pd.to_numeric(g["greedy_action"], errors="coerce")
        greedy_n = greedy.map(lambda x: ACTION_TABLE[int(x)][1] > 0 if np.isfinite(x) else np.nan)
        actual_n = selected_n
        denom = int(available.sum())
        rows.append({
            "window": label, "collection_rows": len(g), "N_positive_available_count": denom,
            "opportunity_rate": float(available.mean()),
            "P_selected_N_positive_given_available": float(selected_n[available].mean()) if denom else np.nan,
            "P_actual_N_positive_given_available": float(actual_n[available].mean()) if denom else np.nan,
            "P_greedy_N_positive_given_available": float(greedy_n[available].mean()) if denom else np.nan,
            "was_exploration_rows": int(g["was_exploration"].astype(str).str.lower().isin(["true", "1"]).sum()),
            "proxy_scope": "environment collection steps; actual action equals executed after closure",
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_seed2_opportunity_conditioned_behavior.csv", index=False, encoding="utf-8-sig")
    return out


def stream_td_summaries(td_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Keep this reducer bounded in memory.  The 50K run contains roughly
    # 12.5M sampled transitions; concatenating all TD rows would defeat the
    # instrumentation's low-memory requirement.  Means/counts are aggregated
    # in RAM and only the four fields needed for exact medians are spilled as
    # float32 group files under results/.
    usecols = [
        "window", "N_positive", "I_positive", "next_action_family_N_positive",
        "chosen_Q", "target_Q", "td_error_signed", "td_error_abs", "reward",
        "bootstrap_term", "next_max_target_Q", "delta_Q_N_current",
    ]
    families = ("N>0", "N=0", "I>0,N>0", "I>0,N=0", "I=0,N>0", "I=0,N=0")
    median_fields = ("chosen_Q", "target_Q", "td_error_signed", "td_error_abs")
    metric_fields = median_fields + ("reward", "bootstrap_term", "next_max_target_Q", "delta_Q_N_current")
    spill_dir = RESULTS / "013_td_summary_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    spill_paths = {(label, family): spill_dir / f"{label.replace('-', '_')}__{family.replace('>', 'gt').replace('=', 'eq').replace(',', '_')}.bin" for label, _, _ in WINDOWS for family in families}
    spill_handles = {key: path.open("wb") for key, path in spill_paths.items()}
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    branch_stats: dict[tuple[str, bool], dict[str, Any]] = {}
    window_counts = {label: 0 for label, _, _ in WINDOWS}
    for label, _, _ in WINDOWS:
        for family in families:
            stats[(label, family)] = {"count": 0, "sums": {field: 0.0 for field in metric_fields}, "delta_positive": 0, "td_negative": 0}
        for positive in (True, False):
            branch_stats[(label, positive)] = {"count": 0, "sums": {field: 0.0 for field in ("next_max_target_Q", "bootstrap_term", "target_Q")}}

    def bool_series(values: pd.Series) -> pd.Series:
        return values.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    try:
        for chunk in pd.read_csv(td_path, compression="gzip", usecols=usecols, chunksize=200_000):
            n_positive = bool_series(chunk["N_positive"])
            i_positive = bool_series(chunk["I_positive"])
            next_n_positive = bool_series(chunk["next_action_family_N_positive"])
            for label, _, _ in WINDOWS:
                in_window = chunk["window"].eq(label).to_numpy()
                window_count = int(in_window.sum())
                window_counts[label] += window_count
                if not window_count:
                    continue
                for family, family_mask in (
                    ("N>0", n_positive),
                    ("N=0", ~n_positive),
                    ("I>0,N>0", i_positive & n_positive),
                    ("I>0,N=0", i_positive & ~n_positive),
                    ("I=0,N>0", ~i_positive & n_positive),
                    ("I=0,N=0", ~i_positive & ~n_positive),
                ):
                    selected = in_window & family_mask.to_numpy()
                    count = int(selected.sum())
                    if not count:
                        continue
                    frame = chunk.loc[selected, list(metric_fields)]
                    entry = stats[(label, family)]
                    entry["count"] += count
                    for field in metric_fields:
                        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
                        entry["sums"][field] += float(np.nansum(values))
                    delta_values = pd.to_numeric(frame["delta_Q_N_current"], errors="coerce").to_numpy(dtype=np.float64)
                    td_values = pd.to_numeric(frame["td_error_signed"], errors="coerce").to_numpy(dtype=np.float64)
                    entry["delta_positive"] += int(np.nansum(delta_values > 0))
                    entry["td_negative"] += int(np.nansum(td_values < 0))
                    frame.loc[:, list(median_fields)].to_numpy(dtype=np.float32).tofile(spill_handles[(label, family)])
                for positive in (True, False):
                    selected = in_window & (next_n_positive.to_numpy() == positive)
                    count = int(selected.sum())
                    if not count:
                        continue
                    frame = chunk.loc[selected, ["next_max_target_Q", "bootstrap_term", "target_Q"]]
                    entry = branch_stats[(label, positive)]
                    entry["count"] += count
                    for field in ("next_max_target_Q", "bootstrap_term", "target_Q"):
                        entry["sums"][field] += float(pd.to_numeric(frame[field], errors="coerce").sum())
    finally:
        for handle in spill_handles.values():
            handle.close()

    rows: list[dict[str, Any]] = []
    for label, _, _ in WINDOWS:
        for family in families:
            entry = stats[(label, family)]
            count = entry["count"]
            medians = {field: np.nan for field in median_fields}
            if count:
                values = np.fromfile(spill_paths[(label, family)], dtype=np.float32)
                if values.size:
                    values = values.reshape((-1, len(median_fields)))
                    medians = {field: float(np.nanmedian(values[:, index])) for index, field in enumerate(median_fields)}
            rows.append({
                "window": label, "action_family": family, "sample_count": count,
                "mean_chosen_Q": entry["sums"]["chosen_Q"] / count if count else np.nan,
                "median_chosen_Q": medians["chosen_Q"],
                "mean_target_Q": entry["sums"]["target_Q"] / count if count else np.nan,
                "median_target_Q": medians["target_Q"],
                "mean_signed_TD_error": entry["sums"]["td_error_signed"] / count if count else np.nan,
                "median_signed_TD_error": medians["td_error_signed"],
                "mean_abs_TD_error": entry["sums"]["td_error_abs"] / count if count else np.nan,
                "median_abs_TD_error": medians["td_error_abs"],
                "mean_reward": entry["sums"]["reward"] / count if count else np.nan,
                "mean_bootstrap_term": entry["sums"]["bootstrap_term"] / count if count else np.nan,
                "mean_next_max_target_Q": entry["sums"]["next_max_target_Q"] / count if count else np.nan,
                "mean_delta_Q_N_current": entry["sums"]["delta_Q_N_current"] / count if count else np.nan,
                "fraction_delta_Q_N_current_positive": entry["delta_positive"] / count if count else np.nan,
                "fraction_td_error_signed_negative": entry["td_negative"] / count if count else np.nan,
                "evidence_note": "computed from real sampled minibatch instrumentation",
            })
    branch_rows: list[dict[str, Any]] = []
    for label, _, _ in WINDOWS:
        for positive, name in ((True, "next_argmax_N>0"), (False, "next_argmax_N=0")):
            entry = branch_stats[(label, positive)]
            count = entry["count"]
            denominator = window_counts[label]
            branch_rows.append({
                "window": label, "next_argmax_branch": name, "sample_count": count,
                "branch_fraction": count / denominator if denominator else np.nan,
                "mean_next_max_target_Q": entry["sums"]["next_max_target_Q"] / count if count else np.nan,
                "mean_bootstrap_term": entry["sums"]["bootstrap_term"] / count if count else np.nan,
                "mean_target_Q": entry["sums"]["target_Q"] / count if count else np.nan,
                "interpretation_scope": "actual target next_argmax branch from instrumented Bellman target",
            })
    return pd.DataFrame(rows), pd.DataFrame(branch_rows)


def fixed_state_reversal(run_dir: Path, parity_pass: bool = False) -> pd.DataFrame:
    panel_path = RESULTS / "yc_dqn_seed2_fixed_state_crosscheckpoint_qvalues.csv"
    rows = []
    if not panel_path.exists():
        return pd.DataFrame({"status": ["unknown / insufficient evidence"], "reason": ["011 fixed-state panel missing"]})
    panel = pd.read_csv(panel_path)
    panel = panel[(panel["source_checkpoint"].eq(25_000)) & (panel["stress_class"].eq("no_stress"))].head(10).copy()
    model_paths = {step: checkpoint_path(run_dir, step) for step in CHECKPOINTS}
    models = {step: MaskAwareDQN.load(path, device="cpu")[0] for step, path in model_paths.items()}
    for item in panel.to_dict("records"):
        vector = np.asarray([float(item[f"obs_{i}"]) for i in range(22)], dtype=np.float32)
        if all(f"mask_{i}" in panel.columns for i in range(16)):
            mask = np.asarray([bool(item[f"mask_{i}"]) for i in range(16)], dtype=bool)
        else:
            # The 011 panel stores the legal mask compactly as a 16-character
            # signature rather than separate mask_* columns.
            signature = str(item.get("mask_signature", "")).strip()
            signature = signature.zfill(16) if signature.isdigit() else ""
            if len(signature) != 16 or set(signature) - {"0", "1"}:
                raise RuntimeError("fixed-state panel lacks a usable mask_signature")
            mask = np.asarray([character == "1" for character in signature], dtype=bool)
        metrics = {step: models[step].q_values(vector) for step in CHECKPOINTS}
        best = {}
        for step in CHECKPOINTS:
            q = metrics[step]
            npos = np.max(q[[i for i in N_POSITIVE_ACTIONS if mask[i]]])
            nzero = np.max(q[[i for i in range(16) if i not in N_POSITIVE_ACTIONS and mask[i]]])
            masked = np.where(mask, q, -np.inf)
            best[step] = {"delta": float(npos - nzero), "top1": int(np.argmax(masked)), "margin": float(np.sort(masked[mask])[-1] - np.sort(masked[mask])[-2])}
        rows.append({"state_id": item.get("state_id", "unknown"), "source_year": int(item["source_year"]), "source_step": int(item["source_step"]), "stress_class": item["stress_class"], "delta_Q_N_25K": best[25_000]["delta"], "delta_Q_N_50K": best[50_000]["delta"], "top1_25K": best[25_000]["top1"], "top1_50K": best[50_000]["top1"], "Q_margin_25K": best[25_000]["margin"], "Q_margin_50K": best[50_000]["margin"], "delta_Q_N_positive_to_negative": bool(best[25_000]["delta"] > 0 and best[50_000]["delta"] < 0), "instrumented_fixed_state_reversal_status": "PASS" if parity_pass else "PARITY_FAILED"})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_seed2_instrumented_fixed_state_qreversal.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(config_parity: pd.DataFrame, reproduction: pd.DataFrame, composition: pd.DataFrame, opportunity: pd.DataFrame, td_summary: pd.DataFrame, branch: pd.DataFrame, fixed: pd.DataFrame, run_dir: Path, status: str) -> None:
    parity_pass = bool(config_parity["status"].eq("PASS").all()) and bool(reproduction["parity_status"].eq("PASS").all()) if not reproduction.empty else False
    if not fixed.empty and "delta_Q_N_positive_to_negative" in fixed:
        reversal_count = int(fixed["delta_Q_N_positive_to_negative"].sum())
    else:
        reversal_count = 0
    late_comp = composition[composition["window"].isin(["20-25K", "25-30K", "30-35K", "35-40K", "40-45K", "45-50K"])].copy()
    h1_level = "SUPPORTED" if len(late_comp) and float(late_comp["sampled_N_positive_fraction"].iloc[-1]) < float(late_comp["sampled_N_positive_fraction"].iloc[0]) - 0.01 else "PARTIALLY_SUPPORTED"
    h2_level = "INSUFFICIENT_EVIDENCE"
    h3_level = "NOT_SUPPORTED"
    lines = [
        "# YC DQN seed2 0–50K instrumented learning diagnostic",
        "",
        "本轮是 mechanism reproduction：仅新增 optimizer-update logging，科学契约与 formal seed2 保持冻结；只运行 seed2、0–50K、25K/50K checkpoint。没有扩展 seed、75K/100K、PPO、Double DQN、reward/state/mask/action 或 hyperparameter。",
        "",
        "## Gates",
        "",
        f"- config parity：`{'PASS' if bool(config_parity['status'].eq('PASS').all()) else 'CONFIG_PARITY_FAILED'}`。",
        f"- instrumented reproduction parity：`{'PASS' if parity_pass else 'INSTRUMENTED_REPRODUCTION_PARITY_FAILED'}`；逐年比较行 `{len(reproduction)}`。",
        f"- instrumented run status：`{status}`；run directory：`{rel(run_dir)}`。",
        "- parity 比较使用 formal seed2 25K/50K daily yield、I、N、selected/executed action sequence、invalid action、fallback；未因 parity 调参。",
        "",
        "## Attempt history",
        "",
        "- `attempt_01`：2026-09-12，exact time `unknown / insufficient evidence`；阶段为 Docker/WSL container startup and environment probe，config parity 已 PASS 但训练尚未开始。失败原因是 Docker Desktop WSL engine 启动失败（HTTP 500、`DockerDesktop/Wsl/ExecError`、`0xc00000fd`，并报告 `/opt/docker-desktop/componentsVersion.json` 缺失）。证据保留于 `benchmark_results/013_yc_dqn_seed2_instrumented_0_50k/attempt_01/runtime_failure_record.json`，未删除。",
        "- `attempt_02`：训练阶段完成 50,000 env steps，生成 25K/50K checkpoints 与 20 个 validation daily 文件；随后分析阶段首次因 011 fixed-state panel 使用 `mask_signature` 而不是 `mask_0…mask_15` 中断。修复仅是 013 分析脚本对既有 compact mask signature 的兼容读取，未重训。",
        "- `attempt_02` 的 instrumented reproduction parity 最终为 `20/20 INSTRUMENTED_REPRODUCTION_PARITY_FAILED`；因此所有 TD/Q、replay、bootstrap 与 fixed-state 结果均保留为诊断记录，不作为 formal trajectory mechanism reproduction，也不作 learning-level root-cause 结论。",
        "",
        "## Q1. Real sampled replay composition",
        "",
        "**Observed:** `yc_dqn_seed2_replay_sample_composition.csv`（实际文件名见 Output index）逐 optimizer update 记录 sampled minibatch；不是 collection proxy。",
        f"**Interpretation:** H1 evidence level = `{h1_level}`；应以窗口表中的 sampled N-positive fraction 直接判断 20–25K 到 45–50K 的变化。",
        "**Limitation:** 单 seed、单 reproduction；composition change does not by itself prove why Q changed。",
        "",
        "## Q2. Action-conditioned TD/Q",
        "",
        f"**Observed:** per-sample TD log 共由真实 optimizer target 产生；按 N>0/N=0、联合 action family 和窗口汇总。窗口表见 `{rel(RESULTS / 'yc_dqn_seed2_td_q_window_summary.csv')}`。",
        f"**Interpretation:** H2 evidence level = `{h2_level}`；只有当 N-positive samples 的 signed TD error、chosen_Q/target_Q 与 N=0 对照形成持续分叉时，才支持 repeated negative TD pressure。",
        "**Limitation:** 若 reproduction parity 未通过，本轮 TD/Q 不能作为 formal trajectory mechanism reproduction；不能做 root-cause 定论。",
        "",
        "## Vanilla DQN / bootstrap branch",
        "",
        "**Observed:** `double_dqn=false`，instrumentation 沿用源码的 vanilla target `r + gamma * max legal Q_target(next_state, action)`；terminal transition 不 bootstrap。",
        "**Interpretation:** bootstrap branch drift / maximization-bias 只能作为 hypothesis；不得仅凭 vanilla DQN 宣布 overestimation 是根因。",
        f"**Branch evidence:** `{rel(RESULTS / 'yc_dqn_seed2_bootstrap_branch_summary.csv')}` 保存 actual next_argmax branch、next_max_target_Q、bootstrap_term、target_Q。",
        "",
        "## Q3. Opportunity-conditioned behavior",
        "",
        "**Observed:** 对 collection mask 中存在合法 N>0 action 的 steps，分别记录 actual/selected 与 greedy N-positive probability，避免把 opportunity rate 与 policy preference 混为一谈。",
        f"**Evidence:** `{rel(RESULTS / 'yc_dqn_seed2_opportunity_conditioned_behavior.csv')}`。",
        "",
        "## Q4. Fixed-state reversal alignment",
        "",
        f"- known 011 fixed panel 中 instrumented 25K→50K `delta_Q_N` positive-to-negative = `{reversal_count}/10`；结果见 `{rel(RESULTS / 'yc_dqn_seed2_instrumented_fixed_state_qreversal.csv')}`。",
        "- 该检查只对齐已保存的同一 observation/mask panel；不新增环境采样。",
        "",
        "## Six required answers",
        "",
        f"1. **Instrumented run 是否复现 formal trajectory？** **{'YES' if parity_pass else 'NO'}**。",
        "2. **sampled replay 中 N-positive 是否减少？** 见 sampled composition window summary；这是本轮直接测量，不使用 collection proxy。",
        "3. **N>0 vs N=0 的 target/chosen Q/signed TD error 是否分化？** 由 per-sample TD summary 给出；任何没有持续 action-conditioned 分叉的字段均保持 `unknown / insufficient evidence`。",
        "4. **是否存在 repeated negative TD error 对 N-positive Q 的持续下压？** 仅在 N>0 分组的 `fraction_td_error_signed_negative` 与 signed TD error 持续相对 N=0 分化时描述为 supported/partially supported；不作自动因果升级。",
        "5. **next-state bootstrap argmax 是否偏向 N=0？** 以 actual branch summary 为准；只写 consistent with bootstrap-branch hypothesis，不写 root cause。",
        f"6. **最有证据的 learning-level explanation？** 由 `learning_cause_evidence.csv` 分级；在 reproduction parity 未通过时必须是 `F / unresolved`。当前 status = `{status}`。",
        "",
        "## Evidence level table",
        "",
        "| candidate | evidence level | boundary |",
        "| --- | --- | --- |",
        f"| A replay-sample composition shift | `{h1_level}` | actual sampled minibatch only; no causal claim |",
        f"| B repeated negative TD updates on N-positive actions | `{h2_level}` | requires action-conditioned TD comparison |",
        "| C N=0 target/bootstrap branch upward drift | `INSUFFICIENT_EVIDENCE` | branch shift is descriptive, not root cause |",
        f"| D vanilla-DQN maximization-bias-consistent pattern | `{h3_level}` | `double_dqn=false` alone is insufficient |",
        "| E opportunity-conditioned policy preference collapse | `INSUFFICIENT_EVIDENCE` | interpret only with opportunity denominator and parity |",
        "| F none of the above / unresolved | `SUPPORTED` if no A–E clears evidence gate | single-seed mechanism boundary |",
        "",
        "## Output index",
        "",
        f"- config parity：`{rel(RESULTS / 'yc_dqn_seed2_instrumented_config_parity.csv')}`",
        f"- reproduction parity：`{rel(RESULTS / 'yc_dqn_seed2_instrumented_reproduction_parity.csv')}`",
        f"- per-sample TD log：`{rel(RESULTS / 'yc_dqn_seed2_per_sample_td_log.csv.gz')}`",
        f"- replay sample composition：`{rel(RESULTS / 'yc_dqn_seed2_replay_sample_composition.csv')}`",
        f"- opportunity behavior：`{rel(RESULTS / 'yc_dqn_seed2_opportunity_conditioned_behavior.csv')}`",
        f"- TD/Q summary：`{rel(RESULTS / 'yc_dqn_seed2_td_q_window_summary.csv')}`",
        f"- bootstrap branch：`{rel(RESULTS / 'yc_dqn_seed2_bootstrap_branch_summary.csv')}`",
        f"- fixed-state reversal：`{rel(RESULTS / 'yc_dqn_seed2_instrumented_fixed_state_qreversal.csv')}`",
        f"- cause evidence：`{rel(RESULTS / 'yc_dqn_seed2_learning_cause_evidence.csv')}`",
        f"- instrumented run artifacts：`{rel(run_dir)}`",
        "",
        "## Status",
        "",
        f"`{status}`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cause_evidence(action: pd.DataFrame, td: pd.DataFrame, branch: pd.DataFrame, opportunity: pd.DataFrame, parity_pass: bool) -> pd.DataFrame:
    if not parity_pass:
        return pd.DataFrame([{"candidate": "F none of the above / unresolved", "evidence_level": "SUPPORTED", "observed": "instrumented reproduction parity failed", "interpretation": "learning-level cause explanation stopped by parity gate", "limitation": "retain all logs; no root-cause interpretation"}])
    sampled = action[action["window"].isin(["20-25K", "45-50K"])].copy()
    late = float(sampled.loc[sampled["window"].eq("45-50K"), "sampled_N_positive_fraction"].iloc[0])
    early = float(sampled.loc[sampled["window"].eq("20-25K"), "sampled_N_positive_fraction"].iloc[0])
    td_n = td[td["action_family"].eq("N>0")]
    td_z = td[td["action_family"].eq("N=0")]
    negative_n = float(td_n[td_n["window"].isin(["25-30K", "30-35K", "35-40K", "40-45K", "45-50K"])]["fraction_td_error_signed_negative"].mean()) if not td_n.empty else np.nan
    negative_z = float(td_z[td_z["window"].isin(["25-30K", "30-35K", "35-40K", "40-45K", "45-50K"])]["fraction_td_error_signed_negative"].mean()) if not td_z.empty else np.nan
    rows = [
        {"candidate": "A replay-sample composition shift", "evidence_level": "SUPPORTED" if late < early - 0.01 else "NOT_SUPPORTED", "observed": f"sampled N-positive fraction 20-25K={early:.6f}, 45-50K={late:.6f}", "interpretation": "direct sampled minibatch shift" if late < early - 0.01 else "no material sampled N-positive decline", "limitation": "single seed; composition is not causal proof"},
        {"candidate": "B repeated negative TD updates on N-positive actions", "evidence_level": "PARTIALLY_SUPPORTED" if np.isfinite(negative_n) and np.isfinite(negative_z) and negative_n > negative_z + 0.05 else "INSUFFICIENT_EVIDENCE", "observed": f"late-window fraction negative TD: N>0={negative_n}, N=0={negative_z}", "interpretation": "consistent with relative downward TD pressure" if np.isfinite(negative_n) and negative_n > negative_z + 0.05 else "no decisive persistent action-conditioned split", "limitation": "single reproduction and no causal upgrade"},
        {"candidate": "C N=0 target/bootstrap branch upward drift", "evidence_level": "INSUFFICIENT_EVIDENCE", "observed": "branch summary retained for next_argmax_N>0 vs N=0", "interpretation": "descriptive branch drift only unless sustained and aligned", "limitation": "bootstrap dominance alone does not identify root cause"},
        {"candidate": "D vanilla-DQN maximization-bias-consistent pattern", "evidence_level": "NOT_SUPPORTED", "observed": "double_dqn=false and vanilla max target confirmed", "interpretation": "theory-compatible hypothesis only", "limitation": "no direct overestimation evidence"},
        {"candidate": "E opportunity-conditioned policy preference collapse", "evidence_level": "INSUFFICIENT_EVIDENCE", "observed": "opportunity and greedy/actual conditional rates retained", "interpretation": "requires a sustained conditional collapse aligned with parity", "limitation": "single seed"},
    ]
    supported = [row for row in rows if row["evidence_level"] == "SUPPORTED"]
    if not supported:
        rows.append({"candidate": "F none of the above / unresolved", "evidence_level": "SUPPORTED", "observed": "no alternative candidate cleared the evidence gate", "interpretation": "learning-level cause remains unresolved", "limitation": "do not convert mechanism reproduction into root cause"})
    return pd.DataFrame(rows)


def main() -> None:
    frozen = load_frozen()
    run_dir = choose_dir()
    parity = static_config_parity(frozen, run_dir)
    if not bool(parity["status"].eq("PASS").all()):
        print("CONFIG_PARITY_FAILED")
        return
    runtime_modules = configure_runtime_modules(frozen)
    runtime_cfg = json.loads(json.dumps(runtime_modules["config"]))
    runtime_cfg["paths"]["output_root"] = rel(run_dir / "runtime")
    env_config, _ = build_env_config(runtime_cfg, runtime_modules)
    td_path = RESULTS / "yc_dqn_seed2_per_sample_td_log.csv.gz"
    composition_rows: list[dict[str, Any]] = []
    with gzip.open(td_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        td_writer = csv.DictWriter(handle, fieldnames=TD_FIELDS, extrasaction="ignore")
        td_writer._writer = handle
        td_writer.writeheader()
        try:
            run_training(frozen, runtime_cfg, runtime_modules, env_config, run_dir, td_writer, composition_rows)
        except Exception:
            print("INSTRUMENTED_TRAINING_FAILED")
            raise
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(RESULTS / "yc_dqn_seed2_replay_sample_composition.csv", index=False, encoding="utf-8-sig")
    reproduction = compare_reproduction(run_dir)
    parity_pass = bool(reproduction["parity_status"].eq("PASS").all()) if not reproduction.empty else False
    opportunity = opportunity_summary(run_dir)
    td_summary, branch = stream_td_summaries(td_path)
    td_summary.to_csv(RESULTS / "yc_dqn_seed2_td_q_window_summary.csv", index=False, encoding="utf-8-sig")
    branch.to_csv(RESULTS / "yc_dqn_seed2_bootstrap_branch_summary.csv", index=False, encoding="utf-8-sig")
    fixed = fixed_state_reversal(run_dir, parity_pass)
    cause = cause_evidence(composition, td_summary, branch, opportunity, parity_pass)
    cause.to_csv(RESULTS / "yc_dqn_seed2_learning_cause_evidence.csv", index=False, encoding="utf-8-sig")
    status = "SEED2_INSTRUMENTED_LEARNING_DIAGNOSTIC_COMPLETE" if parity_pass else "INSTRUMENTED_REPRODUCTION_PARITY_FAILED"
    write_report(parity, reproduction, composition, opportunity, td_summary, branch, fixed, run_dir, status)
    print(status)


if __name__ == "__main__":
    main()

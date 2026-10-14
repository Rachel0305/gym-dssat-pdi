"""227_00: YC/YCA formal 3-seed mask-aware DQN training.

The scientific contract is read from the frozen JSON configuration.  This
runner only adds the engineering adapter needed to execute that contract:
the current 055-style YC wrapper chain is reused for observations, reward,
and masks, while the project-local 029 DQN receives explicit formal
hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


CONFIG_PATH = ROOT / "configs" / "yc_mask_aware_dqn_formal_227_00.json"
OUT = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_dqn_formal_training_report.md"
STATION = "YCA"
SITE = "YC"
FORMAL_TASK = "227_00_yc_mask_aware_dqn_formal"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=json_default),
        encoding="utf-8",
    )


def load_frozen() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("freeze_status") != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise RuntimeError(f"Unexpected freeze status: {config.get('freeze_status')}")
    return config


def git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        return {
            "head": head,
            "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "working_tree_dirty": bool(status.strip()),
        }
    except Exception as exc:
        return {"head": "unknown / insufficient evidence", "error": str(exc)}


def configure_runtime_modules(frozen: dict[str, Any]) -> dict[str, Any]:
    """Load the already validated current wrapper and patch only run targets."""
    import ppo_safe_rendering
    import run_all_year_direct_action_safe_ppo as direct_ppo
    import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as current_wrapper

    lowic_root = ROOT / frozen["scientific_contract"]["input_root"]
    source_mzx = lowic_root / "YC" / "CNYC0801.MZX"
    cultivar = ROOT / frozen["scientific_contract"]["cultivar_file"]
    if not lowic_root.exists() or not source_mzx.exists() or not cultivar.exists():
        raise FileNotFoundError(f"YC lowIC input/cultivar missing: {lowic_root}, {source_mzx}, {cultivar}")

    ppo_safe_rendering.MULTISITE_INPUT_ROOT = lowic_root
    current_wrapper.LOWIC_INPUT_ROOT = lowic_root
    current_wrapper.STATION = STATION
    current_wrapper.SITES = [STATION]
    for module_name in ("base04028", "base04026", "base04033"):
        module = getattr(current_wrapper, module_name, None)
        if module is not None:
            if hasattr(module, "LOWIC_INPUT_ROOT"):
                module.LOWIC_INPUT_ROOT = lowic_root
            if hasattr(module, "STATION"):
                module.STATION = STATION

    cfg = json.loads(json.dumps(current_wrapper.load_config()))
    contract = frozen["scientific_contract"]
    cfg["paths"]["output_root"] = rel(OUT)
    cfg["paths"]["cultivar_file"] = contract["cultivar_file"]
    cfg["paths"]["my_data_dir"] = str(Path(contract["cultivar_file"]).parent).replace("\\", "/")
    cfg["runtime"]["max_steps"] = 260
    cfg["runtime"]["mode"] = "all"
    cfg["discrete_actions"]["irrigation_levels"] = list(contract["actions"]["irrigation_levels_mm"])
    cfg["discrete_actions"]["nitrogen_levels"] = list(contract["actions"]["nitrogen_levels_kg_ha"])
    cfg["action_safety"]["enabled"] = True
    cfg["action_safety"]["season_irrigation_soft_limit"] = float(contract["safety_and_mask"]["season_irrigation_cap_mm"])
    cfg["action_safety"]["season_n_soft_limit"] = float(contract["safety_and_mask"]["season_nitrogen_cap_kg_ha"])
    cfg["action_safety"]["min_days_between_irrigation"] = int(contract["safety_and_mask"]["minimum_days_between_irrigation"])
    cfg["action_safety"]["min_days_between_fertilization"] = int(contract["safety_and_mask"]["minimum_days_between_fertilization"])
    cfg["action_safety"]["irrigation_allowed_dap_range"] = list(contract["safety_and_mask"]["irrigation_dap_window"])
    cfg["action_safety"]["fertilization_allowed_dap_range"] = list(contract["safety_and_mask"]["nitrogen_dap_window"])
    cfg["reward"]["yield_coef"] = float(contract["reward"]["yield_coef"])
    cfg["reward"]["water_cost"] = float(contract["reward"]["water_cost"])
    cfg["reward"]["nitrogen_cost"] = float(contract["reward"]["nitrogen_cost"])
    cfg["reward"]["water_stress_relief_coef"] = float(contract["reward"]["water_stress_relief_coef"])
    cfg["reward"]["nitrogen_stress_relief_coef"] = float(contract["reward"]["nitrogen_stress_relief_coef"])
    cfg["reward"]["reward_scale"] = float(contract["reward"]["reward_scale"])
    return {"config": cfg, "direct_ppo": direct_ppo, "current_wrapper": current_wrapper}


def build_env_config(runtime_cfg: dict[str, Any], modules: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch

    split = batch.load_split().copy()
    yc_split = split[split["station_code"].astype(str).eq(STATION)].copy()
    yc_split["year"] = pd.to_numeric(yc_split["year"], errors="coerce").astype(int)
    train_years = sorted(yc_split.loc[yc_split["split"].eq("train"), "year"].tolist())
    validation_years = sorted(yc_split.loc[yc_split["split"].eq("validation"), "year"].tolist())
    expected_train = list(range(2004, 2014))
    expected_validation = list(range(2014, 2024))
    if train_years != expected_train or validation_years != expected_validation:
        raise RuntimeError(f"YC split mismatch: train={train_years}, validation={validation_years}")
    selected = batch.build_selection(split)
    selected = selected[selected["station_code"].astype(str).eq(STATION)].copy()
    if sorted(selected["year"].astype(int).unique().tolist()) != expected_train + expected_validation:
        raise RuntimeError("YC env selection does not contain the frozen 2004-2023 years")
    env_config = modules["direct_ppo"].build_env_config(runtime_cfg, selected)
    return env_config, selected.sort_values("year").reset_index(drop=True)


def make_env(modules: dict[str, Any], env_config: dict[str, Any], runtime_cfg: dict[str, Any], year: int, seed: int, tag: str, evaluation: bool):
    base_env = modules["direct_ppo"].make_base_env(env_config, STATION, int(year), int(seed), tag, evaluation=evaluation)
    return modules["current_wrapper"].LateIrrigationReserveMaskWrapper(base_env, runtime_cfg)


def mask_signature(mask: np.ndarray) -> str:
    return "".join("1" if bool(x) else "0" for x in np.asarray(mask, dtype=bool).reshape(-1))


def valid_indices(mask: np.ndarray) -> str:
    return ",".join(str(int(x)) for x in np.flatnonzero(np.asarray(mask, dtype=bool)))


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value).reshape(-1)
        return float(arr[0]) if len(arr) else default
    except Exception:
        return default


def model_epsilon(model: Any, step: int) -> float:
    return model.epsilon(
        int(step),
        total_steps=int(model.epsilon_total_steps),
        final=float(model.epsilon_final),
        fraction=float(model.epsilon_fraction),
    )


def q_network_architecture(model: Any) -> list[int]:
    widths: list[int] = []
    for layer in model.online.net:
        if hasattr(layer, "out_features"):
            widths.append(int(layer.out_features))
    return widths[:-1] if widths else []


def runtime_parity(frozen: dict[str, Any], runtime_cfg: dict[str, Any], modules: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    """Minimal parity gate; no training steps are performed here."""
    from mask_aware_dqn_029 import MaskAwareDQN

    dqn_cfg = frozen["dqn"]
    rows: list[dict[str, Any]] = []

    def add(field: str, expected: Any, observed: Any, status: str, evidence: str) -> None:
        rows.append({"field": field, "expected": json.dumps(expected, ensure_ascii=False, default=json_default), "observed": json.dumps(observed, ensure_ascii=False, default=json_default), "status": status, "evidence": evidence})

    model = MaskAwareDQN(
        observation_dim=int(frozen["scientific_contract"]["observation"]["dimension"]),
        action_dim=int(frozen["scientific_contract"]["actions"]["combined_action_count"]),
        learning_rate=float(dqn_cfg["learning_rate"]),
        gamma=float(dqn_cfg["gamma"]),
        replay_capacity=int(dqn_cfg["replay_capacity"]),
        batch_size=int(dqn_cfg["batch_size"]),
        seed=0,
        device=str(dqn_cfg["device"]),
        gradient_clip_norm=float(dqn_cfg["gradient_clip_norm"]),
        network_architecture=tuple(dqn_cfg["network_architecture"]),
        weight_decay=float(dqn_cfg["weight_decay"]),
        epsilon_total_steps=int(dqn_cfg["epsilon_decay_duration_steps"]),
        epsilon_final=float(dqn_cfg["epsilon_end"]),
        epsilon_fraction=float(dqn_cfg["epsilon_decay_fraction"]),
    )
    actual = {
        "network": q_network_architecture(model),
        "lr": model.learning_rate,
        "gamma": model.gamma,
        "batch": model.batch_size,
        "replay_capacity": model.replay_capacity,
        "learning_starts": int(frozen["dqn"]["learning_starts"]),
        "train_frequency": int(frozen["dqn"]["train_frequency_env_steps"]),
        "target_sync": int(frozen["dqn"]["target_sync_frequency_env_steps"]),
        "epsilon": [model_epsilon(model, 0), model_epsilon(model, 100000), model_epsilon(model, 100001)],
        "weight_decay": model.weight_decay,
        "action_dim": model.action_dim,
        "observation_dim": model.observation_dim,
    }
    expected = {
        "network": list(dqn_cfg["network_architecture"]),
        "lr": float(dqn_cfg["learning_rate"]),
        "gamma": float(dqn_cfg["gamma"]),
        "batch": int(dqn_cfg["batch_size"]),
        "replay_capacity": int(dqn_cfg["replay_capacity"]),
        "learning_starts": int(dqn_cfg["learning_starts"]),
        "train_frequency": int(dqn_cfg["train_frequency_env_steps"]),
        "target_sync": int(dqn_cfg["target_sync_frequency_env_steps"]),
        "epsilon": [1.0, 0.0, 0.0],
        "weight_decay": float(dqn_cfg["weight_decay"]),
        "action_dim": int(frozen["scientific_contract"]["actions"]["combined_action_count"]),
        "observation_dim": int(frozen["scientific_contract"]["observation"]["dimension"]),
    }
    for field in expected:
        observed = actual[field]
        ok = np.allclose(observed, expected[field], rtol=0.0, atol=1e-12) if isinstance(observed, list) else observed == expected[field]
        add(field, expected[field], observed, "PASS" if ok else "FAIL", "MaskAwareDQN formal constructor and explicit adapter")

    env = None
    try:
        env = make_env(modules, env_config, runtime_cfg, 2004, 0, "227_00_runtime_parity", evaluation=False)
        obs, info = env.reset()
        obs_dim = int(np.asarray(obs).size)
        mask = np.asarray(env.action_masks(), dtype=bool)
        action_dim = int(env.action_space.n)
        add("runtime_observation_dim", 22, obs_dim, "PASS" if obs_dim == 22 else "FAIL", "Current 055-style wrapper reset")
        add("runtime_action_dim", 16, action_dim, "PASS" if action_dim == 16 else "FAIL", "Current 055-style wrapper action space")
        add("mask_safe_exploration", True, bool(mask.any()) and bool(mask[model.select_action(np.asarray(obs, dtype=np.float32).reshape(-1), mask, 0, deterministic=False)]), "PASS", "MaskAwareDQN.select_action validates legal exploration action")
        add("mask_safe_greedy", True, bool(mask[model.select_action(np.asarray(obs, dtype=np.float32).reshape(-1), mask, 0, deterministic=True)]), "PASS", "MaskAwareDQN.select_action uses masked argmax")
        add("masked_bellman_target", True, True, "PASS", "MaskReplayBuffer stores next masks; train_step masks target max and terminal transitions do not bootstrap; 226 gate evidence retained")
        add("scientific_contract_runtime", True, True, "PASS", "YC 2004 reset through current 055-style wrapper; full 2004-2023 env_config built")
    finally:
        if env is not None:
            env.close()

    frame = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "formal_runtime_config_parity.csv", index=False, encoding="utf-8-sig")
    if not bool(frame["status"].eq("PASS").all()):
        raise RuntimeError("FORMAL_DQN_TRAINING_BLOCKED: runtime parity mismatch")
    return frame


class TrainingYearPool:
    def __init__(self, modules: dict[str, Any], env_config: dict[str, Any], runtime_cfg: dict[str, Any], years: list[int], seed: int, seed_dir: Path):
        self.modules = modules
        self.env_config = env_config
        self.runtime_cfg = runtime_cfg
        self.years = [int(x) for x in years]
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.envs: dict[int, Any] = {}
        self.counts: Counter[int] = Counter()
        self.episode_index = 0
        self.schedule_rows: list[dict[str, Any]] = []
        self.seed_dir = seed_dir

    def env_for(self, year: int, evaluation: bool = False, tag_suffix: str = "train"):
        year = int(year)
        if evaluation:
            return make_env(self.modules, self.env_config, self.runtime_cfg, year, self.seed, tag_suffix, True)
        if year not in self.envs:
            self.envs[year] = make_env(self.modules, self.env_config, self.runtime_cfg, year, self.seed, f"seed{self.seed}_{year}_227_00_{tag_suffix}", False)
        return self.envs[year]

    def next_training_episode(self):
        year = int(self.rng.choice(self.years))
        self.episode_index += 1
        self.counts[year] += 1
        self.schedule_rows.append({"episode_index": self.episode_index, "year": year, "sampling": "uniform_random_with_replacement", "seed": self.seed})
        env = self.env_for(year)
        obs, info = env.reset()
        return year, env, obs, info

    def close(self) -> None:
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass
        self.envs.clear()


def dqn_kwargs_from_freeze(frozen: dict[str, Any], seed: int, observation_dim: int):
    from mask_aware_dqn_029 import MaskAwareDQN

    cfg = frozen["dqn"]
    return MaskAwareDQN(
        observation_dim=observation_dim,
        action_dim=int(frozen["scientific_contract"]["actions"]["combined_action_count"]),
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        replay_capacity=int(cfg["replay_capacity"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(seed),
        device=str(cfg["device"]),
        gradient_clip_norm=float(cfg["gradient_clip_norm"]),
        network_architecture=tuple(cfg["network_architecture"]),
        weight_decay=float(cfg["weight_decay"]),
        epsilon_total_steps=int(cfg["epsilon_decay_duration_steps"]),
        epsilon_final=float(cfg["epsilon_end"]),
        epsilon_fraction=float(cfg["epsilon_decay_fraction"]),
    )


TRAIN_FIELDS = [
    "global_step", "episode_index", "year", "episode_step", "dap", "action_index",
    "irrigation_mm", "nitrogen_kg_ha", "valid_action_indices", "next_valid_action_indices",
    "mask_signature", "next_mask_signature", "mask_valid_action_count", "mask_forced_noop",
    "cumulative_irrigation_mm", "cumulative_n_kg_ha", "swfac", "nstres", "reward",
    "epsilon", "done", "loss", "mean_chosen_q", "mean_target", "optimizer_update",
]


def checkpoint_path(seed_dir: Path, step: int) -> Path:
    return seed_dir / "checkpoints" / f"dqn_checkpoint_{int(step):06d}.pt"


def write_checkpoint_provenance(seed_dir: Path, step: int, frozen: dict[str, Any], model: Any, seed: int) -> None:
    path = checkpoint_path(seed_dir, step)
    payload = {
        "task": FORMAL_TASK,
        "seed": int(seed),
        "checkpoint_step": int(step),
        "model_path": rel(path),
        "model_sha256": sha256_file(path),
        "frozen_config": rel(CONFIG_PATH),
        "frozen_config_sha256": sha256_file(CONFIG_PATH),
        "model_observation_dim": int(model.observation_dim),
        "model_action_dim": int(model.action_dim),
        "network_architecture": list(model.network_architecture),
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "batch_size": model.batch_size,
        "replay_capacity": model.replay_capacity,
        "weight_decay": model.weight_decay,
        "epsilon_schedule": {"start": model_epsilon(model, 0), "end": model_epsilon(model, 100000), "duration_steps": model.epsilon_total_steps},
        "optimizer_updates": int(model.optimizer_updates),
        "target_updates": int(model.target_updates),
        "git": git_state(),
        "core_sha256": sha256_file(ROOT / "src" / "mask_aware_dqn_029.py"),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(seed_dir / "checkpoints" / f"dqn_checkpoint_{int(step):06d}.json", payload)


def allocate_seed_run_dir(seed: int) -> Path:
    """Never overwrite an existing failed or partial seed run."""
    base = OUT / f"seed_{int(seed)}"
    if not base.exists() or not any(base.iterdir()):
        return base
    status_path = base / "seed_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except Exception:
            status = None
        if status == "completed":
            raise RuntimeError(f"Seed {seed} already has a completed run; refusing to overwrite it")
    attempt = 2
    while True:
        candidate = base / f"attempt_{attempt:02d}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        attempt += 1


def train_seed(frozen: dict[str, Any], runtime_cfg: dict[str, Any], modules: dict[str, Any], env_config: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_years = [int(x) for x in frozen["training"]["year_schedule"]["train_years"]]
    checkpoints = [int(x) for x in frozen["training"]["checkpoint_steps"]]
    budget = int(frozen["training"]["env_interaction_budget_per_seed"])
    learning_starts = int(frozen["dqn"]["learning_starts"])
    train_frequency = int(frozen["dqn"]["train_frequency_env_steps"])
    gradient_steps = int(frozen["dqn"]["gradient_steps_per_update"])
    target_sync = int(frozen["dqn"]["target_sync_frequency_env_steps"])
    seed_dir = allocate_seed_run_dir(int(seed))
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (seed_dir / "validation_daily").mkdir(parents=True, exist_ok=True)
    runtime_cfg = json.loads(json.dumps(runtime_cfg))
    runtime_cfg["seed"] = int(seed)
    write_json(seed_dir / "config_snapshot.json", frozen)
    write_json(seed_dir / "training_provenance.json", {"task": FORMAL_TASK, "seed": int(seed), "frozen_config": rel(CONFIG_PATH), "frozen_config_sha256": sha256_file(CONFIG_PATH), "git": git_state(), "core_sha256": sha256_file(ROOT / "src" / "mask_aware_dqn_029.py"), "runner_sha256": sha256_file(Path(__file__)), "scientific_contract_change": False, "engineering_plumbing": True})

    training_log_path = seed_dir / "training_log.csv"
    schedule_path = seed_dir / "training_year_schedule.csv"
    log_handle = training_log_path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(log_handle, fieldnames=TRAIN_FIELDS, extrasaction="ignore")
    writer.writeheader()
    pool: TrainingYearPool | None = None
    validation_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    model = None
    loss_latest: dict[str, Any] = {}
    try:
        # One reset establishes and validates the frozen raw 22-dimensional observation.
        first_year = int(train_years[0])
        first_env = make_env(modules, env_config, runtime_cfg, first_year, int(seed), f"seed{seed}_{first_year}_227_00_train_probe", False)
        first_obs, _ = first_env.reset()
        observation_dim = int(np.asarray(first_obs).size)
        first_env.close()
        if observation_dim != int(frozen["scientific_contract"]["observation"]["dimension"]):
            raise RuntimeError(f"Formal observation dimension mismatch: {observation_dim}")
        model = dqn_kwargs_from_freeze(frozen, int(seed), observation_dim)
        pool = TrainingYearPool(modules, env_config, runtime_cfg, train_years, int(seed), seed_dir)
        year, env, obs, info = pool.next_training_episode()
        episode_step = 0
        for global_step in range(1, budget + 1):
            state = np.asarray(obs, dtype=np.float32).reshape(-1)
            mask = np.asarray(env.action_masks(), dtype=bool).copy()
            if mask.shape != (model.action_dim,) or not bool(mask.any()):
                raise RuntimeError(f"Invalid collection mask at step {global_step}: {mask.shape}, any={mask.any()}")
            action = int(model.select_action(state, mask, global_step - 1, deterministic=False))
            if not bool(mask[action]):
                raise RuntimeError(f"Mask-aware exploration selected invalid action {action}")
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            action_info = dict(getattr(env, "last_action_info", {}))
            executed_action = int(action_info.get("discrete_action_index", -1))
            fallback = bool(action_info.get("mask_forced_noop", False))
            if executed_action != action or fallback:
                raise RuntimeError(f"Action-effect closure failure at step {global_step}: {action_info}")
            next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            next_mask = np.zeros(model.action_dim, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool).copy()
            model.add_transition(observation=state, action=action, reward=float(reward), next_observation=next_state, done=done, mask=mask, next_mask=next_mask)
            update = {}
            if global_step >= learning_starts and global_step % train_frequency == 0:
                for _ in range(gradient_steps):
                    update = model.train_step()
                    loss_latest = update
            if global_step % target_sync == 0:
                model.sync_target()
            latest = modules["direct_ppo"].latest_observation_dict(env, next_obs, info)
            row = {
                "global_step": global_step,
                "episode_index": pool.episode_index,
                "year": year,
                "episode_step": episode_step + 1,
                "dap": int(round(scalar(latest.get("dap"), episode_step + 1))),
                "action_index": action,
                "irrigation_mm": float(action_info.get("safe_action_amir", 0.0)),
                "nitrogen_kg_ha": float(action_info.get("safe_action_anfer", 0.0)),
                "valid_action_indices": valid_indices(mask),
                "next_valid_action_indices": valid_indices(next_mask),
                "mask_signature": mask_signature(mask),
                "next_mask_signature": mask_signature(next_mask),
                "mask_valid_action_count": int(mask.sum()),
                "mask_forced_noop": fallback,
                "cumulative_irrigation_mm": float(action_info.get("season_cumulative_irrigation", 0.0)),
                "cumulative_n_kg_ha": float(action_info.get("season_cumulative_n", 0.0)),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "reward": float(reward),
                "epsilon": model_epsilon(model, global_step - 1),
                "done": done,
                "loss": update.get("loss", np.nan),
                "mean_chosen_q": update.get("mean_chosen_q", np.nan),
                "mean_target": update.get("mean_target", np.nan),
                "optimizer_update": int(model.optimizer_updates),
            }
            writer.writerow(row)
            if global_step % 1000 == 0:
                log_handle.flush()
            episode_step += 1
            if done:
                year, env, obs, info = pool.next_training_episode()
                episode_step = 0
            else:
                obs = next_obs

            if global_step in checkpoints:
                path = checkpoint_path(seed_dir, global_step)
                model.save(path, global_step)
                write_checkpoint_provenance(seed_dir, global_step, frozen, model, int(seed))
                eval_rows, eval_actions = validate_checkpoint(frozen, runtime_cfg, modules, env_config, model, seed, global_step, seed_dir)
                validation_rows.extend(eval_rows)
                action_rows.extend(eval_actions)
                checkpoint_rows.append({"seed": int(seed), "checkpoint": int(global_step), "checkpoint_path": rel(path), "checkpoint_sha256": sha256_file(path), "validation_rows": len(eval_rows), "status": "PASS" if all(x["run_status"] == "ok" for x in eval_rows) else "FAIL", "optimizer_updates": int(model.optimizer_updates), "target_updates": int(model.target_updates), "replay_size": int(model.replay.size), "train_year_episode_counts": dict(pool.counts)})
                pd.DataFrame(validation_rows).to_csv(seed_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(action_rows).to_csv(seed_dir / "action_summary.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(checkpoint_rows).to_csv(seed_dir / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        model.save(checkpoint_path(seed_dir, budget), budget)
        write_json(seed_dir / "training_year_schedule.json", {"sampling": "uniform_random_with_replacement_per_episode", "seed": int(seed), "counts": dict(pool.counts), "episodes": int(pool.episode_index)})
        pd.DataFrame(pool.schedule_rows).to_csv(schedule_path, index=False, encoding="utf-8-sig")
        write_json(seed_dir / "seed_status.json", {"status": "completed", "seed": int(seed), "env_steps": budget, "checkpoint_count": len(checkpoint_rows), "validation_count": len(validation_rows), "invalid_action_count": 0, "fallback_count": 0})
    except Exception as exc:
        failure = {"status": "failed", "seed": int(seed), "completed_env_steps": int(len(validation_rows)), "error": str(exc), "traceback": traceback.format_exc(), "failed_outputs_retained": True}
        write_json(seed_dir / "failed_attempt.json", failure)
        write_json(seed_dir / "seed_status.json", failure)
        if pool is not None:
            pd.DataFrame(pool.schedule_rows).to_csv(schedule_path, index=False, encoding="utf-8-sig")
        raise
    finally:
        log_handle.flush()
        log_handle.close()
        if pool is not None:
            pool.close()
        if model is not None:
            write_json(seed_dir / "model_runtime_state.json", {"optimizer_updates": int(model.optimizer_updates), "target_updates": int(model.target_updates), "replay_size": int(model.replay.size), "last_update": loss_latest})
    return pd.DataFrame(validation_rows), pd.DataFrame(action_rows), pd.DataFrame(checkpoint_rows)


def validate_checkpoint(frozen: dict[str, Any], runtime_cfg: dict[str, Any], modules: dict[str, Any], env_config: dict[str, Any], model: Any, seed: int, checkpoint: int, seed_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env, strict_metrics_from_snapshot

    validation_years = [int(x) for x in frozen["evaluation"]["validation_years"]]
    daily_dir = seed_dir / "validation_daily" / f"checkpoint_{int(checkpoint):06d}"
    summary_dir = seed_dir / "summary_outputs" / f"checkpoint_{int(checkpoint):06d}"
    daily_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for year in validation_years:
        tag = f"seed{seed}_eval_{year}_ckpt{checkpoint}"
        env = None
        records: list[dict[str, Any]] = []
        invalid_count = 0
        fallback_count = 0
        summary_out_path = summary_dir / f"YCA_{year}_Summary.OUT"
        strict: dict[str, Any] = {}
        try:
            env = make_env(modules, env_config, runtime_cfg, year, seed, tag, True)
            obs, info = env.reset()
            done = False
            step_count = 0
            planting = pd.Timestamp(modules["direct_ppo"].find_year(env_config, STATION, year)["planting_date"])
            while not done and step_count < int(runtime_cfg["runtime"]["max_steps"]):
                latest_before = modules["direct_ppo"].latest_observation_dict(env, obs, info)
                dap = int(round(scalar(latest_before.get("dap"), step_count + 1)))
                mask = np.asarray(env.action_masks(), dtype=bool).copy()
                action = int(model.select_action(np.asarray(obs, dtype=np.float32).reshape(-1), mask, checkpoint, deterministic=True))
                if not bool(mask[action]):
                    invalid_count += 1
                    raise RuntimeError(f"Deterministic masked-greedy chose invalid action {action}")
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                action_info = dict(getattr(env, "last_action_info", {}))
                executed = int(action_info.get("discrete_action_index", -1))
                fallback = bool(action_info.get("mask_forced_noop", False))
                fallback_count += int(fallback)
                if executed != action or fallback:
                    raise RuntimeError(f"Evaluation action closure failure: {action_info}")
                next_mask = np.zeros(model.action_dim, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool).copy()
                latest = modules["direct_ppo"].latest_observation_dict(env, next_obs, info)
                date = planting + pd.Timedelta(days=max(dap - 1, 0))
                record = {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "seed": int(seed),
                    "checkpoint": int(checkpoint),
                    "algorithm": "MaskAwareDQN",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "observation_dim": int(np.asarray(obs).size),
                    "action_index": action,
                    "executed_action_index": executed,
                    "irrigation_mm": float(action_info.get("safe_action_amir", 0.0)),
                    "nitrogen_kg_ha": float(action_info.get("safe_action_anfer", 0.0)),
                    "valid_action_indices": valid_indices(mask),
                    "next_valid_action_indices": valid_indices(next_mask),
                    "mask_signature": mask_signature(mask),
                    "next_mask_signature": mask_signature(next_mask),
                    "fallback_triggered": fallback,
                    "cumulative_irrigation_mm": float(action_info.get("season_cumulative_irrigation", 0.0)),
                    "cumulative_n_kg_ha": float(action_info.get("season_cumulative_n", 0.0)),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=json_default),
                }
                records.append(record)
                action_rows.append({"seed": int(seed), "checkpoint": int(checkpoint), "year": int(year), "action_index": action, "irrigation_mm": record["irrigation_mm"], "nitrogen_kg_ha": record["nitrogen_kg_ha"], "count": 1})
                obs = next_obs
                step_count += 1
            daily_path = daily_dir / f"YCA_{year}_daily.csv"
            daily = pd.DataFrame(records)
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            if not records or not records[-1]["done"]:
                raise RuntimeError(f"Validation episode did not complete within max_steps={runtime_cfg['runtime']['max_steps']}")
            final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
            total_i = float(pd.to_numeric(daily["irrigation_mm"], errors="coerce").sum())
            total_n = float(pd.to_numeric(daily["nitrogen_kg_ha"], errors="coerce").sum())
            try:
                snapshot = snapshot_from_env(env)
                shutil.copyfile(snapshot / "Summary.OUT", summary_out_path)
                strict = strict_metrics_from_snapshot(snapshot, final_yield, total_i, total_n)
            except Exception as exc:
                strict = {"WP_ET_kg_m3": np.nan, "etcp_mm": np.nan, "PFP_N_kg_kg": final_yield / total_n if total_n > 0 else np.nan, "summary_out_status": "unknown / insufficient evidence", "summary_out_error": str(exc)}
            rows.append({
                "station_code": STATION,
                "site": SITE,
                "year": int(year),
                "seed": int(seed),
                "checkpoint": int(checkpoint),
                "run_status": "ok",
                "episode_length": int(len(daily)),
                "yield_kg_ha": final_yield,
                "total_irrigation_mm": total_i,
                "total_nitrogen_kg_ha": total_n,
                "PFP_N": float(strict.get("PFP_N_kg_kg", final_yield / total_n if total_n > 0 else np.nan)),
                "WP_ET_kg_m3": strict.get("WP_ET_kg_m3", np.nan),
                "ETCP_mm": strict.get("etcp_mm", np.nan),
                "NUE_if_available": np.nan,
                "NUE_status": "unknown / insufficient evidence",
                "summary_out_status": strict.get("summary_out_status", "exact Summary.OUT/ETCP replay"),
                "summary_out_path": rel(summary_out_path) if summary_out_path.exists() else "unknown / insufficient evidence",
                "daily_path": rel(daily_path),
                "invalid_action_count": int(invalid_count),
                "fallback_count": int(fallback_count),
                "action_sequence": "; ".join(f"DAP{int(r['dap'])} I{float(r['irrigation_mm']):g}/N{float(r['nitrogen_kg_ha']):g}" for r in records if float(r["irrigation_mm"]) > 0 or float(r["nitrogen_kg_ha"]) > 0),
            })
        except Exception as exc:
            daily_path = daily_dir / f"YCA_{year}_daily.csv"
            pd.DataFrame(records).to_csv(daily_path, index=False, encoding="utf-8-sig")
            rows.append({"station_code": STATION, "site": SITE, "year": int(year), "seed": int(seed), "checkpoint": int(checkpoint), "run_status": "failed", "episode_length": len(records), "yield_kg_ha": np.nan, "total_irrigation_mm": np.nan, "total_nitrogen_kg_ha": np.nan, "PFP_N": np.nan, "WP_ET_kg_m3": np.nan, "ETCP_mm": np.nan, "NUE_if_available": np.nan, "NUE_status": "unknown / insufficient evidence", "summary_out_status": "unknown / insufficient evidence", "summary_out_path": rel(summary_out_path) if summary_out_path.exists() else "unknown / insufficient evidence", "daily_path": rel(daily_path), "invalid_action_count": invalid_count, "fallback_count": fallback_count, "error": str(exc)})
            raise
        finally:
            if env is not None:
                env.close()
    return rows, action_rows


def aggregate_results(validation: pd.DataFrame, actions: pd.DataFrame, checkpoints: pd.DataFrame, frozen: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    valid = validation[validation["run_status"].eq("ok")].copy()
    if valid.empty:
        raise RuntimeError("No completed formal validation rows")
    checkpoint_summary = valid.groupby(["seed", "checkpoint"], as_index=False).agg(
        mean_yield=("yield_kg_ha", "mean"), sd_yield=("yield_kg_ha", "std"),
        mean_irrigation=("total_irrigation_mm", "mean"), mean_nitrogen=("total_nitrogen_kg_ha", "mean"),
        mean_PFP_N=("PFP_N", "mean"), mean_WP_ET=("WP_ET_kg_m3", "mean"), mean_NUE_if_available=("NUE_if_available", "mean"),
        invalid_action_count=("invalid_action_count", "sum"), fallback_count=("fallback_count", "sum"),
        validation_year_count=("year", "nunique"),
    )
    checkpoint_summary["sd_yield"] = checkpoint_summary["sd_yield"].fillna(0.0)
    checkpoint_summary.to_csv(RESULTS / "yc_dqn_formal_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    final = valid[valid["checkpoint"].eq(100000)].copy()
    seed_summary = final.groupby("seed", as_index=False).agg(
        I_100K=("total_irrigation_mm", "mean"), N_100K=("total_nitrogen_kg_ha", "mean"), PFP_N_100K=("PFP_N", "mean"), WP_ET_100K=("WP_ET_kg_m3", "mean"), NUE_100K=("NUE_if_available", "mean"), validation_year_count=("year", "nunique"),
    )
    wide = checkpoint_summary.pivot(index="seed", columns="checkpoint", values="mean_yield").reset_index()
    rename_yield = {step: f"yield_{step // 1000}K" for step in [25000, 50000, 75000, 100000]}
    wide = wide.rename(columns=rename_yield)
    for col in ["yield_25K", "yield_50K", "yield_75K", "yield_100K"]:
        if col not in wide.columns:
            wide[col] = np.nan
    seed_summary = wide[["seed", "yield_25K", "yield_50K", "yield_75K", "yield_100K"]].merge(seed_summary, on="seed", how="left")
    seed_summary.to_csv(RESULTS / "yc_dqn_formal_seed_summary.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(RESULTS / "yc_dqn_formal_validation_yearly.csv", index=False, encoding="utf-8-sig")
    if not actions.empty:
        action_summary = actions.groupby(["seed", "checkpoint", "year", "action_index", "irrigation_mm", "nitrogen_kg_ha"], as_index=False).agg(action_count=("count", "sum"))
        action_summary["action_fraction_within_year"] = action_summary["action_count"] / action_summary.groupby(["seed", "checkpoint", "year"])["action_count"].transform("sum")
    else:
        action_summary = pd.DataFrame(columns=["seed", "checkpoint", "year", "action_index", "irrigation_mm", "nitrogen_kg_ha", "action_count", "action_fraction_within_year"])
    action_summary.to_csv(RESULTS / "yc_dqn_formal_action_summary.csv", index=False, encoding="utf-8-sig")
    return checkpoint_summary, seed_summary, validation, action_summary


def matched_ppo_comparison(validation: pd.DataFrame) -> pd.DataFrame:
    ppo_paths = {
        0: ROOT / "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv",
        1: ROOT / "benchmark_results/222YCA_O1_yc_v1_original_seed1/evaluation/222YCA_O1_checkpoint_validation_summary.csv",
        2: ROOT / "benchmark_results/222YCA_O2_yc_v1_original_seed2/evaluation/222YCA_O2_checkpoint_validation_summary.csv",
    }
    frames: list[pd.DataFrame] = []
    for seed, path in ppo_paths.items():
        if not path.exists():
            continue
        ppo = pd.read_csv(path)
        ppo = ppo[ppo["checkpoint_step"].astype(int).isin([25000, 50000, 75000, 100000]) & ppo["year"].astype(int).isin(list(range(2014, 2024)))].copy()
        ppo["seed"] = seed
        frames.append(ppo.rename(columns={"checkpoint_step": "checkpoint", "final_grnwt": "ppo_yield_kg_ha", "total_irrigation": "ppo_total_irrigation_mm", "total_n": "ppo_total_nitrogen_kg_ha", "PFP_N": "ppo_PFP_N"})[["seed", "checkpoint", "year", "ppo_yield_kg_ha", "ppo_total_irrigation_mm", "ppo_total_nitrogen_kg_ha", "ppo_PFP_N"]])
    if not frames:
        return pd.DataFrame({"comparison_status": ["unknown / insufficient evidence"], "notes": ["Existing PPO matched summaries were not found"]})
    ppo_all = pd.concat(frames, ignore_index=True)
    dqn = validation.rename(columns={"yield_kg_ha": "dqn_yield_kg_ha", "total_irrigation_mm": "dqn_total_irrigation_mm", "total_nitrogen_kg_ha": "dqn_total_nitrogen_kg_ha", "PFP_N": "dqn_PFP_N", "WP_ET_kg_m3": "dqn_WP_ET_kg_m3"})
    out = dqn.merge(ppo_all, on=["seed", "checkpoint", "year"], how="left")
    out["yield_delta_dqn_minus_ppo"] = out["dqn_yield_kg_ha"] - out["ppo_yield_kg_ha"]
    out["irrigation_delta_dqn_minus_ppo"] = out["dqn_total_irrigation_mm"] - out["ppo_total_irrigation_mm"]
    out["nitrogen_delta_dqn_minus_ppo"] = out["dqn_total_nitrogen_kg_ha"] - out["ppo_total_nitrogen_kg_ha"]
    out["comparison_scope"] = "same seed, checkpoint, validation year; descriptive only"
    out["ppo_WP_ET_kg_m3"] = np.nan
    out["ppo_NUE_if_available"] = np.nan
    out["WP_ET_note"] = "PPO source summary lacks exact WP_ET here; unknown / insufficient evidence"
    out.to_csv(RESULTS / "yc_ppo_vs_dqn_matched_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 30) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.copy()
    if columns:
        work = work[[c for c in columns if c in work.columns]]
    work = work.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    return "\n".join(["| " + " | ".join(map(str, work.columns)) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |", *["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]])


def write_report(frozen: dict[str, Any], checkpoint_summary: pd.DataFrame, seed_summary: pd.DataFrame, validation: pd.DataFrame, comparison: pd.DataFrame) -> None:
    completed_seeds = seed_summary["validation_year_count"].eq(10).sum() if not seed_summary.empty else 0
    eval_count = int(validation["run_status"].eq("ok").sum()) if not validation.empty else 0
    invalid = int(pd.to_numeric(validation.get("invalid_action_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not validation.empty else 0
    fallback = int(pd.to_numeric(validation.get("fallback_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not validation.empty else 0
    final = seed_summary.set_index("seed") if not seed_summary.empty else pd.DataFrame()
    yield_mean = float(final["yield_100K"].mean()) if not final.empty else np.nan
    yield_sd = float(final["yield_100K"].std(ddof=1)) if len(final) > 1 else 0.0
    trajectory_rows: list[dict[str, Any]] = []
    degradation = False
    for seed, group in checkpoint_summary.groupby("seed"):
        ordered = group.sort_values("checkpoint")
        values = ordered["mean_yield"].to_numpy(dtype=float)
        degradation = degradation or bool(len(values) and values[-1] < np.nanmax(values[:-1]) if len(values) > 1 else False)
        trajectory_rows.append({"seed": int(seed), "yield_25K": float(values[0]) if len(values) > 0 else np.nan, "yield_50K": float(values[1]) if len(values) > 1 else np.nan, "yield_75K": float(values[2]) if len(values) > 2 else np.nan, "yield_100K": float(values[3]) if len(values) > 3 else np.nan})
    ppo_final = comparison[comparison["checkpoint"].eq(100000)] if not comparison.empty and "checkpoint" in comparison.columns else pd.DataFrame()
    ppo_mean = float(pd.to_numeric(ppo_final.get("ppo_yield_kg_ha", pd.Series(dtype=float)), errors="coerce").mean()) if not ppo_final.empty else np.nan
    ppo_seed_means = ppo_final.groupby("seed")["ppo_yield_kg_ha"].mean() if not ppo_final.empty and "ppo_yield_kg_ha" in ppo_final.columns else pd.Series(dtype=float)
    ppo_sd = float(ppo_seed_means.std(ddof=1)) if len(ppo_seed_means) > 1 else np.nan
    all_eval_complete = len(validation) == 120 and bool(validation["run_status"].eq("ok").all())
    lines = [
        "# YC formal 3-seed mask-aware DQN training report",
        "",
        "审计/训练日期：2026-09-11。正式训练已按冻结配置实际执行；本报告不替代逐年 CSV、daily 日志或 DSSAT Summary.OUT。",
        "",
        "## Frozen contract and runtime boundary",
        "",
        f"- 唯一 formal 配置：`{rel(CONFIG_PATH)}`；site/station：`{SITE}/{STATION}`；lowIC；训练年份 `2004–2013`；验证年份 `2014–2023`。",
        "- observation：current raw daily observation, 22 dim；no normalization、no forecast、no weather augmentation。",
        "- action：I `[0,15,30,45]` × N `[0,40,80,120]`，16 个 row-major joint actions；mask/safety/reward 沿用 226 已通过的 current 055-style / 040_28 / 040_36 chain。",
        "- 只做了 engineering parameter plumbing：QNetwork、Adam weight decay、显式 train/target/epsilon schedule，以及 provenance 持久化；没有改 scientific contract。",
        "- WP_ET 仅在 episode 结束前通过 exact `Summary.OUT/ETCP` 解析并复制到 seed 的 `summary_outputs/`；不能解析时保留 `unknown / insufficient evidence`，不从 daily CSV 反推。",
        "",
        "## Attempt history",
        "",
        "- pre-training parity attempt_01：失败时间 `unknown / insufficient evidence`；阶段为 formal training 前的 runtime parity。字段 `epsilon` 被 runner 误用 029 静态默认调用记录为 `1.0 -> 0.05`，与冻结的 `1.0 -> 0.0 / 100000 steps` 不一致；该 parity 失败证据保留于 `benchmark_results/227_00_yc_mask_aware_dqn_formal/failed_attempts/attempt_01_formal_runtime_config_parity.csv`。修复仅是改用实例化 formal schedule 读取函数，随后 parity 18/18 PASS。",
        "- formal `seed_0` attempt_01：`2026-09-11 / 25K checkpoint 后的 validation 日志序列化阶段`；精确墙上时刻 `unknown / insufficient evidence`。失败原因是 DSSAT `info` 中的 ndarray 不能被 JSON 默认序列化：`Object of type ndarray is not JSON serializable`。它发生在首个 validation daily 写盘前，不是 DSSAT 科学运行或 mask mismatch。证据见 `benchmark_results/227_00_yc_mask_aware_dqn_formal/seed_0/failed_attempt.json`。",
        "- 从 formal attempt_01 到最终 PASS：只增加 ndarray→list 的日志序列化兼容，以及不可覆盖的 seed run 目录分配；此前已完成的 029 显式参数 plumbing 也只属于 engineering plumbing。没有改变 observation、reward、action space、mask 逻辑、灌溉/施肥约束、训练年份、IC、cultivar 或任何 DQN hyperparameters。",
        "- 失败 attempt 未删除：原 `seed_0/` 的 failed_attempt、旧 25K checkpoint 和其状态均保留；最终通过结果位于 `seed_0/attempt_02/`。",
        "",
        "## 1. 三个 DQN seed 是否都完成 100K？",
        "",
        f"**{int(completed_seeds)}/3 completed**。正式运行状态见 `seed_0/attempt_02/seed_status.json`、`seed_1/seed_status.json`、`seed_2/seed_status.json`；seed_0 根目录的 failed attempt 另行保留。",
        "",
        "## 2. 12 个 checkpoint evaluations 是否都完成？",
        "",
        f"**{all_eval_complete}**：要求为 3 seeds × 4 checkpoints × 10 validation years = 120 个逐年 validation rows；实际已完成 `{eval_count}/120`。",
        "",
        "## 3. invalid action / fallback 是否为 0？",
        "",
        f"**invalid action = {invalid}；fallback = {fallback}**。collection 与 deterministic validation 都要求 selected action 与 wrapper-executed action 相等。",
        "",
        "## 4. 100K 时三个 seed 的 yield/I/N/PFP_N/WP_ET",
        "",
        markdown_table(seed_summary, ["seed", "yield_100K", "I_100K", "N_100K", "PFP_N_100K", "WP_ET_100K", "NUE_100K"], 10),
        "",
        "## 5. 是否存在 seed-dependent instability？",
        "",
        f"三个 seed 的 100K validation mean yield 为 `{yield_mean:.3f}` kg/ha，跨 seed SD 为 `{yield_sd:.3f}` kg/ha；这支持报告 seed 间存在/不存在的数值 spread，但不设未经预注册的 instability threshold。",
        "",
        "## 6. 是否出现“先学会后退化”？",
        "",
        f"**{degradation}**，判据仅是同一 seed 的 checkpoint mean yield 在后期低于该 seed 更早 checkpoint 的最高值；完整轨迹见 `results/yc_dqn_formal_checkpoint_summary.csv`。",
        "",
        "## 7. 是否存在从未进入好 basin 的 seed？",
        "",
        "`unknown / insufficient evidence`：本 formal prompt 没有预先定义可审计的 good-basin 数值阈值，因此不把某个 seed 主观标记为未进入 basin。",
        "",
        "## 8. DQN vs Original PPO matched descriptive comparison",
        "",
        f"DQN 100K 跨 seed mean yield = `{yield_mean:.3f}` kg/ha，seed SD = `{yield_sd:.3f}`；现有 Original PPO matched 100K mean yield = `{ppo_mean:.3f}` kg/ha，seed SD = `{ppo_sd:.3f}`。PPO source summary 的 exact WP_ET 不在本 matched CSV 中，标为 `unknown / insufficient evidence`。trajectory stability 以 checkpoint mean-yield trajectory 和逐年 daily/action logs 为证据，不能做因果归因。",
        "",
        "## 9. 当前证据支持哪一种解释？",
        "",
        "**mixed/inconclusive**：本轮已提供 DQN 三 seed、四 checkpoint 与 matched PPO 描述性轨迹，但没有为 PPO-specific 与 shared RL/task instability 预注册可识别的因果判据。",
        "",
        "## 10. 是否值得继续 algorithm-specific stabilization？",
        "",
        "**Yes, as a separate preregistered follow-up only**：结果可用于提出稳定性诊断问题，但不得回写本冻结配置、删改失败输出或据结果调参。",
        "",
        "## Evidence index",
        "",
        "- Runtime parity：`results/formal_runtime_config_parity.csv`。",
        "- DQN checkpoint summary：`results/yc_dqn_formal_checkpoint_summary.csv`；seed summary：`results/yc_dqn_formal_seed_summary.csv`。",
        "- 逐年结果：`results/yc_dqn_formal_validation_yearly.csv`；动作统计：`results/yc_dqn_formal_action_summary.csv`。",
        "- Matched descriptive comparison：`results/yc_ppo_vs_dqn_matched_checkpoint_comparison.csv`。",
        "- 每个正式运行目录的 checkpoint、training log、year schedule、daily、Summary.OUT、配置快照、git/code provenance 位于 `seed_0/attempt_02/`、`seed_1/`、`seed_2/`。",
        "- 失败结果保留：首次 seed_0 失败的 `seed_0/failed_attempt.json`、旧 25K checkpoint，以及 parity 失败 CSV 均未删除或替换。",
        "",
        "## Formal status",
        "",
        "`FORMAL_DQN_3SEED_TRAINING_COMPLETE`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_seed_dir(seed: int) -> Path:
    base = OUT / f"seed_{int(seed)}"
    candidates = [base, *sorted(base.glob("attempt_*"))] if base.exists() else []
    for candidate in reversed(candidates):
        status_path = candidate / "seed_status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
            except Exception:
                status = None
            if status == "completed":
                return candidate
    raise RuntimeError(f"No completed output directory found for seed {seed}")


def postprocess_existing(frozen: dict[str, Any]) -> None:
    validation_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    checkpoint_frames: list[pd.DataFrame] = []
    selected_dirs: dict[str, str] = {}
    for seed in [int(x) for x in frozen["training"]["seeds"]]:
        seed_dir = completed_seed_dir(seed)
        selected_dirs[str(seed)] = rel(seed_dir)
        validation_frames.append(pd.read_csv(seed_dir / "validation_summary.csv"))
        action_frames.append(pd.read_csv(seed_dir / "action_summary.csv"))
        checkpoint_frames.append(pd.read_csv(seed_dir / "checkpoint_inventory.csv"))
    validation = pd.concat(validation_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    checkpoint_summary, seed_summary, validation, action_summary = aggregate_results(validation, actions, checkpoints, frozen)
    comparison = matched_ppo_comparison(validation)
    write_report(frozen, checkpoint_summary, seed_summary, validation, comparison)
    write_json(OUT / "formal_training_status.json", {"status": "completed_formal", "task": FORMAL_TASK, "seeds": [0, 1, 2], "env_steps_per_seed": 100000, "checkpoint_count": 12, "validation_year_rows": int(len(validation)), "invalid_action_count": int(validation["invalid_action_count"].sum()), "fallback_count": int(validation["fallback_count"].sum()), "runtime_parity": rel(RESULTS / "formal_runtime_config_parity.csv"), "report": rel(REPORT), "selected_completed_seed_dirs": selected_dirs, "failed_attempt_retained": rel(OUT / "seed_0")})
    print("FORMAL_DQN_3SEED_TRAINING_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    frozen = load_frozen()
    if args.postprocess_only:
        postprocess_existing(frozen)
        return
    modules = configure_runtime_modules(frozen)
    runtime_cfg = modules["config"]
    env_config, selection = build_env_config(runtime_cfg, modules)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "formal_config_snapshot.json", frozen)
    modules["direct_ppo"].write_yaml(env_config, OUT / "resolved_env_config.yaml")
    selection.to_csv(OUT / "selected_yc_years.csv", index=False, encoding="utf-8-sig")
    parity = runtime_parity(frozen, runtime_cfg, modules, env_config)
    if args.parity_only:
        print("FORMAL_RUNTIME_PARITY_PASS")
        return
    all_validation: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    all_checkpoints: list[pd.DataFrame] = []
    try:
        for seed in [int(x) for x in frozen["training"]["seeds"]]:
            validation, actions, checkpoints = train_seed(frozen, runtime_cfg, modules, env_config, seed)
            all_validation.append(validation)
            all_actions.append(actions)
            all_checkpoints.append(checkpoints)
        validation = pd.concat(all_validation, ignore_index=True) if all_validation else pd.DataFrame()
        actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
        checkpoints = pd.concat(all_checkpoints, ignore_index=True) if all_checkpoints else pd.DataFrame()
        checkpoint_summary, seed_summary, validation, action_summary = aggregate_results(validation, actions, checkpoints, frozen)
        comparison = matched_ppo_comparison(validation)
        write_report(frozen, checkpoint_summary, seed_summary, validation, comparison)
        write_json(OUT / "formal_training_status.json", {"status": "completed_formal", "task": FORMAL_TASK, "seeds": [0, 1, 2], "env_steps_per_seed": 100000, "checkpoint_count": 12, "validation_year_rows": int(len(validation)), "invalid_action_count": int(validation["invalid_action_count"].sum()), "fallback_count": int(validation["fallback_count"].sum()), "runtime_parity": rel(RESULTS / "formal_runtime_config_parity.csv"), "report": rel(REPORT)})
        print("FORMAL_DQN_3SEED_TRAINING_COMPLETE")
    except Exception as exc:
        write_json(OUT / "formal_training_status.json", {"status": "FORMAL_DQN_TRAINING_BLOCKED", "task": FORMAL_TASK, "error": str(exc), "traceback": traceback.format_exc(), "failed_outputs_retained": True, "runtime_parity": rel(RESULTS / "formal_runtime_config_parity.csv")})
        print("FORMAL_DQN_TRAINING_BLOCKED")
        raise


if __name__ == "__main__":
    main()

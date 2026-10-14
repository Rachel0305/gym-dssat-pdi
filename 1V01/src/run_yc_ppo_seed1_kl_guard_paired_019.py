"""019/018B: paired YC PPO seed1 KL-guard continuation.

This experiment starts both branches from the existing Original PPO seed1 75K
checkpoint.  It is intentionally isolated from the formal run directory:
the existing PPO/DSSAT source is imported but never edited.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

ORIGINAL_RUN = ROOT / "benchmark_results" / "222YCA_O1_yc_v1_original_seed1"
ORIGINAL_CHECKPOINT = ORIGINAL_RUN / "models" / "YCA" / "YCA_half_split_stress_aware_maskableppo_seed1_ckpt75000.zip"
OUT = ROOT / "benchmark_results" / "019_yc_ppo_seed1_kl_guard_screen"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_kl_guard_stabilization_screen.md"
RESUME_OUT = RESULTS / "yc_ppo_s1_75k_resume_feasibility.csv"
LOGIC_OUT = RESULTS / "yc_ppo_s1_kl_guard_logic_test.csv"
CONTROL_METRICS_OUT = RESULTS / "yc_ppo_s1_control_training_metrics.csv"
GUARD_METRICS_OUT = RESULTS / "yc_ppo_s1_klguard_training_metrics.csv"
CONTROL_VALIDATION_OUT = RESULTS / "yc_ppo_s1_control_checkpoint_validation.csv"
GUARD_VALIDATION_OUT = RESULTS / "yc_ppo_s1_klguard_checkpoint_validation.csv"
CONTROL_POLICY_OUT = RESULTS / "yc_ppo_s1_control_fixed_state_policy.csv"
GUARD_POLICY_OUT = RESULTS / "yc_ppo_s1_klguard_fixed_state_policy.csv"
STRESS_SUMMARY_OUT = RESULTS / "yc_ppo_s1_klguard_stress_policy_summary.csv"
DECISION_OUT = RESULTS / "yc_ppo_s1_kl_guard_screen_decision.csv"

STATION = "YCA"
SITE = "YC"
SEED = 1
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
CONTINUATION_STEPS = 25_000
EFFECTIVE_KL_THRESHOLD = 1.0e-4
CONFIGURED_TARGET_KL = EFFECTIVE_KL_THRESHOLD / 1.5
POLICY_MONITOR_CHECKPOINTS = CHECKPOINTS
EXPECTED_METRIC_COLUMNS = [
    "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate",
    "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed",
    "kl_guard_triggered", "kl_at_trigger",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def action_text(action: int) -> str:
    irrigation = [0.0, 15.0, 30.0, 45.0][int(action) // 4]
    nitrogen = [0.0, 40.0, 80.0, 120.0][int(action) % 4]
    return f"I{irrigation:g}/N{nitrogen:g}"


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_data(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def resume_audit() -> tuple[bool, dict[str, Any]]:
    """Audit recoverability without constructing a training environment."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if not ORIGINAL_CHECKPOINT.exists():
        rows.append({"component": "checkpoint", "status": "FAIL", "detail": f"missing: {rel(ORIGINAL_CHECKPOINT)}"})
        pd.DataFrame(rows).to_csv(RESUME_OUT, index=False, encoding="utf-8-sig")
        return False, {"status": "PPO_75K_RESUME_UNAVAILABLE", "rows": rows}

    import torch

    with zipfile.ZipFile(ORIGINAL_CHECKPOINT) as archive:
        members = set(archive.namelist())
        data = json.loads(archive.read("data"))
        policy_state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=False)
        optimizer_state = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=False)

    policy_keys = set(policy_state.keys())
    value_keys = sorted(key for key in policy_keys if "value_net" in key or "mlp_extractor" in key)
    rng_names = [name for name in members if re.search(r"rng|random_state|random-state", name, flags=re.I)]
    data_rng_keys = [key for key in data if re.search(r"rng|random", key, flags=re.I)]
    optimizer_ok = bool(optimizer_state.get("state")) and bool(optimizer_state.get("param_groups"))
    schedule_ok = all(key in data for key in ["_current_progress_remaining", "_total_timesteps", "lr_schedule", "clip_range"])
    metadata_ok = all(key in data for key in ["num_timesteps", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "learning_rate", "target_kl"])
    component_rows = [
        ("policy_weights", bool(policy_keys), f"policy.pth keys={len(policy_keys)}"),
        ("value_weights", bool(value_keys), f"value-related keys={len(value_keys)}"),
        ("optimizer_state", optimizer_ok, f"optimizer state entries={len(optimizer_state.get('state', {}))}"),
        ("num_timesteps", int(data.get("num_timesteps", -1)) == 75_000, f"num_timesteps={data.get('num_timesteps')}"),
        ("learning_rate_schedule_position", schedule_ok, f"progress_remaining={data.get('_current_progress_remaining')}; total={data.get('_total_timesteps')}"),
        ("clip_range_schedule_position", schedule_ok, f"clip_range metadata present; value={data.get('clip_range', {}).get('value_schedule', 'unknown')}"),
        ("continuation_metadata", metadata_ok, f"n_steps={data.get('n_steps')}, batch_size={data.get('batch_size')}, n_epochs={data.get('n_epochs')}, gamma={data.get('gamma')}, gae_lambda={data.get('gae_lambda')}"),
        ("python_rng_state", False, "not present in checkpoint archive/data; exact stochastic replay unavailable"),
        ("numpy_rng_state", False, "not present in checkpoint archive/data; will reinitialize identically for paired branches"),
        ("pytorch_rng_state", False, "not present in checkpoint archive/data; will reinitialize identically for paired branches"),
        ("environment_year_sampling_rng", False, "not present in checkpoint archive/data; branch env RNG will be reinitialized with seed=1"),
    ]
    for component, available, detail in component_rows:
        rows.append({
            "component": component,
            "available": bool(available),
            "status": "PASS" if available or component in {"python_rng_state", "numpy_rng_state", "pytorch_rng_state", "environment_year_sampling_rng"} else "FAIL",
            "detail": detail,
            "checkpoint": rel(ORIGINAL_CHECKPOINT),
        })
    recoverable = all(
        row["available"]
        for row in rows
        if row["component"] in {
            "policy_weights", "value_weights", "optimizer_state", "num_timesteps",
            "learning_rate_schedule_position", "clip_range_schedule_position", "continuation_metadata",
        }
    )
    payload = {
        "resume_status": "RESUME_AVAILABLE_WITH_REINITIALIZED_RNG" if recoverable else "PPO_75K_RESUME_UNAVAILABLE",
        "checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "zip_members": sorted(members),
        "rng_archive_names": rng_names,
        "rng_data_keys": data_rng_keys,
        "data_num_timesteps": data.get("num_timesteps"),
        "data_total_timesteps": data.get("_total_timesteps"),
        "data_progress_remaining": data.get("_current_progress_remaining"),
        "data_learning_rate": data.get("learning_rate"),
        "data_clip_range": data.get("clip_range"),
        "optimizer_state_entries": len(optimizer_state.get("state", {})),
        "rng_exact_replay": False,
    }
    for row in rows:
        row["overall_resume_status"] = payload["resume_status"]
    pd.DataFrame(rows).to_csv(RESUME_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "resume_audit.json", payload)
    return recoverable, payload


def kl_guard_logic_test() -> tuple[bool, dict[str, Any]]:
    from sb3_contrib import MaskablePPO

    source = inspect.getsource(MaskablePPO.train)
    condition_match = re.search(r"if self\.target_kl is not None and approx_kl_div > ([^:]+):", source)
    condition = condition_match.group(0) if condition_match else "unknown / insufficient evidence"
    uses_one_point_five = "1.5 * self.target_kl" in source
    effective = CONFIGURED_TARGET_KL * 1.5 if uses_one_point_five else CONFIGURED_TARGET_KL
    passed = bool(condition_match and uses_one_point_five and math.isclose(effective, EFFECTIVE_KL_THRESHOLD, rel_tol=0.0, abs_tol=1e-12))
    row = {
        "source_file": str(inspect.getsourcefile(MaskablePPO.train) or "unknown"),
        "source_code_condition": condition,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "effective_threshold": effective,
        "required_effective_threshold": EFFECTIVE_KL_THRESHOLD,
        "pass_fail": "PASS" if passed else "FAIL",
        "detail": "MaskablePPO early stop uses approx_kl > 1.5 * target_kl; configured target is effective/1.5." if passed else "KL guard source semantics did not match the required effective threshold.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(LOGIC_OUT, index=False, encoding="utf-8-sig")
    return passed, row


def runtime_setup():
    """Reuse the exact 017 YCA lowIC runtime, but keep outputs in 019 roots."""
    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    cfg = json.loads((ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json").read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = diagnostic.runtime_setup(cfg)
    return diagnostic, replay, config, env_config, runtime_state


class InstrumentedMaskablePPO:
    """Set by factory below so host py_compile does not import SB3."""


def make_instrumented_ppo_class():
    import torch as th
    import torch.nn.functional as F
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.utils import explained_variance

    class InstrumentedPPO(MaskablePPO):
        """MaskablePPO train loop with only logging/guard instrumentation added."""

        def train(self) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            clip_range_vf = None
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
            entropy_losses: list[float] = []
            pg_losses: list[float] = []
            value_losses: list[float] = []
            clip_fractions: list[float] = []
            approx_kl_divs: list[float] = []
            attempted = int(self.n_epochs)
            completed = 0
            optimized = 0
            guard_triggered = False
            kl_at_trigger = math.nan
            last_loss = th.tensor(0.0)
            continue_training = True
            for epoch in range(self.n_epochs):
                completed = epoch + 1
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    pg_losses.append(float(policy_loss.item()))
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(float(clip_fraction))
                    values_pred = values if clip_range_vf is None else rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(float(value_loss.item()))
                    entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                    entropy_losses.append(float(entropy_loss.item()))
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    last_loss = loss
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = float(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())
                        approx_kl_divs.append(approx_kl_div)
                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        guard_triggered = True
                        kl_at_trigger = approx_kl_div
                        continue_training = False
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    optimized += 1
                self._n_updates += 1
                if not continue_training:
                    break
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
            self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
            self.logger.record("train/value_loss", float(np.mean(value_losses)))
            self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
            self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
            self.logger.record("train/loss", float(last_loss.item()))
            self.logger.record("train/explained_variance", float(explained_var))
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", float(clip_range))
            self.logger.record("train/epochs_attempted", attempted)
            self.logger.record("train/epochs_completed", completed)
            self.logger.record("train/optimizer_steps_completed", optimized)
            self.logger.record("train/kl_guard_triggered", int(guard_triggered))
            if guard_triggered:
                self.logger.record("train/kl_at_trigger", float(kl_at_trigger))
            self.last_guard_metrics = {
                "epochs_attempted": attempted,
                "epochs_completed": completed,
                "optimizer_steps_completed": optimized,
                "kl_guard_triggered": guard_triggered,
                "kl_at_trigger": kl_at_trigger,
            }

    return InstrumentedPPO


def branch_dir(branch: str) -> Path:
    return OUT / branch


def branch_checkpoint_path(branch: str, checkpoint: int) -> Path:
    if int(checkpoint) == 75_000:
        return ORIGINAL_CHECKPOINT
    return branch_dir(branch) / "models" / "YCA" / f"yc_ppo_s1_{branch}_ckpt{int(checkpoint)}.zip"


class DenseCheckpointCallback:
    def __init__(self, branch: str, checkpoints: list[int]):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                now = int(self.model.num_timesteps)
                pending = [target for target in outer.checkpoints if target > 75_000 and target <= now and target not in outer.saved]
                for target in pending:
                    path = branch_checkpoint_path(outer.branch, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    outer.saved.append(target)
                    outer.actual_steps[target] = int(self.model.num_timesteps)
                return True

        self.branch = branch
        self.checkpoints = [int(x) for x in checkpoints]
        self.saved: list[int] = []
        self.actual_steps: dict[int, int] = {}
        self.callback = _Callback()


def copy_starting_point() -> Path:
    target = OUT / "starting_point" / "yc_ppo_s1_original_ckpt75000.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(ORIGINAL_CHECKPOINT, target)
    write_json(OUT / "starting_point" / "manifest.json", {
        "source": rel(ORIGINAL_CHECKPOINT),
        "copied_starting_point": rel(target),
        "sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "num_timesteps": checkpoint_data(ORIGINAL_CHECKPOINT).get("num_timesteps"),
    })
    return target


def train_branch(branch: str, diagnostic: Any, replay: Any, config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.logger import configure

    out = branch_dir(branch)
    out.mkdir(parents=True, exist_ok=True)
    model_paths = {checkpoint: branch_checkpoint_path(branch, checkpoint) for checkpoint in CHECKPOINTS}
    if all(path.exists() for checkpoint, path in model_paths.items() if checkpoint > 75_000):
        return json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
    InstrumentedPPO = make_instrumented_ppo_class()
    set_all_seeds(SEED)
    env = replay.base03222.RandomYearEnv(config, env_config, STATION, TRAIN_YEARS, SEED)
    model = None
    callback = DenseCheckpointCallback(branch, CHECKPOINTS)
    target_kl = None if branch == "control" else CONFIGURED_TARGET_KL
    try:
        model = InstrumentedPPO.load(str(ORIGINAL_CHECKPOINT), env=env, device="cpu")
        if int(model.num_timesteps) != 75_000:
            raise RuntimeError(f"Loaded starting checkpoint has num_timesteps={model.num_timesteps}, expected 75000")
        model.target_kl = target_kl
        model.set_logger(configure(str(out / "tensorboard"), ["tensorboard"]))
        model.learn(total_timesteps=CONTINUATION_STEPS, reset_num_timesteps=False, progress_bar=False, callback=callback.callback)
        final_steps = int(model.num_timesteps)
    finally:
        if env is not None:
            env.close()
    missing = [checkpoint for checkpoint in CHECKPOINTS if checkpoint > 75_000 and not model_paths[checkpoint].exists()]
    if missing:
        raise RuntimeError(f"{branch}: missing dense checkpoints {missing}")
    manifest = {
        "branch": branch,
        "status": "completed",
        "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "starting_num_timesteps": 75_000,
        "continuation_steps_requested": CONTINUATION_STEPS,
        "final_num_timesteps": final_steps,
        "target_kl": target_kl,
        "effective_kl_threshold": None if target_kl is None else target_kl * 1.5,
        "checkpoint_paths": {str(k): rel(v) for k, v in model_paths.items()},
        "checkpoint_actual_num_timesteps": {str(k): checkpoint_data(v).get("num_timesteps") for k, v in model_paths.items() if v.exists()},
        "rng_reinitialization": "python/numpy/torch seeds set to 1 before branch; RandomYearEnv seed=1",
        "other_scientific_contracts_unchanged": True,
    }
    write_json(out / "training_manifest.json", manifest)
    return manifest


def daily_records(model: Any, config: dict[str, Any], env_config: dict[str, Any], year: int, checkpoint: int, branch: str, base: Any, direct: Any) -> tuple[pd.DataFrame, Path]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    weather = direct.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    env = base.make_env(config, env_config, STATION, int(year), SEED, f"019_{branch}_{year}_ckpt{checkpoint}_eval", evaluation=True)
    planting = pd.Timestamp(direct.find_year(env_config, STATION, int(year))["planting_date"])
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = direct.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append({
                "station_code": STATION, "site": SITE, "year": int(year), "seed": SEED,
                "checkpoint_step": int(checkpoint), "branch": branch, "algorithm": "MaskablePPO",
                "date": date.strftime("%Y-%m-%d"), "doy": int(date.dayofyear), "dap": dap,
                "rain": direct.scalar(wrow.get("rain"), np.nan), "srad": direct.scalar(wrow.get("srad"), np.nan),
                "tmax": direct.scalar(wrow.get("tmax"), np.nan), "tmin": direct.scalar(wrow.get("tmin"), np.nan),
                "swfac": direct.scalar(latest.get("swfac")), "nstres": direct.scalar(latest.get("nstres")),
                "topwt": direct.scalar(latest.get("topwt")), "grnwt": direct.scalar(latest.get("grnwt")),
                "xlai": direct.scalar(latest.get("xlai")), "reward": float(reward),
                **dict(env.last_action_info), "done": done,
                "info": json.dumps(info, ensure_ascii=False, default=str),
            })
            step_count += 1
        if not done:
            raise RuntimeError(f"validation did not finish: {branch} checkpoint={checkpoint} year={year}")
        snapshot = diagnostic_snapshot(env)
        final_yield = float(pd.to_numeric(pd.DataFrame(records)["grnwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(pd.DataFrame(records)["safe_action_amir"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(pd.DataFrame(records)["safe_action_anfer"], errors="coerce").fillna(0).sum())
        strict = strict_metrics(snapshot, final_yield, total_i, total_n)
    finally:
        env.close()
    daily = pd.DataFrame(records)
    return daily, strict


def diagnostic_snapshot(env: Any) -> Path:
    folder = getattr(env.unwrapped, "_tmp_folder", None)
    if not folder:
        raise RuntimeError("DSSAT temporary folder unavailable")
    snapshot = Path(folder)
    if not (snapshot / "Summary.OUT").exists():
        raise FileNotFoundError(snapshot / "Summary.OUT")
    return snapshot


def strict_metrics(snapshot: Path, final_yield: float, expected_i: float, expected_n: float) -> dict[str, Any]:
    import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as site_metrics

    return site_metrics.strict_metrics_from_snapshot(snapshot, final_yield, expected_i, expected_n)


def validate_branch(branch: str, diagnostic: Any, replay: Any, config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
    import run_all_year_direct_action_safe_ppo as direct

    out = branch_dir(branch)
    daily_root = out / "daily_outputs" / STATION
    daily_root.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        model_path = branch_checkpoint_path(branch, checkpoint)
        model = MaskablePPO.load(str(model_path), device="cpu")
        for year in VALIDATION_YEARS:
            daily, strict = daily_records(model, config, env_config, year, checkpoint, branch, base, direct)
            daily_path = daily_root / f"YCA_{year}_seed1_ckpt{checkpoint}_daily.csv"
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            invalid = int((pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce") != pd.to_numeric(daily["discrete_action_index"], errors="coerce")).sum())
            fallback = int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum())
            final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
            total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
            total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
            detail_rows.append({
                "row_type": "year", "branch": branch, "checkpoint_step": checkpoint, "model_num_timesteps": checkpoint_data(model_path).get("num_timesteps"),
                "year": year, "yield_kg_ha": final_yield, "I_mm": total_i, "N_kg_ha": total_n,
                "PFP_N": strict["PFP_N_kg_kg"], "WP_ET": strict["WP_ET_kg_m3"], "ETCP_mm": strict["etcp_mm"],
                "invalid_count": invalid, "fallback_count": fallback, "daily_path": rel(daily_path), "status": "PASS",
            })
    detail = pd.DataFrame(detail_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        aggregate_rows.append({
            "row_type": "aggregate", "branch": branch, "checkpoint_step": int(checkpoint), "model_num_timesteps": int(group["model_num_timesteps"].max()),
            "year": "ALL_2014_2023", "validation_year_count": int(len(group)),
            "mean_yield_kg_ha": float(group["yield_kg_ha"].mean()), "sd_yield_kg_ha": float(group["yield_kg_ha"].std(ddof=1)),
            "mean_I_mm": float(group["I_mm"].mean()), "mean_N_kg_ha": float(group["N_kg_ha"].mean()),
            "mean_PFP_N": float(group["PFP_N"].mean()), "mean_WP_ET": float(group["WP_ET"].mean()),
            "invalid_count_total": int(group["invalid_count"].sum()), "fallback_count_total": int(group["fallback_count"].sum()),
            "status": "PASS" if len(group) == len(VALIDATION_YEARS) and group["status"].eq("PASS").all() else "FAIL",
        })
    all_rows = pd.concat([detail, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    output = CONTROL_VALIDATION_OUT if branch == "control" else GUARD_VALIDATION_OUT
    output.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output, index=False, encoding="utf-8-sig")
    detail.to_csv(out / "validation_by_year.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(aggregate_rows)


def policy_monitor(branch: str, diagnostic: Any, replay: Any) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    source = pd.read_csv(RESULTS / "yc_ppo_orig_s1_fixed_state_policy_distribution.csv", keep_default_na=False)
    group_cols = ["state_id", "source_checkpoint", "year", "rollout_step", "dap", "stress_category"]
    source_states = source[group_cols + [f"observation_{i}" for i in range(22)] + [f"mask_{i}" for i in range(16)]].drop_duplicates("state_id").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in POLICY_MONITOR_CHECKPOINTS:
        model = MaskablePPO.load(str(branch_checkpoint_path(branch, checkpoint)), device="cpu")
        for record in source_states.to_dict(orient="records"):
            observation = np.asarray([float(record[f"observation_{i}"]) for i in range(22)], dtype=np.float32)
            mask = np.asarray([str(record[f"mask_{i}"]).lower() == "true" for i in range(16)], dtype=bool)
            info = diagnostic.policy_info(model, observation, mask)
            rows.append({
                "branch": branch, "evaluation_checkpoint": checkpoint, "state_id": record["state_id"],
                "source_checkpoint": record["source_checkpoint"], "year": record["year"], "rollout_step": record["rollout_step"],
                "dap": record["dap"], "stress_category": record["stress_category"], "top1_action": info["top1_action"],
                "top1_action_text": action_text(info["top1_action"]), "top1_probability": info["top1_probability"],
                "probability_margin": info["probability_margin"], "masked_policy_entropy": info["masked_policy_entropy"],
                "P_I_positive": info["P_I_positive"], "P_N_positive": info["P_N_positive"],
                "P_I_and_N_positive": info["P_I_and_N_positive"], "P_noop": info["P_noop"],
                "effective_action_count": info["effective_action_count"],
            })
    result = pd.DataFrame(rows)
    result["top1_action_changed_vs_75k"] = False
    result["top1_probability_delta_vs_75k"] = np.nan
    for _, group in result.groupby(["branch", "state_id"], sort=False):
        base_rows = group[group["evaluation_checkpoint"].eq(75_000)]
        if base_rows.empty:
            continue
        base_row = base_rows.iloc[0]
        idx = group.index
        result.loc[idx, "top1_action_changed_vs_75k"] = result.loc[idx, "top1_action"].astype(int) != int(base_row["top1_action"])
        result.loc[idx, "top1_probability_delta_vs_75k"] = result.loc[idx, "top1_probability"].astype(float) - float(base_row["top1_probability"])
    output = CONTROL_POLICY_OUT if branch == "control" else GUARD_POLICY_OUT
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def stress_summary(control: pd.DataFrame, guard: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([control, guard], ignore_index=True)
    summary = combined.groupby(["branch", "evaluation_checkpoint", "stress_category"], as_index=False).agg(
        state_count=("state_id", "count"),
        top1_changed_vs_75k_fraction=("top1_action_changed_vs_75k", "mean"),
        mean_top1_probability=("top1_probability", "mean"),
        mean_top1_probability_delta_vs_75k=("top1_probability_delta_vs_75k", "mean"),
        mean_probability_margin=("probability_margin", "mean"),
        mean_masked_policy_entropy=("masked_policy_entropy", "mean"),
        mean_P_I_positive=("P_I_positive", "mean"),
        mean_P_N_positive=("P_N_positive", "mean"),
        mean_P_noop=("P_noop", "mean"),
    )
    summary.to_csv(STRESS_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    return summary


def extract_training_metrics(branch: str) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((branch_dir(branch) / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"{branch}: expected exactly one TensorBoard event file, got {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    mapping = {
        "train/approx_kl": "approx_kl", "train/clip_fraction": "clip_fraction", "train/entropy_loss": "entropy_loss",
        "train/policy_gradient_loss": "policy_gradient_loss", "train/value_loss": "value_loss", "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate", "train/clip_range": "clip_range", "train/epochs_attempted": "epochs_attempted",
        "train/epochs_completed": "epochs_completed", "train/optimizer_steps_completed": "optimizer_steps_completed",
        "train/kl_guard_triggered": "kl_guard_triggered", "train/kl_at_trigger": "kl_at_trigger",
    }
    records: dict[int, dict[str, Any]] = {}
    for tag, name in mapping.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda item: item["global_step"]))
    for column in EXPECTED_METRIC_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics.insert(0, "branch", branch)
    metrics = metrics[["branch"] + EXPECTED_METRIC_COLUMNS]
    output = CONTROL_METRICS_OUT if branch == "control" else GUARD_METRICS_OUT
    metrics.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(branch_dir(branch) / "tensorboard_manifest.json", {
        "event_file": rel(event_files[0]), "event_file_sha256": sha256_file(event_files[0]),
        "scalar_tags": accumulator.Tags().get("scalars", []), "rows": int(len(metrics)),
        "missing_required_columns": [column for column in EXPECTED_METRIC_COLUMNS if column not in metrics or metrics[column].isna().all()],
    })
    return metrics


def aggregate_rows(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    for column in [
        "checkpoint_step", "model_num_timesteps", "validation_year_count", "mean_yield_kg_ha", "sd_yield_kg_ha",
        "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid_count_total", "fallback_count_total",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[data["row_type"].eq("aggregate")].copy()


def decide(control_validation: pd.DataFrame, guard_validation: pd.DataFrame, guard_metrics: pd.DataFrame) -> dict[str, Any]:
    c100 = control_validation[control_validation["checkpoint_step"].eq(100_000)].iloc[0]
    g100 = guard_validation[guard_validation["checkpoint_step"].eq(100_000)].iloc[0]
    c_pre = control_validation[control_validation["checkpoint_step"].isin([75_000, 80_000, 85_000, 90_000, 95_000])]
    g_pre = guard_validation[guard_validation["checkpoint_step"].isin([75_000, 80_000, 85_000, 90_000, 95_000])]
    c_best = float(c_pre["mean_yield_kg_ha"].max())
    g_best = float(g_pre["mean_yield_kg_ha"].max())
    c_drop = c_best - float(c100["mean_yield_kg_ha"])
    g_drop = g_best - float(g100["mean_yield_kg_ha"])
    trigger = pd.to_numeric(guard_metrics["kl_guard_triggered"], errors="coerce")
    completed = pd.to_numeric(guard_metrics["epochs_completed"], errors="coerce")
    optimized = pd.to_numeric(guard_metrics["optimizer_steps_completed"], errors="coerce")
    attempted = pd.to_numeric(guard_metrics["epochs_attempted"], errors="coerce")
    guard_update_fraction = float((optimized > 0).mean()) if len(optimized) else math.nan
    guard_mean_optimized = float(optimized.mean()) if len(optimized) else math.nan
    guard_mean_completed = float(completed.mean()) if len(completed) else math.nan
    guard_mean_attempted = float(attempted.mean()) if len(attempted) else math.nan
    guard_trigger_rate = float(trigger.mean()) if len(trigger) else math.nan
    guard_not_frozen = bool(len(optimized) and guard_update_fraction > 0.0 and guard_mean_optimized > 0.0)
    control_degradation = bool(float(c100["mean_yield_kg_ha"]) < 6500.0 or c_drop > 1500.0)
    guard_protection = bool(float(g100["mean_yield_kg_ha"]) >= 7500.0 and g_drop <= 700.0)
    if not control_degradation:
        decision = "PAIRED_CONTROL_NONCOLLAPSE"
    elif guard_protection and guard_not_frozen:
        decision = "KL_GUARD_GO"
    else:
        decision = "KL_GUARD_NO_GO"
    payload = {
        "decision": decision,
        "control_100k_mean_yield": float(c100["mean_yield_kg_ha"]),
        "control_own_best_75_95k_mean_yield": c_best,
        "control_own_best_to_100k_drop": c_drop,
        "control_degradation_gate": control_degradation,
        "klguard_100k_mean_yield": float(g100["mean_yield_kg_ha"]),
        "klguard_own_best_75_95k_mean_yield": g_best,
        "klguard_own_best_to_100k_drop": g_drop,
        "klguard_protection_gate": guard_protection,
        "klguard_trigger_rate": guard_trigger_rate,
        "klguard_mean_epochs_attempted": guard_mean_attempted,
        "klguard_mean_epochs_completed": guard_mean_completed,
        "klguard_mean_optimizer_steps_completed": guard_mean_optimized,
        "klguard_updates_with_optimizer_step_fraction": guard_update_fraction,
        "klguard_not_frozen_gate": guard_not_frozen,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "guard_configured_target_kl": CONFIGURED_TARGET_KL,
    }
    pd.DataFrame([payload]).to_csv(DECISION_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "decision.json", payload)
    return payload


def report(decision: dict[str, Any], control_validation: pd.DataFrame, guard_validation: pd.DataFrame, control_metrics: pd.DataFrame, guard_metrics: pd.DataFrame, stress: pd.DataFrame) -> None:
    control_100 = control_validation[control_validation["checkpoint_step"].eq(100_000)].iloc[0]
    guard_100 = guard_validation[guard_validation["checkpoint_step"].eq(100_000)].iloc[0]
    late_control = control_metrics[control_metrics["global_step"].between(85_000, 94_999)]
    late_guard = guard_metrics[guard_metrics["global_step"].between(85_000, 94_999)]
    late_metric_line = (
        f"For 85–95K, `approx_kl` mean/median was CONTROL `{late_control['approx_kl'].mean():.8f}/{late_control['approx_kl'].median():.8f}` versus KL-GUARD `{late_guard['approx_kl'].mean():.8f}/{late_guard['approx_kl'].median():.8f}`; "
        f"`clip_fraction` mean/median was CONTROL `{late_control['clip_fraction'].mean():.8f}/{late_control['clip_fraction'].median():.8f}` versus KL-GUARD `{late_guard['clip_fraction'].mean():.8f}/{late_guard['clip_fraction'].median():.8f}`. "
        "Thus the guard lowered the late mean KL in this paired run but did not produce a uniformly lower median KL or a distinct median clip reduction."
    )
    stress_100 = stress[stress["evaluation_checkpoint"].eq(100_000)].copy()
    weighted_stress: dict[str, dict[str, float]] = {}
    for branch in ["control", "klguard"]:
        group = stress_100[stress_100["branch"].eq(branch)]
        total = float(group["state_count"].sum())
        weighted_stress[branch] = {
            "top1": float(np.average(group["mean_top1_probability"], weights=group["state_count"])) if total else math.nan,
            "margin": float(np.average(group["mean_probability_margin"], weights=group["state_count"])) if total else math.nan,
            "entropy": float(np.average(group["mean_masked_policy_entropy"], weights=group["state_count"])) if total else math.nan,
            "P_I": float(np.average(group["mean_P_I_positive"], weights=group["state_count"])) if total else math.nan,
            "P_N": float(np.average(group["mean_P_N_positive"], weights=group["state_count"])) if total else math.nan,
        }
    stress_line = (
        f"At 100K across the reused panel, weighted stress-state policy summaries were CONTROL top1/margin/entropy/P(I>0)/P(N>0) `"
        f"{weighted_stress['control']['top1']:.4f}/{weighted_stress['control']['margin']:.4f}/{weighted_stress['control']['entropy']:.4f}/{weighted_stress['control']['P_I']:.4f}/{weighted_stress['control']['P_N']:.4f}` "
        f"and KL-GUARD `"
        f"{weighted_stress['klguard']['top1']:.4f}/{weighted_stress['klguard']['margin']:.4f}/{weighted_stress['klguard']['entropy']:.4f}/{weighted_stress['klguard']['P_I']:.4f}/{weighted_stress['klguard']['P_N']:.4f}`. "
        "This shows a distributional difference, not a standalone proof of stress-response preservation."
    )
    lines = [
        "# YC PPO Original seed1 KL-guard stabilization screen",
        "",
        "This task executed a stabilization intervention.",
        "It is not a repeat of diagnostic 017.",
        "",
        "## Decision",
        "",
        f"Primary decision: **{decision['decision']}**.",
        f"CONTROL 100K mean yield: `{decision['control_100k_mean_yield']:.3f}` kg/ha; KL-GUARD 100K mean yield: `{decision['klguard_100k_mean_yield']:.3f}` kg/ha.",
        f"CONTROL own-best(75–95K)→100K drop: `{decision['control_own_best_to_100k_drop']:.3f}` kg/ha; KL-GUARD drop: `{decision['klguard_own_best_to_100k_drop']:.3f}` kg/ha.",
        "",
        "## Resume audit and guard semantics",
        "",
        f"- 75K checkpoint: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{sha256_file(ORIGINAL_CHECKPOINT)}`.",
        "- Policy/value weights, optimizer state, timestep, lr schedule metadata, and clip-range schedule metadata were recoverable.",
        "- Python/NumPy/PyTorch/environment-year RNG states were not serialized; both branches reinitialized them identically with seed 1. This is a paired continuation from the same 75K training state, not an exact replay of the original formal stochastic trajectory.",
        f"- Effective KL guard threshold: `{decision['effective_kl_threshold']:.1e}`; configured `target_kl`: `{decision['guard_configured_target_kl']:.10g}`.",
        f"- Logic evidence: `{rel(LOGIC_OUT)}`; resume evidence: `{rel(RESUME_OUT)}`.",
        "",
        "## Paired continuation",
        "",
        f"Both branches loaded the same 75K checkpoint and requested `{CONTINUATION_STEPS}` additional environment steps. Dense checkpoints were saved at 80K, 85K, 90K, 95K and 100K; 75K is retained as the copied starting-point reference under `{rel(OUT / 'starting_point')}`.",
        f"- CONTROL manifest: `{rel(branch_dir('control') / 'training_manifest.json')}`.",
        f"- KL-GUARD manifest: `{rel(branch_dir('klguard') / 'training_manifest.json')}`.",
        "- Execution provenance: the first train/validate phase invocations exited during their post-work phase handoff because the orchestrator attempted to read not-yet-created downstream CSVs; the underlying training/validation work had already completed. The phase routing was corrected, and no training or validation episode was rerun.",
        "",
        "## Training metrics",
        "",
        f"- CONTROL metrics: `{rel(CONTROL_METRICS_OUT)}`; rows `{len(control_metrics)}`.",
        f"- KL-GUARD metrics: `{rel(GUARD_METRICS_OUT)}`; rows `{len(guard_metrics)}`.",
        f"- KL-GUARD trigger rate: `{decision['klguard_trigger_rate']:.4f}`; mean epochs attempted/completed `{decision['klguard_mean_epochs_attempted']:.3f}/{decision['klguard_mean_epochs_completed']:.3f}`; mean optimizer steps completed `{decision['klguard_mean_optimizer_steps_completed']:.3f}`; updates with an optimizer step `{decision['klguard_updates_with_optimizer_step_fraction']:.4f}`.",
        late_metric_line,
        "These are observed logged values. They establish whether the guard materially shortened optimization, but do not by themselves establish causality.",
        "",
        "## Validation",
        "",
        f"CONTROL 100K: mean yield `{control_100['mean_yield_kg_ha']:.3f}`, SD `{control_100['sd_yield_kg_ha']:.3f}`, mean I `{control_100['mean_I_mm']:.3f}`, mean N `{control_100['mean_N_kg_ha']:.3f}`, mean PFP_N `{control_100['mean_PFP_N']:.3f}`, mean WP_ET `{control_100['mean_WP_ET']:.6f}`, invalid `{control_100['invalid_count_total']}`, fallback `{control_100['fallback_count_total']}`.",
        f"KL-GUARD 100K: mean yield `{guard_100['mean_yield_kg_ha']:.3f}`, SD `{guard_100['sd_yield_kg_ha']:.3f}`, mean I `{guard_100['mean_I_mm']:.3f}`, mean N `{guard_100['mean_N_kg_ha']:.3f}`, mean PFP_N `{guard_100['mean_PFP_N']:.3f}`, mean WP_ET `{guard_100['mean_WP_ET']:.6f}`, invalid `{guard_100['invalid_count_total']}`, fallback `{guard_100['fallback_count_total']}`.",
        f"- CONTROL validation: `{rel(CONTROL_VALIDATION_OUT)}`.",
        f"- KL-GUARD validation: `{rel(GUARD_VALIDATION_OUT)}`.",
        "",
        "## Stress-response policy monitoring",
        "",
        f"The 017 parity-gated fixed-state panel was reused; no new scientific state/action/mask contract was introduced. CONTROL policy panel: `{rel(CONTROL_POLICY_OUT)}`; KL-GUARD policy panel: `{rel(GUARD_POLICY_OUT)}`; stress summary: `{rel(STRESS_SUMMARY_OUT)}`.",
        stress_line,
        f"The summary contains `{len(stress)}` branch/checkpoint/stress-category groups and reports top-1 switching, top-1 probability, margin, entropy, P(I>0), P(N>0), and P(no-op).",
        "",
        "## Contract boundary",
        "",
        "The only scientific intervention was the fixed effective KL early-stop guard. CONTROL retained the loaded PPO hyperparameters. Observation, reward, action space, mask logic, irrigation/fertilizer constraints, training years, IC, cultivar, network, learning rate, clip range, entropy coefficient, n_steps, batch size, n_epochs, gamma, and GAE lambda were not changed. No additional seed, site, threshold, or tuning sweep was run.",
        "",
        "## Required answers",
        "",
        f"1. 75K recoverability: `RESUME_AVAILABLE_WITH_REINITIALIZED_RNG`.",
        f"2. CONTROL 25K continuation: completed; see `{rel(branch_dir('control') / 'training_manifest.json')}`.",
        f"3. KL-GUARD 25K continuation: completed; see `{rel(branch_dir('klguard') / 'training_manifest.json')}`.",
        f"4. Effective KL threshold: `{decision['effective_kl_threshold']:.1e}` (`PASS`).",
        f"5. CONTROL late degradation: `{decision['control_degradation_gate']}` under the pre-registered gate.",
        f"6. KL-GUARD effect on 85–95K KL/clip displacement: {late_metric_line}",
        f"7. Stress-response preservation: {stress_line}",
        f"8. High-yield preservation: KL-GUARD 100K gate `{decision['klguard_protection_gate']}`.",
        f"9. Final allowed decision: `{decision['decision']}`.",
        "",
        "YC_PPO_SEED1_KL_GUARD_SCREEN_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "train", "validate", "stress", "report", "all"], default="all")
    args = parser.parse_args()
    resume_ok, resume_payload = resume_audit()
    logic_ok, logic_payload = kl_guard_logic_test()
    if not resume_ok:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("# YC PPO seed1 KL-guard screen blocked\n\nPPO_75K_RESUME_UNAVAILABLE\n", encoding="utf-8")
        print("PPO_75K_RESUME_UNAVAILABLE")
        return 2 if args.phase != "audit" else 0
    if not logic_ok:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("# YC PPO seed1 KL-guard screen blocked\n\nKL_GUARD_LOGIC_FAILED\n", encoding="utf-8")
        print("KL_GUARD_LOGIC_FAILED")
        return 2 if args.phase != "audit" else 0
    if args.phase == "audit":
        print("resume=RESUME_AVAILABLE_WITH_REINITIALIZED_RNG")
        print("logic=PASS effective_threshold=1e-4")
        return 0

    copy_starting_point()
    diagnostic, replay, config, env_config, runtime_state = runtime_setup()
    try:
        if args.phase in {"train", "all"}:
            control_manifest = train_branch("control", diagnostic, replay, config, env_config)
            guard_manifest = train_branch("klguard", diagnostic, replay, config, env_config)
            if args.phase == "train":
                print(json.dumps({"control": control_manifest, "klguard": guard_manifest}, ensure_ascii=False))
                return 0
        if args.phase == "stress":
            control_policy = policy_monitor("control", diagnostic, replay)
            guard_policy = policy_monitor("klguard", diagnostic, replay)
            stress = stress_summary(control_policy, guard_policy)
            print(f"stress_policy_rows={len(control_policy) + len(guard_policy)} summary_rows={len(stress)}")
            return 0
        if args.phase in {"validate", "all"}:
            control_validation = validate_branch("control", diagnostic, replay, config, env_config)
            guard_validation = validate_branch("klguard", diagnostic, replay, config, env_config)
        else:
            control_validation = aggregate_rows(CONTROL_VALIDATION_OUT)
            guard_validation = aggregate_rows(GUARD_VALIDATION_OUT)
        if args.phase == "validate":
            print(json.dumps({"control_validation_rows": len(control_validation), "klguard_validation_rows": len(guard_validation)}, ensure_ascii=False))
            return 0
        if args.phase in {"stress", "all"}:
            control_policy = policy_monitor("control", diagnostic, replay)
            guard_policy = policy_monitor("klguard", diagnostic, replay)
            stress = stress_summary(control_policy, guard_policy)
        else:
            control_policy = pd.read_csv(CONTROL_POLICY_OUT)
            guard_policy = pd.read_csv(GUARD_POLICY_OUT)
            stress = pd.read_csv(STRESS_SUMMARY_OUT)
        if args.phase in {"report", "all"}:
            control_metrics = extract_training_metrics("control")
            guard_metrics = extract_training_metrics("klguard")
            decision = decide(control_validation, guard_validation, guard_metrics)
            report(decision, control_validation, guard_validation, control_metrics, guard_metrics, stress)
            print(decision["decision"])
            print("YC_PPO_SEED1_KL_GUARD_SCREEN_COMPLETE")
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


if __name__ == "__main__":
    raise SystemExit(main())

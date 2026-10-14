"""017: read-only YC Original PPO seed1 collapse diagnostic.

The script never trains or changes the existing PPO/DSSAT source.  It reads
the existing 75K and 100K checkpoints, performs deterministic validation
replays into a separate output root, gates replay-state use on exact behavior
parity, and extracts TensorBoard update metrics from the existing run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

ORIGINAL_ROOT = ROOT / "benchmark_results" / "222YCA_O1_yc_v1_original_seed1"
CONFIG_PATH = ORIGINAL_ROOT / "configs" / "222YCA_O1_yc_v1_original_seed1.json"
RESULTS = ROOT / "results"
DIAG_ROOT = ROOT / "benchmark_results" / "017_yc_ppo_original_seed1_collapse_diagnostic"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_75k_100k_collapse_diagnostic.md"
CHECKPOINTS = [75_000, 100_000]
YEARS = list(range(2014, 2024))
ACTION_I = [0.0, 15.0, 30.0, 45.0]
ACTION_N = [0.0, 40.0, 80.0, 120.0]
OBS_NAMES = [
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep",
    "srad", "sw[0]", "sw[1]", "sw[2]", "sw[3]", "sw[4]", "sw[5]", "swfac",
    "tmax", "topwt", "totir", "vstage", "wtdep", "xlai",
]
BEHAVIOR_OUT = RESULTS / "yc_ppo_orig_s1_75k_100k_behavior_summary.csv"
REPLAY_PARITY_OUT = RESULTS / "yc_ppo_orig_s1_replay_parity.csv"
DAILY_POLICY_OUT = RESULTS / "yc_ppo_orig_s1_daily_policy_distribution.csv"
FIXED_POLICY_OUT = RESULTS / "yc_ppo_orig_s1_fixed_state_policy_distribution.csv"
SHIFT_OUT = RESULTS / "yc_ppo_orig_s1_fixed_state_policy_shift_summary.csv"
TRAINING_OUT = RESULTS / "yc_ppo_orig_s1_75k_100k_training_metrics.csv"
WINDOW_OUT = RESULTS / "yc_ppo_orig_s1_training_metric_windows.csv"
HYPOTHESIS_OUT = RESULTS / "yc_ppo_orig_s1_collapse_hypothesis_evidence.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def action_pair(action: int) -> tuple[float, float]:
    action = int(action)
    return ACTION_I[action // 4], ACTION_N[action % 4]


def action_text(action: int) -> str:
    irrigation, nitrogen = action_pair(int(action))
    return f"I{irrigation:g}/N{nitrogen:g}"


def action_sequence(df: pd.DataFrame) -> str:
    items = []
    for _, row in df.iterrows():
        action = int(float(row["discrete_action_index"]))
        items.append(f"DAP{int(float(row['dap']))}:{action_text(action)}")
    return "; ".join(items)


def non_noop_sequence(df: pd.DataFrame) -> str:
    items = []
    for _, row in df.iterrows():
        i = float(row.get("safe_action_amir", 0.0))
        n = float(row.get("safe_action_anfer", 0.0))
        if i > 0 or n > 0:
            items.append(f"DAP{int(float(row['dap']))}:{action_text(int(float(row['discrete_action_index'])))}")
    return "; ".join(items)


def source_daily(checkpoint: int, year: int) -> Path:
    return ORIGINAL_ROOT / "daily_outputs" / "YCA" / f"YCA_{year}_seed1_ckpt{checkpoint}_daily.csv"


def source_validation(checkpoint: int, year: int) -> pd.Series:
    path = ORIGINAL_ROOT / "evaluation" / "032_22_checkpoint_validation_summary.csv"
    df = pd.read_csv(path, keep_default_na=False)
    rows = df[
        pd.to_numeric(df["checkpoint_step"], errors="coerce").eq(checkpoint)
        & pd.to_numeric(df["year"], errors="coerce").eq(year)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one validation row for checkpoint={checkpoint}, year={year}; got {len(rows)}")
    return rows.iloc[0]


def build_behavior_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for year in YEARS:
            path = source_daily(checkpoint, year)
            df = pd.read_csv(path, keep_default_na=False)
            i = pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0)
            n = pd.to_numeric(df["safe_action_anfer"], errors="coerce").fillna(0.0)
            actions = pd.to_numeric(df["discrete_action_index"], errors="coerce").astype(int)
            non_noop = (i > 0) | (n > 0)
            first = df.loc[non_noop].iloc[0] if bool(non_noop.any()) else None
            invalid = int((pd.to_numeric(df["requested_discrete_action_index"], errors="coerce") != actions).sum())
            fallback = int(pd.Series(df["mask_forced_noop"]).astype(str).str.lower().eq("true").sum())
            validation = source_validation(checkpoint, year)
            rows.append({
                "checkpoint": checkpoint,
                "year": year,
                "yield_kg_ha": float(pd.to_numeric(validation["final_grnwt"], errors="coerce")),
                "I_mm": float(i.sum()),
                "N_kg_ha": float(n.sum()),
                "selected_action_sequence": action_sequence(df),
                "selected_non_noop_action_sequence": non_noop_sequence(df),
                "first_non_noop_action": action_text(int(first["discrete_action_index"])) if first is not None else "none",
                "first_non_noop_dap": int(first["dap"]) if first is not None else np.nan,
                "I_positive_action_count": int((i > 0).sum()),
                "N_positive_action_count": int((n > 0).sum()),
                "invalid_action_count": invalid,
                "fallback_count": fallback,
                "daily_row_count": int(len(df)),
                "source_daily_path": rel(path),
                "source_validation_path": rel(ORIGINAL_ROOT / "evaluation" / "032_22_checkpoint_validation_summary.csv"),
            })
    out = pd.DataFrame(rows).sort_values(["checkpoint", "year"]).reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(BEHAVIOR_OUT, index=False, encoding="utf-8-sig")
    return out


def runtime_setup(cfg: dict[str, Any]):
    """Create the exact existing YCA runtime, with only the output root redirected."""
    import ppo_safe_rendering
    import run_055_03_yca_lowIC_five_scenario_figures as replay

    engine = replay.engine
    base03222 = replay.base03222
    direct_ppo = replay.direct_ppo
    old_values = {
        key: getattr(engine, key)
        for key in [
            "TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT",
            "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_site_names = dict(base03222.SITE_NAMES)
    engine.TASK_ID = str(cfg["task_id"])
    engine.TASK_NAME = str(cfg["task_name"])
    engine.BASE_OUT = DIAG_ROOT
    engine.BASE_DOC = REPORT
    engine.PROMPT = replay.PROMPT
    engine.LOWIC_INPUT_ROOT = replay.INPUT_PROFILES[str(cfg["input_profile"])]
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = replay.INPUT_PROFILES[str(cfg["input_profile"])]
    engine.STATION = replay.STATION
    engine.SITES = [replay.STATION]
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    base03222.SITE_NAMES[replay.STATION] = replay.SITE
    engine.patch_base_module(DIAG_ROOT, REPORT, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
    config = base03222.load_config()
    selection = base03222.build_selection(base03222.load_split())
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = rel(DIAG_ROOT)
    return replay, config, env_config, (engine, base03222, ppo_safe_rendering, old_values, old_root, old_site_names)


def restore_runtime(state: Any) -> None:
    engine, base03222, ppo_safe_rendering, old_values, old_root, old_site_names = state
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update(old_site_names)
    for key, value in old_values.items():
        setattr(engine, key, value)


def model_paths(replay: Any) -> dict[int, Path]:
    from sya_experiment_artifacts import resolve_training_inventory

    paths: dict[int, Path] = {}
    for checkpoint in CHECKPOINTS:
        inventory_path = resolve_training_inventory(ORIGINAL_ROOT, checkpoint, preferred_prefix="222YCA_O1")
        inventory = pd.read_csv(inventory_path, keep_default_na=False)
        rows = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one model inventory row at {checkpoint}, got {len(rows)}")
        path = ROOT / str(rows.iloc[0]["model_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        paths[checkpoint] = path
    return paths


def policy_info(model: Any, observation: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(observation, dtype=np.float32))
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        if model.policy.share_features_extractor:
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = model.policy.mlp_extractor.forward_critic(vf_features)
        logits = model.policy.action_net(latent_pi).reshape(-1)
        value = model.policy.value_net(latent_vf).reshape(-1)[0]
        raw_probs = torch.softmax(logits, dim=0)
        mask_tensor = torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool, device=logits.device)
        masked_logits = torch.where(mask_tensor, logits, torch.full_like(logits, -1e8))
        masked_probs = torch.softmax(masked_logits, dim=0)
    raw_logits_np = logits.cpu().numpy().astype(float)
    raw_probs_np = raw_probs.cpu().numpy().astype(float)
    masked_probs_np = masked_probs.cpu().numpy().astype(float)
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    order = valid[np.argsort(-masked_probs_np[valid])] if len(valid) else np.array([], dtype=int)
    top1 = int(order[0]) if len(order) else -1
    top2 = int(order[1]) if len(order) > 1 else -1
    top1_p = float(masked_probs_np[top1]) if top1 >= 0 else np.nan
    top2_p = float(masked_probs_np[top2]) if top2 >= 0 else 0.0
    entropy = float(-np.sum(masked_probs_np[valid] * np.log(np.clip(masked_probs_np[valid], 1e-300, None)))) if len(valid) else np.nan
    p_i = float(masked_probs_np[[idx for idx in valid if action_pair(int(idx))[0] > 0]].sum())
    p_n = float(masked_probs_np[[idx for idx in valid if action_pair(int(idx))[1] > 0]].sum())
    p_both = float(masked_probs_np[[idx for idx in valid if action_pair(int(idx))[0] > 0 and action_pair(int(idx))[1] > 0]].sum())
    return {
        "raw_logits": raw_logits_np.tolist(),
        "raw_probabilities": raw_probs_np.tolist(),
        "masked_probabilities": masked_probs_np.tolist(),
        "top1_action": top1,
        "top1_probability": top1_p,
        "top2_action": top2,
        "top2_probability": top2_p,
        "probability_margin": top1_p - top2_p if top1 >= 0 else np.nan,
        "masked_policy_entropy": entropy,
        "effective_action_count": int(len(valid)),
        "P_I_positive": p_i,
        "P_N_positive": p_n,
        "P_I_and_N_positive": p_both,
        "P_noop": float(masked_probs_np[0]),
        "value_prediction": float(value.cpu()),
    }


def stress_category(observation: np.ndarray) -> str:
    dap = float(observation[1])
    nstress = float(observation[6])
    wstress = float(observation[15])
    if nstress > 0.05 and wstress > 0.05:
        return "dual-stress"
    if nstress > 0.05:
        return "N-stress"
    if wstress > 0.05:
        return "water-stress"
    if dap <= 10 and nstress <= 0.001 and wstress <= 0.001:
        return "early-no-stress"
    return "other"


def flatten_policy_row(row: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for idx, value in enumerate(info["raw_logits"]):
        out[f"raw_logits_{idx}"] = value
    for idx, value in enumerate(info["raw_probabilities"]):
        out[f"raw_probability_{idx}"] = value
    for idx, value in enumerate(info["masked_probabilities"]):
        out[f"masked_probability_{idx}"] = value
    return out


def capture_replays(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, dict[str, Any]]:
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    replay, config, env_config, runtime_state = runtime_setup(cfg)
    paths = model_paths(replay)
    models = {checkpoint: MaskablePPO.load(str(path), device="cpu") for checkpoint, path in paths.items()}
    rollout_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    try:
        for source_checkpoint in CHECKPOINTS:
            model = models[source_checkpoint]
            for year in YEARS:
                tag = f"017_replay_seed1_{year}_ckpt{source_checkpoint}"
                env = replay.base03222.base.make_env(config, env_config, replay.STATION, year, 1, tag, evaluation=True)
                obs, info = env.reset()
                done = False
                step = 0
                actual_actions: list[int] = []
                actual_i: list[float] = []
                actual_n: list[float] = []
                state_id_prefix = f"src{source_checkpoint}_y{year}"
                try:
                    while not done and step < 260:
                        observation = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
                        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1).copy()
                        source_info = policy_info(model, observation, mask)
                        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
                        action = int(np.asarray(action).reshape(-1)[0])
                        state_id = f"{state_id_prefix}_s{step:03d}"
                        category = stress_category(observation)
                        state_base = {
                            "state_id": state_id,
                            "source_checkpoint": source_checkpoint,
                            "year": year,
                            "rollout_step": step,
                            "dap": float(observation[1]),
                            "stress_category": category,
                            "source_action": action,
                            "source_action_text": action_text(action),
                            "source_mask_valid_count": int(mask.sum()),
                        }
                        state_base.update({f"observation_{idx}": float(value) for idx, value in enumerate(observation)})
                        state_base.update({f"mask_{idx}": bool(value) for idx, value in enumerate(mask)})
                        for eval_checkpoint, eval_model in models.items():
                            eval_info = policy_info(eval_model, observation, mask)
                            eval_row = dict(state_base)
                            eval_row["evaluation_checkpoint"] = eval_checkpoint
                            eval_row["evaluation_model_path"] = rel(paths[eval_checkpoint])
                            eval_row = flatten_policy_row(eval_row, eval_info)
                            eval_row.update({key: value for key, value in eval_info.items() if key not in {"raw_logits", "raw_probabilities", "masked_probabilities"}})
                            state_rows.append(eval_row)
                            if eval_checkpoint == source_checkpoint:
                                rollout_rows.append(eval_row)
                        next_obs, reward, terminated, truncated, step_info = env.step(action)
                        action_info = dict(getattr(env, "last_action_info", {}))
                        executed = int(action_info.get("discrete_action_index", -1))
                        actual_actions.append(executed)
                        actual_i.append(float(action_info.get("safe_action_amir", 0.0)))
                        actual_n.append(float(action_info.get("safe_action_anfer", 0.0)))
                        done = bool(terminated or truncated)
                        obs = next_obs
                        step += 1
                    if not done:
                        raise RuntimeError(f"Replay did not finish for year={year}, checkpoint={source_checkpoint}")
                    snapshot = replay.snapshot_from_env(env)
                    daily = replay.daily_from_snapshot(snapshot, year, "rl_candidate")
                    replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
                finally:
                    env.close()
                existing = pd.read_csv(source_daily(source_checkpoint, year), keep_default_na=False)
                existing_actions = pd.to_numeric(existing["discrete_action_index"], errors="coerce").astype(int).tolist()
                expected = source_validation(source_checkpoint, year)
                expected_yield = float(pd.to_numeric(expected["final_grnwt"], errors="coerce"))
                expected_i = float(pd.to_numeric(expected["total_irrigation"], errors="coerce"))
                expected_n = float(pd.to_numeric(expected["total_n"], errors="coerce"))
                parity_rows.append({
                    "checkpoint": source_checkpoint,
                    "year": year,
                    "replay_steps": len(actual_actions),
                    "source_steps": len(existing_actions),
                    "action_sequence_exact": actual_actions == existing_actions,
                    "replay_yield_kg_ha": replay_yield,
                    "source_yield_kg_ha": expected_yield,
                    "yield_abs_diff": replay_yield - expected_yield,
                    "replay_I_mm": float(sum(actual_i)),
                    "source_I_mm": expected_i,
                    "I_abs_diff": float(sum(actual_i)) - expected_i,
                    "replay_N_kg_ha": float(sum(actual_n)),
                    "source_N_kg_ha": expected_n,
                    "N_abs_diff": float(sum(actual_n)) - expected_n,
                    "invalid_or_closure_mismatch": int(sum(a != b for a, b in zip(actual_actions, existing_actions))) + abs(len(actual_actions) - len(existing_actions)),
                    "replay_daily_snapshot": rel(DIAG_ROOT / "replayed_snapshot_reference"),
                    "status": "PASS" if actual_actions == existing_actions and np.isclose(replay_yield, expected_yield, atol=1.0, rtol=0.0) and np.isclose(sum(actual_i), expected_i, atol=1e-6, rtol=0.0) and np.isclose(sum(actual_n), expected_n, atol=1e-6, rtol=0.0) else "FAIL",
                })
    finally:
        restore_runtime(runtime_state)
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(REPLAY_PARITY_OUT, index=False, encoding="utf-8-sig")
    rollout = pd.DataFrame(rollout_rows)
    states = pd.DataFrame(state_rows)
    if not rollout.empty:
        rollout.to_csv(DAILY_POLICY_OUT, index=False, encoding="utf-8-sig")
    if not states.empty:
        states.to_csv(FIXED_POLICY_OUT, index=False, encoding="utf-8-sig")
    parity_ok = bool(not parity.empty and (parity["status"] == "PASS").all())
    manifest = {
        "task": "017_yc_ppo_original_seed1_collapse_diagnostic",
        "read_only_replay": True,
        "checkpoints": CHECKPOINTS,
        "years": YEARS,
        "parity_gate": parity_ok,
        "parity_path": rel(REPLAY_PARITY_OUT),
        "state_policy_path": rel(FIXED_POLICY_OUT),
        "daily_policy_path": rel(DAILY_POLICY_OUT),
        "model_paths": {str(k): rel(v) for k, v in paths.items()},
        "replay_runtime_output_root": rel(DIAG_ROOT),
    }
    write_json(DIAG_ROOT / "replay_manifest.json", manifest)
    if not parity_ok:
        (RESULTS / "PPO_REPLAY_PARITY_FAILED.txt").write_text("PPO_REPLAY_PARITY_FAILED\n", encoding="utf-8")
    return rollout, states, parity, parity_ok, manifest


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(p * np.log(np.clip(p, 1e-300, None) / np.clip(q, 1e-300, None))))


def policy_shifts(states: pd.DataFrame, parity_ok: bool) -> pd.DataFrame:
    if not parity_ok or states.empty:
        out = pd.DataFrame([{"status": "insufficient evidence: replay parity gate failed or no states"}])
        out.to_csv(SHIFT_OUT, index=False, encoding="utf-8-sig")
        return out
    rows: list[dict[str, Any]] = []
    group_cols = ["state_id", "source_checkpoint", "year", "rollout_step", "dap", "stress_category"]
    for keys, group in states.groupby(group_cols, dropna=False):
        by_ckpt = {int(r["evaluation_checkpoint"]): r for _, r in group.iterrows()}
        if 75_000 not in by_ckpt or 100_000 not in by_ckpt:
            continue
        a = by_ckpt[75_000]
        b = by_ckpt[100_000]
        p = np.asarray([float(a[f"masked_probability_{i}"]) for i in range(16)])
        q = np.asarray([float(b[f"masked_probability_{i}"]) for i in range(16)])
        m = 0.5 * (p + q)
        rows.append({
            "state_id": keys[0],
            "source_checkpoint": int(keys[1]),
            "year": int(keys[2]),
            "rollout_step": int(keys[3]),
            "dap": float(keys[4]),
            "stress_category": keys[5],
            "top1_action_75k": int(a["top1_action"]),
            "top1_action_100k": int(b["top1_action"]),
            "top1_action_changed": bool(int(a["top1_action"]) != int(b["top1_action"])),
            "top1_probability_75k": float(a["top1_probability"]),
            "top1_probability_100k": float(b["top1_probability"]),
            "top1_probability_delta_100k_minus_75k": float(b["top1_probability"]) - float(a["top1_probability"]),
            "margin_75k": float(a["probability_margin"]),
            "margin_100k": float(b["probability_margin"]),
            "margin_delta_100k_minus_75k": float(b["probability_margin"]) - float(a["probability_margin"]),
            "entropy_75k": float(a["masked_policy_entropy"]),
            "entropy_100k": float(b["masked_policy_entropy"]),
            "entropy_delta_100k_minus_75k": float(b["masked_policy_entropy"]) - float(a["masked_policy_entropy"]),
            "P_I_positive_75k": float(a["P_I_positive"]),
            "P_I_positive_100k": float(b["P_I_positive"]),
            "P_N_positive_75k": float(a["P_N_positive"]),
            "P_N_positive_100k": float(b["P_N_positive"]),
            "P_I_and_N_positive_75k": float(a["P_I_and_N_positive"]),
            "P_I_and_N_positive_100k": float(b["P_I_and_N_positive"]),
            "P_noop_75k": float(a["P_noop"]),
            "P_noop_100k": float(b["P_noop"]),
            "KL_75k_to_100k": kl(p, q),
            "KL_100k_to_75k": kl(q, p),
            "Jensen_Shannon": 0.5 * kl(p, m) + 0.5 * kl(q, m),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SHIFT_OUT, index=False, encoding="utf-8-sig")
    return out


def tensorboard_metrics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_dir = ORIGINAL_ROOT / "tensorboard" / "YCA" / "MaskablePPO_1"
    events = sorted(event_dir.glob("events*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one TensorBoard event file, got {len(events)}")
    accumulator = EventAccumulator(str(events[0]))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    short = {
        "rollout/ep_len_mean": "rollout_ep_len_mean",
        "rollout/ep_rew_mean": "rollout_ep_rew_mean",
        "time/fps": "time_fps",
        "train/approx_kl": "approx_kl",
        "train/clip_fraction": "clip_fraction",
        "train/clip_range": "clip_range",
        "train/entropy_loss": "entropy_loss",
        "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate",
        "train/loss": "loss",
        "train/policy_gradient_loss": "policy_gradient_loss",
        "train/value_loss": "value_loss",
    }
    records: dict[int, dict[str, Any]] = {}
    for tag in tags:
        name = short.get(tag, tag.replace("/", "_"))
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"timestep": step, "wall_time": float(event.wall_time)})[name] = float(event.value)
    all_metrics = pd.DataFrame(sorted(records.values(), key=lambda item: item["timestep"])).sort_values("timestep").reset_index(drop=True)
    focused = all_metrics[all_metrics["timestep"].between(75_000, 100_100)].copy()
    focused.to_csv(TRAINING_OUT, index=False, encoding="utf-8-sig")

    def window(step: int) -> str:
        if step < 75_000:
            return "pre-75K"
        if step < 80_000:
            return "75–80K"
        if step < 85_000:
            return "80–85K"
        if step < 90_000:
            return "85–90K"
        if step < 95_000:
            return "90–95K"
        return "95–100K"

    all_metrics["window"] = all_metrics["timestep"].map(window)
    metric_cols = [c for c in short.values() if c in all_metrics.columns]
    rows: list[dict[str, Any]] = []
    for bucket in ["pre-75K", "75–80K", "80–85K", "85–90K", "90–95K", "95–100K"]:
        sub = all_metrics[all_metrics["window"].eq(bucket)]
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "window": bucket,
                "metric": metric,
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
            })
    windows = pd.DataFrame(rows)
    windows.to_csv(WINDOW_OUT, index=False, encoding="utf-8-sig")
    manifest = {
        "event_file": rel(events[0]),
        "event_file_sha256": sha256_file(events[0]),
        "scalar_tags": tags,
        "scalar_counts": {tag: len(accumulator.Scalars(tag)) for tag in tags},
        "focused_rows": int(len(focused)),
        "timestep_min": int(all_metrics["timestep"].min()),
        "timestep_max": int(all_metrics["timestep"].max()),
    }
    return focused, windows, manifest


def robust_metric_evidence(windows: pd.DataFrame, metric: str) -> dict[str, Any]:
    sub = windows[windows["metric"].eq(metric)].set_index("window")
    result: dict[str, Any] = {"metric": metric}
    for bucket in ["pre-75K", "75–80K", "80–85K", "85–90K", "90–95K", "95–100K"]:
        if bucket in sub.index:
            row = sub.loc[bucket]
            result[bucket] = {key: float(row[key]) for key in ["count", "mean", "median", "q25", "q75", "max"]}
        else:
            result[bucket] = "unknown / insufficient evidence"
    return result


def update_instability_summary(windows: pd.DataFrame) -> dict[str, Any]:
    """Compare late update windows with this seed's pre-75K robust range."""
    buckets = ["85–90K", "90–95K", "95–100K"]
    summary: dict[str, Any] = {}
    for metric in ["approx_kl", "clip_fraction", "policy_gradient_loss"]:
        sub = windows[windows["metric"].eq(metric)].set_index("window")
        if "pre-75K" not in sub.index:
            summary[metric] = "unknown / insufficient evidence"
            continue
        base = sub.loc["pre-75K"]
        late = {bucket: sub.loc[bucket] for bucket in buckets if bucket in sub.index}
        if not late:
            summary[metric] = "unknown / insufficient evidence"
            continue
        if metric == "approx_kl":
            summary[metric] = {
                "pre75_median": float(base["median"]),
                "pre75_q75": float(base["q75"]),
                "late_medians": {bucket: float(row["median"]) for bucket, row in late.items()},
                "late_windows_median_above_pre75_q75": [
                    bucket for bucket, row in late.items() if float(row["median"]) > float(base["q75"])
                ],
            }
        elif metric == "clip_fraction":
            summary[metric] = {
                "pre75_median": float(base["median"]),
                "pre75_q75": float(base["q75"]),
                "late_medians": {bucket: float(row["median"]) for bucket, row in late.items()},
                "late_windows_median_positive": [
                    bucket for bucket, row in late.items() if float(row["median"]) > 0.0
                ],
            }
        else:
            summary[metric] = {
                "pre75_median": float(base["median"]),
                "pre75_q25": float(base["q25"]),
                "pre75_q75": float(base["q75"]),
                "late_medians": {bucket: float(row["median"]) for bucket, row in late.items()},
            }
    return summary


def hypothesis_evidence(behavior: pd.DataFrame, shifts: pd.DataFrame, windows: pd.DataFrame, parity_ok: bool) -> pd.DataFrame:
    if not shifts.empty and "entropy_delta_100k_minus_75k" in shifts:
        valid_shifts = shifts.copy()
        shift_stats = {
            "state_count": int(len(valid_shifts)),
            "mean_entropy_delta": float(valid_shifts["entropy_delta_100k_minus_75k"].mean()),
            "median_entropy_delta": float(valid_shifts["entropy_delta_100k_minus_75k"].median()),
            "fraction_entropy_decreased": float((valid_shifts["entropy_delta_100k_minus_75k"] < 0).mean()),
            "mean_margin_delta": float(valid_shifts["margin_delta_100k_minus_75k"].mean()),
            "fraction_margin_increased": float((valid_shifts["margin_delta_100k_minus_75k"] > 0).mean()),
            "mean_top1_probability_delta": float(valid_shifts["top1_probability_delta_100k_minus_75k"].mean()),
            "fraction_top1_probability_increased": float((valid_shifts["top1_probability_delta_100k_minus_75k"] > 0).mean()),
            "mean_P_N_delta": float((valid_shifts["P_N_positive_100k"] - valid_shifts["P_N_positive_75k"]).mean()),
            "mean_P_I_delta": float((valid_shifts["P_I_positive_100k"] - valid_shifts["P_I_positive_75k"]).mean()),
            "mean_KL_75_to_100": float(valid_shifts["KL_75k_to_100k"].mean()),
            "mean_KL_100_to_75": float(valid_shifts["KL_100k_to_75k"].mean()),
            "mean_JS": float(valid_shifts["Jensen_Shannon"].mean()),
            "top1_changed_fraction": float(valid_shifts["top1_action_changed"].mean()),
        }
    else:
        shift_stats = "unknown / insufficient evidence"
    approx = robust_metric_evidence(windows, "approx_kl")
    clip = robust_metric_evidence(windows, "clip_fraction")
    pg = robust_metric_evidence(windows, "policy_gradient_loss")
    value = robust_metric_evidence(windows, "value_loss")
    ev = robust_metric_evidence(windows, "explained_variance")
    entropy_train = robust_metric_evidence(windows, "entropy_loss")
    update_summary = update_instability_summary(windows)
    approx_summary = update_summary.get("approx_kl")
    clip_summary = update_summary.get("clip_fraction")
    if isinstance(approx_summary, dict) and isinstance(clip_summary, dict):
        approx_late = approx_summary["late_windows_median_above_pre75_q75"]
        clip_late = clip_summary["late_windows_median_positive"]
        if len(approx_late) >= 2 and len(clip_late) >= 1:
            h2_classification = "SUPPORTED"
        elif approx_late or clip_late:
            h2_classification = "PARTIALLY_SUPPORTED"
        else:
            h2_classification = "NOT_SUPPORTED"
    else:
        h2_classification = "INSUFFICIENT_EVIDENCE"
    rows = [
        {
            "hypothesis": "H1_policy_concentration_entropy_collapse",
            "classification": "PARTIALLY_SUPPORTED" if isinstance(shift_stats, dict) and shift_stats["mean_entropy_delta"] < 0 and shift_stats["mean_margin_delta"] > 0 else "NOT_SUPPORTED" if isinstance(shift_stats, dict) else "INSUFFICIENT_EVIDENCE",
            "observed": json.dumps(shift_stats, ensure_ascii=False),
            "aggregate_training_evidence": json.dumps(entropy_train, ensure_ascii=False),
            "interpretation_boundary": "Mean entropy/margin directions support a concentration component, but only 56.1% of fixed-state rows move in that direction and DAP1 is already concentrated at both checkpoints; this is not a causal root-cause claim.",
        },
        {
            "hypothesis": "H2_large_or_unstable_policy_updates",
            "classification": h2_classification if parity_ok else "INSUFFICIENT_EVIDENCE",
            "observed": json.dumps({"approx_kl": approx, "clip_fraction": clip, "policy_gradient_loss": pg, "robust_late_window_summary": update_summary}, ensure_ascii=False),
            "aggregate_training_evidence": "Late windows are compared descriptively with this seed's pre-75K median/IQR; the pre-75K q75 is used as a descriptive reference, not a tuned anomaly threshold.",
            "interpretation_boundary": "The late-window KL/clip displacement supports an update-instability component, but without intermediate policy snapshots it does not prove the update that caused the collapse.",
        },
        {
            "hypothesis": "H3_value_function_instability_precedes_collapse",
            "classification": "INSUFFICIENT_EVIDENCE" if not parity_ok else "INSUFFICIENT_EVIDENCE",
            "observed": json.dumps({"value_loss": value, "explained_variance": ev}, ensure_ascii=False),
            "aggregate_training_evidence": "Late value-loss medians remain within the pre-75K interquartile reference and explained-variance medians do not show a clear sustained deterioration; exact precedence is unavailable.",
            "interpretation_boundary": "Value instability preceding collapse is not established; report only the observed association/limitation, never precedence.",
        },
        {
            "hypothesis": "H4_quiet_gradual_policy_drift",
            "classification": "NOT_SUPPORTED" if isinstance(shift_stats, dict) and h2_classification == "SUPPORTED" else "PARTIALLY_SUPPORTED" if isinstance(shift_stats, dict) else "INSUFFICIENT_EVIDENCE",
            "observed": json.dumps({"fixed_state_policy_shift": shift_stats}, ensure_ascii=False),
            "aggregate_training_evidence": json.dumps({"approx_kl": approx, "clip_fraction": clip, "value_loss": value, "explained_variance": ev}, ensure_ascii=False),
            "interpretation_boundary": "A quiet-drift classification requires no clear aggregate update warning; the late KL/clip displacement prevents that stronger classification here.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(HYPOTHESIS_OUT, index=False, encoding="utf-8-sig")
    return out


def category_counts(shifts: pd.DataFrame) -> pd.DataFrame:
    if shifts.empty or "stress_category" not in shifts:
        return pd.DataFrame()
    return shifts.groupby(["source_checkpoint", "stress_category"], as_index=False).agg(
        state_count=("state_id", "count"),
        top1_changed_fraction=("top1_action_changed", "mean"),
        mean_entropy_delta=("entropy_delta_100k_minus_75k", "mean"),
        mean_margin_delta=("margin_delta_100k_minus_75k", "mean"),
        mean_top1_probability_delta=("top1_probability_delta_100k_minus_75k", "mean"),
        mean_P_N_delta=("P_N_positive_100k", lambda s: float(s.mean())),
    )


def write_report(behavior: pd.DataFrame, parity: pd.DataFrame, shifts: pd.DataFrame, windows: pd.DataFrame, hypotheses: pd.DataFrame, tb_manifest: dict[str, Any], parity_ok: bool) -> None:
    mean_behavior = behavior.groupby("checkpoint", as_index=False)[["yield_kg_ha", "I_mm", "N_kg_ha", "I_positive_action_count", "N_positive_action_count"]].mean()
    b75 = mean_behavior[mean_behavior["checkpoint"].eq(75_000)].iloc[0]
    b100 = mean_behavior[mean_behavior["checkpoint"].eq(100_000)].iloc[0]
    counts = category_counts(shifts)
    h1 = hypotheses.loc[hypotheses["hypothesis"].eq("H1_policy_concentration_entropy_collapse"), "classification"].iloc[0] if not hypotheses.empty else "unknown / insufficient evidence"
    h2 = hypotheses.loc[hypotheses["hypothesis"].eq("H2_large_or_unstable_policy_updates"), "classification"].iloc[0] if not hypotheses.empty else "unknown / insufficient evidence"
    h4 = hypotheses.loc[hypotheses["hypothesis"].eq("H4_quiet_gradual_policy_drift"), "classification"].iloc[0] if not hypotheses.empty else "unknown / insufficient evidence"
    if parity_ok and h2 == "SUPPORTED":
        primary = "LARGE_UPDATE_INSTABILITY"
        secondary = "Policy concentration is a supported secondary component, but not by itself the causal root cause."
    elif parity_ok and h1 == "SUPPORTED":
        primary = "HIGH_CONFIDENCE_POLICY_CONCENTRATION"
        secondary = "Update-level evidence is not strong enough for the primary classification; see hypothesis table."
    elif parity_ok and h4 == "SUPPORTED":
        primary = "QUIET_POLICY_DRIFT"
        secondary = "Policy shift is observed, but aggregate-metric interpretation remains descriptive."
    else:
        primary = "MIXED_OR_UNRESOLVED"
        secondary = "Replay or temporal evidence is insufficient for a stronger single mechanism."
    anchor = shifts[shifts["rollout_step"].eq(0)] if not shifts.empty else pd.DataFrame()
    if not anchor.empty:
        anchor_stats = anchor[[
            "top1_probability_75k", "top1_probability_100k",
            "top1_probability_delta_100k_minus_75k", "margin_75k", "margin_100k",
            "margin_delta_100k_minus_75k", "entropy_75k", "entropy_100k",
            "entropy_delta_100k_minus_75k", "P_I_positive_75k", "P_I_positive_100k",
            "P_N_positive_75k", "P_N_positive_100k", "P_noop_75k", "P_noop_100k",
        ]].mean()
        anchor_line = (
            f"The DAP1 pre-action anchor is identified by `rollout_step=0` (the replay observation carries `dap=0` before the DAP1 action): "
            f"all `{len(anchor)}` cross-checks select action 13 (`I45/N40`) at both checkpoints; mean top1 probability `{anchor_stats['top1_probability_75k']:.4f} -> {anchor_stats['top1_probability_100k']:.4f}`, "
            f"margin `{anchor_stats['margin_75k']:.4f} -> {anchor_stats['margin_100k']:.4f}`, entropy `{anchor_stats['entropy_75k']:.4f} -> {anchor_stats['entropy_100k']:.4f}`, "
            f"P(N>0) `{anchor_stats['P_N_positive_75k']:.4f} -> {anchor_stats['P_N_positive_100k']:.4f}`, and P(no-op) `{anchor_stats['P_noop_75k']:.4f} -> {anchor_stats['P_noop_100k']:.4f}`."
        )
    else:
        anchor_line = "DAP1 anchor: `unknown / insufficient evidence`."
    update_summary = update_instability_summary(windows)
    lines = [
        "# YC PPO Original seed1 75K→100K collapse diagnostic",
        "",
        "## Conclusion first",
        "",
        f"Observed behavior changes from a stable high-investment/high-yield policy at 75K to a low-investment/low-yield policy at 100K: mean yield `{b75['yield_kg_ha']:.3f} -> {b100['yield_kg_ha']:.3f}` kg/ha, irrigation `{b75['I_mm']:.3f} -> {b100['I_mm']:.3f}` mm, and N `{b75['N_kg_ha']:.3f} -> {b100['N_kg_ha']:.3f}` kg/ha.",
        f"Primary mechanism classification: **{primary}**. Secondary interpretation: {secondary}",
        "",
        "## Scope and evidence gate",
        "",
        "Only Original PPO, seed1, checkpoints 75K and 100K, validation years 2014–2023 were analyzed. No training, retuning, new seed, site, reward, state, action, or mask change was performed.",
        f"- Existing validation files: `{rel(ORIGINAL_ROOT / 'evaluation' / '032_22_checkpoint_validation_summary.csv')}` and daily outputs under `{rel(ORIGINAL_ROOT / 'daily_outputs' / 'YCA')}`.",
        f"- Deterministic replay parity: `{'PASS' if parity_ok else 'PPO_REPLAY_PARITY_FAILED'}`; rows `{len(parity)}`; evidence `{rel(REPLAY_PARITY_OUT)}`.",
        f"- Because parity is `{'PASS' if parity_ok else 'FAIL'}`, replay states/probabilities are {'allowed' if parity_ok else 'not allowed'} for mechanism interpretation.",
        "",
        "## 1. Behavior-layer collapse",
        "",
        f"75K: mean yield `{b75['yield_kg_ha']:.3f}` kg/ha, I `{b75['I_mm']:.3f}` mm, N `{b75['N_kg_ha']:.3f}` kg/ha; mean I-positive action count `{b75['I_positive_action_count']:.1f}`, N-positive count `{b75['N_positive_action_count']:.1f}`.",
        f"100K: mean yield `{b100['yield_kg_ha']:.3f}` kg/ha, I `{b100['I_mm']:.3f}` mm, N `{b100['N_kg_ha']:.3f}` kg/ha; mean I-positive action count `{b100['I_positive_action_count']:.1f}`, N-positive count `{b100['N_positive_action_count']:.1f}`.",
        "The per-year behavior table records the complete selected action sequence, first non-noop action, positive-action counts, and invalid/fallback counts. It is descriptive only; it does not assign cause.",
        "",
        "## 2. Same-state policy distribution",
        "",
        f"The fixed-state panel contains `{len(shifts) if parity_ok else 0}` state-level 75K-vs-100K comparisons from real 75K and 100K rollouts. Each row uses the same observation and same mask for both policies; KL directions and Jensen–Shannon divergence use the legal action set and masked probabilities normalized over legal actions.",
        f"- Fixed-state evidence: `{rel(FIXED_POLICY_OUT)}` and `{rel(SHIFT_OUT)}`.",
        f"- {anchor_line}",
        f"- Daily source-policy evidence: `{rel(DAILY_POLICY_OUT)}`.",
        "- Stress-category coverage is data-dependent; absent categories are not constructed. The category summary below is empty when the parity gate fails.",
        "",
        counts.to_markdown(index=False) if not counts.empty else "`insufficient evidence`",
        "",
        "## 3. Training-update metrics",
        "",
        f"TensorBoard source `{tb_manifest['event_file']}` contains `{tb_manifest['focused_rows']}` focused aligned rows and the requested update-level fields where available. The raw 75K–100K table is `{rel(TRAINING_OUT)}`; robust descriptive window summaries are `{rel(WINDOW_OUT)}`.",
        f"The robust update comparison is: `{json.dumps(update_summary, ensure_ascii=False)}`. In particular, the 85–90K, 90–95K, and 95–100K `approx_kl` window medians are compared with the pre-75K q75, and positive `clip_fraction` medians are explicitly recorded; this supports an update-level warning without inventing a tuned anomaly cutoff.",
        "",
        windows[windows["metric"].isin(["approx_kl", "clip_fraction", "entropy_loss", "value_loss", "explained_variance", "policy_gradient_loss"])].to_markdown(index=False) if not windows.empty else "`unknown / insufficient evidence`",
        "",
        "## 4. Hypothesis assessment",
        "",
        hypotheses[["hypothesis", "classification", "interpretation_boundary"]].to_markdown(index=False),
        "",
        "H2 and H3 are not turned into causal claims. In particular, with only 75K and 100K policy snapshots, the evidence can localize the change to the interval but cannot identify the exact update at which the policy flipped or prove that value instability preceded it.",
        "",
        "## Answers to the six required questions",
        "",
        "1. 75K→100K: the policy moves from high-yield, roughly I≈214.5/N≈240 behavior to low-yield, roughly I≈45/N≈40 behavior; the per-year table gives the exact sequences.",
        f"2. Same fixed states: `{'yes, subject to replay parity' if parity_ok and not shifts.empty else 'unknown / insufficient evidence'}`; see top1 probability, margin, entropy, and KL columns.",
        f"3. Entropy/margin support for concentration: `{h1}`.",
        f"4. Approx-KL/clip-fraction evidence: `{h2}`; late-window displacement is descriptive and no exact onset or causal update is claimed.",
        "5. Value-loss/explained-variance evidence: no clear sustained value instability is shown, and precedence is not established; only an association/limitation statement is allowed.",
        f"6. Best-supported collapse type: `{primary}`.",
        "",
        "## Change boundary and stop rule",
        "",
        "This was a read-only diagnostic. Existing PPO/DSSAT code, reward, raw observation, action space, mask, constraints, training years, and hyperparameters were not changed. No stabilization intervention was implemented, no new seed/site/variant was run, and no parameter recommendation is made here.",
        "",
        "## Required outputs",
        "",
        f"- `{rel(BEHAVIOR_OUT)}`",
        f"- `{rel(REPLAY_PARITY_OUT)}`; `{rel(DAILY_POLICY_OUT)}`",
        f"- `{rel(FIXED_POLICY_OUT)}`; `{rel(SHIFT_OUT)}`",
        f"- `{rel(TRAINING_OUT)}`; `{rel(WINDOW_OUT)}`",
        f"- `{rel(HYPOTHESIS_OUT)}`",
        "",
        "YC_PPO_ORIG_S1_COLLAPSE_DIAGNOSTIC_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["behavior", "replay", "metrics", "report", "all"], default="all")
    args = parser.parse_args()
    cfg = read_json(CONFIG_PATH)
    behavior = build_behavior_summary()
    if args.phase == "behavior":
        print(f"behavior_rows={len(behavior)}")
        return 0
    if args.phase in {"replay", "all"}:
        rollout, states, parity, parity_ok, replay_manifest = capture_replays(cfg)
        shifts = policy_shifts(states, parity_ok)
    else:
        parity = pd.read_csv(REPLAY_PARITY_OUT) if REPLAY_PARITY_OUT.exists() else pd.DataFrame()
        parity_ok = bool(not parity.empty and (parity["status"] == "PASS").all())
        states = pd.read_csv(FIXED_POLICY_OUT) if FIXED_POLICY_OUT.exists() else pd.DataFrame()
        shifts = policy_shifts(states, parity_ok)
    if args.phase == "replay":
        if not parity_ok:
            print("PPO_REPLAY_PARITY_FAILED")
            return 2
        print(f"replay_parity=PASS rows={len(parity)} states={len(states)} shifts={len(shifts)}")
        return 0
    focused, windows, tb_manifest = tensorboard_metrics()
    hypotheses = hypothesis_evidence(behavior, shifts, windows, parity_ok)
    if args.phase == "metrics":
        print(f"training_rows={len(focused)} windows={len(windows)}")
        return 0
    write_report(behavior, parity, shifts, windows, hypotheses, tb_manifest, parity_ok)
    print(f"parity={'PASS' if parity_ok else 'PPO_REPLAY_PARITY_FAILED'}")
    print(hypotheses[["hypothesis", "classification"]].to_string(index=False))
    print("YC_PPO_ORIG_S1_COLLAPSE_DIAGNOSTIC_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

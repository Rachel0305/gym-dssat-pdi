from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frozen_nstep_dqn_config_020_11 import ACTION_TABLE_9, apply_environment_constants, dqn_kwargs
from ppo_evaluate import latest_observation_dict, scalar
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_19"
ENV_SOURCE = ROOT / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1/dqn_env_args.json"
NULL_YIELD = 5408.0
REFERENCE_YIELD = 11205.0
SCHEDULE = {22: 1, 29: 4, 42: 7, 56: 4, 79: 1}


def env_args(label: str) -> dict[str, Any]:
    args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / label
    run.mkdir(parents=True, exist_ok=False)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def make_env(args: dict[str, Any]):
    apply_environment_constants(shared)
    return legacy.make_train_env(args, NULL_YIELD)


def snapshot(env: Any, destination: Path) -> None:
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if not tmp or not Path(tmp).exists():
        raise RuntimeError("PDI temporary output unavailable")
    shutil.copytree(Path(tmp), destination)


def final_yield(snapshot_dir: Path) -> tuple[float, float]:
    frame = parse_dssat_table(snapshot_dir / "PlantGro.OUT")
    gwad = pd.to_numeric(frame["GWAD"], errors="coerce").dropna()
    cwad = pd.to_numeric(frame["CWAD"], errors="coerce").dropna()
    return float(gwad.iloc[-1]), float(cwad.iloc[-1])


def collect_demonstration() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    args = env_args("demonstration_replay")
    env = make_env(args)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(420):
            latest_before = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest_before.get("dap", step)) or 0))
            action = int(SCHEDULE.get(dap, 0))
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            rows.append({
                "step": step, "dap_action": dap, "target_action": action,
                "target_amir": ACTION_TABLE_9[action]["amir"], "target_anfer": ACTION_TABLE_9[action]["anfer"],
                "safe_amir": float(safe.get("amir", 0.0)), "safe_anfer": float(safe.get("anfer", 0.0)),
                "used_irrigation": float(getattr(env, "used_irrigation", np.nan)),
                "used_nitrogen": float(getattr(env, "used_nitrogen", np.nan)),
                "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
    finally:
        snap = OUT / "demonstration_replay" / "pdi_tmp_snapshot_eval"
        snapshot(env, snap)
        env.close()
    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    frame = pd.DataFrame(rows)
    gwad, cwad = final_yield(snap)
    nonzero = frame[frame["target_action"].ne(0)]
    audit = {
        "yield_kg_ha": gwad, "biomass_kg_ha": cwad,
        "irrigation_mm": float(frame["safe_amir"].sum()), "nitrogen_kg_ha": float(frame["safe_anfer"].sum()),
        "observation_count": int(len(obs_array)), "action_count": int(len(action_array)),
        "all_observations_finite": bool(np.isfinite(obs_array).all()),
        "nonzero_event_count": int(len(nonzero)),
        "nonzero_all_uncropped": bool(np.allclose(nonzero["target_amir"], nonzero["safe_amir"]) and np.allclose(nonzero["target_anfer"], nonzero["safe_anfer"])),
        "yield_reproduction_error": abs(gwad - REFERENCE_YIELD),
    }
    passed = audit["yield_reproduction_error"] <= 2 and audit["irrigation_mm"] == 75 and audit["nitrogen_kg_ha"] == 200 and audit["nonzero_event_count"] == 5 and audit["nonzero_all_uncropped"] and audit["all_observations_finite"]
    audit["passed"] = bool(passed)
    frame.to_csv(OUT / "021_19_demonstration_steps.csv", index=False)
    np.savez_compressed(OUT / "021_19_demonstration_dataset.npz", observations=obs_array, actions=action_array)
    (OUT / "021_19_demonstration_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Demonstration audit failed: {audit}")
    return obs_array, action_array, frame, audit


def warmstart(model: DQN, observations: np.ndarray, actions: np.ndarray, epochs: int = 100) -> pd.DataFrame:
    device = model.device
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act = torch.as_tensor(actions, dtype=torch.long, device=device)
    counts = np.bincount(actions, minlength=9).astype(np.float64)
    weights = np.zeros(9, dtype=np.float32)
    present = counts > 0
    weights[present] = len(actions) / (present.sum() * counts[present])
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=1e-4)
    rows = []
    for epoch in range(epochs):
        logits = model.q_net(obs)
        loss = F.cross_entropy(logits, act, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), 10.0)
        optimizer.step()
        pred = logits.argmax(dim=1)
        overall = float((pred == act).float().mean().cpu())
        nz = act.ne(0)
        nonzero = float((pred[nz] == act[nz]).float().mean().cpu())
        rows.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), "overall_accuracy": overall, "nonzero_accuracy": nonzero})
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    return pd.DataFrame(rows)


def free_evaluate(model: DQN) -> tuple[pd.DataFrame, dict[str, Any]]:
    args = env_args("warmstart_free_evaluation")
    env = make_env(args)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(420):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            rows.append({"step": step, "dap_action": dap, "action": action, "safe_amir": float(safe.get("amir", 0.0)), "safe_anfer": float(safe.get("anfer", 0.0)), "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
            if terminated or truncated:
                break
    finally:
        snap = OUT / "warmstart_free_evaluation" / "pdi_tmp_snapshot_eval"
        snapshot(env, snap)
        env.close()
    frame = pd.DataFrame(rows)
    gwad, cwad = final_yield(snap)
    result = {"yield_kg_ha": gwad, "biomass_kg_ha": cwad, "irrigation_mm": float(frame["safe_amir"].sum()), "nitrogen_kg_ha": float(frame["safe_anfer"].sum()), "reward_total": float(frame["reward"].sum())}
    return frame, result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    observations, actions, demo_steps, demo_audit = collect_demonstration()
    train_args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    train_args["log_saving_path"] = str(OUT / "warmstart_model_env.log")
    env = make_env(train_args)
    try:
        model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
        losses = warmstart(model, observations, actions, epochs=100)
        model.save(str(OUT / "dqn_oracle_warmstart_before_rl"))
    finally:
        env.close()
    losses.to_csv(OUT / "021_19_warmstart_losses.csv", index=False)
    final = losses.iloc[-1]
    supervised_pass = bool(final["overall_accuracy"] >= 0.95 and final["nonzero_accuracy"] == 1.0)
    eval_daily, eval_result = free_evaluate(model)
    eval_daily.to_csv(OUT / "021_19_warmstart_free_eval_daily.csv", index=False)
    result = {"demonstration": demo_audit, "supervised": {"epochs": 100, "final_loss": float(final["loss"]), "overall_accuracy": float(final["overall_accuracy"]), "nonzero_accuracy": float(final["nonzero_accuracy"]), "passed": supervised_pass}, "free_evaluation": eval_result, "interpretation": "Warm-start smoke only; no RL training performed."}
    (OUT / "021_19_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

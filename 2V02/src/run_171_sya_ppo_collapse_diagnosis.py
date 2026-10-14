"""Read-only audit of PPO action collapse for task 170.

This script does not train or overwrite existing results.  It replays the five
saved checkpoints on the six weather variants and records policy entropy,
masked action probabilities, action diversity, and observation differences
under a common action sequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_168_sya_weather_aug_pilot_smoke2k_v2 as task168


OUT = ROOT / "benchmark_results" / "171_sya_ppo_collapse_diagnosis"
CONFIG = ROOT / "configs" / "170_sya_weather_aug_entcoef002_20k.json"
MODEL_ROOT = ROOT / "benchmark_results" / "170_sya_weather_aug_entcoef002_20k" / "models" / "SYA"
DAILY_ROOT = ROOT / "benchmark_results" / "170_sya_weather_aug_entcoef002_20k" / "daily_outputs" / "SYA"
CHECKPOINTS = [2000, 5000, 10000, 15000, 20000]
SCENARIO_KEYS = task168.SCENARIO_KEYS


def scalar(value, default=np.nan):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else default
    except Exception:
        return default


def model_path(step: int) -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{int(step)}.zip"


def observation_contract(env) -> dict:
    candidates = []
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidates.append(
            {
                "class": type(current).__name__,
                "module": type(current).__module__,
                "observation_variables": getattr(current, "observation_variables", None),
                "observation_keys": getattr(current, "observation_keys", None),
                "state_variables": getattr(current, "state_variables", None),
            }
        )
        current = getattr(current, "env", None)
    return {
        "space_shape": list(getattr(env.observation_space, "shape", ()) or ()),
        "wrapper_chain": candidates,
        "known_config_note": (
            "The installed gym_dssat_pdi maize config defines rain but the active all-state "
            "observation list used by task 170 was audited separately; this file records the "
            "runtime wrapper attributes without modifying the dependency."
        ),
    }


def policy_distribution(model, obs, mask):
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor, action_masks=np.asarray(mask, dtype=bool))
    probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
    entropy = float(distribution.entropy().detach().cpu().numpy().reshape(-1)[0])
    return probs, entropy


def run_policy_audit(runtime_config, env_config, model, scenario_key: int, step: int) -> tuple[list[dict], dict]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = task168.base03222.base.make_env(
        runtime_config,
        env_config,
        "SYA",
        int(scenario_key),
        0,
        f"SYA_2005_{SCENARIO_KEYS[scenario_key]}_171_eval",
        evaluation=True,
    )
    rows: list[dict] = []
    contract = observation_contract(env)
    try:
        obs, info = env.reset()
        done = False
        t = 0
        while not done and t < int(runtime_config["runtime"]["max_steps"]):
            latest = task168.base03222.base.latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap"), t + 1)))
            mask = get_action_masks(env)
            probs, entropy = policy_distribution(model, obs, mask)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action_index = int(np.asarray(action).reshape(-1)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            action_info = dict(getattr(env, "last_action_info", {}))
            rows.append(
                {
                    "checkpoint_step": int(step),
                    "scenario_key": int(scenario_key),
                    "weather_variant": SCENARIO_KEYS[scenario_key],
                    "t": int(t),
                    "dap": dap,
                    "mask_valid_before": int(np.asarray(mask, dtype=bool).sum()),
                    "mask_valid_after": int(action_info.get("mask_valid_action_count", np.nan)),
                    "action_index": action_index,
                    "safe_action_amir": scalar(action_info.get("safe_action_amir")),
                    "safe_action_anfer": scalar(action_info.get("safe_action_anfer")),
                    "max_action_probability": float(np.max(probs)),
                    "selected_action_probability": float(probs[action_index]),
                    "policy_entropy": float(entropy),
                    "legal_action_probability_mass": float(np.sum(probs[np.asarray(mask, dtype=bool)])),
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                }
            )
            t += 1
            done = bool(terminated or truncated)
    finally:
        env.close()
    return rows, contract


def replay_observations(runtime_config, env_config, action_sequence: list[int]) -> tuple[list[dict], dict]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    rows: list[dict] = []
    contracts: dict = {}
    for scenario_key, variant in SCENARIO_KEYS.items():
        env = task168.base03222.base.make_env(
            runtime_config,
            env_config,
            "SYA",
            int(scenario_key),
            0,
            f"SYA_2005_{variant}_171_replay",
            evaluation=True,
        )
        try:
            contracts[variant] = observation_contract(env)
            obs, info = env.reset()
            for t, action_index in enumerate(action_sequence):
                latest = task168.base03222.base.latest_observation_dict(env, obs, info)
                dap = int(round(scalar(latest.get("dap"), t + 1)))
                mask = get_action_masks(env)
                obs_vec = np.asarray(obs, dtype=float).reshape(-1)
                rows.append(
                    {
                        "scenario_key": int(scenario_key),
                        "weather_variant": variant,
                        "t": int(t),
                        "dap": dap,
                        "action_index_replayed": int(action_index),
                        "mask_valid_before": int(np.asarray(mask, dtype=bool).sum()),
                        **{f"obs_{i}": float(v) for i, v in enumerate(obs_vec)},
                    }
                )
                obs, _, terminated, truncated, info = env.step(int(action_index))
                if terminated or truncated:
                    break
        finally:
            env.close()
    return rows, contracts


def main() -> None:
    from sb3_contrib import MaskablePPO

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)
    old = task168.patch_runtime(manifest)
    try:
        runtime_config = task168.engine.load_config()
        runtime_config["ppo"]["ent_coef"] = float(cfg["ppo_overrides"]["ent_coef"])
        env_config = task168.base03222.direct_ppo.build_env_config(runtime_config, split)

        policy_rows: list[dict] = []
        contract = {}
        for step in CHECKPOINTS:
            model = MaskablePPO.load(str(model_path(step)), device="cpu")
            for scenario_key in SCENARIO_KEYS:
                rows, current_contract = run_policy_audit(runtime_config, env_config, model, scenario_key, step)
                policy_rows.extend(rows)
                contract.setdefault("policy", current_contract)

        policy = pd.DataFrame(policy_rows)
        policy.to_csv(OUT / "171_policy_step_audit.csv", index=False, encoding="utf-8-sig")
        policy["non_noop"] = (policy["safe_action_amir"] > 0) | (policy["safe_action_anfer"] > 0)
        summary = (
            policy.groupby(["checkpoint_step", "weather_variant"], as_index=False)
            .agg(
                steps=("t", "size"),
                distinct_action_count=("action_index", "nunique"),
                non_noop_steps=("non_noop", "sum"),
                mean_policy_entropy=("policy_entropy", "mean"),
                mean_non_noop_entropy=("policy_entropy", lambda s: float(s[policy.loc[s.index, "non_noop"]].mean()) if bool(policy.loc[s.index, "non_noop"].any()) else np.nan),
                mean_max_action_probability=("max_action_probability", "mean"),
                mean_selected_action_probability=("selected_action_probability", "mean"),
                min_mask_valid_before=("mask_valid_before", "min"),
                mean_mask_valid_before=("mask_valid_before", "mean"),
            )
        )
        summary.to_csv(OUT / "171_policy_checkpoint_summary.csv", index=False, encoding="utf-8-sig")

        original_path = DAILY_ROOT / "SYA_2005_seed0_ckpt20000_daily.csv"
        action_sequence = pd.read_csv(original_path, keep_default_na=False)["discrete_action_index"].astype(int).tolist()
        replay_rows, replay_contract = replay_observations(runtime_config, env_config, action_sequence)
        replay = pd.DataFrame(replay_rows)
        obs_cols = [c for c in replay.columns if c.startswith("obs_")]
        original = replay[replay["weather_variant"].eq("original")].set_index("t")[obs_cols]
        replay["obs_l2_vs_original"] = replay.apply(lambda r: float(np.linalg.norm(r[obs_cols].to_numpy(dtype=float) - original.loc[int(r["t"])].to_numpy(dtype=float))), axis=1)
        replay["obs_max_abs_vs_original"] = replay.apply(lambda r: float(np.max(np.abs(r[obs_cols].to_numpy(dtype=float) - original.loc[int(r["t"])].to_numpy(dtype=float)))), axis=1)
        replay.to_csv(OUT / "171_replay_observation_diff.csv", index=False, encoding="utf-8-sig")
        obs_summary = (
            replay[replay["weather_variant"].ne("original")]
            .assign(window=lambda d: pd.cut(d["dap"], bins=[0, 30, 60, 90, 200], labels=["DAP1_30", "DAP31_60", "DAP61_90", "DAP91_plus"]))
            .groupby(["weather_variant", "window"], as_index=False, observed=False)
            .agg(mean_l2=("obs_l2_vs_original", "mean"), max_l2=("obs_l2_vs_original", "max"), mean_max_abs=("obs_max_abs_vs_original", "mean"), rows=("t", "size"))
        )
        obs_summary.to_csv(OUT / "171_replay_observation_summary.csv", index=False, encoding="utf-8-sig")
        contract["replay"] = replay_contract
        contract["action_sequence_source"] = str(original_path.relative_to(ROOT)).replace("\\", "/")
        contract["action_sequence_length"] = len(action_sequence)
        (OUT / "171_runtime_observation_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    finally:
        task168.restore_runtime(old)


if __name__ == "__main__":
    main()

"""Task 193: a deliberately explicit, step-by-step PPO replay.

This file is intentionally written as a slow-motion teaching/debug trace.
It does not train PPO. For one 2005 original-weather scenario, it loads six
existing checkpoints and records, at every DAP:

1. observation and simple environment state;
2. legal-action mask;
3. probability/logit of every legal action;
4. deterministic action selected by PPO;
5. reward and state after that action.

The code reuses only the already audited environment-construction helpers.
The decision loop itself is kept linear and visible for supervisor debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_184_sya_ppo_critic_td_action_value_audit as task184
import run_189_sya_gap1_early_irrigation_reward_critic_audit as task189


TASK_ID = "193"
TASK_NAME = "sya_ppo_slow_motion_step_trace"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_ppo_slow_motion_step_trace.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return default
        number = float(array[0])
        return number if np.isfinite(number) else default
    except Exception:
        return default


def action_text(env: Any, action_index: int) -> str:
    grid = task184.grid_action(env, int(action_index))
    return f"I{grid['irrigation_mm']:g}/N{grid['nitrogen_kg_ha']:g}"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="逐日展开已有 PPO checkpoint 的决策过程")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task193 is read-only and must not train PPO.")

    base_config_path = ROOT / cfg["base_config"]
    base_cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / cfg["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {output_dir.relative_to(ROOT)}")

    model_root = ROOT / cfg["model_root"]
    checkpoints = [int(value) for value in cfg["checkpoint_steps"]]
    scenario_key = int(cfg["scenario_key"])
    weather_variant = str(cfg["weather_variant"])
    seed = int(cfg["seed"])

    for checkpoint in checkpoints:
        model_path = model_root / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{checkpoint}.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

    # The following objects are the existing environment construction chain.
    # They are not changed; only the gap=1 runtime wrapper is temporarily used.
    task178 = task184.task178
    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(base_cfg)
    manifest = task168.read_manifest(base_cfg)
    split = task168.load_split(manifest)
    if not bool(gate.get("physical_response_passed")):
        raise RuntimeError("Physical weather-response gate did not pass.")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evaluation").mkdir()
    (output_dir / "debug").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "configs").mkdir()
    (output_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "configs" / base_config_path.name).write_text(base_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "configs" / "193_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "configs" / "193_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    runtime_patch_owner, original_patch, original_restore = task189.patch_gap1_runtime()
    saved_runtime = runtime_patch_owner.patch_runtime_178(manifest)

    runtime_config = None
    env_config = None
    max_steps = None
    daily_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    live_lines: list[str] = []

    try:
        engine = task168.engine
        base03222 = task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        max_steps = int(runtime_config["runtime"]["max_steps"])

        for checkpoint in checkpoints:
            model_path = model_root / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{checkpoint}.zip"
            model = MaskablePPO.load(str(model_path), device="cpu")
            print(f"[checkpoint {checkpoint}] loaded {model_path.name}", flush=True)

            env = base03222.base.make_env(
                runtime_config,
                env_config,
                "SYA",
                scenario_key,
                seed,
                f"SYA_2005_{weather_variant}_193_ckpt{checkpoint}",
                evaluation=True,
            )

            actions_this_checkpoint: list[str] = []
            first_irrigation_dap = None
            first_i240_dap = None
            try:
                obs, info = env.reset()
                done = False
                step_index = 0

                while not done and step_index < max_steps:
                    observed = dict(getattr(env, "last_obs_dict", {}) or {})
                    obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
                    dap = int(round(scalar(observed.get("dap"), step_index)))
                    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                    probabilities, logits = task184.policy_distribution(model, obs_array, mask)
                    critic_value = task184.state_value(model, obs_array)
                    selected_raw, _ = model.predict(obs_array, action_masks=mask, deterministic=True)
                    selected_action = int(np.asarray(selected_raw).reshape(-1)[0])
                    if selected_action < 0 or selected_action >= len(mask) or not bool(mask[selected_action]):
                        raise RuntimeError(f"Invalid selected action at DAP {dap}: {selected_action}, mask={mask}")

                    selected_grid = task184.grid_action(env, selected_action)
                    if selected_grid["irrigation_mm"] > 0 and first_irrigation_dap is None:
                        first_irrigation_dap = dap
                    if selected_grid["irrigation_mm"] > 0 or selected_grid["nitrogen_kg_ha"] > 0:
                        actions_this_checkpoint.append(f"DAP{dap}:{action_text(env, selected_action)}")

                    legal_actions = []
                    for action_index in np.flatnonzero(mask).astype(int).tolist():
                        legal_actions.append(
                            {
                                "action_index": action_index,
                                "action": action_text(env, action_index),
                                "probability": scalar(probabilities[action_index]),
                                "logit": scalar(logits[action_index]),
                                "rank": task184.probability_rank(probabilities, mask, action_index),
                            }
                        )

                    state_before = task184.current_state(env, obs_array, dap)
                    # The environment's safety_state.cumulative_nitrogen is not
                    # populated reliably in this workflow. The executed-action
                    # record is the trusted source used by Task192 correction.
                    action_info_before = dict(getattr(env, "last_action_info", {}) or {})
                    cumulative_n_before = scalar(
                        action_info_before.get("season_cumulative_n"),
                        state_before["cumulative_nitrogen"],
                    )
                    if not np.isfinite(cumulative_n_before):
                        cumulative_n_before = 0.0
                    state_before["cumulative_nitrogen"] = cumulative_n_before
                    live_probability_text = ", ".join(
                        f"{item['action']}={item['probability']:.4f}" for item in legal_actions
                    )
                    live_line = (
                        f"ckpt={checkpoint} DAP={dap} rain={state_before['rain']:.3f} "
                        f"legal={len(legal_actions)} [{live_probability_text}] "
                        f"choose={action_text(env, selected_action)} V={critic_value:.5f} "
                        f"Icum={state_before['cumulative_irrigation']:.1f} "
                        f"Ncum={state_before['cumulative_nitrogen']:.1f}"
                    )
                    print(live_line, flush=True)
                    live_lines.append(live_line)

                    next_obs, reward, terminated, truncated, info = env.step(selected_action)
                    state_after = task184.current_state(env, next_obs, dap + 1)
                    action_info = dict(getattr(env, "last_action_info", {}) or {})
                    cumulative_n_after = scalar(
                        action_info.get("season_cumulative_n"),
                        cumulative_n_before + selected_grid["nitrogen_kg_ha"],
                    )
                    if not np.isfinite(cumulative_n_after):
                        cumulative_n_after = cumulative_n_before + selected_grid["nitrogen_kg_ha"]
                    state_after["cumulative_nitrogen"] = cumulative_n_after
                    if state_after["cumulative_irrigation"] >= 240.0 - 1e-6 and first_i240_dap is None:
                        first_i240_dap = dap

                    daily_rows.append(
                        {
                            "checkpoint_step": checkpoint,
                            "scenario_key": scenario_key,
                            "weather_variant": weather_variant,
                            "step_index": step_index,
                            "dap": dap,
                            "observation_dim": int(obs_array.size),
                            "observation_vector": json_text(np.round(obs_array, 6).tolist()),
                            "environment_state_before": json_text(observed),
                            "rain": state_before["rain"],
                            "swfac_before": state_before["swfac"],
                            "nstres_before": state_before["nstres"],
                            "grnwt_before": state_before["grnwt"],
                            "cumulative_irrigation_before": state_before["cumulative_irrigation"],
                            "cumulative_nitrogen_before": state_before["cumulative_nitrogen"],
                            "mask": task184.mask_text(mask),
                            "legal_action_count": len(legal_actions),
                            "legal_actions": json_text([item["action"] for item in legal_actions]),
                            "legal_action_probabilities": json_text(legal_actions),
                            "critic_value_before": critic_value,
                            "selected_action_index": selected_action,
                            "selected_action": action_text(env, selected_action),
                            "selected_irrigation_mm": selected_grid["irrigation_mm"],
                            "selected_nitrogen_kg_ha": selected_grid["nitrogen_kg_ha"],
                            "reward_scaled": scalar(reward),
                            "reward_unscaled": scalar(action_info.get("reward_unscaled")),
                            "yield_component": scalar(action_info.get("yield_component")),
                            "resource_cost": scalar(action_info.get("resource_cost")),
                            "stress_relief_bonus": scalar(action_info.get("stress_relief_bonus")),
                            "grnwt_increment": scalar(action_info.get("grnwt_increment"), state_after["grnwt"] - state_before["grnwt"]),
                            "swfac_after": state_after["swfac"],
                            "nstres_after": state_after["nstres"],
                            "grnwt_after": state_after["grnwt"],
                            "cumulative_irrigation_after": state_after["cumulative_irrigation"],
                            "cumulative_nitrogen_after": state_after["cumulative_nitrogen"],
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    for item in legal_actions:
                        probability_rows.append(
                            {
                                "checkpoint_step": checkpoint,
                                "scenario_key": scenario_key,
                                "weather_variant": weather_variant,
                                "step_index": step_index,
                                "dap": dap,
                                **item,
                                "selected": bool(item["action_index"] == selected_action),
                            }
                        )

                    obs = next_obs
                    done = bool(terminated or truncated)
                    step_index += 1

                    if step_index % 20 == 0:
                        pd.DataFrame(daily_rows).to_csv(output_dir / "debug" / "193_daily_step_trace_partial.csv", index=False, encoding="utf-8-sig")
                        pd.DataFrame(probability_rows).to_csv(output_dir / "debug" / "193_policy_action_probabilities_partial.csv", index=False, encoding="utf-8-sig")
            finally:
                env.close()

            summary_rows.append(
                {
                    "checkpoint_step": checkpoint,
                    "scenario_key": scenario_key,
                    "weather_variant": weather_variant,
                    "steps": step_index,
                    "first_irrigation_dap": first_irrigation_dap,
                    "first_i240_dap": first_i240_dap,
                    "unique_nonzero_action_count": len(set(actions_this_checkpoint)),
                    "nonzero_action_sequence": "; ".join(actions_this_checkpoint),
                    "final_row_done": bool(done),
                }
            )

        pd.DataFrame(daily_rows).to_csv(output_dir / "evaluation" / "193_daily_step_trace.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(probability_rows).to_csv(output_dir / "evaluation" / "193_policy_action_probabilities.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(output_dir / "evaluation" / "193_checkpoint_action_summary.csv", index=False, encoding="utf-8-sig")
        (output_dir / "logs" / "193_live_trace.txt").write_text("\n".join(live_lines) + "\n", encoding="utf-8")

        result = {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "status": "completed",
            "training_executed": False,
            "scenario_key": scenario_key,
            "weather_variant": weather_variant,
            "checkpoint_steps": checkpoints,
            "daily_rows": len(daily_rows),
            "probability_rows": len(probability_rows),
            "summary_rows": len(summary_rows),
            "physical_gate_passed": bool(gate.get("physical_response_passed")),
            "outputs": {
                "daily_trace": str((output_dir / "evaluation" / "193_daily_step_trace.csv").relative_to(ROOT)),
                "probabilities": str((output_dir / "evaluation" / "193_policy_action_probabilities.csv").relative_to(ROOT)),
                "summary": str((output_dir / "evaluation" / "193_checkpoint_action_summary.csv").relative_to(ROOT)),
                "live_trace": str((output_dir / "logs" / "193_live_trace.txt").relative_to(ROOT)),
            },
        }
        (output_dir / "193_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        runtime_patch_owner.restore_runtime_178(saved_runtime)
        task178.patch_runtime_178 = original_patch
        task178.restore_runtime_178 = original_restore


if __name__ == "__main__":
    main()

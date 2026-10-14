"""Task 192: frozen-PPO full-season counterfactual critic calibration audit.

For every checkpoint/scenario/target-DAP state, this script recreates the
deterministic PPO history, replaces only the target-day action with one of the
four I0/I15/I30/I45-N0 candidates, then lets the same frozen PPO control every
following day to harvest.  It does not train PPO or modify model/environment
files.  Branches run serially to keep DSSAT memory use bounded.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_184_sya_ppo_critic_td_action_value_audit as task184
import run_189_sya_gap1_early_irrigation_reward_critic_audit as task189

TASK_ID = "192"
TASK_NAME = "sya_critic_fullseason_counterfactual_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_critic_fullseason_counterfactual_audit.md"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
SEED = 0
CHECKPOINTS = [50000, 100000]
TARGET_DAPS = [2, 31, 61, 91]
KEY_ACTIONS = [0, 4, 8, 12]
SCENARIOS = task184.SCENARIOS.copy()
MODEL_ROOT = ROOT / "benchmark_results" / "191_sya_balanced_irrigation_gap1_100k_learning_curve" / "models" / "SYA"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def as_number(value: Any, default: float = np.nan) -> float:
    return task184.number(value, default)


def action_index(value: Any) -> int:
    return task184.one_action(value)


def pair(env: Any, idx: int) -> str:
    grid = task184.grid_action(env, idx)
    return f"I{grid['irrigation_mm']:g}/N{grid['nitrogen_kg_ha']:g}"


def action_text(dap: int, env: Any, idx: int) -> str:
    grid = task184.grid_action(env, idx)
    if grid["irrigation_mm"] <= 1e-9 and grid["nitrogen_kg_ha"] <= 1e-9:
        return ""
    return f"DAP{dap} {pair(env, idx)}"


def model_path(step: int) -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{int(step)}.zip"


def state_snapshot(env: Any, obs: Any, fallback_dap: int) -> dict[str, float]:
    state = task184.current_state(env, obs, fallback_dap)
    # Irrigation is held in safety_state. Nitrogen is exposed by the runtime
    # wrapper through last_action_info as season_cumulative_n.
    details = dict(getattr(env, "last_action_info", {}) or {})
    state["cumulative_nitrogen"] = as_number(
        details.get("season_cumulative_n"), state["cumulative_nitrogen"]
    )
    return state


def action_info(env: Any) -> dict[str, float]:
    info = dict(getattr(env, "last_action_info", {}) or {})
    return {
        "reward_unscaled": as_number(info.get("reward_unscaled"), np.nan),
        "yield_component_unscaled": as_number(info.get("yield_component"), 0.0),
        "resource_cost_unscaled": as_number(info.get("resource_cost"), 0.0),
        "stress_relief_bonus_unscaled": as_number(info.get("stress_relief_bonus"), 0.0),
    }


def save_partial(targets: list[dict[str, Any]], branches: list[dict[str, Any]], progress: list[dict[str, Any]]) -> None:
    pd.DataFrame(targets).to_csv(OUT / "debug" / "192_target_state_values_partial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(branches).to_csv(OUT / "debug" / "192_fullseason_counterfactual_partial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(progress).to_csv(OUT / "debug" / "192_progress.csv", index=False, encoding="utf-8-sig")


def preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    expected = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "training_executed": False,
        "station_code": STATION,
        "seed": SEED,
        "checkpoint_steps": CHECKPOINTS,
        "target_daps": TARGET_DAPS,
        "candidate_action_indices": KEY_ACTIONS,
        "model_root": rel(MODEL_ROOT),
        "continuation_policy": "same_frozen_checkpoint_deterministic",
        "overwrite_existing_output": False,
    }
    for key, expected_value in expected.items():
        if cfg.get(key) != expected_value:
            raise ValueError(f"Task192 config mismatch: {key}={cfg.get(key)!r}, expected {expected_value!r}")
    safety = cfg.get("action_safety_override", {})
    if (int(safety.get("min_days_between_irrigation", -1)), int(safety.get("min_days_between_fertilization", -1)), int(safety.get("season_irrigation_cap_mm", -1))) != (1, 7, 240):
        raise ValueError("Task192 requires Task191 safety: irrigation gap=1, N gap=7, cap=240.")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite output directory: {rel(OUT)}")
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    for checkpoint in CHECKPOINTS:
        if not model_path(checkpoint).exists():
            raise FileNotFoundError(model_path(checkpoint))
    task168 = task184.task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    if not bool(gate.get("physical_response_passed")):
        raise RuntimeError("The prerequisite physical weather-response gate is not passed.")
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)
    if split["year"].astype(int).tolist() != list(SCENARIOS):
        raise ValueError("Scenario manifest is not the required six-scenario fixed order.")
    return gate, manifest, split


def record_target(checkpoint: int, scenario_key: int, weather_variant: str, target_dap: int, env: Any, obs: Any, prefix: list[int], model: Any) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    state = state_snapshot(env, obs, target_dap)
    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
    probs, logits = task184.policy_distribution(model, obs, mask)
    selected_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
    selected = action_index(selected_raw)
    selected_grid = task184.grid_action(env, selected)
    return {
        "checkpoint_step": checkpoint,
        "scenario_key": scenario_key,
        "weather_variant": weather_variant,
        "target_dap": target_dap,
        "prefix_length": len(prefix),
        "prefix_nonzero_action_sequence": "; ".join(filter(None, [action_text(dap, env, idx) for dap, idx in []])),
        "rain": state["rain"],
        "swfac": state["swfac"],
        "nstres": state["nstres"],
        "grnwt": state["grnwt"],
        "topwt": state["topwt"],
        "xlai": state["xlai"],
        "cumulative_irrigation_before": state["cumulative_irrigation"],
        "cumulative_nitrogen_before": state["cumulative_nitrogen"],
        "observation_json": json.dumps(np.asarray(obs, dtype=float).reshape(-1).round(8).tolist()),
        "mask": task184.mask_text(mask),
        "legal_action_count": int(mask.sum()),
        "legal_candidate_count": int(sum(bool(mask[idx]) for idx in KEY_ACTIONS)),
        "actor_selected_action": selected,
        "actor_selected_pair": f"I{selected_grid['irrigation_mm']:g}/N{selected_grid['nitrogen_kg_ha']:g}",
        "actor_selected_probability": float(probs[selected]),
        "critic_value_before": task184.state_value(model, obs),
        "prob_i0_n0": float(probs[0]),
        "prob_i15_n0": float(probs[4]),
        "prob_i30_n0": float(probs[8]),
        "prob_i45_n0": float(probs[12]),
        "logit_i0_n0": float(logits[0]),
        "logit_i15_n0": float(logits[4]),
        "logit_i30_n0": float(logits[8]),
        "logit_i45_n0": float(logits[12]),
    }


def run_branch(base03222: Any, runtime_config: dict[str, Any], env_config: dict[str, Any], model: Any, checkpoint: int, scenario_key: int, weather_variant: str, target_dap: int, prefix: list[int], target_state: dict[str, Any], candidate: int, gamma: float) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = base03222.base.make_env(runtime_config, env_config, STATION, scenario_key, SEED, f"SYA_2005_{weather_variant}_192_ckpt{checkpoint}_dap{target_dap}_a{candidate}", evaluation=True)
    base = {
        "checkpoint_step": checkpoint, "scenario_key": scenario_key, "weather_variant": weather_variant,
        "target_dap": target_dap, "candidate_action_index": candidate, "candidate_pair": pair(env, candidate),
        "candidate_legal": False, "dssat_success": False, "error": "",
    }
    try:
        obs, _ = env.reset()
        prefix_first_cap = np.nan
        for prefix_action in prefix:
            before = state_snapshot(env, obs, -1)
            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            if not bool(mask[prefix_action]):
                raise RuntimeError(f"Prefix action {prefix_action} became illegal at DAP {before['dap']}")
            obs, _, terminated, truncated, _ = env.step(prefix_action)
            after = state_snapshot(env, obs, int(before["dap"]) + 1)
            if not np.isfinite(prefix_first_cap) and after["cumulative_irrigation"] >= 240.0 - 1e-6:
                prefix_first_cap = int(before["dap"])
            if terminated or truncated:
                raise RuntimeError("Episode ended while replaying prefix")
        branch_state = state_snapshot(env, obs, target_dap)
        expected_state = {
            "rain": float(target_state["rain"]),
            "swfac": float(target_state["swfac"]),
            "nstres": float(target_state["nstres"]),
            "grnwt": float(target_state["grnwt"]),
            "cumulative_irrigation": float(target_state["cumulative_irrigation_before"]),
            "cumulative_nitrogen": float(target_state["cumulative_nitrogen_before"]),
        }
        mismatch = max(abs(branch_state[key] - expected_state[key]) for key in expected_state)
        if int(branch_state["dap"]) != target_dap or mismatch > 1e-6:
            raise RuntimeError(f"Prefix mismatch: dap={branch_state['dap']}, max_abs_error={mismatch}")
        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
        base["target_mask"] = task184.mask_text(mask)
        base["candidate_legal"] = bool(mask[candidate])
        if not bool(mask[candidate]):
            base["error"] = "candidate_illegal_under_target_mask"
            return base
        probs, _ = task184.policy_distribution(model, obs, mask)
        value_before = task184.state_value(model, obs)
        before_i = branch_state["cumulative_irrigation"]
        before_n = branch_state["cumulative_nitrogen"]
        before_grnwt = branch_state["grnwt"]
        obs, reward, terminated, truncated, _ = env.step(candidate)
        target_details = action_info(env)
        after = state_snapshot(env, obs, target_dap + 1)
        td = float(reward) + (0.0 if terminated or truncated else gamma * task184.state_value(model, obs))
        reward_scaled = float(reward)
        reward_unscaled = target_details["reward_unscaled"]
        first_i240 = int(target_dap) if after["cumulative_irrigation"] >= 240.0 - 1e-6 else prefix_first_cap
        future_nonzero = [action_text(target_dap, env, candidate)]
        after_target_nonzero: list[str] = []
        mask_at_97_104: dict[int, int] = {}
        action_count_after_target = 0
        while not bool(terminated or truncated):
            state = state_snapshot(env, obs, target_dap + 1)
            dap = int(state["dap"])
            follow_mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            if dap in (97, 104):
                mask_at_97_104[dap] = int(follow_mask.sum())
            selected_raw, _ = model.predict(obs, action_masks=follow_mask, deterministic=True)
            selected = action_index(selected_raw)
            if not bool(follow_mask[selected]):
                raise RuntimeError(f"Frozen PPO returned illegal action {selected} at DAP {dap}")
            obs, step_reward, terminated, truncated, _ = env.step(selected)
            details = action_info(env)
            reward_scaled += float(step_reward)
            reward_unscaled += details["reward_unscaled"]
            action_count_after_target += 1
            follow_text = action_text(dap, env, selected)
            future_nonzero.append(follow_text)
            after_target_nonzero.append(follow_text)
            after = state_snapshot(env, obs, dap + 1)
            if not np.isfinite(first_i240) and after["cumulative_irrigation"] >= 240.0 - 1e-6:
                first_i240 = dap
        final_state = state_snapshot(env, obs, target_dap + 1)
        final_i = final_state["cumulative_irrigation"]
        final_n = final_state["cumulative_nitrogen"]
        final_grnwt = final_state["grnwt"]
        base.update({
            "dssat_success": True,
            "actor_probability_for_candidate": float(probs[candidate]),
            "critic_value_before": value_before,
            "target_reward_scaled": float(reward),
            "target_reward_unscaled": target_details["reward_unscaled"],
            "target_resource_cost_unscaled": target_details["resource_cost_unscaled"],
            "one_step_td_scaled": td,
            "remaining_reward_return_scaled": reward_scaled,
            "remaining_reward_return_unscaled": reward_unscaled,
            "final_grnwt": final_grnwt,
            "total_irrigation": final_i,
            "total_n": final_n,
            "PFP_N": final_grnwt / final_n if final_n > 1e-9 else np.nan,
            "remaining_grnwt_increment": final_grnwt - before_grnwt,
            "remaining_irrigation": final_i - before_i,
            "remaining_n": final_n - before_n,
            "remaining_profit_simple": (final_grnwt - before_grnwt) - 1.1 * (final_i - before_i) - 1.58 * (final_n - before_n),
            "full_season_profit_simple": final_grnwt - 1.1 * final_i - 1.58 * final_n,
            "first_i240_dap": first_i240,
            "dap97_mask_legal_action_count": mask_at_97_104.get(97, np.nan),
            "dap104_mask_legal_action_count": mask_at_97_104.get(104, np.nan),
            "dap97_104_noop_only": bool(mask_at_97_104.get(97) == 1 and mask_at_97_104.get(104) == 1),
            "post_target_action_count": action_count_after_target,
            "future_nonzero_action_sequence": "; ".join(filter(None, future_nonzero)),
            "after_target_nonzero_action_sequence": "; ".join(filter(None, after_target_nonzero)),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "prefix_replay_max_abs_error": mismatch,
        })
        return base
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    finally:
        env.close()


def comparison_tables(branches: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = branches[branches["dssat_success"].astype(bool)].copy()
    key = ["checkpoint_step", "scenario_key", "weather_variant", "target_dap"]
    rows: list[dict[str, Any]] = []
    for values, group in ok.groupby(key, sort=True):
        by_profit = group.sort_values(["remaining_profit_simple", "candidate_action_index"], ascending=[False, True])
        by_return = group.sort_values(["remaining_reward_return_scaled", "candidate_action_index"], ascending=[False, True])
        by_td = group.sort_values(["one_step_td_scaled", "candidate_action_index"], ascending=[False, True])
        true_best, return_best, td_best = by_profit.iloc[0], by_return.iloc[0], by_td.iloc[0]
        actor = targets.set_index(key).loc[values]
        rows.append({
            **dict(zip(key, values)),
            "successful_candidate_count": len(group),
            "true_profit_top_action": int(true_best.candidate_action_index),
            "true_profit_top_pair": true_best.candidate_pair,
            "true_profit_top_remaining_profit": float(true_best.remaining_profit_simple),
            "reward_return_top_action": int(return_best.candidate_action_index),
            "td_top_action": int(td_best.candidate_action_index),
            "td_top_pair": td_best.candidate_pair,
            "td_top_matches_true_profit_top": bool(int(td_best.candidate_action_index) == int(true_best.candidate_action_index)),
            "reward_return_top_matches_true_profit_top": bool(int(return_best.candidate_action_index) == int(true_best.candidate_action_index)),
            "td_top_matches_reward_return_top": bool(int(td_best.candidate_action_index) == int(return_best.candidate_action_index)),
            "actor_selected_action": int(actor.actor_selected_action),
            "actor_selected_pair": str(actor.actor_selected_pair),
            "actor_matches_true_profit_top": bool(int(actor.actor_selected_action) == int(true_best.candidate_action_index)),
            "actor_matches_td_top": bool(int(actor.actor_selected_action) == int(td_best.candidate_action_index)),
            "td_minus_noop_remaining_profit": float(td_best.remaining_profit_simple - group.loc[group.candidate_action_index.eq(0), "remaining_profit_simple"].iloc[0]),
        })
    aligned = pd.DataFrame(rows)
    calibration = aligned[[*key, "successful_candidate_count", "td_top_action", "td_top_pair", "true_profit_top_action", "true_profit_top_pair", "td_top_matches_true_profit_top", "reward_return_top_action", "reward_return_top_matches_true_profit_top", "td_top_matches_reward_return_top", "td_minus_noop_remaining_profit"]].copy()
    actor = aligned[[*key, "actor_selected_action", "actor_selected_pair", "td_top_action", "true_profit_top_action", "actor_matches_true_profit_top", "actor_matches_td_top"]].copy()
    return calibration, actor


def write_doc(result: dict[str, Any], calibration: pd.DataFrame, actor: pd.DataFrame) -> None:
    def rate(frame: pd.DataFrame, column: str) -> str:
        return "NA" if frame.empty else f"{100.0 * frame[column].astype(bool).mean():.1f}% ({int(frame[column].astype(bool).sum())}/{len(frame)})"
    lines = [
        "# Task192：SYA PPO critic—整季收益反事实校准审计记录", "",
        "## 1. 实验目的", "",
        "检验冻结的50K和100K PPO在早中期状态上的局部 one-step TD 评价，是否与候选动作后继续由同一 PPO 控制至收获得到的真实剩余 profit 一致。", "",
        "## 2. 与Task189的区别", "",
        "Task189 在候选动作执行一步后停止，以 `即时 reward + gamma×V(next)` 作为局部代理。本轮在候选动作后继续使用同一冻结 PPO运行到收获；因此可比较 TD、actor 与真实剩余整季 profit 的动作排序。它仍不是全局最优控制搜索。", "",
        "## 3. 输入、冻结条件和运行方式", "",
        f"- checkpoint：Task191 `{result['checkpoint_steps']}`，模型根目录：`{result['model_root']}`。Task191的50K已与Task190严格复现，故优先在同一目录取50K和100K。",
        "- 天气：2005派生六情景 original、early_dry、mid_dry、late_dry、early_wet、mid_wet。",
        "- 目标日：DAP2、31、61、91；候选为目标日合法的 I0/I15/I30/I45，且N均为0。",
        "- 目标日前：确定性 PPO 历史动作；目标日：仅替换候选动作；目标日后：同一 checkpoint 的确定性 PPO 动作直至收获。",
        "- reward return 是目标日（含）到收获的环境 reward 累加；raw remaining profit 为 `GRNWT增量 - 1.1×后续灌溉 - 1.58×后续N`。两者分别报告，不能混用。", "",
        "## 4. 代码、配置、输入和命令", "",
        f"- prompt：`{rel(PROMPT)}`", f"- 实际config：`{result['actual_config']}`", f"- 主脚本：`{rel(Path(__file__))}`", "- 复用方法：`src/run_184_sya_ppo_critic_td_action_value_audit.py`（目标状态/TD）与`src/run_189_sya_gap1_early_irrigation_reward_critic_audit.py`（gap=1 runtime）。", "",
        "```powershell", "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_192_sya_critic_fullseason_counterfactual_audit.py --config configs/192_sya_critic_fullseason_counterfactual_audit.json", "```", "",
        "## 5. 执行结果", "",
        f"- preflight：`{result['preflight_status']}`；物理天气 gate：`{result['physical_gate_passed']}`。",
        f"- 目标状态：`{result['target_state_count']}`；候选记录：`{result['candidate_record_count']}`；DSSAT成功/失败：`{result['dssat_success_count']}/{result['dssat_failure_count']}`。",
        f"- one-step TD top 与真实 profit top 一致：{rate(calibration, 'td_top_matches_true_profit_top')}。",
        f"- actor 与真实 profit top 一致：{rate(actor, 'actor_matches_true_profit_top')}。", "",
        "## 6. 状态级对照", "", calibration.round(5).to_markdown(index=False) if not calibration.empty else "无成功分支。", "",
        "## 7. 已验证事实、解释边界和下一步", "",
        result["conclusion"], "",
        "已验证的是：在这些固定状态、固定四个候选动作、以及‘候选动作后仍跟随当前PPO’的条件下，TD/actor排序与真实结果的对应关系。未验证的是候选动作后重新优化策略的全局最优收益，也不能仅凭该审计断言具体代码存在bug。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    global OUT
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    requested_out = Path(str(cfg.get("output_dir", rel(OUT))))
    if requested_out.is_absolute() or ".." in requested_out.parts:
        raise ValueError("Task192 output_dir must be a project-relative directory.")
    OUT = ROOT / requested_out
    gate, manifest, split = preflight(cfg)
    OUT.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "debug", "evaluation", "logs"):
        (OUT / folder).mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "192_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Step 0/4] Preflight passed: frozen 50K/100K, six scenarios, no training.", flush=True)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    task178 = task184.task178
    task168 = task178.task177.task174.task173.task172.task168
    old_globals = {key: getattr(task178, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    task178.TASK_ID, task178.TASK_NAME, task178.OUT, task178.DOC, task178.PROMPT = TASK_ID, TASK_NAME, OUT, DOC, PROMPT
    patched_task178, original_patch, original_restore = task189.patch_gap1_runtime()
    saved_runtime: Any = None
    targets: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    try:
        saved_runtime = patched_task178.patch_runtime_178(manifest)
        engine, base03222 = task168.engine, task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        max_steps = int(runtime_config["runtime"]["max_steps"])
        for checkpoint in CHECKPOINTS:
            model = MaskablePPO.load(str(model_path(checkpoint)), device="cpu")
            gamma = float(model.gamma)
            print(f"[Step 1/4] Loaded frozen checkpoint {checkpoint}.", flush=True)
            for scenario_key, weather_variant in SCENARIOS.items():
                for target_dap in TARGET_DAPS:
                    ref = base03222.base.make_env(runtime_config, env_config, STATION, scenario_key, SEED, f"SYA_2005_{weather_variant}_192_ckpt{checkpoint}_reference_dap{target_dap}", evaluation=True)
                    prefix: list[int] = []
                    prefix_nonzero: list[str] = []
                    try:
                        obs, _ = ref.reset()
                        found = False
                        for _ in range(max_steps):
                            state = state_snapshot(ref, obs, -1)
                            mask = np.asarray(get_action_masks(ref), dtype=bool).reshape(-1)
                            selected_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
                            selected = action_index(selected_raw)
                            if int(state["dap"]) == target_dap:
                                found = True
                                target = record_target(checkpoint, scenario_key, weather_variant, target_dap, ref, obs, prefix, model)
                                target["prefix_nonzero_action_sequence"] = "; ".join(prefix_nonzero)
                                targets.append(target)
                                break
                            prefix.append(selected)
                            text = action_text(int(state["dap"]), ref, selected)
                            if text:
                                prefix_nonzero.append(text)
                            obs, _, terminated, truncated, _ = ref.step(selected)
                            if terminated or truncated:
                                raise RuntimeError("Reference episode ended before target DAP")
                        if not found:
                            raise RuntimeError(f"Target DAP {target_dap} was not reached")
                    except Exception as exc:
                        targets.append({"checkpoint_step": checkpoint, "scenario_key": scenario_key, "weather_variant": weather_variant, "target_dap": target_dap, "reference_success": False, "error": f"{type(exc).__name__}: {exc}"})
                        progress.append({"checkpoint_step": checkpoint, "weather_variant": weather_variant, "target_dap": target_dap, "status": "reference_failed", "error": str(exc)})
                        save_partial(targets, branches, progress)
                        continue
                    finally:
                        ref.close()
                    print(f"[Step 2/4] ckpt={checkpoint} scenario={weather_variant} DAP={target_dap}; serial branches.", flush=True)
                    for candidate in KEY_ACTIONS:
                        row = run_branch(base03222, runtime_config, env_config, model, checkpoint, scenario_key, weather_variant, target_dap, prefix, target, candidate, gamma)
                        branches.append(row)
                    progress.append({"checkpoint_step": checkpoint, "weather_variant": weather_variant, "target_dap": target_dap, "status": "completed", "candidate_records": len(KEY_ACTIONS), "successes": int(sum(bool(r.get("dssat_success")) for r in branches[-len(KEY_ACTIONS):]))})
                    save_partial(targets, branches, progress)
    except Exception:
        (OUT / "192_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if saved_runtime is not None:
            patched_task178.restore_runtime_178(saved_runtime)
        patched_task178.patch_runtime_178 = original_patch
        patched_task178.restore_runtime_178 = original_restore
        for key, value in old_globals.items():
            setattr(task178, key, value)

    target_frame, branch_frame = pd.DataFrame(targets), pd.DataFrame(branches)
    calibration, actor = comparison_tables(branch_frame, target_frame)
    target_frame.to_csv(OUT / "evaluation" / "192_target_state_values.csv", index=False, encoding="utf-8-sig")
    branch_frame.to_csv(OUT / "evaluation" / "192_fullseason_counterfactual.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUT / "evaluation" / "192_critic_calibration_summary.csv", index=False, encoding="utf-8-sig")
    actor.to_csv(OUT / "evaluation" / "192_actor_vs_true_best_action.csv", index=False, encoding="utf-8-sig")
    success = int(branch_frame.get("dssat_success", pd.Series(dtype=bool)).astype(bool).sum())
    failures = int(len(branch_frame) - success)
    td_profit_rate = float(calibration["td_top_matches_true_profit_top"].mean()) if not calibration.empty else np.nan
    actor_profit_rate = float(actor["actor_matches_true_profit_top"].mean()) if not actor.empty else np.nan
    reward_profit_rate = float(calibration["reward_return_top_matches_true_profit_top"].mean()) if not calibration.empty else np.nan
    td_reward_rate = float(calibration["td_top_matches_reward_return_top"].mean()) if not calibration.empty else np.nan
    same_errors = int(((calibration.pivot_table(index=["scenario_key", "weather_variant", "target_dap"], columns="checkpoint_step", values="td_top_matches_true_profit_top", aggfunc="first").fillna(True) == False).all(axis=1)).sum()) if not calibration.empty else 0
    label = "inconclusive"
    conclusion = "可完成的分支不足以给出方向性解释；应先检查失败分支和输入一致性。"
    if np.isfinite(td_reward_rate) and np.isfinite(reward_profit_rate) and td_reward_rate >= 0.5 and reward_profit_rate < 0.5:
        label = "supports_reward_objective_misalignment"
        conclusion = "在本轮固定状态与候选动作中，one-step TD更常与后续环境reward排序一致、却与raw remaining profit排序不一致；这支持优先检查reward的产量、资源成本和应力项权重，而不能先把原因归为critic。"
    elif np.isfinite(td_profit_rate) and td_profit_rate < 0.5 and same_errors > 0:
        label = "supports_critic_or_temporal_credit_miscalibration"
        conclusion = "在多个状态中，one-step TD最高动作与真实剩余profit最高动作不一致，且50K和100K存在相同误判状态；这支持critic价值/时间信用分配未正确反映当前PPO后续控制下的长期后果。它不是具体代码bug的证明。"
    elif np.isfinite(actor_profit_rate) and actor_profit_rate < 0.5:
        label = "supports_actor_policy_misalignment"
        conclusion = "真实剩余profit的最佳候选动作在本轮状态中并不总等于actor动作，而TD也未给出稳定解释；应优先检查actor对观测、熵和策略更新的使用。"
    else:
        conclusion = "在本轮固定状态、四个候选动作和冻结PPO后续控制的范围内，尚未出现足够强的单一误判模式；需要按DAP、天气和mask进一步分层。"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}", "status": "completed", "training_executed": False,
        "actual_config": rel(config_path),
        "preflight_status": "passed", "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "model_root": rel(MODEL_ROOT), "checkpoint_steps": CHECKPOINTS, "target_daps": TARGET_DAPS,
        "scenario_count": len(SCENARIOS), "target_state_count": int(len(target_frame)), "candidate_record_count": int(len(branch_frame)),
        "dssat_success_count": success, "dssat_failure_count": failures,
        "td_top_matches_true_profit_top_rate": td_profit_rate,
        "actor_matches_true_profit_top_rate": actor_profit_rate,
        "reward_return_top_matches_true_profit_top_rate": reward_profit_rate,
        "td_top_matches_reward_return_top_rate": td_reward_rate,
        "same_td_misclassification_50k_100k_states": same_errors,
        "primary_label": label,
        "conclusion": conclusion,
        "outputs": {"target_states": rel(OUT / "evaluation" / "192_target_state_values.csv"), "branches": rel(OUT / "evaluation" / "192_fullseason_counterfactual.csv"), "calibration": rel(OUT / "evaluation" / "192_critic_calibration_summary.csv"), "actor": rel(OUT / "evaluation" / "192_actor_vs_true_best_action.csv")},
    }
    (OUT / "192_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(result, calibration, actor)
    print("[Step 4/4] Completed full-season counterfactual audit.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

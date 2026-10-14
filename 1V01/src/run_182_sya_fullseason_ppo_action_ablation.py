"""Task 182: full-season ablation of actual PPO irrigation events.

The main experiment is kept as a readable linear workflow:

1. load the already-trained task-177 PPO-2000 and the six 2005 weather scenarios;
2. generate a complete deterministic PPO action schedule for each scenario;
3. replay that fixed schedule from DAP 0 to harvest under four treatments;
4. delete only DAP 97 and/or DAP 104 irrigation events in the ablation branches;
5. compare final yield, reward, irrigation, nitrogen and stress against full PPO.

No PPO training is performed here.  Future actions are not re-optimized after
an ablation; this is an event-contribution test, not a closed-loop compensation
test.
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
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_178_sya_weather_signal_pathway_audit as task178


TASK_ID = "182"
TASK_NAME = "sya_fullseason_ppo_action_ablation"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_fullseason_ppo_action_ablation.md"
MODEL = (
    ROOT
    / "benchmark_results"
    / "177_sya_dense_yield_credit_smoke2k"
    / "models"
    / "SYA"
    / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
)

STATION = "SYA"
SEED = 0
CHECKPOINT = 2000
NO_OP_ACTION = 0
TARGET_ABLATION_DAPS = [97, 104]
SCENARIOS = {
    2005: "original",
    2006: "early_dry",
    2007: "mid_dry",
    2008: "late_dry",
    2009: "early_wet",
    2010: "mid_wet",
}
TREATMENTS = [
    {"treatment": "full_ppo", "drop_daps": []},
    {"treatment": "drop_dap97", "drop_daps": [97]},
    {"treatment": "drop_dap104", "drop_daps": [104]},
    {"treatment": "drop_dap97_dap104", "drop_daps": [97, 104]},
]


def number(value: Any, default: float = np.nan) -> float:
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
        result = float(values[0]) if values.size else default
        return result if np.isfinite(result) else default
    except Exception:
        return default


def one_action(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def mask_text(mask: np.ndarray) -> str:
    return "".join("1" if bool(item) else "0" for item in np.asarray(mask, dtype=bool).reshape(-1))


def grid_action(env, action_index: int) -> dict[str, float]:
    raw = dict(list(getattr(env, "grid"))[int(action_index)])
    return {
        "irrigation_mm": number(raw.get("amir"), 0.0),
        "nitrogen_kg_ha": number(raw.get("anfer"), 0.0),
    }


def current_state(env, obs: Any, fallback_dap: int = -1) -> dict[str, float]:
    observed = dict(getattr(env, "last_obs_dict", {}) or {})
    obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
    safety = getattr(env, "safety_state", None)
    return {
        "dap": int(round(number(observed.get("dap"), fallback_dap))),
        "rain": number(obs_array[-1], 0.0),
        "swfac": number(observed.get("swfac")),
        "nstres": number(observed.get("nstres")),
        "grnwt": number(observed.get("grnwt"), 0.0),
        "topwt": number(observed.get("topwt")),
        "xlai": number(observed.get("xlai")),
        "cumulative_irrigation": number(getattr(safety, "cumulative_irrigation", np.nan)),
        "cumulative_nitrogen": number(getattr(safety, "cumulative_nitrogen", np.nan)),
    }


def nonzero_signature(frame: pd.DataFrame, action_i_col: str, action_n_col: str) -> str:
    if frame.empty:
        return ""
    out: list[str] = []
    for row in frame.itertuples(index=False):
        irrigation = float(getattr(row, action_i_col))
        nitrogen = float(getattr(row, action_n_col))
        if irrigation > 1e-9 or nitrogen > 1e-9:
            out.append(f"DAP{int(row.dap)} I{irrigation:g}/N{nitrogen:g}")
    return "; ".join(out)


def episode_summary(daily: pd.DataFrame, scenario_key: int, weather_variant: str, treatment: str, drop_daps: list[int]) -> dict[str, Any]:
    irrigation = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    nitrogen = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    rewards = pd.to_numeric(daily["reward"], errors="coerce").fillna(0.0)
    reward_unscaled = pd.to_numeric(daily["reward_unscaled"], errors="coerce").fillna(0.0)
    yield_component = pd.to_numeric(daily["yield_component"], errors="coerce").fillna(0.0)
    resource_cost = pd.to_numeric(daily["resource_cost"], errors="coerce").fillna(0.0)
    stress_bonus = pd.to_numeric(daily["stress_relief_bonus"], errors="coerce").fillna(0.0)
    swfac = pd.to_numeric(daily["swfac_after"], errors="coerce")
    nstres = pd.to_numeric(daily["nstres_after"], errors="coerce")
    final_grnwt = float(pd.to_numeric(daily["grnwt_after"], errors="coerce").iloc[-1])
    total_i = float(irrigation.sum())
    total_n = float(nitrogen.sum())
    return {
        "checkpoint_step": CHECKPOINT,
        "scenario_key": int(scenario_key),
        "weather_variant": weather_variant,
        "treatment": treatment,
        "drop_daps": ",".join(str(x) for x in drop_daps),
        "episode_length": int(len(daily)),
        "completed_to_terminal": bool(daily["terminated"].astype(bool).iloc[-1] or daily["truncated"].astype(bool).iloc[-1]),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_grnwt - 1.1 * total_i - 1.58 * total_n,
        "PFP_N": final_grnwt / total_n if total_n > 1e-9 else np.nan,
        "reward_sum": float(rewards.sum()),
        "reward_unscaled_sum": float(reward_unscaled.sum()),
        "yield_component_sum": float(yield_component.sum()),
        "resource_cost_sum": float(resource_cost.sum()),
        "stress_relief_bonus_sum": float(stress_bonus.sum()),
        "irrigation_event_count": int((irrigation > 1e-9).sum()),
        "n_event_count": int((nitrogen > 1e-9).sum()),
        "first_irrigation_dap": int(daily.loc[irrigation > 1e-9, "dap"].iloc[0]) if (irrigation > 1e-9).any() else -1,
        "first_n_dap": int(daily.loc[nitrogen > 1e-9, "dap"].iloc[0]) if (nitrogen > 1e-9).any() else -1,
        "max_swfac": float(swfac.max()),
        "swfac_days_gt_0p001": int((swfac > 0.001).sum()),
        "swfac_days_gt_0p01": int((swfac > 0.01).sum()),
        "swfac_days_gt_0p05": int((swfac > 0.05).sum()),
        "max_nstres": float(nstres.max()),
        "nstres_days_gt_0p001": int((nstres > 0.001).sum()),
        "nstres_days_gt_0p01": int((nstres > 0.01).sum()),
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()),
        "ablated_event_count": int(daily["ablated_event"].astype(bool).sum()),
        "invalid_action_rows": int((~daily["applied_action_legal"].astype(bool)).sum()),
        "dap_schedule_mismatch_rows": int(daily["dap_schedule_mismatch"].astype(bool).sum()),
        "applied_action_sequence": nonzero_signature(daily, "safe_action_amir", "safe_action_anfer"),
        "baseline_action_sequence": nonzero_signature(daily, "scheduled_irrigation_mm", "scheduled_nitrogen_kg_ha"),
    }


def run(config_path: Path) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # Step 0. Read the explicit experiment contract.
    # ------------------------------------------------------------------
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 182 is read-only and must not train PPO.")
    if int(cfg["checkpoint_step"]) != CHECKPOINT:
        raise ValueError("Task 182 must use PPO checkpoint 2000.")
    if [int(x) for x in cfg["target_ablation_daps"]] != TARGET_ABLATION_DAPS:
        raise ValueError("Target ablation DAPs drifted from [97, 104].")
    if int(cfg["replacement_action_index"]) != NO_OP_ACTION:
        raise ValueError("Task 182 expects action 0 as the no-op replacement.")
    if bool(cfg.get("closed_loop_compensation")):
        raise ValueError("Task 182 is fixed-schedule ablation, not closed-loop compensation.")
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {relative(OUT)}")

    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "debug").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "logs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "182_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[Step 0/5] Contract checked: no training; PPO-2000; fixed-schedule ablation.", flush=True)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    # ------------------------------------------------------------------
    # Step 1. Patch only the already-audited task-177 runtime.
    # ------------------------------------------------------------------
    saved_task178_globals = {
        "TASK_ID": task178.TASK_ID,
        "TASK_NAME": task178.TASK_NAME,
        "OUT": task178.OUT,
        "DOC": task178.DOC,
        "PROMPT": task178.PROMPT,
    }
    task178.TASK_ID = TASK_ID
    task178.TASK_NAME = TASK_NAME
    task178.OUT = OUT
    task178.DOC = DOC
    task178.PROMPT = PROMPT
    saved_runtime = task178.patch_runtime_178(manifest)

    schedule_rows: list[dict[str, Any]] = []
    schedule_summary_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    deleted_event_rows: list[dict[str, Any]] = []

    try:
        engine = task168.engine
        base03222 = task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        model = MaskablePPO.load(str(MODEL), device="cpu")
        max_steps = int(runtime_config["runtime"]["max_steps"])

        # --------------------------------------------------------------
        # Step 2. For each weather scenario, generate the closed-loop PPO
        # action schedule once. This schedule becomes the fixed treatment
        # backbone in Step 3.
        # --------------------------------------------------------------
        schedules: dict[int, pd.DataFrame] = {}
        for scenario_key, weather_variant in SCENARIOS.items():
            print(f"[Step 1/5] Generate baseline PPO schedule: {weather_variant}", flush=True)
            env = base03222.base.make_env(
                runtime_config,
                env_config,
                STATION,
                scenario_key,
                SEED,
                f"SYA_2005_{weather_variant}_182_schedule",
                evaluation=True,
            )
            local_rows: list[dict[str, Any]] = []
            cumulative_reward = 0.0
            cumulative_reward_unscaled = 0.0
            try:
                obs, info = env.reset()
                for step_index in range(max_steps):
                    state_before = current_state(env, obs, step_index)
                    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                    action_raw, _ = model.predict(
                        np.asarray(obs, dtype=np.float32).reshape(-1),
                        action_masks=mask,
                        deterministic=True,
                    )
                    action_index = one_action(action_raw)
                    if not bool(mask[action_index]):
                        raise RuntimeError(
                            f"Baseline PPO action {action_index} invalid at {weather_variant} DAP {state_before['dap']}"
                        )
                    scheduled = grid_action(env, action_index)
                    obs_after, reward, terminated, truncated, info = env.step(action_index)
                    state_after = current_state(env, obs_after, step_index + 1)
                    action_info = dict(getattr(env, "last_action_info", {}) or {})
                    reward_unscaled = number(action_info.get("reward_unscaled"), reward)
                    cumulative_reward += float(reward)
                    cumulative_reward_unscaled += reward_unscaled
                    row = {
                        "checkpoint_step": CHECKPOINT,
                        "scenario_key": scenario_key,
                        "weather_variant": weather_variant,
                        "step_index": step_index,
                        "dap": int(state_before["dap"]),
                        "mask": mask_text(mask),
                        "ppo_action_index": action_index,
                        "ppo_action_legal": True,
                        "scheduled_irrigation_mm": scheduled["irrigation_mm"],
                        "scheduled_nitrogen_kg_ha": scheduled["nitrogen_kg_ha"],
                        "safe_action_amir": number(action_info.get("safe_action_amir"), scheduled["irrigation_mm"]),
                        "safe_action_anfer": number(action_info.get("safe_action_anfer"), scheduled["nitrogen_kg_ha"]),
                        "rain_before": state_before["rain"],
                        "swfac_before": state_before["swfac"],
                        "swfac_after": state_after["swfac"],
                        "nstres_before": state_before["nstres"],
                        "nstres_after": state_after["nstres"],
                        "grnwt_before": state_before["grnwt"],
                        "grnwt_after": state_after["grnwt"],
                        "reward": float(reward),
                        "reward_unscaled": reward_unscaled,
                        "cumulative_reward": cumulative_reward,
                        "cumulative_reward_unscaled": cumulative_reward_unscaled,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    local_rows.append(row)
                    schedule_rows.append(row)
                    obs = obs_after
                    if bool(terminated or truncated):
                        break
            finally:
                env.close()

            schedule = pd.DataFrame(local_rows)
            schedules[scenario_key] = schedule
            schedule_summary_rows.append(
                {
                    "scenario_key": scenario_key,
                    "weather_variant": weather_variant,
                    "episode_length": int(len(schedule)),
                    "final_grnwt": float(schedule["grnwt_after"].iloc[-1]),
                    "total_irrigation": float(pd.to_numeric(schedule["safe_action_amir"], errors="coerce").sum()),
                    "total_n": float(pd.to_numeric(schedule["safe_action_anfer"], errors="coerce").sum()),
                    "action_sequence": nonzero_signature(schedule, "safe_action_amir", "safe_action_anfer"),
                }
            )
            pd.DataFrame(schedule_rows).to_csv(
                OUT / "debug" / "182_baseline_ppo_schedule_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

        pd.DataFrame(schedule_rows).to_csv(
            OUT / "debug" / "182_baseline_ppo_schedule.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(schedule_summary_rows).to_csv(
            OUT / "evaluation" / "182_schedule_reference_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # --------------------------------------------------------------
        # Step 3. Replay the fixed schedule under four treatments.
        # --------------------------------------------------------------
        expected_branches = len(SCENARIOS) * len(TREATMENTS)
        branch_count = 0
        for scenario_key, weather_variant in SCENARIOS.items():
            schedule = schedules[scenario_key].reset_index(drop=True)
            if len(schedule) < 1:
                raise RuntimeError(f"Empty baseline schedule: {weather_variant}")
            for treatment_item in TREATMENTS:
                branch_count += 1
                treatment = str(treatment_item["treatment"])
                drop_daps = [int(x) for x in treatment_item["drop_daps"]]
                print(
                    f"[Step 2/5] Branch {branch_count}/{expected_branches}: "
                    f"{weather_variant}, treatment={treatment}",
                    flush=True,
                )
                env = base03222.base.make_env(
                    runtime_config,
                    env_config,
                    STATION,
                    scenario_key,
                    SEED,
                    f"SYA_2005_{weather_variant}_182_{treatment}",
                    evaluation=True,
                )
                local_daily_rows: list[dict[str, Any]] = []
                cumulative_reward = 0.0
                cumulative_reward_unscaled = 0.0
                cumulative_yield_component = 0.0
                cumulative_resource_cost = 0.0
                cumulative_stress_bonus = 0.0
                try:
                    obs, info = env.reset()
                    for step_index in range(max_steps):
                        if step_index >= len(schedule):
                            raise RuntimeError(f"Schedule exhausted before env termination: {weather_variant}")
                        state_before = current_state(env, obs, step_index)
                        scheduled_row = schedule.iloc[step_index]
                        scheduled_dap = int(scheduled_row["dap"])
                        scheduled_action = int(scheduled_row["ppo_action_index"])
                        scheduled_grid = grid_action(env, scheduled_action)

                        applied_action = scheduled_action
                        ablated_event = False
                        if int(state_before["dap"]) in drop_daps and scheduled_grid["irrigation_mm"] > 1e-9:
                            applied_action = NO_OP_ACTION
                            ablated_event = True

                        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                        applied_legal = bool(mask[applied_action])
                        if not applied_legal:
                            raise RuntimeError(
                                f"Applied action {applied_action} invalid in {weather_variant} "
                                f"{treatment} at DAP {state_before['dap']}; mask={mask_text(mask)}"
                            )
                        applied_grid = grid_action(env, applied_action)
                        obs_after, reward, terminated, truncated, info = env.step(applied_action)
                        state_after = current_state(env, obs_after, step_index + 1)
                        action_info = dict(getattr(env, "last_action_info", {}) or {})

                        reward_unscaled = number(action_info.get("reward_unscaled"), reward)
                        yield_component = number(action_info.get("yield_component"), 0.0)
                        resource_cost = number(action_info.get("resource_cost"), 0.0)
                        stress_bonus = number(action_info.get("stress_relief_bonus"), 0.0)
                        safe_i = number(action_info.get("safe_action_amir"), applied_grid["irrigation_mm"])
                        safe_n = number(action_info.get("safe_action_anfer"), applied_grid["nitrogen_kg_ha"])

                        cumulative_reward += float(reward)
                        cumulative_reward_unscaled += reward_unscaled
                        cumulative_yield_component += yield_component
                        cumulative_resource_cost += resource_cost
                        cumulative_stress_bonus += stress_bonus

                        row = {
                            "checkpoint_step": CHECKPOINT,
                            "scenario_key": scenario_key,
                            "weather_variant": weather_variant,
                            "treatment": treatment,
                            "drop_daps": ",".join(str(x) for x in drop_daps),
                            "step_index": step_index,
                            "dap": int(state_before["dap"]),
                            "scheduled_dap": scheduled_dap,
                            "dap_schedule_mismatch": bool(int(state_before["dap"]) != scheduled_dap),
                            "mask": mask_text(mask),
                            "scheduled_action_index": scheduled_action,
                            "scheduled_irrigation_mm": scheduled_grid["irrigation_mm"],
                            "scheduled_nitrogen_kg_ha": scheduled_grid["nitrogen_kg_ha"],
                            "applied_action_index": applied_action,
                            "applied_action_legal": applied_legal,
                            "applied_irrigation_mm": applied_grid["irrigation_mm"],
                            "applied_nitrogen_kg_ha": applied_grid["nitrogen_kg_ha"],
                            "ablated_event": ablated_event,
                            "safe_action_amir": safe_i,
                            "safe_action_anfer": safe_n,
                            "rain_before": state_before["rain"],
                            "swfac_before": state_before["swfac"],
                            "swfac_after": state_after["swfac"],
                            "nstres_before": state_before["nstres"],
                            "nstres_after": state_after["nstres"],
                            "grnwt_before": state_before["grnwt"],
                            "grnwt_after": state_after["grnwt"],
                            "grnwt_increment": number(action_info.get("grnwt_increment"), state_after["grnwt"] - state_before["grnwt"]),
                            "topwt_after": state_after["topwt"],
                            "xlai_after": state_after["xlai"],
                            "reward": float(reward),
                            "reward_unscaled": reward_unscaled,
                            "yield_component": yield_component,
                            "resource_cost": resource_cost,
                            "stress_relief_bonus": stress_bonus,
                            "cumulative_reward": cumulative_reward,
                            "cumulative_reward_unscaled": cumulative_reward_unscaled,
                            "cumulative_yield_component": cumulative_yield_component,
                            "cumulative_resource_cost": cumulative_resource_cost,
                            "cumulative_stress_relief_bonus": cumulative_stress_bonus,
                            "season_cumulative_irrigation": state_after["cumulative_irrigation"],
                            "season_cumulative_nitrogen": state_after["cumulative_nitrogen"],
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        daily_rows.append(row)
                        local_daily_rows.append(row)

                        if ablated_event:
                            deleted_event_rows.append(
                                {
                                    "scenario_key": scenario_key,
                                    "weather_variant": weather_variant,
                                    "treatment": treatment,
                                    "dap": int(state_before["dap"]),
                                    "scheduled_action_index": scheduled_action,
                                    "scheduled_irrigation_mm": scheduled_grid["irrigation_mm"],
                                    "scheduled_nitrogen_kg_ha": scheduled_grid["nitrogen_kg_ha"],
                                    "applied_action_index": applied_action,
                                    "applied_irrigation_mm": applied_grid["irrigation_mm"],
                                    "applied_nitrogen_kg_ha": applied_grid["nitrogen_kg_ha"],
                                }
                            )

                        obs = obs_after
                        if bool(terminated or truncated):
                            break
                finally:
                    env.close()

                local_daily = pd.DataFrame(local_daily_rows)
                summary_rows.append(
                    episode_summary(local_daily, scenario_key, weather_variant, treatment, drop_daps)
                )
                progress_rows.append(
                    {
                        "branch_number": branch_count,
                        "weather_variant": weather_variant,
                        "treatment": treatment,
                        "status": "completed",
                        "rows": int(len(local_daily)),
                        "ablated_event_count": int(local_daily["ablated_event"].astype(bool).sum()),
                    }
                )

                pd.DataFrame(daily_rows).to_csv(
                    OUT / "debug" / "182_daily_action_ablation_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                pd.DataFrame(summary_rows).to_csv(
                    OUT / "debug" / "182_episode_summary_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                pd.DataFrame(progress_rows).to_csv(
                    OUT / "debug" / "182_progress.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
    except Exception:
        (OUT / "182_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        task178.restore_runtime_178(saved_runtime)
        for key, value in saved_task178_globals.items():
            setattr(task178, key, value)

    # ------------------------------------------------------------------
    # Step 4. Build final comparison tables against full PPO.
    # ------------------------------------------------------------------
    daily_frame = pd.DataFrame(daily_rows)
    summary_frame = pd.DataFrame(summary_rows)
    deleted_event_frame = pd.DataFrame(deleted_event_rows)

    full = summary_frame[summary_frame["treatment"].eq("full_ppo")][
        [
            "scenario_key",
            "weather_variant",
            "final_grnwt",
            "profit_simple",
            "reward_sum",
            "reward_unscaled_sum",
            "total_irrigation",
            "total_n",
            "max_swfac",
            "max_nstres",
            "swfac_days_gt_0p05",
            "nstres_days_gt_0p05",
        ]
    ].rename(
        columns={
            "final_grnwt": "full_ppo_final_grnwt",
            "profit_simple": "full_ppo_profit_simple",
            "reward_sum": "full_ppo_reward_sum",
            "reward_unscaled_sum": "full_ppo_reward_unscaled_sum",
            "total_irrigation": "full_ppo_total_irrigation",
            "total_n": "full_ppo_total_n",
            "max_swfac": "full_ppo_max_swfac",
            "max_nstres": "full_ppo_max_nstres",
            "swfac_days_gt_0p05": "full_ppo_swfac_days_gt_0p05",
            "nstres_days_gt_0p05": "full_ppo_nstres_days_gt_0p05",
        }
    )
    effects = summary_frame.merge(full, on=["scenario_key", "weather_variant"], how="left", validate="many_to_one")
    effects["delta_final_grnwt_vs_full"] = effects["final_grnwt"] - effects["full_ppo_final_grnwt"]
    effects["delta_profit_simple_vs_full"] = effects["profit_simple"] - effects["full_ppo_profit_simple"]
    effects["delta_reward_sum_vs_full"] = effects["reward_sum"] - effects["full_ppo_reward_sum"]
    effects["delta_reward_unscaled_sum_vs_full"] = effects["reward_unscaled_sum"] - effects["full_ppo_reward_unscaled_sum"]
    effects["delta_total_irrigation_vs_full"] = effects["total_irrigation"] - effects["full_ppo_total_irrigation"]
    effects["delta_total_n_vs_full"] = effects["total_n"] - effects["full_ppo_total_n"]
    effects["delta_max_swfac_vs_full"] = effects["max_swfac"] - effects["full_ppo_max_swfac"]
    effects["delta_max_nstres_vs_full"] = effects["max_nstres"] - effects["full_ppo_max_nstres"]
    effects["delta_swfac_days_gt_0p05_vs_full"] = effects["swfac_days_gt_0p05"] - effects["full_ppo_swfac_days_gt_0p05"]
    effects["delta_nstres_days_gt_0p05_vs_full"] = effects["nstres_days_gt_0p05"] - effects["full_ppo_nstres_days_gt_0p05"]

    non_full_effects = effects[~effects["treatment"].eq("full_ppo")].copy()
    max_yield_loss = float((-non_full_effects["delta_final_grnwt_vs_full"]).max())
    min_yield_delta = float(non_full_effects["delta_final_grnwt_vs_full"].min())
    max_profit_gain = float(non_full_effects["delta_profit_simple_vs_full"].max())
    max_reward_unscaled_gain = float(non_full_effects["delta_reward_unscaled_sum_vs_full"].max())
    any_drop_reduces_yield = bool((non_full_effects["delta_final_grnwt_vs_full"] < -1e-6).any())
    any_drop_improves_profit = bool((non_full_effects["delta_profit_simple_vs_full"] > 1e-6).any())
    all_branches_complete = bool(summary_frame["completed_to_terminal"].all())
    all_actions_legal = bool(summary_frame["invalid_action_rows"].sum() == 0)
    all_daps_match_schedule = bool(summary_frame["dap_schedule_mismatch_rows"].sum() == 0)
    expected_deleted_events = len(SCENARIOS) * (1 + 1 + 2)
    observed_deleted_events = int(len(deleted_event_frame))

    if any_drop_reduces_yield and not any_drop_improves_profit:
        diagnosis = "ppo_irrigation_events_have_yield_value_under_fixed_schedule"
    elif any_drop_reduces_yield and any_drop_improves_profit:
        diagnosis = "ppo_irrigation_events_trade_yield_for_resource_cost_under_fixed_schedule"
    elif not any_drop_reduces_yield and any_drop_improves_profit:
        diagnosis = "ppo_irrigation_events_have_no_yield_value_and_reduce_profit_under_fixed_schedule"
    else:
        diagnosis = "ppo_irrigation_event_ablation_has_no_material_final_effect_under_fixed_schedule"

    # ------------------------------------------------------------------
    # Step 5. Save outputs and a concise machine-readable result.
    # ------------------------------------------------------------------
    daily_frame.to_csv(OUT / "evaluation" / "182_daily_action_ablation.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(OUT / "evaluation" / "182_episode_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "evaluation" / "182_ablation_effects_vs_full_ppo.csv", index=False, encoding="utf-8-sig")
    deleted_event_frame.to_csv(OUT / "evaluation" / "182_deleted_event_check.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "mode": "read_only_fullseason_fixed_schedule_ablation",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "checkpoint_step": CHECKPOINT,
        "station": STATION,
        "scenario_count": len(SCENARIOS),
        "treatment_count": len(TREATMENTS),
        "branch_count": int(len(summary_frame)),
        "daily_rows": int(len(daily_frame)),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "closed_loop_compensation": False,
        "quality_control": {
            "all_branches_complete": all_branches_complete,
            "all_actions_legal": all_actions_legal,
            "all_daps_match_schedule": all_daps_match_schedule,
            "expected_deleted_events": expected_deleted_events,
            "observed_deleted_events": observed_deleted_events,
        },
        "diagnosis": {
            "label": diagnosis,
            "any_drop_reduces_yield": any_drop_reduces_yield,
            "any_drop_improves_profit": any_drop_improves_profit,
            "min_delta_final_grnwt_vs_full": min_yield_delta,
            "max_yield_loss_vs_full": max_yield_loss,
            "max_profit_gain_vs_full": max_profit_gain,
            "max_reward_unscaled_gain_vs_full": max_reward_unscaled_gain,
            "interpretation": "Fixed future PPO action schedule; not a closed-loop compensation experiment.",
        },
        "outputs": {
            "baseline_schedule": relative(OUT / "debug" / "182_baseline_ppo_schedule.csv"),
            "schedule_reference_summary": relative(OUT / "evaluation" / "182_schedule_reference_summary.csv"),
            "daily_ablation": relative(OUT / "evaluation" / "182_daily_action_ablation.csv"),
            "episode_summary": relative(OUT / "evaluation" / "182_episode_summary.csv"),
            "effects_vs_full_ppo": relative(OUT / "evaluation" / "182_ablation_effects_vs_full_ppo.csv"),
            "deleted_event_check": relative(OUT / "evaluation" / "182_deleted_event_check.csv"),
            "progress": relative(OUT / "debug" / "182_progress.csv"),
        },
    }
    (OUT / "182_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    DOC.write_text(
        "\n".join(
            [
                "# Task 182：SYA PPO 实际灌水动作全季消融记录",
                "",
                "> 这是自动生成的简要记录；正式解释请在运行后按 docs/EXPERIMENT_RECORD_TEMPLATE.md 扩展。",
                "",
                f"- 状态：completed；分支：{len(summary_frame)}；逐日记录：{len(daily_frame)}。",
                f"- 模型：task 177 PPO checkpoint {CHECKPOINT}。",
                "- 处理：full_ppo、drop_dap97、drop_dap104、drop_dap97_dap104。",
                f"- diagnosis：`{diagnosis}`。",
                f"- 最小产量差值 drop-full：`{min_yield_delta}` kg/ha。",
                f"- 最大 simple profit 增益 drop-full：`{max_profit_gain}`。",
                "",
                "## 输出",
                "",
                f"- `{relative(OUT / '182_result.json')}`",
                f"- `{relative(OUT / 'debug' / '182_baseline_ppo_schedule.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '182_daily_action_ablation.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '182_episode_summary.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '182_ablation_effects_vs_full_ppo.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '182_deleted_event_check.csv')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[Step 5/5] Completed. diagnosis={diagnosis}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

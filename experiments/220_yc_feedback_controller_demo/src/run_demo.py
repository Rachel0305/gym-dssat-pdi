"""Run experiment 220: YC2019 external daily feedback controller demo.

The runner is deliberately serial. One DSSAT environment is created per
candidate and closed before the next candidate starts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(EXPERIMENT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import ppo_safe_rendering
from controller import ControllerParams, DailyStateFeedbackController
from nsga2 import feasible_pareto, run_nsga2
from ppo_action_safety import normalize_action

baseline = None
direct_ppo = None
siteppo = None
parse_management_events = None


def _load_runtime_modules() -> None:
    global baseline, direct_ppo, siteppo, parse_management_events
    if baseline is not None:
        return
    import run_all_year_direct_action_safe_ppo as _direct_ppo
    import run_multisite_input_ic1_four_baseline_rebuild_034_00 as _baseline
    import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as _siteppo
    from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events as _parse_management_events

    direct_ppo = _direct_ppo
    baseline = _baseline
    siteppo = _siteppo
    parse_management_events = _parse_management_events

CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "experiment_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RESULTS = EXPERIMENT_ROOT / "results"
LOGS = EXPERIMENT_ROOT / "logs"
INPUT_ROOT = PROJECT_ROOT / CONFIG["input_root"]
BASELINE_DAILY = PROJECT_ROOT / "benchmark_results" / "055_02_yca_lowIC_four_baselines_static_level1" / "evaluation" / "055_02_baseline_daily.csv"
MAX_STEPS = int(CONFIG["runtime"]["max_steps"])
EXPERT_REFERENCE_YIELD = float(CONFIG["reference_yield_kg_ha"])
YIELD_FLOOR = EXPERT_REFERENCE_YIELD * float(CONFIG["yield_constraint_fraction"])
PARAMETER_SPACE = CONFIG["parameter_space"]

SCOPE_DECLARATION = """本任务是 **proof-of-concept demo**，目的仅为验证一条技术链是否能跑通：

> 外部 daily state-feedback controller（逐日读取胁迫状态、独立决定水/氮管理动作）
> + NSGA-II（自动标定 controller 参数）
> + DSSAT
> → 能否产生合理的、随生育期胁迫状态动态触发的水氮管理方案，以及合理的产量-资源 Pareto trade-off。

**本次 demo 不需要证明、也不要在报告中声称证明了以下任何一项：**

1. 该方法适用于全部五站点（本次只做单站点）；
2. 该方法优于课题组已发表的 NSGA-III 全国/全国县域框架；
3. 该方法优于已有 PPO 结果；
4. 水氮联合触发（本次水、氮两个 controller 彼此独立，不耦合）；
5. 跨年份/跨气象年的泛化能力（本次只用单一年份做 smoke test，不做 held-out 验证）。
"""

SUCCESS_CRITERIA = [
    "① daily controller 确实逐日读取状态并做出响应",
    "② action 确实被 DSSAT 正确接收并执行",
    "③ 改变 threshold 等参数会导致明显不同的管理行为",
    "④ 不同 controller 参数组合会形成不同的 yield-N-water trade-off",
    "⑤ NSGA-II 能找到一个正常展开（非退化、非全部挤在一点）的 Pareto front",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def ensure_dirs() -> None:
    for path in [RESULTS, LOGS, RESULTS / "configs", RESULTS / "snapshots", RESULTS / "plots"]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_env_config() -> dict[str, Any]:
    """Resolve the existing observed-year environment for exactly YC2019."""
    _load_runtime_modules()
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
    run_config = direct_ppo.load_yaml(baseline.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = 0
    run_config["paths"]["output_root"] = rel(RESULTS)
    run_config["runtime"]["max_steps"] = MAX_STEPS
    split = pd.read_csv(baseline.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selected = split[(split["station_code"].astype(str) == "YCA") & (split["year"] == int(CONFIG["year"]))].copy()
    if len(selected) != 1:
        raise RuntimeError("033_04 split registry does not contain exactly one YCA2019 row")
    # 034_00 的 builder 会先把 split 行与 scenario pool 合并；直接调用
    # direct_ppo.build_env_config 会因 split registry 不含 scenario_type 而失败。
    env_config = baseline.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = rel(RESULTS)
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = 0
    env_config["input_profile"] = CONFIG["input_profile"]
    direct_ppo.write_yaml(env_config, RESULTS / "configs" / "resolved_yca2019_env_config.yaml")
    return env_config


def make_controller_params(genome: dict[str, Any]) -> ControllerParams:
    fixed = CONFIG["fixed_windows"]
    return ControllerParams(
        n_threshold=float(genome["n_threshold"]),
        n_dose=int(genome["n_dose"]),
        n_min_interval_days=int(genome["n_min_interval_days"]),
        n_season_budget=int(genome["n_season_budget"]),
        n_start_dap=int(fixed["n_start_dap"]),
        n_stop_dap=int(fixed["n_stop_dap"]),
        water_threshold=float(genome["water_threshold"]),
        irrigation_dose=int(genome["irrigation_dose"]),
        irrigation_min_interval_days=int(genome["irrigation_min_interval_days"]),
        irrigation_season_budget=int(genome["irrigation_season_budget"]),
        irrigation_start_dap=int(fixed["irrigation_start_dap"]),
        irrigation_stop_dap=int(fixed["irrigation_stop_dap"]),
    )


def make_env(env_config: dict[str, Any], candidate_id: str):
    import gym
    from sb3_wrapper import GymDssatWrapper

    year_info = direct_ppo.find_year(env_config, "YCA", int(CONFIG["year"]))
    env_args = ppo_safe_rendering.build_env_args(
        station="YCA",
        year=int(CONFIG["year"]),
        planting_date=year_info["planting_date"],
        seed=0,
        config=env_config,
        run_tag=f"controller_{candidate_id}",
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
        linked_management=True,
    )
    env_args["log_saving_path"] = str(LOGS / f"dssat_{candidate_id}.log")
    template_text = Path(env_args["fileX_template_path"]).read_text(encoding="utf-8", errors="replace")
    if not any(" 1 MA" in line and " L     L " in line for line in template_text.splitlines()):
        raise RuntimeError("rendered candidate template did not contain linked IRRIG=L/FERTI=L management")
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    return env, env_args


def _state_value(state: dict[str, Any], key: str) -> float:
    value = baseline.scalar(state.get(key), np.nan)
    if not math.isfinite(float(value)):
        raise ValueError(f"missing_or_nonfinite_{key}")
    return float(value)


def _event_totals(snapshot: Path) -> tuple[float, float, int, pd.DataFrame]:
    events = parse_management_events(snapshot / "MgmtEvent.OUT")
    if events.empty:
        return 0.0, 0.0, 0, events
    operations = events["operation"].astype(str).str.lower()
    irrigation = float(pd.to_numeric(events.loc[operations.str.startswith("irrigation"), "amount"], errors="coerce").fillna(0.0).sum())
    nitrogen = float(pd.to_numeric(events.loc[operations.str.startswith("fertilizer"), "amount"], errors="coerce").fillna(0.0).sum())
    return irrigation, nitrogen, int(len(events)), events


def _sanity_checks(
    params: ControllerParams,
    trace: list[dict[str, Any]],
    final_yield: float,
    actual_irrigation: float,
    actual_nitrogen: float,
    event_count: int,
    done: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_actions = [float(row["nitrogen_action_kg_ha"]) for row in trace]
    i_actions = [float(row["irrigation_action_mm"]) for row in trace]
    n_daps = [int(row["dap"]) for row in trace if float(row["nitrogen_action_kg_ha"]) > 0]
    i_daps = [int(row["dap"]) for row in trace if float(row["irrigation_action_mm"]) > 0]

    def add(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "status": "pass" if passed else "fail", "details": details})

    add("episode_completed", bool(done), f"done={done}, trace_days={len(trace)}")
    add("finite_yield", math.isfinite(final_yield) and final_yield >= 0, f"grain_yield={final_yield}")
    add("controller_n_budget", sum(n_actions) <= params.n_season_budget + 1e-9, f"controller_n={sum(n_actions):.6f}; budget={params.n_season_budget}")
    add("irrigation_budget", sum(i_actions) <= params.irrigation_season_budget + 1e-9, f"irrigation={sum(i_actions):.6f}; budget={params.irrigation_season_budget}")
    add("n_action_windows", all(params.n_start_dap <= dap <= params.n_stop_dap for dap in n_daps), f"daps={n_daps}")
    add("irrigation_action_windows", all(params.irrigation_start_dap <= dap <= params.irrigation_stop_dap for dap in i_daps), f"daps={i_daps}")
    add("n_intervals", all(b - a >= params.n_min_interval_days for a, b in zip(n_daps, n_daps[1:])), f"daps={n_daps}")
    add("irrigation_intervals", all(b - a >= params.irrigation_min_interval_days for a, b in zip(i_daps, i_daps[1:])), f"daps={i_daps}")
    expected_n = float(CONFIG["basal_n_kg_ha"]) + sum(n_actions)
    expected_i = float(CONFIG["preseason_irrigation_mm"]) + sum(i_actions)
    add("n_action_sum_closure", abs(expected_n - actual_nitrogen) <= 1.0, f"expected={expected_n:.6f}; DSSAT events={actual_nitrogen:.6f}")
    add("irrigation_action_sum_closure", abs(expected_i - actual_irrigation) <= 1.0, f"expected={expected_i:.6f}; DSSAT events={actual_irrigation:.6f}")
    add("dssat_action_events_present", event_count > 0, f"event_count={event_count}")
    return rows


def _write_candidate_log(candidate_id: str, row: dict[str, Any], trace: list[dict[str, Any]], error_text: str = "") -> None:
    log_row = {key: value for key, value in row.items() if not key.startswith("_")}
    log_row["error_text"] = error_text
    log_row["trace_days"] = len(trace)
    write_json(LOGS / f"candidate_{candidate_id}.json", log_row)
    if row.get("status") == "invalid":
        write_csv(LOGS / f"candidate_{candidate_id}_daily_trace.csv", trace)


def evaluate_candidate(
    env_config: dict[str, Any],
    genome: dict[str, Any],
    candidate_id: str,
    save_trace: bool = False,
    snapshot_destination: Path | None = None,
) -> dict[str, Any]:
    _load_runtime_modules()
    params = make_controller_params(genome)
    controller = DailyStateFeedbackController(params)
    trace: list[dict[str, Any]] = []
    env = None
    error_text = ""
    done = False
    final_yield = math.nan
    actual_irrigation = math.nan
    actual_nitrogen = math.nan
    event_count = 0
    try:
        env, env_args = make_env(env_config, candidate_id)
        obs, info = env.reset()
        step_count = 0
        basal_sent = False
        planting = pd.Timestamp(direct_ppo.find_year(env_config, "YCA", int(CONFIG["year"]))["planting_date"])
        while not done and step_count < MAX_STEPS:
            latest_pre = baseline.latest_observation_dict(env, obs, info)
            dap_raw = baseline.scalar(latest_pre.get("dap"), step_count + 1)
            dap = int(round(float(dap_raw))) if math.isfinite(float(dap_raw)) and float(dap_raw) > 0 else step_count + 1
            swfac_pre = _state_value(latest_pre, "swfac")
            nstres_pre = _state_value(latest_pre, "nstres")
            decision = controller.decide(dap, swfac_pre, nstres_pre)
            basal = float(CONFIG["basal_n_kg_ha"]) if dap == 1 and not basal_sent else 0.0
            action_real = {"amir": float(decision.irrigation_action_mm), "anfer": float(decision.nitrogen_action_kg_ha) + basal}
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest_post = baseline.latest_observation_dict(env, obs, info)
            trace.append(
                {
                    "candidate_id": candidate_id,
                    "site": CONFIG["site"],
                    "year": CONFIG["year"],
                    "date": (planting + pd.Timedelta(days=max(dap - 1, 0))).strftime("%Y-%m-%d"),
                    "dap": dap,
                    "swfac": swfac_pre,
                    "nstres": nstres_pre,
                    "swfac_post": baseline.scalar(latest_post.get("swfac"), np.nan),
                    "nstres_post": baseline.scalar(latest_post.get("nstres"), np.nan),
                    "n_threshold": params.n_threshold,
                    "water_threshold": params.water_threshold,
                    "water_window_ok": decision.water_window_ok,
                    "n_window_ok": decision.n_window_ok,
                    "water_condition": decision.water_condition,
                    "n_condition": decision.n_condition,
                    "water_stress_condition": decision.water_condition,
                    "n_stress_condition": decision.n_condition,
                    "water_interval_ok": decision.water_interval_ok,
                    "n_interval_ok": decision.n_interval_ok,
                    "water_budget_ok": decision.water_budget_ok,
                    "n_budget_ok": decision.n_budget_ok,
                    "irrigation_action_mm": decision.irrigation_action_mm,
                    "nitrogen_action_kg_ha": decision.nitrogen_action_kg_ha,
                    "irrigation_action": decision.irrigation_action_mm,
                    "nitrogen_action": decision.nitrogen_action_kg_ha,
                    "basal_n_action_kg_ha": basal,
                    "action_irrigation_sent_mm": action_real["amir"],
                    "action_nitrogen_sent_kg_ha": action_real["anfer"],
                    "cumulative_irrigation_mm": decision.cumulative_irrigation_mm,
                    "cumulative_controller_n_kg_ha": decision.cumulative_controller_n_kg_ha,
                    "cumulative_irrigation": decision.cumulative_irrigation_mm,
                    "cumulative_n": decision.cumulative_controller_n_kg_ha,
                    "reward": float(reward),
                    "done": done,
                    "grnwt_post": baseline.scalar(latest_post.get("grnwt"), np.nan),
                    "topwt_post": baseline.scalar(latest_post.get("topwt"), np.nan),
                }
            )
            basal_sent = basal_sent or basal > 0
            step_count += 1
        if trace:
            final_yield = float(pd.to_numeric(pd.Series([trace[-1].get("grnwt_post")]), errors="coerce").iloc[0])
        if not done:
            raise RuntimeError(f"episode_not_finished_within_{MAX_STEPS}_steps")
        snapshot = siteppo.snapshot_from_env(env)
        metrics = baseline.metrics_from_snapshot(snapshot, final_yield)
        actual_irrigation, actual_nitrogen, event_count, events = _event_totals(snapshot)
        checks = _sanity_checks(params, trace, final_yield, actual_irrigation, actual_nitrogen, event_count, done)
        failed = [item for item in checks if item["status"] != "pass"]
        status = "valid" if not failed else "invalid"
        if snapshot_destination is not None:
            if snapshot_destination.exists():
                shutil.rmtree(snapshot_destination)
            shutil.copytree(snapshot, snapshot_destination)
        row = {
            "candidate_id": candidate_id,
            **genome,
            "n_start_dap": params.n_start_dap,
            "n_stop_dap": params.n_stop_dap,
            "irrigation_start_dap": params.irrigation_start_dap,
            "irrigation_stop_dap": params.irrigation_stop_dap,
            "status": status,
            "valid": status == "valid",
            "feasible": status == "valid" and final_yield >= YIELD_FLOOR,
            "invalid_reason": "; ".join(item["check"] + ":" + item["details"] for item in failed),
            "grain_yield_kg_ha": final_yield,
            "yield_constraint_floor_kg_ha": YIELD_FLOOR,
            "total_nitrogen_kg_ha": actual_nitrogen,
            "total_irrigation_mm": actual_irrigation,
            "basal_n_kg_ha": float(CONFIG["basal_n_kg_ha"]),
            "controller_n_kg_ha": float(sum(float(item["nitrogen_action_kg_ha"]) for item in trace)),
            "controller_irrigation_mm": float(sum(float(item["irrigation_action_mm"]) for item in trace)),
            "n_action_count": int(sum(float(item["nitrogen_action_kg_ha"]) > 0 for item in trace)),
            "irrigation_action_count": int(sum(float(item["irrigation_action_mm"]) > 0 for item in trace)),
            "dssat_event_count": event_count,
            "summary_irrigation_total_mm": float(metrics["actual_irrigation_mm"]),
            "summary_nitrogen_total_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "etcp_mm": float(metrics["etcp_mm"]),
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
            "PFP_N_kg_kg": float(final_yield / actual_nitrogen) if actual_nitrogen > 0 else math.nan,
            "max_swfac_pre": float(max(item["swfac"] for item in trace)) if trace else math.nan,
            "max_nstres_pre": float(max(item["nstres"] for item in trace)) if trace else math.nan,
            "sanity_checks": checks,
            "snapshot_path": rel(snapshot_destination) if snapshot_destination is not None else "",
        }
    except Exception as exc:
        error_text = traceback.format_exc()
        row = {
            "candidate_id": candidate_id,
            **genome,
            "n_start_dap": int(CONFIG["fixed_windows"]["n_start_dap"]),
            "n_stop_dap": int(CONFIG["fixed_windows"]["n_stop_dap"]),
            "irrigation_start_dap": int(CONFIG["fixed_windows"]["irrigation_start_dap"]),
            "irrigation_stop_dap": int(CONFIG["fixed_windows"]["irrigation_stop_dap"]),
            "status": "invalid",
            "valid": False,
            "feasible": False,
            "invalid_reason": f"{type(exc).__name__}: {exc}",
            "grain_yield_kg_ha": math.nan,
            "yield_constraint_floor_kg_ha": YIELD_FLOOR,
            "total_nitrogen_kg_ha": math.nan,
            "total_irrigation_mm": math.nan,
            "basal_n_kg_ha": float(CONFIG["basal_n_kg_ha"]),
            "controller_n_kg_ha": float(sum(float(item["nitrogen_action_kg_ha"]) for item in trace)),
            "controller_irrigation_mm": float(sum(float(item["irrigation_action_mm"]) for item in trace)),
            "n_action_count": int(sum(float(item["nitrogen_action_kg_ha"]) > 0 for item in trace)),
            "irrigation_action_count": int(sum(float(item["irrigation_action_mm"]) > 0 for item in trace)),
            "dssat_event_count": event_count,
            "summary_irrigation_total_mm": math.nan,
            "summary_nitrogen_total_kg_ha": math.nan,
            "etcp_mm": math.nan,
            "WP_ET_kg_m3": math.nan,
            "PFP_N_kg_kg": math.nan,
            "max_swfac_pre": float(max((item["swfac"] for item in trace), default=math.nan)),
            "max_nstres_pre": float(max((item["nstres"] for item in trace), default=math.nan)),
            "sanity_checks": [],
            "snapshot_path": "",
        }
    finally:
        if env is not None:
            env.close()
    _write_candidate_log(candidate_id, row, trace, error_text)
    if save_trace:
        row["_trace"] = trace
    return row


def _scenario_totals(scenario: str) -> dict[str, float]:
    if not BASELINE_DAILY.exists():
        return {"total_n": math.nan, "total_i": math.nan, "max_n": math.nan, "max_i": math.nan}
    frame = pd.read_csv(BASELINE_DAILY, keep_default_na=False)
    frame = frame[(frame.get("site", "") == CONFIG["site"]) & (pd.to_numeric(frame.get("year"), errors="coerce") == int(CONFIG["year"]))]
    if "scenario" in frame.columns:
        frame = frame[frame["scenario"].astype(str).eq(scenario)]
    ncol = "nitrogen_executed_kg_ha" if "nitrogen_executed_kg_ha" in frame.columns else "nitrogen_requested_kg_ha"
    icol = "irrigation_executed_mm" if "irrigation_executed_mm" in frame.columns else "irrigation_requested_mm"
    if frame.empty or ncol not in frame or icol not in frame:
        return {"total_n": math.nan, "total_i": math.nan, "max_n": math.nan, "max_i": math.nan}
    n = pd.to_numeric(frame[ncol], errors="coerce").fillna(0.0)
    i = pd.to_numeric(frame[icol], errors="coerce").fillna(0.0)
    return {"total_n": float(n.sum()), "total_i": float(i.sum()), "max_n": float(n.max()), "max_i": float(i.max())}


def write_parameter_space_audit() -> Path:
    expert = _scenario_totals("official_extension_expert")
    farmer = _scenario_totals("recorded_farmer_template")
    rows = [
        {
            "site": CONFIG["site"],
            "station": CONFIG["station"],
            "year": CONFIG["year"],
            "expert_total_n_kg_ha": expert["total_n"],
            "expert_max_single_n_kg_ha": expert["max_n"],
            "expert_total_irrigation_mm": expert["total_i"],
            "expert_max_single_irrigation_mm": expert["max_i"],
            "recorded_template_total_n_kg_ha": farmer["total_n"],
            "recorded_template_max_single_n_kg_ha": farmer["max_n"],
            "recorded_template_total_irrigation_mm": farmer["total_i"],
            "recorded_template_max_single_irrigation_mm": farmer["max_i"],
            "fixed_basal_n_kg_ha": CONFIG["basal_n_kg_ha"],
            "fixed_preseason_irrigation_mm": CONFIG["preseason_irrigation_mm"],
            "n_threshold_range": str(PARAMETER_SPACE["n_threshold"]),
            "water_threshold_range": str(PARAMETER_SPACE["water_threshold"]),
            "n_dose_values": str(PARAMETER_SPACE["n_dose"]),
            "irrigation_dose_values": str(PARAMETER_SPACE["irrigation_dose"]),
            "n_interval_range_days": str(PARAMETER_SPACE["n_min_interval_days"]),
            "irrigation_interval_range_days": str(PARAMETER_SPACE["irrigation_min_interval_days"]),
            "n_budget_values_kg_ha": str(PARAMETER_SPACE["n_season_budget"]),
            "irrigation_budget_values_mm": str(PARAMETER_SPACE["irrigation_season_budget"]),
            "range_basis": "YC2019 null stress distribution anchors; thresholds expanded beyond observed P05/P95; dose/budget sets retain explicit discrete values",
        }
    ]
    out = EXPERIMENT_ROOT / "configs" / "parameter_space_audit.csv"
    write_csv(out, rows)
    write_json(RESULTS / "configs" / "experiment_config_resolved.json", CONFIG)
    return out


def _manual_genomes() -> list[dict[str, Any]]:
    return [
        {
            "n_threshold": 0.04,
            "n_dose": 40,
            "n_min_interval_days": 7,
            "n_season_budget": 250,
            "water_threshold": 0.15,
            "irrigation_dose": 30,
            "irrigation_min_interval_days": 7,
            "irrigation_season_budget": 180,
        },
        {
            "n_threshold": 0.10,
            "n_dose": 30,
            "n_min_interval_days": 7,
            "n_season_budget": 200,
            "water_threshold": 0.25,
            "irrigation_dose": 25,
            "irrigation_min_interval_days": 7,
            "irrigation_season_budget": 150,
        },
        {
            "n_threshold": 0.25,
            "n_dose": 60,
            "n_min_interval_days": 14,
            "n_season_budget": 100,
            "water_threshold": 0.85,
            "irrigation_dose": 40,
            "irrigation_min_interval_days": 14,
            "irrigation_season_budget": 60,
        },
    ]


def run_unit() -> None:
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(EXPERIMENT_ROOT / "src" / "tests"),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(
        RESULTS / "unit_test_result.json",
        {
            "successful": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
        },
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


def run_smoke(env_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, genome in enumerate(_manual_genomes(), start=1):
        print(f"[220 smoke] candidate M{index:03d}", flush=True)
        rows.append(evaluate_candidate(env_config, genome, f"M{index:03d}"))
    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if not key.startswith("_") and key != "sanity_checks"}
            for row in rows
        ]
    )
    write_csv(RESULTS / "smoke_candidate_summary.csv", frame)
    action_pairs = {
        (int(row.get("n_action_count", 0)), int(row.get("irrigation_action_count", 0)))
        for row in rows
        if row.get("status") == "valid"
    }
    closure_pass = all(
        row.get("status") == "valid"
        and any(item.get("check") == "n_action_sum_closure" and item.get("status") == "pass" for item in row.get("sanity_checks", []))
        and any(item.get("check") == "irrigation_action_sum_closure" and item.get("status") == "pass" for item in row.get("sanity_checks", []))
        for row in rows
    )
    smoke_pass = bool(all(row.get("status") == "valid" for row in rows) and len(action_pairs) >= 2 and closure_pass)
    write_json(
        RESULTS / "smoke_result.json",
        {
            "smoke_passed": smoke_pass,
            "candidate_count": len(rows),
            "distinct_action_count_pairs": len(action_pairs),
            "closure_passed": closure_pass,
            "criterion_1_state_response": len(action_pairs) >= 2,
            "criterion_2_dssat_execution": closure_pass,
        },
    )
    if not smoke_pass:
        raise RuntimeError("220 smoke did not pass; NSGA-II was not started")
    return frame


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
        for key, value in row.items()
        if not key.startswith("_")
    }


def _balanced_solution(front: list[dict[str, Any]]) -> dict[str, Any]:
    if not front:
        raise RuntimeError("NSGA-II produced no feasible Pareto candidates")
    frame = pd.DataFrame([_flatten(row) for row in front])
    y = pd.to_numeric(frame["grain_yield_kg_ha"], errors="coerce")
    n = pd.to_numeric(frame["total_nitrogen_kg_ha"], errors="coerce")
    i = pd.to_numeric(frame["total_irrigation_mm"], errors="coerce")

    def norm(series: pd.Series, maximize: bool) -> pd.Series:
        low, high = float(series.min()), float(series.max())
        if abs(high - low) < 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (high - series) / (high - low) if maximize else (series - low) / (high - low)

    frame["balanced_distance"] = np.sqrt(norm(y, True) ** 2 + norm(n, False) ** 2 + norm(i, False) ** 2)
    chosen_id = str(frame.sort_values(["balanced_distance", "grain_yield_kg_ha"], ascending=[True, False]).iloc[0]["candidate_id"])
    return next(row for row in front if str(row["candidate_id"]) == chosen_id)


def _unique_objective_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one parameterization per objective point for plotting/reporting."""
    unique: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            round(float(row["grain_yield_kg_ha"]), 6),
            round(float(row["total_nitrogen_kg_ha"]), 6),
            round(float(row["total_irrigation_mm"]), 6),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def write_pareto_plot(front: list[dict[str, Any]], balanced_id: str) -> None:
    frame = pd.DataFrame([_flatten(row) for row in front])
    if frame.empty:
        return
    frame["grain_yield_kg_ha"] = pd.to_numeric(frame["grain_yield_kg_ha"], errors="coerce")
    frame["total_nitrogen_kg_ha"] = pd.to_numeric(frame["total_nitrogen_kg_ha"], errors="coerce")
    frame["total_irrigation_mm"] = pd.to_numeric(frame["total_irrigation_mm"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=160)
    points = ax.scatter(
        frame["total_nitrogen_kg_ha"],
        frame["grain_yield_kg_ha"],
        c=frame["total_irrigation_mm"],
        s=45 + frame["total_irrigation_mm"] * 0.7,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
    )
    selected = frame[frame["candidate_id"].astype(str).eq(str(balanced_id))]
    if not selected.empty:
        ax.scatter(
            selected["total_nitrogen_kg_ha"],
            selected["grain_yield_kg_ha"],
            marker="*",
            s=180,
            color="red",
            edgecolor="black",
            label="balanced",
        )
    ax.axhline(YIELD_FLOOR, color="grey", linestyle="--", linewidth=0.8, label="0.97 reference yield")
    ax.set_xlabel("Total nitrogen (kg N ha$^{-1}$)")
    ax.set_ylabel("Grain yield (kg ha$^{-1}$)")
    ax.set_title("YC2019 controller Pareto front; color/size = total irrigation")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(points, ax=ax, label="Total irrigation (mm)")
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "pareto_front_yield_n_water.png")
    plt.close(fig)


def write_balanced_outputs(env_config: dict[str, Any], balanced: dict[str, Any]) -> dict[str, Any]:
    rerun_id = f"BALANCED_{balanced['candidate_id']}"
    snapshot_destination = RESULTS / "snapshots" / "balanced_solution"
    rerun = evaluate_candidate(
        env_config,
        {key: balanced[key] for key in PARAMETER_SPACE},
        rerun_id,
        save_trace=True,
        snapshot_destination=snapshot_destination,
    )
    trace = rerun.pop("_trace", [])
    trace_frame = pd.DataFrame(trace)
    write_csv(RESULTS / "balanced_solution_daily_decision_trace.csv", trace_frame)
    write_csv(RESULTS / "balanced_solution_summary.csv", [_flatten(rerun)])

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), dpi=160, sharex=True)
    if not trace_frame.empty:
        axes[0].plot(trace_frame["dap"], trace_frame["swfac"], label="SWFAC pre-action", color="#1f77b4")
        axes[0].plot(trace_frame["dap"], trace_frame["nstres"], label="NSTRES pre-action", color="#d62728")
        axes[0].axhline(float(balanced["water_threshold"]), color="#1f77b4", linestyle="--", alpha=0.55)
        axes[0].axhline(float(balanced["n_threshold"]), color="#d62728", linestyle="--", alpha=0.55)
        irrigation_days = trace_frame[trace_frame["irrigation_action_mm"] > 0]
        n_days = trace_frame[trace_frame["nitrogen_action_kg_ha"] > 0]
        axes[0].scatter(irrigation_days["dap"], irrigation_days["swfac"], marker="v", color="#1f77b4", label="irrigation action")
        axes[0].scatter(n_days["dap"], n_days["nstres"], marker="^", color="#d62728", label="N action")
        axes[1].step(trace_frame["dap"], trace_frame["cumulative_irrigation_mm"], where="post", label="cumulative irrigation (mm)", color="#2ca02c")
        axes[1].step(trace_frame["dap"], trace_frame["cumulative_controller_n_kg_ha"], where="post", label="cumulative controller N (kg/ha)", color="#9467bd")
    axes[0].set_ylabel("Stress index")
    axes[1].set_ylabel("Cumulative action")
    axes[1].set_xlabel("DAP")
    axes[0].set_title("Balanced YC2019 daily state-feedback trajectory")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "balanced_solution_daily_trajectory.png")
    plt.close(fig)

    parameter_rows = []
    full_params = make_controller_params({key: balanced[key] for key in PARAMETER_SPACE}).as_dict()
    optimized = set(PARAMETER_SPACE)
    for name, value in full_params.items():
        parameter_rows.append(
            {
                "parameter": name,
                "value": value,
                "role": "optimized" if name in optimized else "fixed",
                "unit_or_semantics": "stress index"
                if "threshold" in name
                else "kg N/ha"
                if name in {"n_dose", "n_season_budget"}
                else "mm"
                if name in {"irrigation_dose", "irrigation_season_budget"}
                else "DAP"
                if "dap" in name or "interval" in name
                else "",
            }
        )
    write_csv(RESULTS / "balanced_solution_parameters.csv", parameter_rows)

    expert = _scenario_totals("official_extension_expert")
    farmer = _scenario_totals("recorded_farmer_template")
    comparisons = [
        {
            "comparison_case": "balanced_controller",
            "data_status": "simulated_DSSAT",
            "grain_yield_kg_ha": rerun.get("grain_yield_kg_ha"),
            "total_nitrogen_kg_ha": rerun.get("total_nitrogen_kg_ha"),
            "total_irrigation_mm": rerun.get("total_irrigation_mm"),
            "note": "single-year controller candidate; not a field measurement",
        },
        {
            "comparison_case": "official_extension_expert",
            "data_status": "simulated_DSSAT_reference",
            "grain_yield_kg_ha": EXPERT_REFERENCE_YIELD,
            "total_nitrogen_kg_ha": expert["total_n"],
            "total_irrigation_mm": expert["total_i"],
            "note": "used for simulated yield constraint; not field measured",
        },
        {
            "comparison_case": "recorded_farmer_template",
            "data_status": "template_simulation_not_year_specific",
            "grain_yield_kg_ha": 6557.869873046875,
            "total_nitrogen_kg_ha": farmer["total_n"],
            "total_irrigation_mm": farmer["total_i"],
            "note": "template reused, not validated YC2019 field measurement",
        },
        {
            "comparison_case": "field_measured",
            "data_status": "unavailable",
            "grain_yield_kg_ha": math.nan,
            "total_nitrogen_kg_ha": math.nan,
            "total_irrigation_mm": math.nan,
            "note": "no independent YC2019 field-measured record was found; normalized Euclidean comparison not computed",
        },
    ]
    write_csv(RESULTS / "balanced_solution_comparison.csv", comparisons)
    return rerun


def write_results_doc(
    all_rows: list[dict[str, Any]],
    front: list[dict[str, Any]],
    balanced: dict[str, Any],
    smoke: dict[str, Any],
    elapsed: float,
) -> Path:
    valid = [row for row in all_rows if row.get("status") == "valid"]
    front_flat = pd.DataFrame([_flatten(row) for row in front])
    if not front_flat.empty:
        y_values = pd.to_numeric(front_flat["grain_yield_kg_ha"], errors="coerce")
        n_values = pd.to_numeric(front_flat["total_nitrogen_kg_ha"], errors="coerce")
        i_values = pd.to_numeric(front_flat["total_irrigation_mm"], errors="coerce")
        y_range = float(y_values.max() - y_values.min())
        n_range = float(n_values.max() - n_values.min())
        i_range = float(i_values.max() - i_values.min())
    else:
        y_range = n_range = i_range = math.nan
    criterion_rows = [
        {
            "criterion": SUCCESS_CRITERIA[0],
            "status": "pass"
            if any(int(row.get("n_action_count", 0)) + int(row.get("irrigation_action_count", 0)) > 0 for row in valid)
            else "fail",
            "evidence": "daily trace records pre-action SWFAC/NSTRES, conditions, intervals, budgets, and action markers",
        },
        {
            "criterion": SUCCESS_CRITERIA[1],
            "status": "pass" if all(row.get("status") == "valid" for row in all_rows if row.get("dssat_event_count", 0) > 0) else "review",
            "evidence": "MgmtEvent.OUT event totals and action-sum closure are retained per candidate",
        },
        {
            "criterion": SUCCESS_CRITERIA[2],
            "status": "pass"
            if len({(row.get("n_action_count"), row.get("irrigation_action_count")) for row in all_rows if row.get("status") == "valid"}) >= 2
            else "fail",
            "evidence": "three predeclared manual threshold/action candidates",
        },
        {
            "criterion": SUCCESS_CRITERIA[3],
            "status": "pass" if n_range > 0 and i_range > 0 and y_range > 0 else "fail",
            "evidence": f"final feasible front ranges: yield={y_range:.3f}, N={n_range:.3f}, irrigation={i_range:.3f}",
        },
        {
            "criterion": SUCCESS_CRITERIA[4],
            "status": "pass" if len(front) >= 3 and (y_range > 0 or n_range > 0 or i_range > 0) else "fail",
            "evidence": f"feasible Pareto candidates={len(front)}; population=32; generations=20",
        },
    ]
    invalid = [row for row in all_rows if row.get("status") != "valid"]
    feasible_objective_count = len(
        {
            (
                round(float(row["grain_yield_kg_ha"]), 6),
                round(float(row["total_nitrogen_kg_ha"]), 6),
                round(float(row["total_irrigation_mm"]), 6),
            )
            for row in all_rows
            if row.get("valid") and row.get("feasible")
        }
    )
    lines = [
        "# YC feedback-controller demo results",
        "",
        SCOPE_DECLARATION.rstrip(),
        "",
        '**本次 demo 真正的成功判据是以下五条，不是“打赢 farmer/expert”：**',
        "",
        *[f"{item}" for item in SUCCESS_CRITERIA],
        "",
        "## 1. 执行范围和资源保护",
        "",
        f"- site-year: YC/YCA {CONFIG['year']}；仅一个年份，未做 held-out 或跨年份泛化。",
        f"- NSGA-II: population={CONFIG['optimizer']['population']}, generations={CONFIG['optimizer']['generations']}, seed={CONFIG['optimizer']['seed']}；仅一个 optimizer seed，尚未评估跨 seed 稳健性。",
        f"- 实际唯一 DSSAT candidate evaluations: {len(all_rows)}；全程串行，elapsed {elapsed:.1f} s。",
        "- 原始输入未改写；每个候选使用实验目录下渲染副本和单独 DSSAT 日志。",
        "",
        "## 2. 关键约束与 basal 处理",
        "",
        f"- reference yield = {EXPERT_REFERENCE_YIELD:.3f} kg/ha，constraint floor = {YIELD_FLOOR:.3f} kg/ha；参照是同一 DSSAT/gym-DSSAT 环境中的 simulated official extension expert。",
        f"- 固定 DAP1 basal N = {CONFIG['basal_n_kg_ha']:.1f} kg/ha，controller N budget 只约束季内反馈 N；最终 total_nitrogen = basal + controller actions。",
        f"- 固定播前灌溉 = {CONFIG['preseason_irrigation_mm']:.1f} mm；最终 total_irrigation = preseason + controller actions。",
        "- 水和氮使用独立 threshold、interval、budget，不存在 joint trigger。",
        "",
        "## 3. 五条判据逐条结果",
        "",
        pd.DataFrame(criterion_rows).to_markdown(index=False),
        "",
        "## 4. NSGA-II Pareto front",
        "",
        f"- feasible front candidates: {len(front)}；valid candidate records: {len(valid)}；invalid records retained: {len(invalid)}。",
        f"- feasible candidate cloud has {feasible_objective_count} distinct objective triples, but only {len(front)} nondominated objective point(s) remain after the 0.97 reference-yield constraint。",
        f"- front ranges: yield {y_range:.3f}, total N {n_range:.3f}, total irrigation {i_range:.3f}。",
        f"- balanced candidate: {balanced.get('candidate_id')}; rerun outputs are in results/balanced_solution_* and results/snapshots/balanced_solution/.",
        "",
        "## 5. 输出文件",
        "",
        *[
            f"- {rel(path)}"
            for path in [
                EXPERIMENT_ROOT / "configs" / "parameter_space_audit.csv",
                RESULTS / "candidate_evaluations.csv",
                RESULTS / "pareto_front.csv",
                RESULTS / "plots" / "pareto_front_yield_n_water.png",
                RESULTS / "balanced_solution_daily_decision_trace.csv",
                RESULTS / "plots" / "balanced_solution_daily_trajectory.png",
                RESULTS / "balanced_solution_comparison.csv",
                RESULTS / "balanced_solution_parameters.csv",
                RESULTS / "sanity_check_summary.csv",
            ]
        ],
        "",
        "## 6. 外部参照与未验证事项",
        "",
        "- farmer/expert 条目仅作为审计过的 simulated/template reference；本轮没有独立 YC2019 田间实测记录，因此没有伪造 measured comparison，也没有计算 normalized Euclidean distance。",
        "- 结果不支持五站点适用性、优于 NSGA-III、优于 PPO、水氮耦合触发、跨年份泛化等结论；这些均尚待正式实验验证。",
        "- 没有做多 optimizer seed 稳健性检验，也没有运行 optional 40/30 confirmation。",
        "",
        "## 7. Smoke 记录",
        "",
        f"- smoke_passed={smoke.get('smoke_passed')}; distinct action-count pairs={smoke.get('distinct_action_count_pairs')}; DSSAT closure={smoke.get('closure_passed')}。",
    ]
    out = PROJECT_ROOT / "docs" / "2026-09-09_yc_feedback_controller_demo_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_nsga(env_config: dict[str, Any], smoke: dict[str, Any]) -> None:
    if not smoke.get("smoke_passed"):
        raise RuntimeError("smoke_result.json is not passed; NSGA-II blocked")
    start = time.time()
    unique_rows: dict[str, dict[str, Any]] = {}
    counter = [0]

    def evaluate(genome: dict[str, Any]) -> dict[str, Any]:
        counter[0] += 1
        candidate_id = f"N{counter[0]:04d}"
        print(
            f"[220 NSGA-II] candidate {candidate_id}/{CONFIG['optimizer']['population'] * (CONFIG['optimizer']['generations'] + 1)}",
            flush=True,
        )
        row = evaluate_candidate(env_config, genome, candidate_id)
        unique_rows[candidate_id] = row
        return row

    initial = _manual_genomes()
    initial.extend(
        [
            {
                "n_threshold": 0.03,
                "n_dose": 30,
                "n_min_interval_days": 3,
                "n_season_budget": 300,
                "water_threshold": 0.12,
                "irrigation_dose": 20,
                "irrigation_min_interval_days": 3,
                "irrigation_season_budget": 240,
            },
            {
                "n_threshold": 0.06,
                "n_dose": 50,
                "n_min_interval_days": 5,
                "n_season_budget": 250,
                "water_threshold": 0.20,
                "irrigation_dose": 30,
                "irrigation_min_interval_days": 5,
                "irrigation_season_budget": 240,
            },
            {
                "n_threshold": 0.15,
                "n_dose": 40,
                "n_min_interval_days": 10,
                "n_season_budget": 200,
                "water_threshold": 0.45,
                "irrigation_dose": 40,
                "irrigation_min_interval_days": 10,
                "irrigation_season_budget": 180,
            },
        ]
    )
    history, final_population = run_nsga2(
        PARAMETER_SPACE,
        evaluate,
        population_size=int(CONFIG["optimizer"]["population"]),
        generations=int(CONFIG["optimizer"]["generations"]),
        seed=int(CONFIG["optimizer"]["seed"]),
        initial_genomes=initial,
    )
    all_rows = list(unique_rows.values())
    final_front = _unique_objective_rows(feasible_pareto(final_population))
    if not final_front:
        final_front = _unique_objective_rows(feasible_pareto(all_rows))
    balanced = _balanced_solution(final_front)
    write_csv(RESULTS / "candidate_evaluations.csv", [_flatten(row) for row in all_rows])
    write_csv(RESULTS / "nsga_history.csv", [_flatten(row) for row in history])
    write_csv(RESULTS / "pareto_front.csv", [_flatten(row) for row in final_front])
    write_pareto_plot(final_front, str(balanced["candidate_id"]))
    balanced_rerun = write_balanced_outputs(env_config, balanced)
    sanity_rows: list[dict[str, Any]] = []
    for row in all_rows:
        checks = row.get("sanity_checks", [])
        sanity_rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "status": row.get("status"),
                "invalid_reason": row.get("invalid_reason", ""),
                "check_count": len(checks),
                "failed_check_count": sum(item.get("status") != "pass" for item in checks),
                "checks_json": json.dumps(checks, ensure_ascii=False),
            }
        )
    write_csv(RESULTS / "sanity_check_summary.csv", sanity_rows)
    smoke_summary = json.loads((RESULTS / "smoke_result.json").read_text(encoding="utf-8"))
    results_doc = write_results_doc(all_rows, final_front, balanced, smoke_summary, time.time() - start)
    write_json(
        RESULTS / "220_result.json",
        {
            "status": "completed",
            "site": CONFIG["site"],
            "year": CONFIG["year"],
            "unique_candidate_count": len(all_rows),
            "final_front_count": len(final_front),
            "balanced_candidate_id": balanced["candidate_id"],
            "results_doc": rel(results_doc),
            "elapsed_seconds": time.time() - start,
        },
    )
    print(
        json.dumps(
            {
                "result_doc": rel(results_doc),
                "unique_candidate_count": len(all_rows),
                "final_front_count": len(final_front),
                "balanced": balanced["candidate_id"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def finalize_existing() -> None:
    """Rebuild the report/front from completed evaluations without DSSAT reruns."""
    candidate_path = RESULTS / "candidate_evaluations.csv"
    result_path = RESULTS / "220_result.json"
    if not candidate_path.exists():
        raise RuntimeError("candidate_evaluations.csv is missing; run --mode nsga first")
    frame = pd.read_csv(candidate_path)
    rows = frame.to_dict(orient="records")
    for row in rows:
        for key in ["valid", "feasible"]:
            if key in row:
                row[key] = str(row[key]).strip().lower() == "true"
    front = _unique_objective_rows(feasible_pareto(rows))
    if not front:
        raise RuntimeError("no feasible Pareto point found in completed evaluations")
    balanced = _balanced_solution(front)
    write_csv(RESULTS / "pareto_front.csv", [_flatten(row) for row in front])
    write_pareto_plot(front, str(balanced["candidate_id"]))
    smoke = json.loads((RESULTS / "smoke_result.json").read_text(encoding="utf-8"))
    old_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    results_doc = write_results_doc(rows, front, balanced, smoke, float(old_result.get("elapsed_seconds", 0.0)))
    write_json(
        result_path,
        {
            "status": "completed_finalized",
            "site": CONFIG["site"],
            "year": CONFIG["year"],
            "unique_candidate_count": len(rows),
            "final_front_count": len(front),
            "final_front_distinct_objective_count": len(front),
            "balanced_candidate_id": balanced["candidate_id"],
            "results_doc": rel(results_doc),
            "elapsed_seconds": float(old_result.get("elapsed_seconds", 0.0)),
        },
    )
    print(json.dumps({"result_doc": rel(results_doc), "final_front_count": len(front), "balanced": balanced["candidate_id"]}, ensure_ascii=False))


def refresh_balanced() -> None:
    """Re-run only the selected balanced candidate to refresh trace aliases."""
    env_config = load_env_config()
    front_path = RESULTS / "pareto_front.csv"
    if not front_path.exists():
        raise RuntimeError("pareto_front.csv is missing; finalize the NSGA-II results first")
    front = pd.read_csv(front_path)
    if front.empty:
        raise RuntimeError("pareto_front.csv is empty")
    balanced = front.iloc[0].to_dict()
    write_balanced_outputs(env_config, balanced)
    print("balanced outputs refreshed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parameter-audit", "unit", "smoke", "nsga", "finalize", "refresh-balanced"], required=True)
    args = parser.parse_args()
    ensure_dirs()
    write_parameter_space_audit()
    if args.mode == "parameter-audit":
        print(rel(EXPERIMENT_ROOT / "configs" / "parameter_space_audit.csv"))
        return
    if args.mode == "unit":
        run_unit()
        return
    if args.mode == "finalize":
        finalize_existing()
        return
    if args.mode == "refresh-balanced":
        refresh_balanced()
        return
    env_config = load_env_config()
    if args.mode == "smoke":
        run_smoke(env_config)
        return
    smoke_path = RESULTS / "smoke_result.json"
    if not smoke_path.exists():
        raise RuntimeError("run --mode smoke first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    run_nsga(env_config, smoke)


if __name__ == "__main__":
    main()

"""090_00: read-only FQ action-grid DSSAT counterfactual audit.

Runs a small, fixed set of water/nitrogen schedules for representative years.
It deliberately does not train PPO or modify source input files. 2018 uses the
isolated corrected weather copy from task 088; all other years use the original
originIC input root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import ppo_safe_rendering
import run_sya_lowIC_binary_timing_maskableppo_042_10 as formal_fq_engine


BASELINE_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
CORRECTED_ROOT = ROOT / "benchmark_results" / "088_00_fq2018_input_quality_correction" / "corrected_input"
TASK_ID = "090_00"
STATION = "FQA"
SITE = "FQ"
CHECKPOINT_PPO = "ppo"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_setup(config_path: Path, out_root: Path, use_formal_fq_safety: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    config = direct_ppo.load_yaml(config_path)
    config = json.loads(json.dumps(config))
    config["seed"] = 0
    config["paths"]["output_root"] = rel(out_root)
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = 2014
    config["runtime"]["max_steps"] = 260
    config["use_formal_fq_safety"] = bool(use_formal_fq_safety)
    if use_formal_fq_safety:
        config["action_safety"]["season_irrigation_soft_limit"] = 240.0
    split = batch_ppo.load_split()
    selection = batch_ppo.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = rel(out_root)
    return config, env_config


def water_schedule(total: int) -> dict[int, float]:
    if total not in {150, 210, 240}:
        raise ValueError(total)
    n_events = total // 15
    return {1 + 7 * idx: 15.0 for idx in range(n_events)}


def nitrogen_schedule(total: int) -> dict[int, float]:
    schedules = {
        80: {1: 40.0, 30: 40.0},
        120: {1: 40.0, 30: 40.0, 50: 40.0},
        160: {1: 40.0, 30: 40.0, 50: 40.0, 65: 40.0},
        240: {1: 80.0, 30: 40.0, 50: 80.0, 65: 40.0},
    }
    return schedules[total]


def action_index(config: dict[str, Any], amir: float, anfer: float) -> int:
    grid = base032.action_grid(config)
    distances = [abs(float(x["amir"]) - amir) + abs(float(x["anfer"]) - anfer) for x in grid]
    return int(np.argmin(distances))


def run_one(config: dict[str, Any], env_config: dict[str, Any], out_root: Path, year: int, irrigation: int, nitrogen: int) -> dict[str, Any]:
    tag = f"FQ_{year}_{TASK_ID}_I{irrigation}_N{nitrogen}"
    schedule_i = water_schedule(irrigation)
    schedule_n = nitrogen_schedule(nitrogen)
    original_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    original_wrapper = base032.StressAwareDiscreteWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = CORRECTED_ROOT if year == 2018 else BASELINE_ROOT
    if bool(config.get("use_formal_fq_safety", False)):
        base032.StressAwareDiscreteWrapper = formal_fq_engine.base04036.LateIrrigationReserveMaskWrapper
    env = None
    records: list[dict[str, Any]] = []
    try:
        env = base032.make_env(config, env_config, STATION, year, 0, tag, evaluation=True)
        obs, info = env.reset()
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, year)["planting_date"])
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1), step_count + 1)
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            amir = float(schedule_i.get(dap, 0.0))
            anfer = float(schedule_n.get(dap, 0.0))
            idx = action_index(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(idx)
            done = bool(terminated or truncated)
            latest_after = base032.latest_observation_dict(env, obs, info)
            records.append({
                "station_code": STATION,
                "site": SITE,
                "year": year,
                "dap": dap,
                "date": (planting + pd.Timedelta(days=max(dap - 1, 0))).strftime("%Y-%m-%d"),
                "requested_irrigation_mm": amir,
                "requested_nitrogen_kg_ha": anfer,
                "safe_irrigation_mm": float(env.last_action_info.get("safe_action_amir", 0.0)),
                "safe_nitrogen_kg_ha": float(env.last_action_info.get("safe_action_anfer", 0.0)),
                "grnwt": scalar(latest_after.get("grnwt")),
                "topwt": scalar(latest_after.get("topwt")),
                "swfac": scalar(latest_after.get("swfac")),
                "nstres": scalar(latest_after.get("nstres")),
                "reward": float(reward),
                "done": done,
            })
            step_count += 1
        if not records or not done:
            raise RuntimeError(f"incomplete run year={year}, I={irrigation}, N={nitrogen}, steps={step_count}")
        daily = pd.DataFrame(records)
        scenario = f"I{irrigation}_N{nitrogen}"
        daily_path = out_root / "daily_outputs" / SITE / str(year) / f"{scenario}_daily.csv"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        tmp_snapshot = baseline034.siteppo.snapshot_from_env(env)
        snapshot = out_root / "snapshots" / SITE / str(year) / scenario
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise FileExistsError(snapshot)
        shutil.copytree(tmp_snapshot, snapshot)
        final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        metrics = baseline034.metrics_from_snapshot(snapshot, final_yield)
        safe_i = float(pd.to_numeric(daily["safe_irrigation_mm"], errors="coerce").sum())
        safe_n = float(pd.to_numeric(daily["safe_nitrogen_kg_ha"], errors="coerce").sum())
        return {
            "station_code": STATION,
            "site": SITE,
            "year": year,
            "scenario": scenario,
            "irrigation_plan_mm": irrigation,
            "nitrogen_plan_kg_ha": nitrogen,
            "grain_yield_kg_ha": final_yield,
            "biomass_kg_ha": scalar(daily["topwt"].iloc[-1]),
            "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "actual_nitrogen_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "safe_irrigation_mm": safe_i,
            "safe_nitrogen_kg_ha": safe_n,
            "etcp_mm": float(metrics["etcp_mm"]),
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
            "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if math.isfinite(float(metrics["PFP_N_kg_kg"])) else np.nan,
            "season_last_dap": int(pd.to_numeric(daily["dap"], errors="coerce").max()),
            "interface_pass": bool(abs(safe_i - float(metrics["actual_irrigation_mm"])) < 1e-6 and abs(safe_n - float(metrics["actual_nitrogen_kg_ha"])) < 1e-6),
            "grid_pass": bool(all(abs(float(x) - round(float(x) / 15.0) * 15.0) < 1e-6 for x in daily["safe_irrigation_mm"]) and all(float(x) in {0.0, 40.0, 80.0, 120.0} for x in daily["safe_nitrogen_kg_ha"])),
            "source_weather": "088_corrected_FQ2018" if year == 2018 else "original_FQ_WTH",
            "daily_path": rel(daily_path),
            "snapshot_path": rel(snapshot),
        }
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = original_root
        base032.StressAwareDiscreteWrapper = original_wrapper


def pareto_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pareto_efficient"] = False
    for year, group in out.groupby("year", sort=True):
        idxs = list(group.index)
        for i in idxs:
            a = group.loc[i]
            dominated = False
            for j in idxs:
                if i == j:
                    continue
                b = group.loc[j]
                no_worse = (
                    b["grain_yield_kg_ha"] >= a["grain_yield_kg_ha"]
                    and b["WP_ET_kg_m3"] >= a["WP_ET_kg_m3"]
                    and (pd.isna(a["PFP_N_kg_kg"]) or pd.isna(b["PFP_N_kg_kg"]) or b["PFP_N_kg_kg"] >= a["PFP_N_kg_kg"])
                    and b["actual_nitrogen_kg_ha"] <= a["actual_nitrogen_kg_ha"]
                    and b["actual_irrigation_mm"] <= a["actual_irrigation_mm"]
                )
                strict = (
                    b["grain_yield_kg_ha"] > a["grain_yield_kg_ha"]
                    or b["WP_ET_kg_m3"] > a["WP_ET_kg_m3"]
                    or (pd.notna(a["PFP_N_kg_kg"]) and pd.notna(b["PFP_N_kg_kg"]) and b["PFP_N_kg_kg"] > a["PFP_N_kg_kg"])
                    or b["actual_nitrogen_kg_ha"] < a["actual_nitrogen_kg_ha"]
                    or b["actual_irrigation_mm"] < a["actual_irrigation_mm"]
                )
                if no_worse and strict:
                    dominated = True
                    break
            out.loc[i, "pareto_efficient"] = not dominated
    return out


def make_decision(grid: pd.DataFrame, ppo: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    floor = float(config["yield_floor_ratio_vs_ppo"])
    low_n = float(config["low_n_threshold_kg_ha"])
    for year, group in grid.groupby("year", sort=True):
        p = ppo[ppo["year"].astype(int).eq(int(year))].iloc[0]
        candidates = group[
            (group["actual_nitrogen_kg_ha"] <= low_n + 1e-6)
            & (group["grain_yield_kg_ha"] >= float(p["grain_yield_kg_ha"]) * floor)
            & ((group["WP_ET_kg_m3"] > float(p["WP_ET_kg_m3"]) + 1e-9) | (group["PFP_N_kg_kg"] > float(p["PFP_N_kg_kg"]) + 1e-9))
            & group["interface_pass"].astype(bool)
            & group["grid_pass"].astype(bool)
        ]
        best = candidates.sort_values(["grain_yield_kg_ha", "WP_ET_kg_m3"], ascending=False).iloc[0] if len(candidates) else None
        rows.append({
            "year": int(year),
            "ppo_yield_kg_ha": float(p["grain_yield_kg_ha"]),
            "ppo_WP_ET_kg_m3": float(p["WP_ET_kg_m3"]),
            "ppo_PFP_N_kg_kg": float(p["PFP_N_kg_kg"]),
            "low_n_candidate_count": int(len(candidates)),
            "rescue_signal": bool(len(candidates) > 0),
            "selected_scenario": str(best["scenario"]) if best is not None else "",
            "selected_yield_kg_ha": float(best["grain_yield_kg_ha"]) if best is not None else np.nan,
            "selected_WP_ET_kg_m3": float(best["WP_ET_kg_m3"]) if best is not None else np.nan,
            "selected_PFP_N_kg_kg": float(best["PFP_N_kg_kg"]) if best is not None else np.nan,
            "selected_N_kg_ha": float(best["actual_nitrogen_kg_ha"]) if best is not None else np.nan,
            "selected_I_mm": float(best["actual_irrigation_mm"]) if best is not None else np.nan,
        })
    return pd.DataFrame(rows)


def finalize_existing_rows(cfg: dict[str, Any], out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in [int(x) for x in cfg["years"]]:
        for irrigation in [int(x) for x in cfg["irrigation_totals_mm"]]:
            for nitrogen in [int(x) for x in cfg["nitrogen_totals_kg_ha"]]:
                scenario = f"I{irrigation}_N{nitrogen}"
                daily_path = out_root / "daily_outputs" / SITE / str(year) / f"{scenario}_daily.csv"
                snapshot = out_root / "snapshots" / SITE / str(year) / scenario
                if not daily_path.exists() or not snapshot.exists():
                    raise FileNotFoundError(f"missing existing artifact: {daily_path} or {snapshot}")
                daily = pd.read_csv(daily_path, keep_default_na=False)
                final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
                metrics = baseline034.metrics_from_snapshot(snapshot, final_yield)
                safe_i = float(pd.to_numeric(daily["safe_irrigation_mm"], errors="coerce").sum())
                safe_n = float(pd.to_numeric(daily["safe_nitrogen_kg_ha"], errors="coerce").sum())
                grid_ok = bool(
                    all(abs(float(x) - round(float(x) / 15.0) * 15.0) < 1e-6 for x in daily["safe_irrigation_mm"])
                    and all(float(x) in {0.0, 40.0, 80.0, 120.0} for x in daily["safe_nitrogen_kg_ha"])
                )
                rows.append({
                    "station_code": STATION,
                    "site": SITE,
                    "year": year,
                    "scenario": scenario,
                    "irrigation_plan_mm": irrigation,
                    "nitrogen_plan_kg_ha": nitrogen,
                    "grain_yield_kg_ha": final_yield,
                    "biomass_kg_ha": scalar(daily.get("topwt", pd.Series([np.nan])).iloc[-1]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
                    "safe_irrigation_mm": safe_i,
                    "safe_nitrogen_kg_ha": safe_n,
                    "etcp_mm": float(metrics["etcp_mm"]),
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if math.isfinite(float(metrics["PFP_N_kg_kg"])) else np.nan,
                    "season_last_dap": int(pd.to_numeric(daily["dap"], errors="coerce").max()),
                    "interface_pass": bool(abs(safe_i - float(metrics["actual_irrigation_mm"])) < 1e-6 and abs(safe_n - float(metrics["actual_nitrogen_kg_ha"])) < 1e-6),
                    "grid_pass": grid_ok,
                    "source_weather": "088_corrected_FQ2018" if year == 2018 else "original_FQ_WTH",
                    "daily_path": rel(daily_path),
                    "snapshot_path": rel(snapshot),
                })
    return rows


def run(config_path: Path, dry_run: bool = False, finalize_existing: bool = False) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    required = [
        ROOT / cfg["inputs"]["correction_manifest_088"],
        ROOT / cfg["inputs"]["ppo_paper_summary_089"],
        ROOT / cfg["inputs"]["split_years"],
    ]
    if not CORRECTED_ROOT.exists() or not BASELINE_ROOT.exists():
        raise FileNotFoundError("missing original/corrected FQ input root")
    if any(not p.exists() for p in required):
        raise FileNotFoundError([str(p) for p in required if not p.exists()])
    out_root = (ROOT / cfg["output_root"]).resolve()
    if dry_run:
        return {
            "task_id": cfg["task_id"],
            "dry_run": True,
            "years": cfg["years"],
            "n_runs": len(cfg["years"]) * len(cfg["irrigation_totals_mm"]) * len(cfg["nitrogen_totals_kg_ha"]),
            "output_root": str(out_root),
            "training_allowed": False,
            "external_n_allowed": False,
            "corrected_2018_root_exists": CORRECTED_ROOT.exists(),
        }
    if finalize_existing:
        if not out_root.exists():
            raise FileNotFoundError(f"cannot finalize missing output root: {out_root}")
        grid_rows = finalize_existing_rows(cfg, out_root)
    else:
        if out_root.exists():
            raise FileExistsError(f"refusing to overwrite output root: {out_root}")
        out_root.mkdir(parents=True, exist_ok=False)
        (out_root / "tables").mkdir(parents=True, exist_ok=True)
        config, env_config = load_setup(
            ROOT / cfg["inputs"]["base_env_config"],
            out_root,
            bool(cfg.get("use_formal_fq_safety", False)),
        )
        run_id = str(cfg["task_id"])
        (out_root / "configs").mkdir(parents=True, exist_ok=True)
        direct_ppo.write_yaml(config, out_root / "configs" / f"{run_id}_effective_config.yaml")
        direct_ppo.write_yaml(env_config, out_root / "configs" / f"{run_id}_effective_env_config.yaml")
        grid_rows = []
    start = time.time()
    if not finalize_existing:
        for year in [int(x) for x in cfg["years"]]:
            for irrigation in [int(x) for x in cfg["irrigation_totals_mm"]]:
                for nitrogen in [int(x) for x in cfg["nitrogen_totals_kg_ha"]]:
                    grid_rows.append(run_one(config, env_config, out_root, year, irrigation, nitrogen))
    grid = pd.DataFrame(grid_rows)
    expected_runs = len(cfg["years"]) * len(cfg["irrigation_totals_mm"]) * len(cfg["nitrogen_totals_kg_ha"])
    if len(grid) != expected_runs:
        raise AssertionError(f"unexpected grid row count: {len(grid)}")
    if not bool(grid["interface_pass"].astype(bool).all()) or not bool(grid["grid_pass"].astype(bool).all()):
        raise AssertionError("interface/grid gate failed")
    grid = pareto_flags(grid)
    ppo = pd.read_csv(ROOT / cfg["inputs"]["ppo_paper_summary_089"], keep_default_na=False)
    ppo["scenario"] = ppo["scenario"].astype(str)
    ppo = ppo[ppo["scenario"].eq(CHECKPOINT_PPO)].copy()
    ppo["year"] = pd.to_numeric(ppo["year"], errors="coerce").astype(int)
    ppo["grain_yield_kg_ha"] = pd.to_numeric(ppo["grain_yield_kg_ha"], errors="coerce")
    ppo["WP_ET_kg_m3"] = pd.to_numeric(ppo["WP_ET_kg_m3"], errors="coerce")
    ppo["PFP_N_kg_kg"] = pd.to_numeric(ppo["PFP_N_kg_kg"], errors="coerce")
    decision = make_decision(grid, ppo, cfg)
    run_id = str(cfg["task_id"])
    grid.to_csv(out_root / "tables" / f"{run_id}_fq_action_grid_summary.csv", index=False, encoding="utf-8-sig")
    grid[grid["pareto_efficient"]].to_csv(out_root / "tables" / f"{run_id}_fq_action_grid_pareto_candidates.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(out_root / "tables" / f"{run_id}_fq_action_grid_year_decision.csv", index=False, encoding="utf-8-sig")
    elapsed = time.time() - start
    manifest_path = ROOT / cfg["inputs"]["correction_manifest_088"]
    result = {
        "task_id": cfg["task_id"],
        "status": "ok",
        "elapsed_seconds": elapsed,
        "n_runs": int(len(grid)),
        "n_years": int(grid["year"].nunique()),
        "years": [int(x) for x in sorted(grid["year"].unique())],
        "interface_pass_all": bool(grid["interface_pass"].astype(bool).all()),
        "grid_pass_all": bool(grid["grid_pass"].astype(bool).all()),
        "external_n_included": False,
        "training_run": False,
        "use_formal_fq_safety": bool(cfg.get("use_formal_fq_safety", False)),
        "corrected_2018_used": bool((grid.loc[grid["year"].eq(2018), "source_weather"] == "088_corrected_FQ2018").all()),
        "rescue_signal_years": int(decision["rescue_signal"].astype(bool).sum()),
        "rescue_signal": bool(decision["rescue_signal"].astype(bool).sum() >= 2),
        "decision_rows": json_safe(decision.to_dict(orient="records")),
        "input_sha256": {
            "correction_manifest_088": sha256(manifest_path),
            "ppo_paper_summary_089": sha256(ROOT / cfg["inputs"]["ppo_paper_summary_089"]),
            "base_env_config": sha256(ROOT / cfg["inputs"]["base_env_config"]),
        },
        "output_root": rel(out_root),
        "next_step": "one FQ-specific 2K PPO variant only if rescue_signal is true; otherwise stop tuning",
    }
    (out_root / f"{run_id}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    manifest = {**result, "correction_manifest": json.loads(manifest_path.read_text(encoding="utf-8"))}
    (out_root / f"{run_id}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return result


def write_doc(result: dict[str, Any], out_root: Path, doc_path: Path) -> None:
    run_id = str(result["task_id"])
    decision = pd.read_csv(out_root / "tables" / f"{run_id}_fq_action_grid_year_decision.csv")
    pareto = pd.read_csv(out_root / "tables" / f"{run_id}_fq_action_grid_pareto_candidates.csv")
    lines = [
        f"# {run_id} FQ 水氮动作格反事实审计记录",
        "",
        "## 结论",
        "",
        f"- 完成 {result['n_runs']} 个 DSSAT 前向情景；无训练、无 external-N，接口和动作网格全通过。",
        f"- formal FQ I240 分阶段/后期保留安全层：`{result.get('use_formal_fq_safety', False)}`。",
        f"- 2018 使用 088 隔离修正天气副本：`{result['corrected_2018_used']}`。",
        f"- 满足预注册低氮可行条件的代表年份：{result['rescue_signal_years']} 个；整体 rescue signal：`{result['rescue_signal']}`。",
        "",
        "## 逐年判断",
        "",
        md_table(decision, 20),
        "",
        "## Pareto 候选预览",
        "",
        md_table(pareto[[c for c in ["year", "scenario", "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"] if c in pareto.columns]], 80),
        "",
        "## 解释边界",
        "",
        "- 这些是固定水氮计划的 DSSAT 反事实，不是 PPO 训练结果。",
        "- 低氮候选若只在少数年份出现，不能直接宣称 FQ PPO 可跨年稳定改善。",
        "- 只有 rescue signal 达到预注册阈值，才允许设计一个单因素 FQ 专用 2K PPO smoke；本任务不批准 25K/100K。",
        "",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/090_00_fq_action_grid_counterfactual_audit.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--finalize-existing", action="store_true", help="Summarize completed isolated DSSAT artifacts without rerunning them")
    args = ap.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = run(config_path.resolve(), dry_run=args.dry_run, finalize_existing=args.finalize_existing)
    if not args.dry_run:
        cfg_for_doc = json.loads(config_path.read_text(encoding="utf-8"))
        doc_rel = cfg_for_doc.get("doc_output", "docs/090_00_fq_action_grid_counterfactual_audit_record.md")
        write_doc(result, ROOT / cfg_for_doc["output_root"], ROOT / doc_rel)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()

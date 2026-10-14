"""040_25: irrigation budget feasibility audit for SYA lowIC PPO.

This is a fixed-schedule counterfactual, not RL training.  It replays the
040_24 checkpoint-100000 PPO actions and then adds pre-registered late-season
irrigation increments to test whether the low yield is mainly a water-budget
problem.
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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_staged_irrigation_reserve_mask_040_24 as staged04024
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_evaluate import latest_observation_dict


TASK = "040_25_sya_lowIC_irrigation_budget_feasibility_audit"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
LOWIC_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
PPO04024 = ROOT / "benchmark_results" / "040_24_sya_lowIC_ppo_staged_irrigation_reserve_mask"
PPO04024_EVAL = PPO04024 / "evaluation" / "040_24_checkpoint_validation_summary.csv"
BASELINE_SUMMARY = ROOT / "benchmark_results" / "040_21_sya_lowIC_four_baseline_rebuild" / "evaluation" / "040_21_baseline_summary.csv"

STATION = "SYA"
SITE = "SY"
SEED = 0
CHECKPOINT_STEP = 100_000
MAX_STEPS = 260

BRANCHES = {
    "original_150": [],
    "plus45_to_195": [(82, 45.0)],
    "plus90_to_240": [(82, 45.0), (89, 45.0)],
    "plus135_to_285": [(82, 45.0), (89, 45.0), (96, 45.0)],
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs() -> None:
    for d in ["configs", "tables", "daily_outputs", "snapshots", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 160) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def configure_runtime() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    baseline034.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    base032.StressAwareDiscreteWrapper = staged04024.StagedIrrigationReserveMaskWrapper
    batch03222.base.StressAwareDiscreteWrapper = staged04024.StagedIrrigationReserveMaskWrapper
    cfg = batch03222.load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["output_root"] = rel(OUT)
    cfg["runtime"]["max_steps"] = MAX_STEPS
    cfg["runtime"]["smoke_station"] = STATION
    split = batch03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = batch03222.build_selection(split)
    env_config = direct_ppo.build_env_config(cfg, selection)
    env_config["paths"]["output_root"] = rel(OUT)
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    return cfg, env_config, selection


def action_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    return base032.action_grid(config)


def action_index_for(config: dict[str, Any], irrigation: float, nitrogen: float) -> int:
    for idx, action in enumerate(action_grid(config)):
        if abs(float(action["amir"]) - float(irrigation)) < 1e-9 and abs(float(action["anfer"]) - float(nitrogen)) < 1e-9:
            return int(idx)
    raise ValueError(f"找不到动作档位 I{irrigation}/N{nitrogen}")


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_ppo_source() -> pd.DataFrame:
    df = pd.read_csv(PPO04024_EVAL, keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    rows = df[df["checkpoint_step"].eq(CHECKPOINT_STEP)].copy().sort_values("year").reset_index(drop=True)
    if len(rows) != 10:
        raise RuntimeError(f"040_25 需要 10 个 SYA 验证年 ckpt100000 行，实际 {len(rows)}")
    return rows


def actions_by_dap(daily_csv_path: str) -> dict[int, tuple[float, float]]:
    daily = pd.read_csv(ROOT / daily_csv_path, keep_default_na=False)
    out: dict[int, tuple[float, float]] = {}
    for _, row in daily.iterrows():
        dap = int(round(float(row["dap"])))
        i = float(row.get("safe_action_amir", 0.0) or 0.0)
        n = float(row.get("safe_action_anfer", 0.0) or 0.0)
        if abs(i) > 1e-9 or abs(n) > 1e-9:
            out[dap] = (i, n)
    return out


def apply_extra_irrigation(base_actions: dict[int, tuple[float, float]], extras: list[tuple[int, float]]) -> dict[int, tuple[float, float]]:
    out = dict(base_actions)
    for dap, extra_i in extras:
        old_i, old_n = out.get(int(dap), (0.0, 0.0))
        # Extras are placed at DAPs that should not already contain irrigation.
        # If the branch encounters an existing N-only action, combine I with N.
        if old_i > 1e-9:
            raise RuntimeError(f"DAP{dap} 已有灌溉 {old_i}，不能叠加补水以免超过单次档位")
        out[int(dap)] = (float(extra_i), float(old_n))
    return out


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def summary_metrics(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is not None:
            candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise RuntimeError(f"Summary.OUT 无可用 HWAM：{snapshot}")
    _, neg_idx, row = min(candidates, key=lambda x: (x[0], x[1]))
    ircm = float(siteppo.num(row, "IRCM") or 0.0)
    nicm = float(siteppo.num(row, "NICM") or 0.0)
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else (float(final_yield) / float(etcp) / 10.0 if etcp and float(etcp) > 0 else math.nan)
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and float(ypnam) >= 0 else (float(final_yield) / nicm if nicm > 0 else math.nan)
    return {
        "summary_row_index": int(-neg_idx),
        "summary_match_score": abs(float(siteppo.num(row, "HWAM") or final_yield) - float(final_yield)),
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": float(etcp) if etcp is not None else math.nan,
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
    }


def replay_schedule(
    config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    branch: str,
    schedule: dict[int, tuple[float, float]],
    irrigation_limit: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = json.loads(json.dumps(config))
    cfg["action_safety"]["season_irrigation_soft_limit"] = float(irrigation_limit)
    records: list[dict[str, Any]] = []
    weather = direct_ppo.weather_for_daily(cfg)
    run_tag = f"{STATION}_{year}_040_25_{branch}"
    env = base032.make_env(cfg, env_config, STATION, int(year), SEED, run_tag, evaluation=True)
    snapshot = OUT / "snapshots" / STATION / str(year) / branch
    daily_path = OUT / "daily_outputs" / f"{STATION}_{year}_{branch}_daily.csv"
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < MAX_STEPS:
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            i, n = schedule.get(dap, (0.0, 0.0))
            action_idx = action_index_for(cfg, i, n)
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "branch": branch,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(siteppo.snapshot_from_env(env), snapshot)
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
    total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
    total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
    irrigation_by_bucket = {
        "irrigation_dap1_30": float(pd.to_numeric(daily.loc[pd.to_numeric(daily["dap"], errors="coerce") <= 30, "safe_action_amir"], errors="coerce").fillna(0).sum()),
        "irrigation_dap31_60": float(pd.to_numeric(daily.loc[(pd.to_numeric(daily["dap"], errors="coerce") > 30) & (pd.to_numeric(daily["dap"], errors="coerce") <= 60), "safe_action_amir"], errors="coerce").fillna(0).sum()),
        "irrigation_dap61_90": float(pd.to_numeric(daily.loc[(pd.to_numeric(daily["dap"], errors="coerce") > 60) & (pd.to_numeric(daily["dap"], errors="coerce") <= 90), "safe_action_amir"], errors="coerce").fillna(0).sum()),
        "irrigation_dap91_plus": float(pd.to_numeric(daily.loc[pd.to_numeric(daily["dap"], errors="coerce") > 90, "safe_action_amir"], errors="coerce").fillna(0).sum()),
    }
    summary = {
        "station_code": STATION,
        "site": SITE,
        "year": int(year),
        "branch": branch,
        "run_status": "ok",
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "PFP_N": final_y / total_n if total_n > 0 else math.nan,
        "irrigation_limit_used": float(irrigation_limit),
        "daily_csv_path": rel(daily_path),
        "snapshot_path": rel(snapshot),
        **irrigation_by_bucket,
        **stress_summary(daily),
        **summary_metrics(snapshot, final_y),
    }
    return daily, summary


def add_expert_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    if not BASELINE_SUMMARY.exists() or summary.empty:
        return summary
    base = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    expert = base[base["scenario"].eq("official_extension_expert")].copy()
    for c in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]:
        expert[c] = pd.to_numeric(expert[c], errors="coerce")
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        yr = int(row["year"])
        ex = expert[expert["year"].astype(int).eq(yr)].iloc[0]
        out = row.to_dict()
        out["expert_yield"] = float(ex["grain_yield_kg_ha"])
        out["gap_yield_vs_expert"] = float(row["final_grnwt"]) - float(ex["grain_yield_kg_ha"])
        out["gap_WP_ET_vs_expert"] = float(row["WP_ET_kg_m3"]) - float(ex["WP_ET_kg_m3"])
        out["gap_PFP_N_vs_expert"] = float(row["PFP_N_kg_kg"]) - float(ex["PFP_N_kg_kg"])
        out["water_saved_vs_expert_mm"] = float(ex["actual_irrigation_mm"]) - float(row["total_irrigation"])
        out["n_saved_vs_expert_kg_ha"] = float(ex["actual_nitrogen_kg_ha"]) - float(row["total_n"])
        rows.append(out)
    return pd.DataFrame(rows)


def write_record(plan: pd.DataFrame, summary: pd.DataFrame, by_branch: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# 040_25 SYA lowIC 灌溉预算可行性审计记录",
        "",
        "## 结论边界",
        "",
        "- 本任务不训练，只重放 040_24 checkpoint 100000 的固定动作，并在 DAP60 后补水。",
        "- 它不能作为新 PPO 策略成功结果，只用于判断水量是否是主要限制因子。",
        f"- 耗时：{elapsed:.1f} 秒",
        "",
        "## 预注册补水计划",
        "",
        md_table(plan, 80),
        "",
        "## 分支均值",
        "",
        md_table(by_branch, 80),
        "",
        "## 年度结果",
        "",
        md_table(summary, 240),
        "",
        "## 输出",
        "",
        f"- summary：`{rel(OUT / 'tables' / '040_25_counterfactual_summary.csv')}`",
        f"- daily：`{rel(OUT / 'tables' / '040_25_counterfactual_daily.csv')}`",
        f"- snapshots：`{rel(OUT / 'snapshots')}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dry_run: bool) -> dict[str, Any]:
    ensure_dirs()
    config, env_config, selection = configure_runtime()
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_25_resolved_env_config.yaml")
    selection.to_csv(OUT / "configs" / "040_25_selection.csv", index=False, encoding="utf-8-sig")
    source = load_ppo_source()
    plan_rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        year = int(row["year"])
        base_actions = actions_by_dap(str(row["daily_csv_path"]))
        base_total_i = sum(v[0] for v in base_actions.values())
        for branch, extras in BRANCHES.items():
            planned_total_i = base_total_i + sum(v for _, v in extras)
            plan_rows.append(
                {
                    "year": year,
                    "branch": branch,
                    "source_checkpoint": CHECKPOINT_STEP,
                    "source_daily_csv_path": row["daily_csv_path"],
                    "base_total_irrigation": base_total_i,
                    "extra_irrigation_plan": "; ".join([f"DAP{d}+{amt:g}" for d, amt in extras]),
                    "planned_total_irrigation": planned_total_i,
                    "irrigation_limit_used": max(160.0, planned_total_i),
                    "base_action_sequence": row.get("action_sequence", ""),
                }
            )
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(OUT / "tables" / "040_25_counterfactual_plan.csv", index=False, encoding="utf-8-sig")
    if dry_run:
        result = {
            "task": TASK,
            "mode": "dry_run",
            "planned_rows": int(len(plan)),
            "plan_csv": rel(OUT / "tables" / "040_25_counterfactual_plan.csv"),
            "next_step_allowed": bool(len(plan) == len(source) * len(BRANCHES)),
        }
        (OUT / "040_25_dry_run_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    daily_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    for _, row in source.iterrows():
        year = int(row["year"])
        base_actions = actions_by_dap(str(row["daily_csv_path"]))
        for branch, extras in BRANCHES.items():
            print(f"[040_25] {year} {branch}", flush=True)
            try:
                schedule = apply_extra_irrigation(base_actions, extras)
                limit = max(160.0, sum(v[0] for v in schedule.values()))
                daily, summary = replay_schedule(config, env_config, year, branch, schedule, limit)
                daily_frames.append(daily)
                summaries.append(summary)
            except Exception:
                failures.append({"year": year, "branch": branch, "traceback": traceback.format_exc()[-4000:]})
    elapsed = time.time() - start
    summary = add_expert_gaps(pd.DataFrame(summaries))
    if not summary.empty:
        original = summary[summary["branch"].eq("original_150")][["year", "final_grnwt", "max_swfac", "WP_ET_kg_m3"]].rename(
            columns={"final_grnwt": "original_yield", "max_swfac": "original_max_swfac", "WP_ET_kg_m3": "original_WP_ET"}
        )
        summary = summary.merge(original, on="year", how="left")
        summary["delta_yield_vs_original"] = pd.to_numeric(summary["final_grnwt"], errors="coerce") - pd.to_numeric(summary["original_yield"], errors="coerce")
        summary["delta_WP_ET_vs_original"] = pd.to_numeric(summary["WP_ET_kg_m3"], errors="coerce") - pd.to_numeric(summary["original_WP_ET"], errors="coerce")
        summary["delta_max_swfac_vs_original"] = pd.to_numeric(summary["max_swfac"], errors="coerce") - pd.to_numeric(summary["original_max_swfac"], errors="coerce")
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    summary.to_csv(OUT / "tables" / "040_25_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT / "tables" / "040_25_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "tables" / "040_25_failures.csv", index=False, encoding="utf-8-sig")
    by_branch = pd.DataFrame()
    if not summary.empty:
        by_branch = summary.groupby("branch", as_index=False).agg(
            n=("year", "count"),
            mean_yield=("final_grnwt", "mean"),
            mean_delta_yield_vs_original=("delta_yield_vs_original", "mean"),
            mean_gap_yield_vs_expert=("gap_yield_vs_expert", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_WP_ET=("WP_ET_kg_m3", "mean"),
            mean_gap_WP_ET_vs_expert=("gap_WP_ET_vs_expert", "mean"),
            mean_PFP_N=("PFP_N_kg_kg", "mean"),
            mean_max_swfac=("max_swfac", "mean"),
            max_swfac=("max_swfac", "max"),
        )
    by_branch.to_csv(OUT / "tables" / "040_25_counterfactual_by_branch.csv", index=False, encoding="utf-8-sig")
    write_record(plan, summary, by_branch, elapsed)
    result = {
        "task": TASK,
        "mode": "full",
        "summary_rows": int(len(summary)),
        "failed_rows": int(len(failures_df)),
        "summary_csv": rel(OUT / "tables" / "040_25_counterfactual_summary.csv"),
        "by_branch_csv": rel(OUT / "tables" / "040_25_counterfactual_by_branch.csv"),
        "record_md": rel(DOC),
        "elapsed_seconds": elapsed,
    }
    (OUT / "040_25_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not failures_df.empty:
        print(failures_df.to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.dry_run)


if __name__ == "__main__":
    main()


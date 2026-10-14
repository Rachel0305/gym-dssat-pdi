from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base03200
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo02707


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "036_03"
OUT = ROOT / "benchmark_results" / "036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018"
TABLES = OUT / "tables"
SNAPSHOTS = OUT / "snapshots"
DOC = ROOT / "docs" / "036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018_record.md"
PROMPT = ROOT / "prompts" / "036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018.md"

PPO_03601 = (
    ROOT
    / "benchmark_results"
    / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun"
    / "evaluation"
    / "036_01_checkpoint_validation_summary.csv"
)
CORRECTED_03602 = (
    ROOT
    / "benchmark_results"
    / "036_02_correct_03601_baseline_metric_postprocess"
    / "tables"
    / "036_02_corrected_checkpoint_validation_summary.csv"
)
BASELINE_COVERAGE_03602 = (
    ROOT
    / "benchmark_results"
    / "036_02_correct_03601_baseline_metric_postprocess"
    / "tables"
    / "036_02_baseline_coverage_by_station_year.csv"
)


def ensure_dirs() -> None:
    for p in [TABLES, SNAPSHOTS, DOC.parent]:
        p.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else default
    except Exception:
        try:
            return float(value)
        except Exception:
            return default


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def replay_one(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = str(row["station_code"])
    year = int(row["year"])
    seed = int(row.get("seed", 0))
    checkpoint = int(row["checkpoint_step"])
    model_path = ROOT / str(row["model_path"])
    run_tag = f"{station}_{year}_{TASK_ID}_ckpt{checkpoint}"
    out: dict[str, Any] = {
        "station_code": station,
        "year": year,
        "seed": seed,
        "checkpoint_step": checkpoint,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else str(model_path),
    }
    if not model_path.exists():
        out["run_status"] = "missing_model"
        return out

    env = None
    try:
        model = MaskablePPO.load(str(model_path), device="cpu")
        weather = direct_ppo.weather_for_daily(config)
        env = base03200.make_env(config, env_config, station, year, seed, run_tag, evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        records: list[dict[str, Any]] = []
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = base03200.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base03200.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "seed": seed,
                    "checkpoint_step": checkpoint,
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
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1

        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{station}{year} ckpt{checkpoint} did not finish within max_steps")
        daily_ref_path = ROOT / str(row["daily_csv_path"])
        summary = base03200.summarize_daily("MaskablePPO", daily, daily_ref_path, model_path)
        snapshot_tmp = siteppo02707.snapshot_from_env(env)
        metrics = baseline03400.metrics_from_snapshot(snapshot_tmp, float(summary["final_grnwt"]))
        out.update(summary)
        out.update(stress_summary(daily))
        out.update(metrics)
        out["run_status"] = "ok_replay"
        out["episode_length_replay"] = int(len(daily))
        out["summary_snapshot_tmp"] = str(snapshot_tmp)
        if station == "FQA" and year == 2018:
            target = SNAPSHOTS / station / str(year) / f"ckpt{checkpoint}"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(snapshot_tmp, target)
            out["saved_snapshot_path"] = str(target.relative_to(ROOT)).replace("\\", "/")
        else:
            out["saved_snapshot_path"] = ""
        return out
    except Exception:
        out["run_status"] = "failed"
        out["notes"] = traceback.format_exc()[-4000:]
        return out
    finally:
        if env is not None:
            env.close()


def add_baseline_gaps(metrics: pd.DataFrame, corrected: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "station_code",
        "year",
        "available_rows",
        "available_groups",
        "available_scenarios",
        "max_yield_available",
        "max_wp_et_available",
        "max_pfp_n_available",
    ]
    out = corrected.drop(columns=[c for c in ["WP_ET_kg_m3", "etcp_mm", "PFP_N_kg_kg"] if c in corrected.columns], errors="ignore")
    merge_cols = ["station_code", "year", "checkpoint_step"]
    mcols = merge_cols + [
        "run_status",
        "episode_length_replay",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "summary_match_score",
        "summary_row_index",
        "saved_snapshot_path",
    ]
    out = out.merge(metrics[mcols], on=merge_cols, how="left", suffixes=("", "_replay"))
    out = out.merge(coverage[base_cols], on=["station_code", "year"], how="left", suffixes=("", "_coverage"))
    out["gap_yield_vs_available_baseline_max_with_wp"] = pd.to_numeric(out["final_grnwt"], errors="coerce") - pd.to_numeric(out["max_yield_available"], errors="coerce")
    out["gap_wp_et_vs_available_baseline_max_with_wp"] = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce") - pd.to_numeric(out["max_wp_et_available"], errors="coerce")
    out["gap_pfp_n_vs_available_baseline_max_with_wp"] = pd.to_numeric(out["PFP_N_kg_kg"], errors="coerce") - pd.to_numeric(out["max_pfp_n_available"], errors="coerce")
    out["any_metric_win_available_baseline_with_wp"] = (
        (out["gap_yield_vs_available_baseline_max_with_wp"] > 0)
        | (out["gap_wp_et_vs_available_baseline_max_with_wp"] > 0)
        | (out["gap_pfp_n_vs_available_baseline_max_with_wp"] > 0)
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["run_status_replay"].astype(str).eq("ok_replay")].copy() if "run_status_replay" in df.columns else df.copy()
    if "run_status_replay" not in ok.columns and "run_status" in ok.columns:
        ok = ok[ok["run_status"].astype(str).eq("ok_replay")].copy()
    return (
        ok.groupby(["station_code", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_ETCP_mm=("etcp_mm", "mean"),
            mean_WP_ET_kg_m3=("WP_ET_kg_m3", "mean"),
            mean_PFP_N_summary=("PFP_N_kg_kg", "mean"),
            any_metric_win_count=("any_metric_win_available_baseline_with_wp", lambda s: int(pd.Series(s).fillna(False).sum())),
            yield_win_count=("gap_yield_vs_available_baseline_max_with_wp", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            wp_et_win_count=("gap_wp_et_vs_available_baseline_max_with_wp", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            pfp_n_win_count=("gap_pfp_n_vs_available_baseline_max_with_wp", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            mean_gap_yield=("gap_yield_vs_available_baseline_max_with_wp", "mean"),
            mean_gap_wp_et=("gap_wp_et_vs_available_baseline_max_with_wp", "mean"),
            mean_gap_pfp_n=("gap_pfp_n_vs_available_baseline_max_with_wp", "mean"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


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


def write_record(metrics: pd.DataFrame, full: pd.DataFrame, by: pd.DataFrame, fqa: pd.DataFrame) -> None:
    failed = metrics[~metrics["run_status"].astype(str).eq("ok_replay")].copy()
    lines = [
        "# 036_03 重放 036_01 checkpoint 补算 WP_ET 记录",
        "",
        "## 结论先说",
        "",
        f"- 重放行数：{len(metrics)}；失败行数：{len(failed)}。",
        "- 本任务不训练，只做确定性 checkpoint 评估重放。",
        "- `WP_ET_kg_m3` 来自 DSSAT `Summary.OUT` 的 `YPEM×0.1` 优先口径，缺失时用 `final_yield/ETCP/10`。",
        "- 只保存 FQA2018 四个异常 checkpoint 的完整 DSSAT snapshot，避免大量占用空间。",
        "",
        "## 站点 checkpoint 汇总",
        "",
        md_table(by, 80),
        "",
        "## FQA2018 异常审计",
        "",
        md_table(fqa, 20),
        "",
        "## 失败重放行",
        "",
        md_table(failed[["station_code", "year", "checkpoint_step", "run_status", "notes"]] if len(failed) and "notes" in failed.columns else failed, 40),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_replay_metrics.csv`",
        "- `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_corrected_checkpoint_validation_with_wp_et.csv`",
        "- `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_by_station_checkpoint_with_wp_et.csv`",
        "- `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_fqa2018_summary_audit.csv`",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = batch03222.load_config()
    split = batch03222.load_split()
    selection = batch03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    rows = pd.read_csv(PPO_03601, keep_default_na=False)
    corrected = pd.read_csv(CORRECTED_03602, keep_default_na=False)
    coverage = pd.read_csv(BASELINE_COVERAGE_03602, keep_default_na=False)
    metrics_rows: list[dict[str, Any]] = []
    for idx, row in rows.iterrows():
        metrics_rows.append(replay_one(config, env_config, row))
        if (idx + 1) % 20 == 0:
            partial = pd.DataFrame(metrics_rows)
            partial.to_csv(TABLES / "036_03_replay_metrics_partial.csv", index=False, encoding="utf-8-sig")
            print(f"{idx + 1}/{len(rows)} replayed", flush=True)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(TABLES / "036_03_replay_metrics.csv", index=False, encoding="utf-8-sig")
    full = add_baseline_gaps(metrics, corrected, coverage)
    full.to_csv(TABLES / "036_03_corrected_checkpoint_validation_with_wp_et.csv", index=False, encoding="utf-8-sig")
    by = summarize(full)
    by.to_csv(TABLES / "036_03_by_station_checkpoint_with_wp_et.csv", index=False, encoding="utf-8-sig")
    fqa = full[(full["station_code"].eq("FQA")) & (pd.to_numeric(full["year"], errors="coerce").eq(2018))].copy()
    fqa_cols = [
        "station_code",
        "year",
        "checkpoint_step",
        "run_status_replay",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "summary_match_score",
        "summary_row_index",
        "saved_snapshot_path",
    ]
    fqa[[c for c in fqa_cols if c in fqa.columns]].to_csv(TABLES / "036_03_fqa2018_summary_audit.csv", index=False, encoding="utf-8-sig")
    write_record(metrics, full, by, fqa[[c for c in fqa_cols if c in fqa.columns]])
    payload = {
        "task": TASK_ID,
        "rows": int(len(metrics)),
        "failed_rows": int((~metrics["run_status"].astype(str).eq("ok_replay")).sum()) if "run_status" in metrics else len(metrics),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "metrics_csv": (TABLES / "036_03_replay_metrics.csv").relative_to(ROOT).as_posix(),
        "with_wp_csv": (TABLES / "036_03_corrected_checkpoint_validation_with_wp_et.csv").relative_to(ROOT).as_posix(),
        "by_station_csv": (TABLES / "036_03_by_station_checkpoint_with_wp_et.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / "036_03_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(by.to_string(index=False))


if __name__ == "__main__":
    main()

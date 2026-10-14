"""040_23: re-evaluate existing 040_20 PPO checkpoints and complete WP_ET.

No training is performed.  The script loads frozen MaskablePPO checkpoints from
040_20, evaluates them under the same lowIC input root and hard early-irrigation
reserve mask, saves DSSAT snapshots, and reads ETCP/WP_ET/PFP_N from Summary.OUT.
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
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch03222
import run_sya_lowIC_ppo_early_irrigation_hard_reserve_mask_040_13 as hard04013
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_evaluate import latest_observation_dict


TASK = "040_23_sya_lowIC_ppo_snapshot_metric_completion"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
PPO_OUT = ROOT / "benchmark_results" / "040_13_sya_lowIC_ppo_early_irrigation_hard_reserve_mask_lowic_rework_04020_100k"
PPO_EVAL = PPO_OUT / "evaluation" / "040_13_lowic_rework_04020_100k_checkpoint_validation_summary.csv"
BASELINE_SUMMARY = ROOT / "benchmark_results" / "040_21_sya_lowIC_four_baseline_rebuild" / "evaluation" / "040_21_baseline_summary.csv"
LOWIC_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION = "SYA"
SITE = "SY"
SEED = 0
MAX_STEPS = 260


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs() -> None:
    for d in ["configs", "evaluation", "daily_outputs/SYA", "snapshots/SYA", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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


def configure_runtime() -> tuple[dict[str, Any], dict[str, Any]]:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    baseline034.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    base032.StressAwareDiscreteWrapper = hard04013.EarlyIrrigationHardReserveMaskWrapper
    batch03222.base.StressAwareDiscreteWrapper = hard04013.EarlyIrrigationHardReserveMaskWrapper

    cfg = batch03222.load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["output_root"] = rel(OUT)
    cfg["runtime"]["max_steps"] = MAX_STEPS
    split = batch03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = batch03222.build_selection(split)
    env_config = direct_ppo.build_env_config(cfg, selection)
    env_config["paths"]["output_root"] = rel(OUT)
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_23_resolved_env_config.yaml")
    return cfg, env_config


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


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
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise RuntimeError(f"Summary.OUT 没有可用 HWAM：{snapshot}")
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


def action_sequence(daily: pd.DataFrame) -> str:
    parts: list[str] = []
    dap = pd.to_numeric(daily["dap"], errors="coerce")
    irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    nit = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    for d, i, n in zip(dap, irr, nit):
        if float(i) > 1e-9 or float(n) > 1e-9:
            parts.append(f"DAP{int(d)} I{float(i):g}/N{float(n):g}")
    return "; ".join(parts)


def evaluate_one(config: dict[str, Any], env_config: dict[str, Any], model_path: Path, checkpoint_step: int, year: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    run_tag = f"{STATION}_{year}_040_23_ckpt{checkpoint_step}_eval"
    env = base032.make_env(config, env_config, STATION, int(year), SEED, run_tag, evaluation=True)
    records: list[dict[str, Any]] = []
    snapshot = OUT / "snapshots" / STATION / str(year) / f"ckpt{checkpoint_step}"
    daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_seed{SEED}_ckpt{checkpoint_step}_daily.csv"
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < MAX_STEPS:
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
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
                    "seed": SEED,
                    "checkpoint_step": int(checkpoint_step),
                    "algorithm": "MaskablePPO",
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
        tmp = siteppo.snapshot_from_env(env)
        shutil.copytree(tmp, snapshot)
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
    total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
    total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
    out = {
        "station_code": STATION,
        "site": SITE,
        "year": int(year),
        "seed": SEED,
        "checkpoint_step": int(checkpoint_step),
        "algorithm": "MaskablePPO",
        "run_status": "ok",
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "PFP_N": final_y / total_n if total_n > 0 else math.nan,
        "daily_csv_path": rel(daily_path),
        "model_path": rel(model_path),
        "snapshot_path": rel(snapshot),
        "action_sequence": action_sequence(daily),
        "irrigation_event_count": int((pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0) > 0).sum()),
        "n_event_count": int((pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0) > 0).sum()),
        **stress_summary(daily),
        **summary_metrics(snapshot, final_y),
    }
    return out


def add_baseline_gaps(ppo: pd.DataFrame) -> pd.DataFrame:
    if not BASELINE_SUMMARY.exists() or ppo.empty:
        return ppo
    base = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    for col in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for _, row in ppo.iterrows():
        yr = int(row["year"])
        subset = base[base["year"].astype(int).eq(yr)].copy()
        out = row.to_dict()
        if len(subset) >= 4:
            expert = subset[subset["scenario"].eq("official_extension_expert")].iloc[0]
            out["baseline_rows"] = int(len(subset))
            out["gap_yield_vs_expert"] = float(row["final_grnwt"]) - float(expert["grain_yield_kg_ha"])
            out["gap_WP_ET_vs_expert"] = float(row["WP_ET_kg_m3"]) - float(expert["WP_ET_kg_m3"])
            out["gap_PFP_N_vs_expert"] = float(row["PFP_N_kg_kg"]) - float(expert["PFP_N_kg_kg"])
            out["water_saved_vs_expert_mm"] = float(expert["actual_irrigation_mm"]) - float(row["total_irrigation"])
            out["n_saved_vs_expert_kg_ha"] = float(expert["actual_nitrogen_kg_ha"]) - float(row["total_n"])
            out["gap_yield_vs_four_max"] = float(row["final_grnwt"]) - float(subset["grain_yield_kg_ha"].max())
            out["gap_WP_ET_vs_four_max"] = float(row["WP_ET_kg_m3"]) - float(subset["WP_ET_kg_m3"].max())
            out["gap_PFP_N_vs_four_max"] = float(row["PFP_N_kg_kg"]) - float(subset["PFP_N_kg_kg"].max())
            out["any_metric_win_four"] = bool(
                out["gap_yield_vs_four_max"] > 0
                or out["gap_WP_ET_vs_four_max"] > 0
                or out["gap_PFP_N_vs_four_max"] > 0
            )
        else:
            out["baseline_rows"] = int(len(subset))
        rows.append(out)
    return pd.DataFrame(rows)


def write_record(completed: pd.DataFrame, by_checkpoint: pd.DataFrame, elapsed: float) -> None:
    view_cols = [
        "checkpoint_step",
        "n",
        "mean_yield",
        "mean_irrigation",
        "mean_n",
        "mean_WP_ET",
        "mean_PFP_N",
        "mean_gap_yield_vs_expert",
        "mean_gap_WP_ET_vs_expert",
        "mean_gap_PFP_N_vs_expert",
        "any_metric_win_four_count",
        "max_swfac",
        "max_nstres",
    ]
    year_cols = [
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "gap_yield_vs_expert",
        "gap_WP_ET_vs_expert",
        "gap_PFP_N_vs_expert",
        "any_metric_win_four",
        "max_swfac",
        "max_nstres",
    ]
    lines = [
        "# 040_23 SYA lowIC PPO snapshot 与 WP_ET 指标补全记录",
        "",
        "## 结论边界",
        "",
        "- 本任务只加载 040_20 已有 PPO checkpoint 冻结重评估，没有重新训练。",
        "- 目的只是补齐 DSSAT `Summary.OUT`、`ETCP`、`WP_ET`，并和 040_21 四基线同源比较。",
        "",
        f"- 耗时：{elapsed:.1f} 秒",
        "",
        "## checkpoint 均值",
        "",
        md_table(by_checkpoint[view_cols] if not by_checkpoint.empty else by_checkpoint, 80),
        "",
        "## 年度结果",
        "",
        md_table(completed[year_cols] if not completed.empty else completed, 200),
        "",
        "## 输出文件",
        "",
        f"- 补全 summary：`{rel(OUT / 'evaluation' / '040_23_ppo_metric_completed_summary.csv')}`",
        f"- checkpoint 汇总：`{rel(OUT / 'evaluation' / '040_23_ppo_metric_completed_by_checkpoint.csv')}`",
        f"- daily 输出目录：`{rel(OUT / 'daily_outputs' / 'SYA')}`",
        f"- snapshot 输出目录：`{rel(OUT / 'snapshots' / 'SYA')}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(checkpoints: list[int] | None) -> dict[str, Any]:
    ensure_dirs()
    config, env_config = configure_runtime()
    inventory = pd.read_csv(PPO_EVAL, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="coerce").astype(int)
    inventory["year"] = pd.to_numeric(inventory["year"], errors="coerce").astype(int)
    if checkpoints:
        inventory = inventory[inventory["checkpoint_step"].isin(checkpoints)].copy()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    for _, inv in inventory[["checkpoint_step", "model_path", "year"]].drop_duplicates().sort_values(["checkpoint_step", "year"]).iterrows():
        step = int(inv["checkpoint_step"])
        year = int(inv["year"])
        model_path = ROOT / str(inv["model_path"])
        print(f"[040_23] eval SYA{year} ckpt{step}", flush=True)
        try:
            rows.append(evaluate_one(config, env_config, model_path, step, year))
        except Exception:
            failures.append({"year": year, "checkpoint_step": step, "model_path": rel(model_path), "traceback": traceback.format_exc()[-4000:]})
    completed = add_baseline_gaps(pd.DataFrame(rows))
    elapsed = time.time() - start
    failures_df = pd.DataFrame(failures)
    completed.to_csv(OUT / "evaluation" / "040_23_ppo_metric_completed_summary.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "040_23_failures.csv", index=False, encoding="utf-8-sig")
    by_checkpoint = pd.DataFrame()
    if not completed.empty:
        by_checkpoint = completed.groupby("checkpoint_step", as_index=False).agg(
            n=("year", "count"),
            mean_yield=("final_grnwt", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n=("total_n", "mean"),
            mean_WP_ET=("WP_ET_kg_m3", "mean"),
            mean_PFP_N=("PFP_N_kg_kg", "mean"),
            mean_gap_yield_vs_expert=("gap_yield_vs_expert", "mean"),
            mean_gap_WP_ET_vs_expert=("gap_WP_ET_vs_expert", "mean"),
            mean_gap_PFP_N_vs_expert=("gap_PFP_N_vs_expert", "mean"),
            any_metric_win_four_count=("any_metric_win_four", lambda s: int(pd.Series(s).fillna(False).sum())),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
    by_checkpoint.to_csv(OUT / "evaluation" / "040_23_ppo_metric_completed_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    write_record(completed, by_checkpoint, elapsed)
    result = {
        "task": TASK,
        "evaluated_rows": int(len(completed)),
        "failed_rows": int(len(failures_df)),
        "output_root": rel(OUT),
        "record_md": rel(DOC),
        "elapsed_seconds": elapsed,
    }
    (OUT / "040_23_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not failures_df.empty:
        print(failures_df.to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="", help="逗号分隔，例如 100000；默认补全部 checkpoint")
    args = parser.parse_args()
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    run(checkpoints or None)


if __name__ == "__main__":
    main()


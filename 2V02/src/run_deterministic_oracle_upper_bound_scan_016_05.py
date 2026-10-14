from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diagnose_fq2007_2008_rescue_016_05 import (
    FQ_INPUT_ROOT,
    MZX_NAME as FQ_MZX_NAME,
    TEMPLATE_TRNO as FQ_TEMPLATE_TRNO,
    align_general_sdate_to_icdat,
    full_date_shift_fq_template_to_year,
)
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT as MULTISITE_INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "deterministic_oracle_upper_bound_scan_016_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_016_05_deterministic_oracle_upper_bound_scan_record.md"

DEFAULT_CASES = ["FQ:2007", "FQ:2008", "FQ:2016", "YC:2014"]
IRR_TOTALS = [0.0, 60.0, 120.0]
N_TOTALS = [0.0, 150.0, 300.0]
TIMING_MODES = ["early", "critical"]


def split_capped(total: float, dates: list[int], cap: float) -> dict[int, float]:
    if total <= 0:
        return {}
    remaining = float(total)
    out: dict[int, float] = {}
    for dap in dates:
        if remaining <= 1e-9:
            break
        amt = min(cap, remaining)
        out[dap] = amt
        remaining -= amt
    if remaining > 1e-6:
        out[dates[-1]] = out.get(dates[-1], 0.0) + remaining
    return out


def schedule_for(total_i: float, total_n: float, timing: str) -> dict[int, dict[str, float]]:
    if timing == "early":
        irrigation_dates = [25, 40, 55, 70]
        nitrogen_dates = [1, 25, 45]
    elif timing == "critical":
        irrigation_dates = [45, 60, 75, 90]
        nitrogen_dates = [1, 40, 65]
    else:
        raise ValueError(timing)
    irr = split_capped(total_i, irrigation_dates, cap=30.0)
    nit = split_capped(total_n, nitrogen_dates, cap=100.0)
    all_daps = sorted(set(irr) | set(nit))
    return {dap: {"amir": irr.get(dap, 0.0), "anfer": nit.get(dap, 0.0)} for dap in all_daps}


def prepare_fq_text(year: int) -> tuple[str, int]:
    source = (FQ_INPUT_ROOT / FQ_MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    shifted = full_date_shift_fq_template_to_year(source, year)
    yy = f"{year % 100:02d}"
    shifted = align_general_sdate_to_icdat(shifted, FQ_TEMPLATE_TRNO, f"{yy}152")
    return set_management_for_treatment(shifted, FQ_TEMPLATE_TRNO, "L", "L"), FQ_TEMPLATE_TRNO


def prepare_yc_text(year: int) -> tuple[str, int]:
    cfg = SITE_CONFIG["YC"]
    trno = cfg["treatments"][year]
    source = (MULTISITE_INPUT_ROOT / "YC" / cfg["mzx"]).read_text(encoding="latin-1", errors="ignore")
    return set_management_for_treatment(source, trno, "L", "L"), trno


def prepare_run_dir(site: str, year: int, scenario: str) -> Path:
    run_dir = OUT_DIR / "runs" / site / str(year) / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    if site == "FQ":
        text, trno = prepare_fq_text(year)
        input_src = FQ_INPUT_ROOT
        mzx_name = FQ_MZX_NAME
    elif site == "YC":
        text, trno = prepare_yc_text(year)
        input_src = MULTISITE_INPUT_ROOT / "YC"
        mzx_name = SITE_CONFIG["YC"]["mzx"]
    else:
        raise ValueError(site)

    filex = input_dir / f"{site}{year}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
        if src.is_file() and src.name != mzx_name:
            shutil.copyfile(src, input_dir / src.name)

    aux = [
        str(p)
        for p in input_dir.iterdir()
        if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}
    ]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": int(trno),
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"site": site, "year": year, "scenario": scenario, "trno": int(trno)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, schedule: dict[int, dict[str, float]], max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", 0)) or 0))
            real_action = schedule.get(dap_before, {"amir": 0.0, "anfer": 0.0})
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, real_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": meta["site"],
                    "year": meta["year"],
                    "scenario": meta["scenario"],
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(real_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(real_action.get("anfer", 0.0)),
                    "reward": scalar(reward),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(run_dir / "daily.csv", index=False, encoding="utf-8-sig")


def summarize(run_dir: Path, site: str, year: int, scenario: str, status: str, message: str = "") -> dict[str, Any]:
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    daily_path = run_dir / "daily.csv"
    daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    gwad = np.nan
    cwad = np.nan
    if not plantgro.empty:
        if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty:
            gwad = float(plantgro["GWAD"].dropna().iloc[-1])
        if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty:
            cwad = float(plantgro["CWAD"].dropna().iloc[-1])
    return {
        "site": site,
        "year": year,
        "scenario": scenario,
        "status": status,
        "message": message,
        "final_gwad": gwad,
        "final_cwad": cwad,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else np.nan,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty and "swfac" in daily else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty and "nstres" in daily else np.nan,
        "last_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty and "dap" in daily else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def run_one(site: str, year: int, total_i: float, total_n: float, timing: str, timeout_s: int) -> dict[str, Any]:
    scenario = f"I{int(total_i)}_N{int(total_n)}_{timing}"
    run_dir = prepare_run_dir(site, year, scenario)
    schedule = schedule_for(total_i, total_n, timing)
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        str(run_dir),
        "--schedule",
        str(run_dir / "schedule.json"),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (run_dir / "child_timeout_stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="ignore")
        (run_dir / "child_timeout_stderr.txt").write_text(exc.stderr or "", encoding="utf-8", errors="ignore")
        return summarize(run_dir, site, year, scenario, "timeout", f"timeout_after_{timeout_s}s")
    (run_dir / "child_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    (run_dir / "child_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        return summarize(run_dir, site, year, scenario, "failed", f"returncode={proc.returncode}")
    return summarize(run_dir, site, year, scenario, "ok", "")


def parse_case(value: str) -> tuple[str, int]:
    site, year = value.split(":", 1)
    return site.upper(), int(year)


def write_record(summary: pd.DataFrame, best: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    def table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in df.itertuples(index=False):
            vals = []
            for val in row:
                if isinstance(val, float):
                    vals.append("" if np.isnan(val) else f"{val:.2f}")
                else:
                    vals.append(str(val))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    lines = [
        "# 016_05 确定性水氮上界扫描记录",
        "",
        "## 目的",
        "",
        "不训练 DQN/PPO，仅用人工确定性水氮时序组合检查当前站点年份是否存在可达的高产/节水/节氮策略空间。",
        "",
        "## 扫描组合",
        "",
        f"- 灌溉总量：{IRR_TOTALS}",
        f"- 施氮总量：{N_TOTALS}",
        f"- 时机模式：{TIMING_MODES}",
        "- 单次灌溉上限 30 mm，单次施氮上限 100 kg/ha。",
        "",
        "## 每个站点年份的最高产量组合",
        "",
        table(best),
        "",
        "## 全部结果路径",
        "",
        f"- summary CSV: `{(OUT_DIR / '016_05_deterministic_oracle_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- best CSV: `{(OUT_DIR / '016_05_deterministic_oracle_best_by_year.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--timeout-s", type=int, default=90)
    args = parser.parse_args()

    if args.child:
        schedule = json.loads(args.schedule.read_text(encoding="utf-8")) if args.schedule else {}
        schedule = {int(k): v for k, v in schedule.items()}
        child_run(args.child, schedule=schedule)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in args.cases:
        site, year = parse_case(case)
        for total_i in IRR_TOTALS:
            for total_n in N_TOTALS:
                for timing in TIMING_MODES:
                    print(f"[016_05] {site}{year} I={total_i} N={total_n} timing={timing}", flush=True)
                    rows.append(run_one(site, year, total_i, total_n, timing, timeout_s=args.timeout_s))
    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "016_05_deterministic_oracle_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ok = summary[summary["status"].eq("ok") & summary["final_gwad"].notna()].copy()
    best = ok.sort_values(["site", "year", "final_gwad"], ascending=[True, True, False]).groupby(["site", "year"], as_index=False).head(1)
    best_path = OUT_DIR / "016_05_deterministic_oracle_best_by_year.csv"
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    write_record(summary, best)
    print(best.to_string(index=False))
    print(f"[016_05] summary: {summary_path}")
    print(f"[016_05] record: {DOC_PATH}")


if __name__ == "__main__":
    main()


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

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_deterministic_oracle_upper_bound_scan_016_05 import prepare_fq_text, parse_dssat_table

OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq_n_timing_response_diagnostic_016_07"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_016_07_fq_n_timing_response_diagnostic_record.md"
YEARS = [2008, 2016]
TOTAL_N = 150.0
TIMING_MODES = {
    "early": {1: 50.0, 20: 50.0, 40: 50.0},
    "critical": {40: 50.0, 55: 50.0, 70: 50.0},
    "late": {55: 50.0, 70: 50.0, 85: 50.0},
    "split_early_late": {1: 50.0, 40: 50.0, 70: 50.0},
    "single_critical": {55: 150.0},
}
IRRIGATION = {45: 30.0, 70: 30.0}


def schedule_for(mode: str) -> dict[int, dict[str, float]]:
    nit = TIMING_MODES[mode]
    daps = sorted(set(IRRIGATION) | set(nit))
    return {dap: {"amir": IRRIGATION.get(dap, 0.0), "anfer": nit.get(dap, 0.0)} for dap in daps}


def prepare_run_dir(year: int, mode: str) -> Path:
    scenario = f"I60_N150_{mode}"
    run_dir = OUT_DIR / "runs" / str(year) / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    text, trno = prepare_fq_text(year)
    filex = input_dir / f"FQ{year}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    input_src = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "FQ"
    for src in input_src.iterdir():
        if src.is_file() and src.name != "CNFQ0801.MZX":
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
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
    return run_dir


def child_run(run_dir: Path, schedule: dict[int, dict[str, float]], max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
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
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
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


def run_one(year: int, mode: str, timeout_s: int) -> dict[str, Any]:
    run_dir = prepare_run_dir(year, mode)
    schedule = schedule_for(mode)
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
    except subprocess.TimeoutExpired:
        return {
            "year": year,
            "timing_mode": mode,
            "status": "timeout",
            "final_gwad": np.nan,
            "final_cwad": np.nan,
            "max_water_stress": np.nan,
            "max_nitrogen_stress": np.nan,
            "last_dap": np.nan,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        }
    (run_dir / "child_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    (run_dir / "child_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
    daily = pd.read_csv(run_dir / "daily.csv") if (run_dir / "daily.csv").exists() else pd.DataFrame()
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    gwad = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan
    cwad = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan
    return {
        "year": year,
        "timing_mode": mode,
        "status": "ok" if proc.returncode == 0 else f"failed_{proc.returncode}",
        "final_gwad": gwad,
        "final_cwad": cwad,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "last_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def write_record(df: pd.DataFrame) -> None:
    def table(x: pd.DataFrame) -> str:
        cols = list(x.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in x.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append("" if np.isnan(v) else f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# 016_07 FQ 施氮时机响应诊断",
        "",
        "固定灌溉为 `I60 critical`，固定总施氮为 `N150`，只改变施氮时机。",
        "",
        table(df),
        "",
        "## 初步判读",
        "",
    ]
    for year, g in df.groupby("year"):
        best = g["final_gwad"].max()
        best_modes = ", ".join(g.loc[g["final_gwad"].eq(best), "timing_mode"].tolist())
        lines.append(f"- FQ{year}: 最佳产量 `{best:.1f}`，对应时机模式 `{best_modes}`。")
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--timeout-s", type=int, default=90)
    args = parser.parse_args()

    if args.child:
        schedule = json.loads(args.schedule.read_text(encoding="utf-8")) if args.schedule else {}
        schedule = {int(k): v for k, v in schedule.items()}
        child_run(args.child, schedule)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in YEARS:
        for mode in TIMING_MODES:
            print(f"[016_07] FQ{year} mode={mode}", flush=True)
            rows.append(run_one(year, mode, args.timeout_s))
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "016_07_fq_n_timing_response_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_record(df)
    print(df.to_string(index=False))
    print(out_csv)
    print(DOC_PATH)


if __name__ == "__main__":
    main()


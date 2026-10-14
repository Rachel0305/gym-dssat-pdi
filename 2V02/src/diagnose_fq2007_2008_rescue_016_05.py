from __future__ import annotations

import argparse
import json
import re
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
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    INPUT_ROOT as FQ_INPUT_ROOT,
    MZX_NAME,
    TEMPLATE_TRNO,
    prepare_text_for_shifted_scenario,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    parse_dssat_table,
    prepare_text_for_scenario,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2007_2008_rescue_016_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_016_05_fq2007_2008_rescue_record.md"
YEARS = [2007, 2008]
VARIANTS = ["old_shift_exp2", "old_shift_exp1", "full_date_shift_exp2", "full_date_shift_exp2_sdate_icdat"]


def full_date_shift_fq_template_to_year(source: str, year: int) -> str:
    """Shift both 07xxx and 08xxx DSSAT dates to the target year.

    The old FQ transfer helper only shifted 08xxx dates because it treated FQ2008
    treatment 2 as the template. The source file also contains shared 07xxx
    initial-condition and simulation-control dates, which can leave the generated
    FileX internally inconsistent for some target years.
    """

    yy = f"{year % 100:02d}"
    text = source
    text = text.replace("CNFQ0801", f"CNFQ{yy}01")
    text = text.replace("CNFQ2008", f"CNFQ{year}")
    text = text.replace("CNFQ2007", f"CNFQ{year}")
    text = text.replace("Sim2008", f"Sim{year}")
    text = text.replace("Sim2007", f"Sim{year}")
    text = text.replace(" 2008", f" {year}")
    text = text.replace(" 2007", f" {year}")
    text = re.sub(r"\b(?:07|08)(\d{3})\b", rf"{yy}\1", text)
    return text


def align_general_sdate_to_icdat(text: str, trno: int, icdat: str) -> str:
    out: list[str] = []
    changed = False
    in_general = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N GENERAL") and "SDATE" in stripped:
            in_general = True
            out.append(line)
            continue
        if in_general and re.match(rf"^\s*{trno}\s+GE\b", line):
            parts = line.split()
            # @N GENERAL columns: N GE NYERS NREPS START SDATE RSEED SNAME SMODEL
            # parts indices:       0 1  2     3     4     5     6     7     8
            if len(parts) >= 6:
                parts[5] = icdat
                out.append(
                    f" {int(parts[0]):1d} GE {parts[2]:>14} {parts[3]:>5} {parts[4]:>5} {parts[5]:>5} {parts[6]:>5} {parts[7]}"
                )
                changed = True
                continue
        if in_general and stripped.startswith("@"):
            in_general = False
        out.append(line)
    if not changed:
        raise RuntimeError(f"Could not align SDATE for treatment {trno}")
    return "\n".join(out) + "\n"


def prepare_variant_text(year: int, variant: str) -> tuple[str, int]:
    source = (FQ_INPUT_ROOT / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    if variant == "old_shift_exp2":
        return prepare_text_for_shifted_scenario(year, "null"), 2
    if variant == "old_shift_exp1":
        shifted = prepare_text_for_shifted_scenario(year, "recorded_shifted")
        return prepare_text_for_scenario(shifted, 1, "null"), 1
    if variant == "full_date_shift_exp2":
        shifted = full_date_shift_fq_template_to_year(source, year)
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "null"), TEMPLATE_TRNO
    if variant == "full_date_shift_exp2_sdate_icdat":
        shifted = full_date_shift_fq_template_to_year(source, year)
        yy = f"{year % 100:02d}"
        shifted = align_general_sdate_to_icdat(shifted, TEMPLATE_TRNO, f"{yy}152")
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "null"), TEMPLATE_TRNO
    raise ValueError(variant)


def prepare_run_dir(year: int, variant: str) -> Path:
    run_dir = OUT_DIR / "runs" / str(year) / variant
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    text, experiment_number = prepare_variant_text(year, variant)
    filex = input_dir / f"CNFQ{year}_{variant}_null.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in FQ_INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
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
        "experiment_number": int(experiment_number),
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "site": "FQ",
                "year": year,
                "variant": variant,
                "scenario": "null",
                "experiment_number": int(experiment_number),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": meta["site"],
                    "year": meta["year"],
                    "variant": meta["variant"],
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
    pd.DataFrame(rows).to_csv(run_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")


def summarize_run(run_dir: Path, year: int, variant: str, status: str, message: str = "") -> dict[str, Any]:
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    daily_path = run_dir / "gym_post_state_daily.csv"
    daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    gwad = np.nan
    cwad = np.nan
    if not plantgro.empty:
        if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty:
            gwad = float(plantgro["GWAD"].dropna().iloc[-1])
        if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty:
            cwad = float(plantgro["CWAD"].dropna().iloc[-1])
    return {
        "year": year,
        "variant": variant,
        "status": status,
        "message": message,
        "n_daily_rows": int(len(daily)),
        "final_gwad": gwad,
        "final_cwad": cwad,
        "last_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty and "dap" in daily else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty and "swfac" in daily else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty and "nstres" in daily else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def run_one(year: int, variant: str, timeout_s: int) -> dict[str, Any]:
    run_dir = prepare_run_dir(year, variant)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(run_dir), "--max-steps", "360"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        (run_dir / "child_timeout_stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="ignore")
        (run_dir / "child_timeout_stderr.txt").write_text(exc.stderr or "", encoding="utf-8", errors="ignore")
        return summarize_run(run_dir, year, variant, "timeout", f"timeout_after_{timeout_s}s")
    (run_dir / "child_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    (run_dir / "child_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        return summarize_run(run_dir, year, variant, "failed", f"returncode={proc.returncode}")
    return summarize_run(run_dir, year, variant, "ok", "")


def write_record(summary: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    def simple_markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in df.itertuples(index=False):
            vals = []
            for val in row:
                if isinstance(val, float):
                    vals.append("" if np.isnan(val) else f"{val:.3f}")
                else:
                    vals.append(str(val))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    lines = [
        "# 016_05 FQ2007/FQ2008 抢救 smoke 记录",
        "",
        "## 目的",
        "",
        "检查 FQ2007/FQ2008 在旧多年迁移实验中失败是否由日期迁移不完整导致。",
        "",
        "## 变体",
        "",
        "- `old_shift_exp2`：旧迁移逻辑 + treatment 2。",
        "- `old_shift_exp1`：旧迁移逻辑 + treatment 1。",
        "- `full_date_shift_exp2`：将 `07xxx/08xxx` 日期统一平移到目标年份 + treatment 2。",
        "- `full_date_shift_exp2_sdate_icdat`：在上一变体基础上，将 treatment 2 的 `SDATE` 对齐到 `ICDAT=yy152`。",
        "",
        "## 结果摘要",
        "",
        simple_markdown_table(summary),
        "",
        "## 初步判读",
        "",
    ]
    ok_full = summary[(summary["variant"].eq("full_date_shift_exp2_sdate_icdat")) & (summary["status"].eq("ok"))]
    if len(ok_full) == len(YEARS):
        lines += [
            "`full_date_shift_exp2` 对 2007/2008 均可运行，说明这两个年份本身可以抢救；旧排除原因更可能是 FileX 日期迁移不完整，而不是年份不可用。",
        ]
    else:
        lines += [
            "`full_date_shift_exp2` 仍未全部通过，需要继续检查 treatment 指针、ICDAT/SDATE 与 FileX 多处理结构。",
        ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    args = parser.parse_args()

    if args.child:
        child_run(args.child, max_steps=args.max_steps)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for year in args.years:
        for variant in args.variants:
            print(f"[016_05] smoke year={year} variant={variant}", flush=True)
            rows.append(run_one(year, variant, timeout_s=args.timeout_s))
    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "016_05_fq2007_2008_rescue_smoke_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_record(summary)
    print(summary.to_string(index=False))
    print(f"[016_05] summary: {summary_path}")
    print(f"[016_05] record: {DOC_PATH}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


WINDOWS_RUN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "run_CNYC0802_DSSAT480_IC0_null_2000_2023"
OUT_DIR = WINDOWS_RUN_DIR / "analysis_windows480_vs_pdi_yearly_ic0"
YEARS = list(range(2008, 2024))


def parse_table_out(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    current_run: int | None = None
    current_treatment: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            run_match = re.match(r"\*RUN\s+(\d+)\s*:\s*(.*?)\s{2,}", line)
            if run_match:
                current_run = int(run_match.group(1))
                current_treatment = run_match.group(2).strip()
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d{4}\s+\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    row = dict(zip(header, parts[: len(header)]))
                    row["RUNNO"] = current_run
                    row["TNAM"] = current_treatment
                    rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    if {"RUNNO", "YEAR", "DOY", "DAP"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["RUNNO", "YEAR", "DOY", "DAP"], keep="last").reset_index(drop=True)
    return df


def parse_summary(path: Path) -> pd.DataFrame:
    header = None
    rows = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    rows.append(parts[: len(header)])
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def render_single_year_mzx(year: int, out_path: Path) -> None:
    yy = year % 100
    yy2 = f"{yy:02d}"
    text = (WINDOWS_RUN_DIR / "CNYC0801.MZX").read_text(encoding="latin1", errors="ignore")
    # Keep the file structure intact but make treatment 1 a one-year run.
    replacements = {
        "Sim2008": f"Sim{year}",
        "CNYC2008": f"CNYC{year}",
        "CNYC0801": f"CNYC{yy2}01",
        " 1 08170 08176": f" 1 {yy2}170 {yy2}176",
        " 1 08163 IR003": f" 1 {yy2}163 IR003",
        " 1 08211 FE005": f" 1 {yy2}211 FE005",
        " 1 08153   -99": f" 1 {yy2}153   -99",
        " 1 GE             23     1     S 08153": f" 1 GE              1     1     S {yy2}153",
        " 1 PL          08001 08001": f" 1 PL          {yy2}001 {yy2}001",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Also update descriptive free text year tokens in treatment 1 rows where fixed
    # widths are not critical.
    text = re.sub(r"(?m)^ 1 MZ ZD0985 Yucheng  No985$", " 1 MZ ZD0985 Yucheng  No985", text)
    out_path.write_text(text, encoding="latin1", errors="ignore")


def run_pdi_for_year(year: int, timeout_s: int = 90) -> dict[str, Any]:
    year_dir = OUT_DIR / "pdi_yearly_runs" / str(year)
    input_dir = year_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    filex = input_dir / f"CNYC{year % 100:02d}01.MZX"
    render_single_year_mzx(year, filex)

    aux: list[str] = []
    for src in [
        WINDOWS_RUN_DIR / f"CNYC{year % 100:02d}01.WTH",
        PROJECT_ROOT / "my_data" / "YC.SOL",
        PROJECT_ROOT / "my_data" / "MZCER048.CUL",
        PROJECT_ROOT / "my_data" / "CNYC.CLI",
    ]:
        if src.exists():
            dst = input_dir / src.name
            shutil.copyfile(src, dst)
            aux.append(str(dst))
        else:
            raise FileNotFoundError(str(src))

    env_args = {
        "log_saving_path": str(year_dir / f"pdi_{year}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (year_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")

    # Execute one year in a child process so a stuck PDI run can be killed by
    # subprocess timeout when this script is launched from the host/container.
    # When this function is called inside the child mode, actually run gym.
    return {"year": year, "year_dir": str(year_dir)}


def child_run_year(year: int, max_steps: int = 450) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    year_dir = OUT_DIR / "pdi_yearly_runs" / str(year)
    env_args = json.loads((year_dir / "env_args.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    rows = []
    done = False
    for step in range(max_steps):
        action = {name: 0.0 for name in env.formator.action_names}
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
        obs, reward, terminated, truncated, info = env.step(norm)
        done = bool(terminated or truncated)
        latest = latest_observation_dict(env, obs, info)
        yrdoy = scalar(latest.get("yrdoy"))
        rows.append(
            {
                "year_requested": year,
                "step": step,
                "yrdoy": yrdoy,
                "year": int(yrdoy // 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "dap": scalar(latest.get("dap")),
                "topwt": scalar(latest.get("topwt")),
                "grnwt": scalar(latest.get("grnwt")),
                "xlai": scalar(latest.get("xlai")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "reward": scalar(reward),
                "done": done,
            }
        )
        if done:
            break
    pd.DataFrame(rows).to_csv(year_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, year_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()


def run_child_processes(years: list[int]) -> pd.DataFrame:
    statuses = []
    for year in years:
        run_pdi_for_year(year)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-year", str(year)]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=100, capture_output=True, text=True)
            statuses.append(
                {
                    "year": year,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-500:],
                    "stderr_tail": proc.stderr[-1000:],
                    "timed_out": False,
                }
            )
        except subprocess.TimeoutExpired as exc:
            statuses.append(
                {
                    "year": year,
                    "returncode": None,
                    "stdout_tail": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                    "timed_out": True,
                }
            )
    out = pd.DataFrame(statuses)
    out.to_csv(OUT_DIR / "pdi_yearly_run_status.csv", index=False, encoding="utf-8-sig")
    return out


def collect_pdi_raw(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        pg = OUT_DIR / "pdi_yearly_runs" / str(year) / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if not pg.exists():
            continue
        df = parse_table_out(pg)
        df["requested_year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    windows_pg = parse_table_out(WINDOWS_RUN_DIR / "PlantGro.OUT")
    windows_pg = windows_pg[windows_pg["YEAR"].isin(YEARS)].copy()
    pdi_pg = collect_pdi_raw(YEARS)
    pdi_pg.to_csv(OUT_DIR / "pdi_yearly_PlantGro_daily_parsed.csv", index=False, encoding="utf-8-sig")
    windows_pg.to_csv(OUT_DIR / "windows480_PlantGro_daily_parsed.csv", index=False, encoding="utf-8-sig")
    if pdi_pg.empty:
        return pd.DataFrame(), pdi_pg
    merged = windows_pg.merge(
        pdi_pg[["YEAR", "DOY", "DAP", "CWAD", "GWAD", "LAID", "WSPD", "NSTD"]],
        on=["YEAR", "DOY", "DAP"],
        how="outer",
        suffixes=("_windows", "_pdi"),
        indicator=True,
    )
    merged = merged.rename(columns={"YEAR": "year", "DOY": "doy", "DAP": "dap"})
    for col in ["CWAD", "GWAD", "LAID", "WSPD", "NSTD"]:
        w = f"{col}_windows"
        p = f"{col}_pdi"
        if w in merged and p in merged:
            merged[f"diff_{col}_windows_minus_pdi"] = merged[w] - merged[p]
    merged.to_csv(OUT_DIR / "daily_windows480_vs_pdi_yearly_by_DAP.csv", index=False, encoding="utf-8-sig")
    return merged, pdi_pg


def plot_year(merged: pd.DataFrame, year: int, out_path: Path) -> None:
    sub = merged[merged["year"].eq(year)].sort_values("dap")
    if sub.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    specs = [
        ("CWAD / biomass", "CWAD_windows", "CWAD_pdi", "kg/ha"),
        ("GWAD / grain", "GWAD_windows", "GWAD_pdi", "kg/ha"),
        ("WSPD / water stress", "WSPD_windows", "WSPD_pdi", "index"),
        ("NSTD / nitrogen stress", "NSTD_windows", "NSTD_pdi", "index"),
    ]
    for ax, (title, wcol, pcol, ylabel) in zip(axes, specs):
        ax.plot(sub["dap"], sub[wcol], color="#804126", linewidth=3.0, label="Windows DSSAT 4.8.0")
        ax.plot(sub["dap"], sub[pcol], color="#2E4780", linewidth=2.0, linestyle=(0, (4, 2)), marker="o", markersize=2.4, markevery=max(1, len(sub) // 18), label="PDI DSSAT 4.8.0 yearly")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("DAP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.suptitle(f"YC {year} IC=0 null: Windows DSSAT 4.8.0 vs PDI yearly run", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = run_child_processes(YEARS)
    merged, pdi_pg = make_comparison()
    figures_dir = OUT_DIR / "high_contrast_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not merged.empty:
        for year in sorted(merged["year"].dropna().astype(int).unique().tolist()):
            if year in YEARS:
                plot_year(merged, year, figures_dir / f"YC_IC0_windows480_vs_pdi_yearly_{year}.png")
        summary_rows = []
        for year, g in merged.groupby("year"):
            row = {"year": int(year), "rows": int(len(g))}
            for col in ["CWAD", "GWAD", "WSPD", "NSTD"]:
                d = f"diff_{col}_windows_minus_pdi"
                if d in g:
                    row[f"max_abs_{col}_diff"] = float(pd.to_numeric(g[d], errors="coerce").abs().max())
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "annual_max_daily_diff_windows480_vs_pdi_yearly.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"out_dir": str(OUT_DIR), "status": status.to_dict("records"), "pdi_rows": int(len(pdi_pg)), "merged_rows": int(len(merged))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child-year":
        child_run_year(int(sys.argv[2]))
    else:
        main()

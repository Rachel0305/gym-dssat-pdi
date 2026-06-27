from __future__ import annotations

import json
import math
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


TEMPLATE_MZX = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0407_DSSAT480_IC0_null_2004_2023" / "CNHL0407.MZX"
WEATHER_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0405_IC0_null"
FALLBACK_WEATHER_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_ic1_pdi_yearly_2004_2023"
YEARS = list(range(2004, 2024))


def find_weather(year: int) -> Path:
    yy = year % 100
    name = f"CNHL{yy:02d}01.WTH"
    for base in [WEATHER_DIR, FALLBACK_WEATHER_DIR, PROJECT_ROOT / "my_data"]:
        p = base / name
        if p.exists():
            return p
    matches = list(PROJECT_ROOT.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(name)


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


def parse_weather_out(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = parse_table_out(path)
    keep = [c for c in ["YEAR", "DOY", "DAS", "PRED"] if c in df.columns]
    return df[keep].rename(columns={"YEAR": "year", "DOY": "doy", "DAS": "das", "PRED": "rain"})


def render_single_year_mzx(year: int, out_path: Path) -> None:
    yy2 = f"{year % 100:02d}"
    text = TEMPLATE_MZX.read_text(encoding="latin1", errors="ignore")
    replacements = {
        "CNHL0407MZ": f"CNHL{yy2}I1MZ",
        "Sim2004": f"Sim{year}_IC1",
        " 1 1 1 0 Sim": " 1 1 1 0 Sim",
        "                    1  1  0  0  1  0  1": "                    1  1  0  1  1  0  1",
        "CNHL2004": f"CNHL{year}",
        " 1    MZ 04125": f" 1    MZ {yy2}125",
        " 1 04125 04132": f" 1 {yy2}125 {yy2}132",
        " 1 04125   -99": f" 1 {yy2}125   -99",
        " 1 04125 FE005": f" 1 {yy2}125 FE005",
        " 1 GE             20     1     S 04125": f" 1 GE              1     1     S {yy2}125",
        " 1 PL          04001 04001": f" 1 PL          {yy2}001 {yy2}001",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    out_path.write_text(text, encoding="latin1", errors="ignore")


def prepare_year(year: int) -> Path:
    year_dir = OUT_DIR / "pdi_yearly_runs" / str(year)
    input_dir = year_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    filex = input_dir / f"CNHL{year % 100:02d}I1.MZX"
    render_single_year_mzx(year, filex)
    aux = []
    for src in [
        find_weather(year),
        PROJECT_ROOT / "my_data" / "HL.SOL",
        PROJECT_ROOT / "my_data" / "MZCER048.CUL",
    ]:
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux.append(str(dst))
    env_args = {
        "log_saving_path": str(year_dir / f"pdi_hla_ic1_{year}.log"),
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
    return year_dir


def child_run_year(year: int, max_steps: int = 450) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    year_dir = OUT_DIR / "pdi_yearly_runs" / str(year)
    env_args = json.loads((year_dir / "env_args.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    rows = []
    for step in range(max_steps):
        action = {name: 0.0 for name in env.formator.action_names}
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
        obs, reward, terminated, truncated, info = env.step(norm)
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
                "done": bool(terminated or truncated),
            }
        )
        if terminated or truncated:
            break
    pd.DataFrame(rows).to_csv(year_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, year_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()


def run_years() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for year in YEARS:
        prepare_year(year)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-year", str(year)]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=100, capture_output=True, text=True)
            statuses.append({"year": year, "returncode": proc.returncode, "timed_out": False, "stderr_tail": proc.stderr[-1000:]})
        except subprocess.TimeoutExpired as exc:
            statuses.append({"year": year, "returncode": None, "timed_out": True, "stderr_tail": str(exc)[-1000:]})
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "hla_ic1_pdi_yearly_run_status.csv", index=False, encoding="utf-8-sig")
    return status


def collect_outputs() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        pg = OUT_DIR / "pdi_yearly_runs" / str(year) / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if not pg.exists():
            continue
        df = parse_table_out(pg)
        df["requested_year"] = year
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(OUT_DIR / "hla_ic1_pdi_PlantGro_daily_parsed.csv", index=False, encoding="utf-8-sig")
    return out


def make_plot_data(pg: pd.DataFrame) -> pd.DataFrame:
    if pg.empty:
        return pd.DataFrame()
    data = pg.rename(
        columns={
            "YEAR": "year",
            "DOY": "doy",
            "DAP": "dap",
            "CWAD": "cwad",
            "GWAD": "gwad",
            "WSPD": "wspd",
            "NSTD": "nstd",
        }
    )
    rain_frames = []
    for year in YEARS:
        weather = OUT_DIR / "pdi_yearly_runs" / str(year) / "pdi_tmp_snapshot" / "Weather.OUT"
        if weather.exists():
            w = parse_weather_out(weather)
            rain_frames.append(w)
    rain = pd.concat(rain_frames, ignore_index=True) if rain_frames else pd.DataFrame()
    if not rain.empty:
        data = data.merge(rain[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    data = data[data["year"].isin(YEARS)].sort_values(["year", "dap"]).reset_index(drop=True)
    data.to_csv(OUT_DIR / "hla_ic1_daily_rain_stress_growth_for_plots.csv", index=False, encoding="utf-8-sig")
    return data


def plot_rain_stress(data: pd.DataFrame) -> Path:
    years = sorted(data["year"].dropna().astype(int).unique().tolist())
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel())
    colors = {"rain": "#C5CAD3", "wspd": "#D62728", "nstd": "#2CA02C"}
    for ax, year in zip(axes, years):
        sub = data[data["year"].eq(year)].sort_values("dap")
        ax2 = ax.twinx()
        ax2.bar(sub["dap"], sub.get("rain", pd.Series([0] * len(sub))), width=1.0, color=colors["rain"], alpha=0.45, edgecolor="#7A828F", linewidth=0.25)
        ax.plot(sub["dap"], sub["wspd"], color=colors["wspd"], linewidth=2.0, label="WSPD")
        ax.plot(sub["dap"], sub["nstd"], color=colors["nstd"], linewidth=2.0, linestyle=(0, (4, 2)), label="NSTD")
        ax.set_title(str(year), loc="left", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        rain_max = pd.to_numeric(sub.get("rain", pd.Series([0])), errors="coerce").max()
        ax2.set_ylim(0, max(20, rain_max * 1.15 if pd.notna(rain_max) else 20))
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax2.tick_params(axis="y", labelsize=8)
        if ax in axes[-ncols:]:
            ax.set_xlabel("DAP", fontsize=9)
    for ax in axes[len(years) :]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=colors["rain"], alpha=0.45, edgecolor="#7A828F", label="Rainfall (mm/day, right axis)"),
            Line2D([0], [0], color=colors["wspd"], lw=2.0, label="WSPD"),
            Line2D([0], [0], color=colors["nstd"], lw=2.0, linestyle=(0, (4, 2)), label="NSTD"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Hailun IC=1 PDI yearly: rainfall and DSSAT stress indices", y=1.018, fontsize=14)
    fig.text(0.01, 0.985, "Left axis: WSPD/NSTD. Right axis: rainfall. Lower WSPD/NSTD means stronger stress.", fontsize=9, color="#6F768A")
    out = OUT_DIR / "hla_ic1_rainfall_wspd_nstd_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_growth(data: pd.DataFrame) -> Path:
    years = sorted(data["year"].dropna().astype(int).unique().tolist())
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel())
    for ax, year in zip(axes, years):
        sub = data[data["year"].eq(year)].sort_values("dap")
        ax.plot(sub["dap"], sub["cwad"], color="#2E4780", linewidth=2.1, label="CWAD")
        ax.plot(sub["dap"], sub["gwad"], color="#804126", linewidth=2.1, linestyle=(0, (4, 2)), label="GWAD")
        final = pd.to_numeric(sub["gwad"], errors="coerce").dropna()
        title = f"{year}"
        if not final.empty:
            title += f" | final GWAD={final.iloc[-1]:.0f}"
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if ax in axes[-ncols:]:
            ax.set_xlabel("DAP", fontsize=9)
    for ax in axes[len(years) :]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="#2E4780", lw=2.1, label="CWAD biomass"),
            Line2D([0], [0], color="#804126", lw=2.1, linestyle=(0, (4, 2)), label="GWAD grain"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Hailun IC=1 PDI yearly: biomass and grain trajectories", y=1.018, fontsize=14)
    out = OUT_DIR / "hla_ic1_cwad_gwad_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    status = run_years()
    pg = collect_outputs()
    data = make_plot_data(pg)
    outputs = []
    if not data.empty:
        outputs = [str(plot_rain_stress(data)), str(plot_growth(data))]
    summary = {}
    if not data.empty:
        summary = {
            "years": sorted(data["year"].dropna().astype(int).unique().tolist()),
            "min_wspd": float(pd.to_numeric(data["wspd"], errors="coerce").min()),
            "max_wspd": float(pd.to_numeric(data["wspd"], errors="coerce").max()),
            "min_nstd": float(pd.to_numeric(data["nstd"], errors="coerce").min()),
            "max_nstd": float(pd.to_numeric(data["nstd"], errors="coerce").max()),
        }
    print(json.dumps({"out_dir": str(OUT_DIR), "status": status.to_dict("records"), "rows": int(len(data)), "summary": summary, "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child-year":
        child_run_year(int(sys.argv[2]))
    else:
        main()

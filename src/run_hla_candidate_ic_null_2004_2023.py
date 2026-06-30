"""Run and compare HLA candidate initial condition null simulations.

Candidate IC:
    SH2O = SLLL + 0.55 * (SDUL - SLLL)
    SNH4/SNO3 = 0.25 * HLA 2004 record-based profile

This script performs forward DSSAT/PDI simulations only. It does not train PPO.
It then compares candidate null results with:
    - IC=0 null, Windows DSSAT 4.8.0 CNHL0407 output
    - original IC=1 null, PDI/gym DSSAT 4.8.0 output
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


YEARS = list(range(2004, 2024))
WATER_FRACTION = 0.55
N_SCALE = 0.25

IC0_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0407_DSSAT480_IC0_null_2004_2023"
ORIG_IC1_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_ic1_yearly_diagnostics_2004_2023" / "null"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "candidate_ic055_n025_null_2004_2023"
SOIL_FILE = PROJECT_ROOT / "my_data" / "HL.SOL"

BASE_N_PROFILE = {
    20: (22.7, 10.1),
    40: (10.8, 11.3),
    60: (15.1, 12.5),
    90: (8.2, 8.1),
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "IC0_null": "#2E4780",
    "Original_IC1_null": "#CC6F47",
    "Candidate_IC055_N025_null": "#386411",
}

LINESTYLES = {
    "IC0_null": "-",
    "Original_IC1_null": "--",
    "Candidate_IC055_N025_null": "-.",
}

LABELS = {
    "IC0_null": "IC=0 null",
    "Original_IC1_null": "Original IC=1 null",
    "Candidate_IC055_N025_null": "Candidate IC 0.55 + 0.25N null",
}


def parse_dssat_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    columns: list[str] | None = None
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                columns = stripped.split()
                if columns and columns[0] == "@":
                    columns = columns[1:]
                elif columns and columns[0].startswith("@"):
                    columns[0] = columns[0].lstrip("@")
                continue
            if columns is None:
                continue
            if stripped.startswith("*") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < len(columns):
                parts = parts + [""] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=columns or [])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def parse_hl_soil_water_limits() -> dict[int, tuple[float, float]]:
    lines = SOIL_FILE.read_text(encoding="latin-1", errors="ignore").splitlines()
    in_hl = False
    header: list[str] | None = None
    limits: dict[int, tuple[float, float]] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*HL99001200"):
            in_hl = True
            header = None
            continue
        if in_hl and stripped.startswith("*") and not stripped.startswith("*HL99001200"):
            break
        if not in_hl:
            continue
        if stripped.startswith("@") and "SLB" in stripped and "SLLL" in stripped:
            header = stripped.replace("@", "", 1).split()
            continue
        if header and stripped and stripped[0].isdigit():
            parts = stripped.split()
            row = dict(zip(header, parts))
            slb = int(float(row["SLB"]))
            limits[slb] = (float(row["SLLL"]), float(row["SDUL"]))
    if not limits:
        raise RuntimeError(f"Could not parse HLA soil water limits from {SOIL_FILE}")
    return limits


def candidate_profile() -> dict[int, tuple[float, float, float]]:
    water_limits = parse_hl_soil_water_limits()
    profile = {}
    for depth, (base_nh4, base_no3) in BASE_N_PROFILE.items():
        slll, sdul = water_limits[depth]
        sh2o = slll + WATER_FRACTION * (sdul - slll)
        profile[depth] = (round(sh2o, 3), round(base_nh4 * N_SCALE, 2), round(base_no3 * N_SCALE, 2))
    return profile


def update_initial_conditions(text: str) -> str:
    profile = candidate_profile()
    out: list[str] = []
    in_ic_layer_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@C") and "ICBL" in stripped and "SH2O" in stripped:
            out.append(line)
            in_ic_layer_block = True
            continue
        if in_ic_layer_block:
            if stripped.startswith("*") or stripped.startswith("@"):
                in_ic_layer_block = False
                out.append(line)
                continue
            parts = stripped.split()
            if len(parts) >= 5 and parts[0] == "1":
                depth = int(float(parts[1]))
                if depth in profile:
                    sh2o, snh4, sno3 = profile[depth]
                    out.append(f" 1 {depth:5d} {sh2o:5.3f} {snh4:5.2f} {sno3:5.2f}")
                    continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_run(year: int) -> Path:
    run_dir = OUT_DIR / "runs" / str(year)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    base_input = ORIG_IC1_DIR / str(year) / "input"
    mzx_files = sorted(base_input.glob("*.MZX"))
    if not mzx_files:
        raise FileNotFoundError(f"No base MZX in {base_input}")
    text = mzx_files[0].read_text(encoding="latin-1", errors="ignore")
    text = update_initial_conditions(text)
    filex = input_dir / f"CNHL{year % 100:02d}C1.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    aux = []
    for src in [*base_input.glob("*.WTH"), *base_input.glob("*.SOL"), *base_input.glob("*.CUL")]:
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux.append(str(dst))

    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")
    (run_dir / "candidate_ic_metadata.json").write_text(
        json.dumps(
            {
                "year": year,
                "water_fraction": WATER_FRACTION,
                "n_scale": N_SCALE,
                "initial_profile": candidate_profile(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 280) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
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
    pd.DataFrame(rows).to_csv(run_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()


def standardize_plantgro(path: Path, source: str, requested_year: int | None = None) -> pd.DataFrame:
    df = parse_dssat_table(path)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "source": source,
            "year": pd.to_numeric(df.get("YEAR"), errors="coerce"),
            "doy": pd.to_numeric(df.get("DOY"), errors="coerce"),
            "dap": pd.to_numeric(df.get("DAP"), errors="coerce"),
            "wspd": pd.to_numeric(df.get("WSPD"), errors="coerce"),
            "nstd": pd.to_numeric(df.get("NSTD"), errors="coerce"),
            "gwad": pd.to_numeric(df.get("GWAD"), errors="coerce"),
            "cwad": pd.to_numeric(df.get("CWAD"), errors="coerce"),
            "lai": pd.to_numeric(df.get("LAID"), errors="coerce"),
        }
    )
    if requested_year is not None:
        out = out[out["year"] == requested_year].copy()
    out = out.dropna(subset=["year", "dap"])
    out = out.drop_duplicates(subset=["source", "year", "dap"], keep="last")
    return out


def collect_all_daily() -> pd.DataFrame:
    frames = []
    frames.append(standardize_plantgro(IC0_DIR / "PlantGro.OUT", "IC0_null"))
    for year in YEARS:
        orig = ORIG_IC1_DIR / str(year) / "pdi_tmp_snapshot" / "PlantGro.OUT"
        cand = OUT_DIR / "runs" / str(year) / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if orig.exists():
            frames.append(standardize_plantgro(orig, "Original_IC1_null", year))
        if cand.exists():
            frames.append(standardize_plantgro(cand, "Candidate_IC055_N025_null", year))
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    daily.to_csv(OUT_DIR / "hla_ic0_origic1_candidate_null_daily_2004_2023.csv", index=False, encoding="utf-8-sig")
    return daily


def build_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, year), sub in daily.groupby(["source", "year"]):
        sub = sub.sort_values("dap")
        final = sub.iloc[-1]
        rows.append(
            {
                "source": source,
                "year": int(year),
                "final_dap": final["dap"],
                "grain_yield_gwad": final["gwad"],
                "biomass_cwad": final["cwad"],
                "max_wspd": sub["wspd"].max(),
                "mean_wspd": sub["wspd"].mean(),
                "max_nstd": sub["nstd"].max(),
                "mean_nstd": sub["nstd"].mean(),
                "max_lai": sub["lai"].max(),
                "formed_grain": bool(final["gwad"] > 0),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "hla_ic0_origic1_candidate_null_summary_long_2004_2023.csv", index=False, encoding="utf-8-sig")
    wide = summary.pivot_table(
        index="year",
        columns="source",
        values=["grain_yield_gwad", "biomass_cwad", "final_dap", "max_wspd", "mean_wspd", "max_nstd", "mean_nstd"],
        aggfunc="last",
    )
    wide.columns = [f"{metric}_{source}" for metric, source in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT_DIR / "hla_ic0_origic1_candidate_null_summary_wide_2004_2023.csv", index=False, encoding="utf-8-sig")
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"])
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])


def plot_yearly_yield(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=TOKENS["surface"])
    for source in ["IC0_null", "Original_IC1_null", "Candidate_IC055_N025_null"]:
        sub = summary[summary["source"] == source].sort_values("year")
        ax.plot(
            sub["year"],
            sub["grain_yield_gwad"],
            color=COLORS[source],
            linestyle=LINESTYLES[source],
            marker="o",
            linewidth=1.8,
            label=LABELS[source],
        )
    for year in [2004, 2009, 2012]:
        ax.axvline(year, color="#7A828F", linestyle=":", linewidth=1)
    ax.set_title("HLA null grain yield: IC=0 vs original IC=1 vs candidate IC", loc="left", color=TOKENS["ink"])
    ax.set_xlabel("Year")
    ax.set_ylabel("GWAD / grain yield (kg/ha)")
    years = sorted(summary["year"].unique())
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mticker.FixedLocator(years))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hla_ic0_origic1_candidate_null_yield_2004_2023.png", dpi=220)
    plt.close(fig)


def plot_focus_years(daily: pd.DataFrame) -> None:
    for year in [2004, 2009, 2012]:
        sub = daily[daily["year"] == year].copy()
        fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), sharex=True, facecolor=TOKENS["surface"])
        metrics = [
            ("wspd", "Water stress WSPD", "Stress index"),
            ("nstd", "Nitrogen stress NSTD", "Stress index"),
            ("gwad", "Grain weight GWAD", "kg/ha"),
        ]
        for ax, (metric, title, ylabel) in zip(axes, metrics):
            for source in ["IC0_null", "Original_IC1_null", "Candidate_IC055_N025_null"]:
                sdf = sub[sub["source"] == source].sort_values("dap")
                if sdf.empty:
                    continue
                ax.plot(
                    sdf["dap"],
                    sdf[metric],
                    color=COLORS[source],
                    linestyle=LINESTYLES[source],
                    linewidth=1.8,
                    label=LABELS[source],
                )
            ax.set_title(title, color=TOKENS["ink"], loc="left", fontsize=11)
            ax.set_ylabel(ylabel)
            style_axes(ax)
        axes[-1].set_xlabel("DAP")
        axes[0].legend(frameon=False, loc="upper left", ncol=3)
        fig.suptitle(f"HLA {year} null daily comparison", x=0.01, ha="left", color=TOKENS["ink"])
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_DIR / f"hla_{year}_ic0_origic1_candidate_null_daily.png", dpi=220)
        plt.close(fig)


def parent_run(skip_existing: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for year in YEARS:
        run_dir = prepare_run(year)
        daily_path = run_dir / "gym_post_state_daily.csv"
        snapshot_path = run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if skip_existing and daily_path.exists() and snapshot_path.exists():
            statuses.append({"year": year, "returncode": 0, "skipped_existing": True, "timed_out": False, "stderr_tail": ""})
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-run-dir", str(run_dir)]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=100, capture_output=True, text=True)
            statuses.append(
                {
                    "year": year,
                    "returncode": proc.returncode,
                    "skipped_existing": False,
                    "timed_out": False,
                    "stderr_tail": proc.stderr[-1000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            statuses.append(
                {
                    "year": year,
                    "returncode": None,
                    "skipped_existing": False,
                    "timed_out": True,
                    "stderr_tail": str(exc)[-1000:],
                }
            )
    pd.DataFrame(statuses).to_csv(OUT_DIR / "hla_candidate_ic_null_run_status.csv", index=False, encoding="utf-8-sig")
    daily = collect_all_daily()
    summary = build_summary(daily)
    plot_yearly_yield(summary)
    plot_focus_years(daily)
    print(summary.pivot(index="year", columns="source", values="grain_yield_gwad").to_string())
    print(f"Wrote outputs to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run-dir", type=Path)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.child_run_dir:
        child_run(args.child_run_dir)
    else:
        parent_run(skip_existing=not args.rerun)


if __name__ == "__main__":
    main()

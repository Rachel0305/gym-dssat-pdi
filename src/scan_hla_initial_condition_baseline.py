"""Small HLA initial-condition baseline scan.

Runs only null forward simulations for HLA 2004 and 2012.
No PPO training is performed.

The scan varies:
- initial water as SLLL + water_fraction * (SDUL - SLLL)
- initial mineral N as scale * record-based HLA 2004 SNH4/SNO3 profile
"""

from __future__ import annotations

import argparse
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


BASE_INPUT_ROOT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_ic1_yearly_diagnostics_2004_2023"
    / "null"
)
SOIL_FILE = PROJECT_ROOT / "my_data" / "HL.SOL"
OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "initial_condition_baseline_scan_010_12"
)

YEARS = [2004, 2009, 2012]
WATER_FRACTIONS = [0.35, 0.45, 0.55, 0.65]
N_SCALES = [0.25, 0.50, 0.75, 1.00]

BASE_N_PROFILE = {
    20: (22.7, 10.1),
    40: (10.8, 11.3),
    60: (15.1, 12.5),
    90: (8.2, 8.1),
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


def initial_profile(water_fraction: float, n_scale: float) -> dict[int, tuple[float, float, float]]:
    water_limits = parse_hl_soil_water_limits()
    profile = {}
    for depth, (base_nh4, base_no3) in BASE_N_PROFILE.items():
        slll, sdul = water_limits[depth]
        sh2o = slll + water_fraction * (sdul - slll)
        profile[depth] = (round(sh2o, 3), round(base_nh4 * n_scale, 2), round(base_no3 * n_scale, 2))
    return profile


def update_initial_conditions(text: str, water_fraction: float, n_scale: float) -> str:
    profile = initial_profile(water_fraction, n_scale)
    lines = text.splitlines()
    out: list[str] = []
    in_ic_layer_block = False
    for line in lines:
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


def prepare_run(year: int, water_fraction: float, n_scale: float) -> Path:
    tag = f"Y{year}_W{water_fraction:.2f}_N{n_scale:.2f}".replace(".", "p")
    run_dir = OUT_DIR / "runs" / tag
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    base_input = BASE_INPUT_ROOT / str(year) / "input"
    mzx_files = sorted(base_input.glob("*.MZX"))
    if not mzx_files:
        raise FileNotFoundError(f"No base MZX in {base_input}")
    base_mzx = mzx_files[0]
    text = base_mzx.read_text(encoding="latin-1", errors="ignore")
    text = update_initial_conditions(text, water_fraction, n_scale)
    filex = input_dir / f"CNHL{year % 100:02d}S1.MZX"
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
    (run_dir / "scan_metadata.json").write_text(
        json.dumps(
            {
                "year": year,
                "water_fraction": water_fraction,
                "n_scale": n_scale,
                "initial_profile": initial_profile(water_fraction, n_scale),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 260) -> None:
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


def collect_results() -> pd.DataFrame:
    rows = []
    for meta_path in sorted((OUT_DIR / "runs").glob("*/scan_metadata.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        daily_path = run_dir / "gym_post_state_daily.csv"
        if not daily_path.exists():
            rows.append({**meta, "status": "missing_daily"})
            continue
        daily = pd.read_csv(daily_path)
        final = daily.sort_values("dap").iloc[-1]
        rows.append(
            {
                "status": "ok",
                "run": run_dir.name,
                "year": meta["year"],
                "water_fraction": meta["water_fraction"],
                "n_scale": meta["n_scale"],
                "final_dap": final["dap"],
                "grain_yield_gwad": final["grnwt"],
                "biomass_cwad": final["topwt"],
                "max_wspd": daily["swfac"].max(),
                "mean_wspd": daily["swfac"].mean(),
                "max_nstd": daily["nstres"].max(),
                "mean_nstd": daily["nstres"].mean(),
                "max_lai": daily["xlai"].max(),
                "formed_grain": final["grnwt"] > 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "hla_ic_baseline_scan_summary.csv", index=False, encoding="utf-8-sig")
    return out


def plot_heatmaps(summary: pd.DataFrame) -> None:
    years = sorted(summary["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(5.8 * len(years), 4.8), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        sub = summary[summary["year"] == year]
        pivot = sub.pivot(index="n_scale", columns="water_fraction", values="grain_yield_gwad").sort_index(ascending=False)
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrBr")
        ax.set_title(f"HLA {year} null grain yield")
        ax.set_xlabel("Initial water fraction")
        ax.set_ylabel("Initial mineral N scale")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85, label="GWAD kg/ha")
    fig.savefig(OUT_DIR / "hla_ic_baseline_scan_yield_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(years), figsize=(5.8 * len(years), 4.8), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        sub = summary[summary["year"] == year]
        pivot = sub.pivot(index="n_scale", columns="water_fraction", values="biomass_cwad").sort_index(ascending=False)
        im = ax.imshow(pivot.values, aspect="auto", cmap="Greens")
        ax.set_title(f"HLA {year} null biomass")
        ax.set_xlabel("Initial water fraction")
        ax.set_ylabel("Initial mineral N scale")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85, label="CWAD kg/ha")
    fig.savefig(OUT_DIR / "hla_ic_baseline_scan_biomass_heatmap.png", dpi=220)
    plt.close(fig)


def parent_run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for year in YEARS:
        for water_fraction in WATER_FRACTIONS:
            for n_scale in N_SCALES:
                run_dir = prepare_run(year, water_fraction, n_scale)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child-run-dir", str(run_dir)]
                try:
                    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=90, capture_output=True, text=True)
                    statuses.append(
                        {
                            "run": run_dir.name,
                            "year": year,
                            "water_fraction": water_fraction,
                            "n_scale": n_scale,
                            "returncode": proc.returncode,
                            "timed_out": False,
                            "stderr_tail": proc.stderr[-1000:],
                        }
                    )
                except subprocess.TimeoutExpired as exc:
                    statuses.append(
                        {
                            "run": run_dir.name,
                            "year": year,
                            "water_fraction": water_fraction,
                            "n_scale": n_scale,
                            "returncode": None,
                            "timed_out": True,
                            "stderr_tail": str(exc)[-1000:],
                        }
                    )
    pd.DataFrame(statuses).to_csv(OUT_DIR / "hla_ic_baseline_scan_run_status.csv", index=False, encoding="utf-8-sig")
    summary = collect_results()
    if not summary.empty:
        plot_heatmaps(summary[summary["status"] == "ok"].copy())
        print(summary.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run-dir", type=Path)
    args = parser.parse_args()
    if args.child_run_dir:
        child_run(args.child_run_dir)
    else:
        parent_run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
import shutil
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


WINDOWS_RUN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "UFGA_Windows480_vs_gym_pdi"
OUT_DIR = WINDOWS_RUN_DIR / "analysis_ufga_windows480_vs_pdi_with_rain"


def plain(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def parse_dssat_table(path: Path) -> pd.DataFrame:
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
                    row: dict[str, Any] = dict(zip(header, parts[: len(header)]))
                    row["RUNNO"] = current_run
                    row["TNAM"] = current_treatment
                    rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    if {"RUNNO", "YEAR", "DOY", "DAP"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["RUNNO", "YEAR", "DOY", "DAP"], keep="last").reset_index(drop=True)
    elif {"RUNNO", "YEAR", "DOY", "DAS"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["RUNNO", "YEAR", "DOY", "DAS"], keep="last").reset_index(drop=True)
    return df


def parse_summary(path: Path) -> pd.DataFrame:
    return parse_dssat_table(path)


def make_windows_daily() -> pd.DataFrame:
    plantgro = parse_dssat_table(WINDOWS_RUN_DIR / "PlantGro.OUT")
    weather = parse_dssat_table(WINDOWS_RUN_DIR / "Weather.OUT")
    keep = [
        "RUNNO",
        "TNAM",
        "YEAR",
        "DOY",
        "DAS",
        "DAP",
        "CWAD",
        "GWAD",
        "LAID",
        "WSPD",
        "WSGD",
        "NSTD",
    ]
    daily = plantgro[[col for col in keep if col in plantgro.columns]].copy()
    daily = daily.rename(
        columns={
            "RUNNO": "run",
            "TNAM": "treatment",
            "YEAR": "year",
            "DOY": "doy",
            "DAS": "das",
            "DAP": "dap",
            "CWAD": "cwad",
            "GWAD": "gwad",
            "LAID": "laid",
            "WSPD": "wspd",
            "WSGD": "wsgd",
            "NSTD": "nstd",
        }
    )
    if not weather.empty and {"RUNNO", "YEAR", "DOY", "PRED"}.issubset(weather.columns):
        rain = weather[["RUNNO", "YEAR", "DOY", "PRED"]].rename(
            columns={"RUNNO": "run", "YEAR": "year", "DOY": "doy", "PRED": "rain"}
        )
        daily = daily.merge(rain, on=["run", "year", "doy"], how="left")
    return daily


def run_pdi_official_example(experiment_number: int, max_steps: int = 700) -> tuple[pd.DataFrame, Path]:
    import gym
    from gym_dssat_pdi.envs.utils import utils as pdi_utils
    from sb3_wrapper import GymDssatWrapper

    pdi_dir = OUT_DIR / "pdi_official_runtime" / f"experiment_{experiment_number}"
    pdi_dir.mkdir(parents=True, exist_ok=True)
    filex = pdi_dir / "UFGA8201.MZX"
    shutil.copyfile(WINDOWS_RUN_DIR / "UFGA8201.MZX", filex)

    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_args = {
        "log_saving_path": str(log_dir / f"ufga_official_pdi_experiment_{experiment_number}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": int(experiment_number),
        "auxiliary_file_paths": [],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (OUT_DIR / f"ufga_pdi_env_args_experiment_{experiment_number}.json").write_text(
        json.dumps(env_args, indent=2),
        encoding="utf-8",
    )

    post_rows: list[dict[str, Any]] = []
    original_post_treat = pdi_utils._post_treat_state

    def spy_post_treat(state, cultivar="maize"):
        post = original_post_treat(state, cultivar)
        row = {"call_index": len(post_rows), "cultivar": cultivar}
        for key in ["yrdoy", "dap", "topwt", "grnwt", "xlai", "swfac", "nstres", "trnu"]:
            row[f"post_{key}"] = plain((post or {}).get(key))
        post_rows.append(row)
        return post

    pdi_utils._post_treat_state = spy_post_treat
    env = None
    tmp_snapshot = OUT_DIR / f"pdi_tmp_snapshot_experiment_{experiment_number}"
    if tmp_snapshot.exists():
        shutil.rmtree(tmp_snapshot)
    records: list[dict[str, Any]] = []
    try:
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            records.append(
                {
                    "experiment_number": int(experiment_number),
                    "step_index": step,
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
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                }
            )
            step += 1
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, tmp_snapshot, dirs_exist_ok=True)
    finally:
        pdi_utils._post_treat_state = original_post_treat
        if env is not None:
            env.close()
    pdi_post = pd.DataFrame(records)
    pdi_post.to_csv(OUT_DIR / f"pdi_gym_post_state_daily_experiment_{experiment_number}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(post_rows).to_csv(OUT_DIR / f"pdi_gym_post_treat_calls_experiment_{experiment_number}.csv", index=False, encoding="utf-8-sig")
    return pdi_post, tmp_snapshot


def plot_windows_stress(daily: pd.DataFrame, out_path: Path) -> None:
    runs = sorted(pd.to_numeric(daily["run"], errors="coerce").dropna().astype(int).unique().tolist())
    colors = ["#2E4780", "#804126", "#386411", "#8A3A6F", "#736422", "#464C55"]
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    rain_source = daily[daily["run"].eq(runs[0])].sort_values("dap") if runs else daily.iloc[0:0]
    axes[0].bar(
        rain_source["dap"],
        rain_source["rain"],
        color="#A3BEFA",
        edgecolor="#2E4780",
        linewidth=0.35,
        width=1.0,
        label=f"Rainfall, run {runs[0] if runs else ''}",
    )
    axes[0].set_title("Daily rainfall", loc="left", fontsize=12, color="#1F2430")
    axes[0].set_ylabel("Rainfall (mm/day)")
    axes[0].grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    specs = [
        ("Water stress index WSPD", "wspd", "WSPD (0 = strongest stress in DSSAT output convention)"),
        ("Nitrogen stress index NSTD", "nstd", "NSTD (0 = strongest stress in DSSAT output convention)"),
        ("Grain weight GWAD", "gwad", "GWAD (kg/ha)"),
    ]
    for ax, (title, col, ylabel) in zip(axes[1:], specs):
        for i, run in enumerate(runs):
            sub = daily[daily["run"].eq(run)].sort_values("dap")
            label = f"Run {run}: {str(sub['treatment'].dropna().iloc[0])[:28]}" if not sub.empty else f"Run {run}"
            ax.plot(
                sub["dap"],
                sub[col],
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=2.2,
                label=label,
            )
        ax.set_title(title, loc="left", fontsize=12, color="#1F2430")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("DAP")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("UFGA official example: rainfall, stress indices, and grain trajectories", fontsize=14, y=0.995)
    fig.text(
        0.01,
        0.965,
        "Source: Windows DSSAT 4.8.0 PlantGro.OUT and Weather.OUT. Rainfall is shown once because all treatments use the same weather year.",
        fontsize=9,
        color="#6F768A",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pdi_raw_vs_windows(windows_daily: pd.DataFrame, pdi_pg: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    pdi = pdi_pg.rename(
        columns={
            "RUNNO": "run",
            "TNAM": "treatment",
            "YEAR": "year",
            "DOY": "doy",
            "DAP": "dap",
            "CWAD": "pdi_cwad",
            "GWAD": "pdi_gwad",
            "WSPD": "pdi_wspd",
            "NSTD": "pdi_nstd",
        }
    )
    pdi = pdi[[col for col in ["run", "year", "doy", "dap", "pdi_cwad", "pdi_gwad", "pdi_wspd", "pdi_nstd"] if col in pdi.columns]]
    win = windows_daily.rename(
        columns={
            "cwad": "windows_cwad",
            "gwad": "windows_gwad",
            "wspd": "windows_wspd",
            "nstd": "windows_nstd",
        }
    )
    merged = win.merge(pdi, on=["run", "year", "doy", "dap"], how="outer", indicator=True)
    for metric in ["cwad", "gwad", "wspd", "nstd"]:
        w = f"windows_{metric}"
        p = f"pdi_{metric}"
        if w in merged.columns and p in merged.columns:
            merged[f"diff_{metric}_windows_minus_pdi"] = merged[w] - merged[p]
    merged.to_csv(OUT_DIR / "ufga_daily_windows480_vs_pdi_raw.csv", index=False, encoding="utf-8-sig")

    if merged.empty:
        return merged
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    specs = [
        ("Water stress WSPD", "windows_wspd", "pdi_wspd", "index"),
        ("Nitrogen stress NSTD", "windows_nstd", "pdi_nstd", "index"),
        ("Biomass CWAD", "windows_cwad", "pdi_cwad", "kg/ha"),
        ("Grain GWAD", "windows_gwad", "pdi_gwad", "kg/ha"),
    ]
    # Plot run 1 first to avoid unreadable six-run overload.
    sub = merged[merged["run"].eq(1)].sort_values("dap")
    for ax, (title, wcol, pcol, ylabel) in zip(axes, specs):
        ax.plot(sub["dap"], sub[wcol], color="#804126", linewidth=3.0, label="Windows DSSAT 4.8.0")
        ax.plot(sub["dap"], sub[pcol], color="#2E4780", linewidth=2.0, linestyle=(0, (4, 2)), marker="o", markersize=2.5, markevery=max(1, len(sub) // 18), label="PDI DSSAT 4.8.0 raw")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("DAP")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.22), ncol=2, frameon=False)
    fig.suptitle("UFGA official example run 1: Windows 4.8.0 vs PDI 4.8.0 raw outputs", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows_daily = make_windows_daily()
    windows_daily.to_csv(OUT_DIR / "ufga_windows480_daily_stress_growth.csv", index=False, encoding="utf-8-sig")
    summary = (
        windows_daily.groupby(["run", "treatment"], dropna=False)
        .agg(
            n_days=("dap", "count"),
            final_dap=("dap", "max"),
            final_gwad=("gwad", "last"),
            min_wspd=("wspd", "min"),
            max_wspd=("wspd", "max"),
            mean_wspd=("wspd", "mean"),
            days_wspd_eq0=("wspd", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
            min_nstd=("nstd", "min"),
            max_nstd=("nstd", "max"),
            mean_nstd=("nstd", "mean"),
            days_nstd_eq0=("nstd", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "ufga_windows480_stress_summary_by_treatment.csv", index=False, encoding="utf-8-sig")
    plot_windows_stress(windows_daily, OUT_DIR / "ufga_windows480_stress_growth_by_treatment.png")

    pdi_status: dict[str, Any] = {"attempted": True}
    try:
        pdi_frames: list[pd.DataFrame] = []
        snapshots: dict[int, str] = {}
        experiments = sorted(pd.to_numeric(windows_daily["run"], errors="coerce").dropna().astype(int).unique().tolist())
        for experiment_number in experiments:
            _, tmp_snapshot = run_pdi_official_example(experiment_number=experiment_number)
            snapshots[int(experiment_number)] = str(tmp_snapshot)
            if (tmp_snapshot / "PlantGro.OUT").exists():
                pdi_one = parse_dssat_table(tmp_snapshot / "PlantGro.OUT")
                # PDI emits one selected treatment at a time. Reset RUNNO to the
                # requested experiment number so it can be aligned with the
                # multi-treatment Windows output.
                pdi_one["RUNNO"] = int(experiment_number)
                pdi_frames.append(pdi_one)
        pdi_status["tmp_snapshots"] = snapshots
        if pdi_frames:
            pdi_pg = pd.concat(pdi_frames, ignore_index=True)
            pdi_pg.to_csv(OUT_DIR / "ufga_pdi_raw_PlantGro_daily_all_experiments.csv", index=False, encoding="utf-8-sig")
            merged = plot_pdi_raw_vs_windows(windows_daily, pdi_pg, OUT_DIR / "ufga_run1_windows480_vs_pdi_raw_high_contrast.png")
            pdi_status["pdi_raw_rows"] = int(len(pdi_pg))
            pdi_status["merged_rows"] = int(len(merged))
            for metric in ["cwad", "gwad", "wspd", "nstd"]:
                col = f"diff_{metric}_windows_minus_pdi"
                if col in merged:
                    pdi_status[f"max_abs_{col}"] = float(pd.to_numeric(merged[col], errors="coerce").abs().max())
            by_run = []
            for run, group in merged.groupby("run", dropna=False):
                row: dict[str, Any] = {"run": int(run) if pd.notna(run) else run, "rows": int(len(group))}
                for metric in ["cwad", "gwad", "wspd", "nstd"]:
                    col = f"diff_{metric}_windows_minus_pdi"
                    if col in group:
                        row[f"max_abs_{metric}_diff"] = float(pd.to_numeric(group[col], errors="coerce").abs().max())
                by_run.append(row)
            pd.DataFrame(by_run).to_csv(OUT_DIR / "ufga_windows480_vs_pdi_raw_maxdiff_by_experiment.csv", index=False, encoding="utf-8-sig")
            pdi_status["maxdiff_by_experiment_csv"] = str(OUT_DIR / "ufga_windows480_vs_pdi_raw_maxdiff_by_experiment.csv")
        else:
            pdi_status["error"] = "No PDI PlantGro.OUT files found for any experiment"
    except Exception as exc:
        pdi_status["error"] = repr(exc)

    (OUT_DIR / "ufga_pdi_attempt_status.json").write_text(json.dumps(pdi_status, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# UFGA official example Windows DSSAT 4.8.0 and PDI/gym diagnostic",
        "",
        "Purpose: check whether DSSAT official example stress-index trajectories also show near-zero water/nitrogen stress values, and whether PDI DSSAT raw output matches Windows DSSAT raw output.",
        "",
        "Outputs:",
        "- `ufga_windows480_daily_stress_growth.csv`: parsed Windows DSSAT daily PlantGro stress/growth values.",
        "- `ufga_windows480_stress_summary_by_treatment.csv`: treatment-level stress summary.",
        "- `ufga_windows480_stress_growth_by_treatment.png`: high-contrast treatment stress/growth plot.",
        "- `ufga_pdi_attempt_status.json`: whether the PDI/gym official-example run succeeded.",
        "- If PDI succeeds: `ufga_daily_windows480_vs_pdi_raw.csv`, `ufga_windows480_vs_pdi_raw_maxdiff_by_experiment.csv`, and `ufga_run1_windows480_vs_pdi_raw_high_contrast.png`.",
    ]
    (OUT_DIR / "README_UFGA_official_example_diagnostic.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "windows_rows": int(len(windows_daily)), **pdi_status}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

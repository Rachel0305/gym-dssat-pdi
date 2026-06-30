from __future__ import annotations

import argparse
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


SOURCE_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_ic1_yearly_diagnostics_2004_2023"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_new_cultivar_candidate_year_screening"
NEW_CUL = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "cultivar_calibration_HLA2004_480"
    / "input_corrected_package"
    / "MZCER048.CUL"
)

YEARS = [2007, 2010, 2015]
SCENARIOS = ["null", "auto_irrig"]


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
    elif {"YEAR", "DOY", "DAP"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["YEAR", "DOY", "DAP"], keep="last").reset_index(drop=True)
    return df


def parse_mgmt_events(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(rows)
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9 or not parts[0].isdigit() or not parts[3].isdigit():
                continue
            try:
                run = int(parts[0])
                day = int(parts[2].rstrip(","))
                year = int(parts[3])
                doy = int(parts[4])
                das = int(parts[5])
                dap = int(parts[6])
            except ValueError:
                continue
            op_tokens = parts[8:]
            if op_tokens and op_tokens[0].isdigit():
                op_tokens = op_tokens[1:]
            operation = " ".join(op_tokens)
            quantity = 0.0
            unit = ""
            qmatch = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mm|kg/ha|kg|%)", operation)
            if qmatch:
                quantity = float(qmatch.group(1))
                unit = qmatch.group(2)
            rows.append(
                {
                    "run": run,
                    "date_label": f"{parts[1]} {day}, {year}",
                    "year": year,
                    "doy": doy,
                    "das": das,
                    "dap": dap,
                    "operation": operation,
                    "quantity": quantity,
                    "unit": unit,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["year", "doy", "dap", "operation", "quantity", "unit"], keep="last").reset_index(drop=True)
    return df


def source_input_dir(scenario: str, year: int) -> Path:
    d = SOURCE_ROOT / scenario / str(year) / "input"
    if not d.exists():
        raise FileNotFoundError(f"Missing source input dir: {d}")
    return d


def prepare_case(scenario: str, year: int) -> Path:
    src_dir = source_input_dir(scenario, year)
    case_dir = OUT_DIR / scenario / str(year)
    input_dir = case_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)

    if not NEW_CUL.exists():
        raise FileNotFoundError(str(NEW_CUL))
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")

    mzx_files = sorted(input_dir.glob("*.MZX"))
    if len(mzx_files) != 1:
        raise RuntimeError(f"Expected one MZX in {input_dir}, found {len(mzx_files)}")
    filex = mzx_files[0]

    aux_paths = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(case_dir / f"{scenario}_{year}_pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux_paths,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def child_run(scenario: str, year: int, max_steps: int = 330) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = OUT_DIR / scenario / str(year)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
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
            rows.append(
                {
                    "scenario": scenario,
                    "requested_year": year,
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
            step += 1
    finally:
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(case_dir / f"{scenario}_{year}_gym_post_state_daily.csv", index=False, encoding="utf-8-sig")


def run_cases(timeout: int) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for scenario in SCENARIOS:
        for year in YEARS:
            case_dir = prepare_case(scenario, year)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", scenario, str(year)]
            try:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout, capture_output=True, text=True)
                statuses.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "returncode": proc.returncode,
                        "timed_out": False,
                        "stdout_tail": proc.stdout[-1000:],
                        "stderr_tail": proc.stderr[-1000:],
                        "case_dir": str(case_dir),
                    }
                )
            except subprocess.TimeoutExpired as exc:
                statuses.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "returncode": None,
                        "timed_out": True,
                        "stdout_tail": "",
                        "stderr_tail": str(exc)[-1000:],
                        "case_dir": str(case_dir),
                    }
                )
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "hla_new_cultivar_candidate_run_status.csv", index=False, encoding="utf-8-sig")
    return status


def daily_from_case(scenario: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = OUT_DIR / scenario / str(year) / "pdi_tmp_snapshot"
    plant_path = raw_dir / "PlantGro.OUT"
    if not plant_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    plant = parse_table_out(plant_path)
    weather = parse_table_out(raw_dir / "Weather.OUT") if (raw_dir / "Weather.OUT").exists() else pd.DataFrame()
    events = parse_mgmt_events(raw_dir / "MgmtEvent.OUT")
    plant = plant.rename(columns={"YEAR": "year", "DOY": "doy", "DAP": "dap", "WSPD": "wspd", "NSTD": "nstd", "CWAD": "cwad", "GWAD": "gwad"})
    weather = weather.rename(columns={"YEAR": "year", "DOY": "doy", "PRED": "rain"})
    daily = plant[[c for c in ["year", "doy", "dap", "wspd", "nstd", "cwad", "gwad"] if c in plant.columns]].copy()
    if {"year", "doy", "rain"}.issubset(weather.columns):
        daily = daily.merge(weather[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    else:
        daily["rain"] = 0.0
    daily["scenario"] = scenario
    daily["requested_year"] = year
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if not events.empty:
        for _, ev in events.iterrows():
            op = str(ev["operation"])
            dap = int(ev["dap"])
            qty = float(ev["quantity"])
            if "Irrigation" in op:
                daily.loc[daily["dap"].eq(dap), "irrigation_mm"] += qty
            if "Fertil" in op:
                daily.loc[daily["dap"].eq(dap), "fertilizer_kg_ha"] += qty
    events["scenario"] = scenario
    events["requested_year"] = year
    return daily, events


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames = []
    event_frames = []
    for scenario in SCENARIOS:
        for year in YEARS:
            daily, events = daily_from_case(scenario, year)
            if not daily.empty:
                daily_frames.append(daily)
            if not events.empty:
                event_frames.append(events)
    data = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    data.to_csv(OUT_DIR / "hla_new_cultivar_candidate_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "hla_new_cultivar_candidate_events.csv", index=False, encoding="utf-8-sig")
    if data.empty:
        return data, events, pd.DataFrame()
    summary = (
        data.sort_values(["scenario", "requested_year", "dap"])
        .groupby(["scenario", "requested_year"], as_index=False)
        .agg(
            days=("dap", "size"),
            final_dap=("dap", "last"),
            final_gwad=("gwad", "last"),
            final_cwad=("cwad", "last"),
            rain_total=("rain", "sum"),
            irrigation_total=("irrigation_mm", "sum"),
            fertilizer_total=("fertilizer_kg_ha", "sum"),
            max_wspd=("wspd", "max"),
            mean_wspd=("wspd", "mean"),
            max_nstd=("nstd", "max"),
            mean_nstd=("nstd", "mean"),
        )
    )
    summary.to_csv(OUT_DIR / "hla_new_cultivar_candidate_summary.csv", index=False, encoding="utf-8-sig")
    return data, events, summary


def plot_year_process(data: pd.DataFrame, year: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
    colors = {"null": "#2E4780", "auto_irrig": "#CC6F47"}
    labels = {"null": "Null", "auto_irrig": "DSSAT auto"}
    for ax, stress_col, title in [
        (axes[0], "wspd", "Water stress WSPD"),
        (axes[1], "nstd", "Nitrogen stress NSTD"),
    ]:
        ax2 = ax.twinx()
        yr = data[data["requested_year"].eq(year)].copy()
        for scenario in SCENARIOS:
            sub = yr[yr["scenario"].eq(scenario)].sort_values("dap")
            if sub.empty:
                continue
            x = pd.to_numeric(sub["dap"], errors="coerce")
            ax.plot(
                x,
                pd.to_numeric(sub[stress_col], errors="coerce"),
                color=colors[scenario],
                linewidth=2.1,
                linestyle="-" if scenario == "null" else (0, (5, 2)),
                label=labels[scenario],
            )
            if scenario == "auto_irrig":
                ax2.bar(x, pd.to_numeric(sub["irrigation_mm"], errors="coerce").fillna(0), width=2.4, color="#A3BEFA", edgecolor="#2E4780", alpha=0.82, label="Auto irrigation")
                ax2.bar(x, pd.to_numeric(sub["fertilizer_kg_ha"], errors="coerce").fillna(0), width=3.0, color="#F0986E", edgecolor="#804126", alpha=0.75, label="Auto fertilizer")
        rain_source = yr[yr["scenario"].eq("null")].sort_values("dap")
        if not rain_source.empty:
            ax2.bar(
                pd.to_numeric(rain_source["dap"], errors="coerce"),
                pd.to_numeric(rain_source["rain"], errors="coerce").fillna(0),
                width=1.0,
                color="#C5CAD3",
                edgecolor="#464C55",
                alpha=0.35,
                label="Rain",
            )
        ax.set_ylabel(title)
        ax.set_ylim(-0.03, 1.03)
        ax2.set_ylabel("Rain / input (mm or kg ha$^{-1}$)")
        max_amt = max(float(yr["rain"].max() if "rain" in yr else 0), float(yr["irrigation_mm"].max()), float(yr["fertilizer_kg_ha"].max()), 10.0)
        ax2.set_ylim(0, max_amt * 1.25)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    axes[-1].set_xlabel("DAP")
    line_handles = [
        plt.Line2D([0], [0], color=colors["null"], linewidth=2.1, linestyle="-", label="Null"),
        plt.Line2D([0], [0], color=colors["auto_irrig"], linewidth=2.1, linestyle=(0, (5, 2)), label="DSSAT auto"),
    ]
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C5CAD3", alpha=0.35, label="Rain"),
        plt.Rectangle((0, 0), 1, 1, color="#A3BEFA", alpha=0.82, label="Auto irrigation"),
        plt.Rectangle((0, 0), 1, 1, color="#F0986E", alpha=0.75, label="Auto fertilizer"),
    ]
    fig.suptitle(f"HLA {year} candidate screening, new HY0006 cultivar", fontsize=13)
    all_handles = line_handles + bar_handles
    fig.legend(all_handles, [h.get_label() for h in all_handles], loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"hla_{year}_candidate_process_rain_auto_stress.png"
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return out


def plot_yield(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    order = ["null", "auto_irrig"]
    labels = {"null": "Null", "auto_irrig": "DSSAT auto"}
    colors = {"null": "#A3BEFA", "auto_irrig": "#F0986E"}
    for ax, col, title in [(axes[0], "final_gwad", "Grain yield GWAD"), (axes[1], "final_cwad", "Biomass CWAD")]:
        pivot = summary.pivot(index="requested_year", columns="scenario", values=col).reindex(columns=order)
        x = np.arange(len(pivot.index))
        width = 0.34
        for i, sc in enumerate(order):
            vals = pivot[sc].to_numpy()
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[sc], edgecolor="#464C55", linewidth=0.7, label=labels[sc])
            for xi, val in zip(x + (i - 0.5) * width, vals):
                if np.isfinite(val):
                    ax.text(xi, val, f"{val:.0f}", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(y)) for y in pivot.index])
        ax.set_ylabel("kg ha$^{-1}$")
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("HLA candidate year screening yield summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "hla_candidate_yield_biomass_summary.png"
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(status: pd.DataFrame, summary: pd.DataFrame, figures: list[Path]) -> Path:
    report = OUT_DIR / "README.md"
    lines = [
        "# HLA 新品种参数候选年份筛选",
        "",
        "## 目的",
        "",
        "在不训练 PPO 的前提下，用 PDI/gym-DSSAT forward check 初筛 HLA 2007、2010、2015 是否适合作为后续水氮优化年份。",
        "",
        "## 口径",
        "",
        "- 运行环境：Docker 内 PDI/gym-DSSAT；",
        "- 品种参数：`DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL`；",
        "- 初始条件/模板：沿用 `hla_ic1_yearly_diagnostics_2004_2023` 中的 IC=1 年度输入；",
        "- 情景：null 与 DSSAT auto；2010/2015 未构造 recorded 管理。",
        "",
        "## 运行状态",
        "",
        dataframe_to_markdown(status),
        "",
    ]
    if not summary.empty:
        lines += [
            "## 汇总结果",
            "",
            dataframe_to_markdown(summary.round(3)),
            "",
            "## 初步判读",
            "",
        ]
        for year in YEARS:
            sub = summary[summary["requested_year"].eq(year)].set_index("scenario")
            if {"null", "auto_irrig"}.issubset(sub.index):
                gain = float(sub.loc["auto_irrig", "final_gwad"] - sub.loc["null", "final_gwad"])
                lines.append(f"- {year}: DSSAT auto 相对 null 的 GWAD 差值为 {gain:.1f} kg/ha。")
    lines += [
        "",
        "## 图件",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(PROJECT_ROOT)}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=2, metavar=("SCENARIO", "YEAR"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing PDI snapshots and redraw/report.")
    args = parser.parse_args()
    if args.child:
        child_run(args.child[0], int(args.child[1]))
        return
    status_path = OUT_DIR / "hla_new_cultivar_candidate_run_status.csv"
    if args.collect_only and status_path.exists():
        status = pd.read_csv(status_path, keep_default_na=False)
    else:
        status = run_cases(timeout=args.timeout)
    data, events, summary = collect()
    figures: list[Path] = []
    if not data.empty:
        figures.extend(plot_year_process(data, year) for year in YEARS)
    if not summary.empty:
        figures.append(plot_yield(summary))
    report = write_report(status, summary, figures)
    manifest = {
        "out_dir": str(OUT_DIR),
        "status_csv": str(OUT_DIR / "hla_new_cultivar_candidate_run_status.csv"),
        "daily_csv": str(OUT_DIR / "hla_new_cultivar_candidate_daily.csv"),
        "events_csv": str(OUT_DIR / "hla_new_cultivar_candidate_events.csv"),
        "summary_csv": str(OUT_DIR / "hla_new_cultivar_candidate_summary.csv"),
        "report": str(report),
        "figures": [str(p) for p in figures],
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

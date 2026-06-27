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


TEMPLATES = {
    "auto_irrig": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0408_DSSAT480_2004" / "CNHL0408.MZX",
    "null": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0409_DSSAT480_2004" / "CNHL0409.MZX",
}
WEATHER_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0405_IC0_null"
SOURCE_AUX = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_ic1_yearly_diagnostics_2004_2023"


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
    elif {"YEAR", "DOY"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["YEAR", "DOY"], keep="last").reset_index(drop=True)
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


def replace_dates(text: str, year: int, scenario: str) -> str:
    yy = f"{year % 100:02d}"
    full = f"{year}"
    # Keep the 2004 IC profile values but move date fields to the target year.
    text = text.replace("CNHL0408", f"CNHL{yy}A1")
    text = text.replace("CNHL0409", f"CNHL{yy}N1")
    text = text.replace("CNHL2004", f"CNHL{full}")
    text = text.replace("CNHL0401", f"CNHL{yy}01")
    # Keep treatment name short to avoid shifting fixed-width factor columns.
    text = text.replace("Sim2004", f"Sim{full}")
    text = re.sub(r"\b04(?=\d{3}\b)", yy, text)
    return text


def find_weather(year: int) -> Path:
    p = WEATHER_DIR / f"CNHL{year % 100:02d}01.WTH"
    if p.exists():
        return p
    matches = list(PROJECT_ROOT.rglob(f"CNHL{year % 100:02d}01.WTH"))
    if matches:
        return matches[0]
    raise FileNotFoundError(p.name)


def prepare_case(scenario: str, year: int) -> Path:
    case_dir = OUT_DIR / scenario / str(year)
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    template = TEMPLATES[scenario]
    if not template.exists():
        raise FileNotFoundError(str(template))
    filex = input_dir / f"CNHL{year % 100:02d}{'A1' if scenario == 'auto_irrig' else 'N1'}.MZX"
    text = template.read_text(encoding="latin1", errors="ignore")
    filex.write_text(replace_dates(text, year, scenario), encoding="latin1")

    aux_paths = []
    for src in [find_weather(year), SOURCE_AUX / "SOIL.SOL", PROJECT_ROOT / "my_data" / "MZCER048.CUL"]:
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux_paths.append(str(dst))

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


def run_cases(years: list[int], scenarios: list[str], timeout: int) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for scenario in scenarios:
        for year in years:
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
                        "stdout_tail": proc.stdout[-800:],
                        "stderr_tail": proc.stderr[-800:],
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
                        "stderr_tail": str(exc)[-800:],
                        "case_dir": str(case_dir),
                    }
                )
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "hla_ic1_yearly_pdi_run_status.csv", index=False, encoding="utf-8-sig")
    return status


def daily_from_case(scenario: str, year: int) -> pd.DataFrame:
    raw_dir = OUT_DIR / scenario / str(year) / "pdi_tmp_snapshot"
    plant_path = raw_dir / "PlantGro.OUT"
    if not plant_path.exists():
        return pd.DataFrame()
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
    return daily


def collect(years: list[int], scenarios: list[str]) -> pd.DataFrame:
    frames = [daily_from_case(sc, yr) for sc in scenarios for yr in years]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    data.to_csv(OUT_DIR / "hla_ic1_yearly_daily_values.csv", index=False, encoding="utf-8-sig")
    if data.empty:
        return data
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
    summary.to_csv(OUT_DIR / "hla_ic1_yearly_summary.csv", index=False, encoding="utf-8-sig")
    return data


def plot_scenario(data: pd.DataFrame, scenario: str) -> Path:
    sub_all = data[data["scenario"].eq(scenario)].copy()
    years = sorted(sub_all["requested_year"].dropna().astype(int).unique().tolist())
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(3.2 * nrows, 4)), sharex=False)
    axes = list(np.ravel(axes))
    for ax, year in zip(axes, years):
        sub = sub_all[sub_all["requested_year"].eq(year)].sort_values("dap")
        ax2 = ax.twinx()
        x = pd.to_numeric(sub["dap"], errors="coerce")
        rain = pd.to_numeric(sub["rain"], errors="coerce").fillna(0.0)
        irrig = pd.to_numeric(sub["irrigation_mm"], errors="coerce").fillna(0.0)
        fert = pd.to_numeric(sub["fertilizer_kg_ha"], errors="coerce").fillna(0.0)
        ax2.bar(x, rain, width=1.0, color="#BFC5D2", edgecolor="#69707D", alpha=0.45, linewidth=0.25)
        ax2.bar(x, irrig, width=2.4, color="#1F77B4", edgecolor="#0B3D70", alpha=0.80, linewidth=0.25)
        ax2.bar(x, fert, width=3.0, color="#FF7F0E", edgecolor="#9A4B00", alpha=0.75, linewidth=0.25)
        ax.plot(x, pd.to_numeric(sub["wspd"], errors="coerce"), color="#D62728", linewidth=2.0, label="WSPD")
        ax.plot(x, pd.to_numeric(sub["nstd"], errors="coerce"), color="#2CA02C", linestyle=(0, (5, 2)), linewidth=2.0, label="NSTD")
        ax.set_title(str(year), loc="left", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        max_amt = max(float(rain.max()), float(irrig.max()), float(fert.max()), 10.0)
        ax2.set_ylim(0, max_amt * 1.2)
        ax.grid(True, axis="y", color="#E4E7EF", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    for ax in axes[len(years) :]:
        ax.axis("off")
    fig.suptitle(f"HLA IC=1 yearly diagnostic: {scenario} (red=WSPD, green=NSTD; bars=rain/irrigation/fertilizer)", fontsize=12)
    handles = [
        plt.Line2D([0], [0], color="#D62728", lw=2, label="WSPD"),
        plt.Line2D([0], [0], color="#2CA02C", lw=2, linestyle=(0, (5, 2)), label="NSTD"),
        plt.Rectangle((0, 0), 1, 1, color="#BFC5D2", alpha=0.45, label="Rain"),
        plt.Rectangle((0, 0), 1, 1, color="#1F77B4", alpha=0.8, label="Irrigation"),
        plt.Rectangle((0, 0), 1, 1, color="#FF7F0E", alpha=0.75, label="Fertilizer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"hla_ic1_yearly_{scenario}_rain_irrig_fert_wspd_nstd.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=2, metavar=("SCENARIO", "YEAR"))
    parser.add_argument("--years", default="2004-2023")
    parser.add_argument("--scenarios", default="auto_irrig,null")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.child:
        child_run(args.child[0], int(args.child[1]))
        return
    if "-" in args.years:
        start, end = [int(x) for x in args.years.split("-", 1)]
        years = list(range(start, end + 1))
    else:
        years = [int(x) for x in args.years.split(",") if x.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    status = run_cases(years, scenarios, args.timeout)
    data = collect(years, scenarios)
    figures = []
    if not data.empty:
        for sc in scenarios:
            figures.append(str(plot_scenario(data, sc)))
    manifest = {
        "status_csv": str(OUT_DIR / "hla_ic1_yearly_pdi_run_status.csv"),
        "daily_csv": str(OUT_DIR / "hla_ic1_yearly_daily_values.csv"),
        "summary_csv": str(OUT_DIR / "hla_ic1_yearly_summary.csv"),
        "figures": figures,
        "statuses": status.to_dict(orient="records"),
        "note": "Diagnostic only: 2004 initial soil profile is shifted to each target year.",
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

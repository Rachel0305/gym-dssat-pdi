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
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table, prepare_text_for_scenario, set_treatment_pointers


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "LC"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-07_017_11_lc_fixed_input_year_screening_record.md"

MZX_NAME = "CNLC0801.MZX"
YEARS = {2008: 1, 2009: 2, 2010: 3, 2011: 4}
SCENARIOS = ["null", "recorded", "dssat_auto"]
ORIGINAL_SOIL_ID = "LC99001200"
FIXED_SOIL_ID = "LC990012007"


def yy_date(year: int, doy: int = 121) -> str:
    return f"{year % 100:02d}{doy:03d}"


def rebuild_initial_conditions(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().upper().startswith("*INITIAL CONDITIONS"):
            out.append(line)
            i += 1
            if i < len(lines) and lines[i].strip().startswith("@C   PCR"):
                header1 = lines[i]
                out.append(header1)
                i += 1
            else:
                raise RuntimeError("LC initial condition header line 1 not found")
            if i >= len(lines):
                raise RuntimeError("LC initial condition summary row not found")
            summary_parts = lines[i].split()
            i += 1
            if i < len(lines) and lines[i].strip().startswith("@C  ICBL"):
                header2 = lines[i]
                out.append(header2)
                i += 1
            else:
                raise RuntimeError("LC initial condition header line 2 not found")
            profile_rows: list[list[str]] = []
            while i < len(lines) and not lines[i].strip().startswith("*"):
                if lines[i].strip() and not lines[i].strip().startswith("@"):
                    profile_rows.append(lines[i].split())
                i += 1
            for year, trno in YEARS.items():
                parts = list(summary_parts)
                parts[0] = str(trno)
                parts[2] = yy_date(year)
                out.append(
                    f"{int(parts[0]):2d}    {parts[1]:<2} {parts[2]:>5} {parts[3]:>5} {parts[4]:>5} {parts[5]:>5} "
                    f"{parts[6]:>5} {parts[7]:>5} {parts[8]:>5} {parts[9]:>5} {parts[10]:>5} {parts[11]:>5} {parts[12]:>5} {parts[13]:>5}"
                )
            out.append(header2)
            for year, trno in YEARS.items():
                for row in profile_rows:
                    r = list(row)
                    r[0] = str(trno)
                    out.append(f"{int(r[0]):2d} {int(float(r[1])):5d} {float(r[2]):5.2f} {float(r[3]):5.1f} {float(r[4]):5.1f}")
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def update_treatment_ic_pointers(text: str) -> str:
    out: list[str] = []
    in_treatments = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            in_treatments = True
            out.append(line)
            continue
        if in_treatments and stripped.startswith("*"):
            in_treatments = False
        if in_treatments and re.match(r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\S+", line):
            parts = line.split()
            trno = int(parts[0])
            if trno in YEARS.values():
                while len(parts) < 18:
                    parts.append("0")
                parts[8] = str(trno)
                out.append(
                    f" {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<25} "
                    f"{parts[5]:>2} {parts[6]:>2} {parts[7]:>2} {parts[8]:>2} {parts[9]:>2} {parts[10]:>2} {parts[11]:>2} "
                    f"{parts[12]:>2} {parts[13]:>2} {parts[14]:>2} {parts[15]:>2} {parts[16]:>2} {parts[17]:>2}"
                )
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def update_simulation_sdates(text: str) -> str:
    trno_to_year = {trno: year for year, trno in YEARS.items()}
    out: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+GE\b", line)
        if match:
            trno = int(match.group(1))
            year = trno_to_year.get(trno)
            if year is not None:
                out.append(f"{trno:2d} GE              1     1     S {yy_date(year)}  2150 {year}")
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def update_planting_dates(text: str) -> str:
    """Align planting rows to each treatment year in copied run inputs."""
    trno_to_year = {trno: year for year, trno in YEARS.items()}
    out: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(\s*)(\d+)(\s+PL\s+)(\d{5})(\s+)(\d{5})(.*)$", line)
        if match:
            indent, trno_s, prefix, pfrst, gap, plast, rest = match.groups()
            trno = int(trno_s)
            year = trno_to_year.get(trno)
            if year is not None:
                yy = year % 100
                pfrst = f"{yy:02d}{int(pfrst[-3:]):03d}"
                plast = f"{yy:02d}{int(plast[-3:]):03d}"
                line = f"{indent}{trno_s}{prefix}{pfrst}{gap}{plast}{rest}"
        out.append(line)
    return "\n".join(out) + "\n"


def update_reported_event_dates(text: str) -> str:
    """Align reported irrigation/fertilizer event years to treatment years."""
    trno_to_year = {trno: year for year, trno in YEARS.items()}
    out: list[str] = []
    in_event_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@I IDATE") or stripped.startswith("@F FDATE"):
            in_event_section = True
            out.append(line)
            continue
        if in_event_section:
            if stripped.startswith("@") or stripped.startswith("*"):
                in_event_section = False
                out.append(line)
                continue
            match = re.match(r"^(\s*)(\d+)(\s+)(\d{5})(.*)$", line)
            if match:
                indent, trno_s, gap, date, rest = match.groups()
                trno = int(trno_s)
                year = trno_to_year.get(trno)
                if year is not None:
                    date = f"{year % 100:02d}{int(date[-3:]):03d}"
                    line = f"{indent}{trno_s}{gap}{date}{rest}"
        out.append(line)
    return "\n".join(out) + "\n"


def fixed_source_text() -> str:
    text = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    text = text.replace(ORIGINAL_SOIL_ID, FIXED_SOIL_ID)
    text = rebuild_initial_conditions(text)
    text = update_treatment_ic_pointers(text)
    text = update_simulation_sdates(text)
    text = update_planting_dates(text)
    text = update_reported_event_dates(text)
    return text


def prepare_run(year: int, scenario: str) -> Path:
    trno = YEARS[year]
    run_dir = OUT_DIR / "runs" / str(year) / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    text = fixed_source_text()
    text = prepare_text_for_scenario(text, trno, scenario)
    if scenario == "dssat_auto":
        text = set_treatment_pointers(text, trno, "0", "0")
    filex = input_dir / f"CNLC{year % 100:02d}01.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux_candidates = [
        *sorted(input_dir.glob("CNLC*.WTH")),
        input_dir / "SOIL.SOL",
        input_dir / "MZCER048.CUL",
        input_dir / "CNLC.CLI",
        input_dir / "CNLC.PRM",
        input_dir / "CNLC.wdb",
    ]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": [str(p) for p in aux_candidates if p.exists()],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"year": year, "trno": trno, "scenario": scenario}, indent=2), encoding="utf-8")
    return run_dir


def child_run(run_dir: Path, max_steps: int = 380) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": "LC",
                    "requested_year": meta["year"],
                    "scenario": meta["scenario"],
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
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


def parse_events(run_dir: Path, year: int, scenario: str) -> pd.DataFrame:
    path = run_dir / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(columns=["site", "requested_year", "scenario", "dap", "operation", "amount", "unit"])
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap = np.nan
        if len(parts) >= 7:
            try:
                dap = int(parts[6])
            except ValueError:
                pass
        amount = 0.0
        unit = ""
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
        rows.append({"site": "LC", "requested_year": year, "scenario": scenario, "dap": dap, "operation": raw.strip(), "amount": amount, "unit": unit})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["site", "requested_year", "scenario", "dap", "operation", "amount", "unit"])


def parse_case_summary(run_dir: Path, year: int, scenario: str) -> dict[str, Any]:
    summary_out = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "Summary.OUT")
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    row: dict[str, Any] = {"site": "LC", "station": "Luancheng", "year": year, "scenario": scenario}
    if not summary_out.empty:
        s = summary_out.iloc[-1]
        for key in ["HWAM", "CWAM", "ADAT", "MDAT", "IRCM", "NICM", "PRCP", "ETCP"]:
            val = pd.to_numeric(pd.Series([s.get(key)]), errors="coerce").iloc[0]
            row[key] = np.nan if pd.isna(val) else float(val)
    if not plantgro.empty:
        for source, target in [("GWAD", "final_gwad"), ("CWAD", "final_cwad")]:
            if source in plantgro.columns and not plantgro[source].dropna().empty:
                row[target] = float(plantgro[source].dropna().iloc[-1])
        for source, target in [("WSPD", "max_water_stress"), ("NSTD", "max_nitrogen_stress")]:
            if source in plantgro.columns and not plantgro[source].dropna().empty:
                row[target] = float(plantgro[source].dropna().max())
        if "DAP" in plantgro.columns and not plantgro["DAP"].dropna().empty:
            row["final_dap"] = float(plantgro["DAP"].dropna().iloc[-1])
    events = parse_events(run_dir, year, scenario)
    row["event_irrigation_total"] = float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0
    row["event_fertilizer_total"] = float(events.loc[events["unit"].astype(str).str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0
    return row


def run_cases(timeout: int) -> pd.DataFrame:
    statuses: list[dict[str, Any]] = []
    for year in YEARS:
        for scenario in SCENARIOS:
            run_dir = prepare_run(year, scenario)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(run_dir)]
            try:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout, capture_output=True, text=True)
                statuses.append(
                    {
                        "year": year,
                        "scenario": scenario,
                        "returncode": proc.returncode,
                        "timed_out": False,
                        "stdout_tail": proc.stdout[-1200:],
                        "stderr_tail": proc.stderr[-1200:],
                        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                    }
                )
            except subprocess.TimeoutExpired as exc:
                statuses.append(
                    {
                        "year": year,
                        "scenario": scenario,
                        "returncode": None,
                        "timed_out": True,
                        "stdout_tail": "",
                        "stderr_tail": str(exc)[-1200:],
                        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                    }
                )
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "017_11_lc_fixed_input_status.csv", index=False, encoding="utf-8-sig")
    return status


def collect_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for year in YEARS:
        for scenario in SCENARIOS:
            run_dir = OUT_DIR / "runs" / str(year) / scenario
            summaries.append(parse_case_summary(run_dir, year, scenario))
            daily_path = run_dir / "gym_post_state_daily.csv"
            if daily_path.exists():
                daily_frames.append(pd.read_csv(daily_path))
            events = parse_events(run_dir, year, scenario)
            if not events.empty:
                event_frames.append(events)
    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    pivot = summary.pivot_table(index="year", columns="scenario", values="final_gwad", aggfunc="first")
    if "null" in pivot:
        summary = summary.merge(pivot[["null"]].rename(columns={"null": "null_final_gwad"}), on="year", how="left")
        summary["yield_gain_vs_null"] = summary["final_gwad"] - summary["null_final_gwad"]
    summary.to_csv(OUT_DIR / "017_11_lc_fixed_input_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "017_11_lc_fixed_input_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "017_11_lc_fixed_input_events.csv", index=False, encoding="utf-8-sig")
    return summary, daily, events


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_"
    safe = df.copy().fillna("")
    cols = list(safe.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_record(status: pd.DataFrame, summary: pd.DataFrame) -> None:
    view_cols = [
        "year",
        "scenario",
        "final_gwad",
        "final_cwad",
        "event_irrigation_total",
        "event_fertilizer_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "yield_gain_vs_null",
    ]
    view = summary[[c for c in view_cols if c in summary.columns]].copy()
    candidate = summary[(summary["scenario"].isin(["recorded", "dssat_auto"])) & (summary.get("yield_gain_vs_null", 0) >= 100)]
    lines = [
        "# 017_11 LC fixed-input yearly baseline screening",
        "",
        "## Purpose",
        "",
        "Screen LC 2008-2011 after temporary input fixes. This stage does not train DQN and does not modify the source LC input package.",
        "",
        "## Temporary fixes",
        "",
        f"- Soil ID: `{ORIGINAL_SOIL_ID}` -> `{FIXED_SOIL_ID}`.",
        "- IC levels were cloned by treatment year: 08121, 09121, 10121, 11121.",
        "- Treatment IC pointers and simulation SDATE were aligned to the cloned IC levels.",
        "",
        "## Run status",
        "",
        markdown_table(status[["year", "scenario", "returncode", "timed_out", "run_dir"]]),
        "",
        "## Summary",
        "",
        markdown_table(view),
        "",
        "## Decision",
        "",
    ]
    if candidate.empty:
        lines += [
            "- Under the current fixed input package, no LC baseline year shows >=100 kg/ha yield gain from recorded or DSSAT auto over null.",
            "- LC should not enter DQN training yet. If LC remains required, the next step is an explicit initial-condition sensitivity diagnostic, not algorithm training.",
        ]
    else:
        years = sorted(candidate["year"].unique())
        lines += [
            f"- Candidate years with baseline optimization space: {years}.",
            "- These years can enter low-cost DQN smoke testing after visual process plots are checked.",
        ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=str)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.child:
        child_run(Path(args.child))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = run_cases(timeout=args.timeout)
    summary, daily, events = collect_outputs()
    write_record(status, summary)
    print(status[["year", "scenario", "returncode", "timed_out"]].to_string(index=False))
    print(summary[["year", "scenario", "final_gwad", "event_irrigation_total", "event_fertilizer_total", "yield_gain_vs_null"]].to_string(index=False))
    print(f"[done] {OUT_DIR}")


if __name__ == "__main__":
    main()

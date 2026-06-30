"""HLA 2007/2009 low-cost management-response check.

Forward simulations only:
- no PPO training
- no cultivar search
- no calibration-effect scatterplots

Runs TRNO 1 (2007) and TRNO 2 (2009) from the corrected HLA calibration MZX
under three management scenarios: null, recorded, and DSSAT auto.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


INPUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "cultivar_calibration_HLA2004_480"
    / "input_corrected_package"
)
FILEX_NAME = "CNHL0701_corrected_IC123.MZX"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2007_2009_management_space_check"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-28_hla_2007_2009_management_space_check.md"

YEARS = {2007: 1, 2009: 2}
SCENARIOS = ["null", "recorded", "dssat_auto"]


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
            if columns is None or stripped.startswith("*") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < len(columns):
                parts += [""] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=columns or [])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def read_hy0006_line() -> str:
    for line in (INPUT_DIR / "MZCER048.CUL").read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("HY0006 "):
            return line
    raise RuntimeError("HY0006 not found")


def set_treatment_pointers(text: str, trno: int, mi: str, mf: str) -> str:
    out = []
    changed = False
    in_treatments = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            in_treatments = True
            out.append(line)
            continue
        if in_treatments and re.match(rf"^\s*{trno}\s+1\s+1\s+0\s+\S+", line):
            parts = line.split()
            year = parts[4].replace("Sim", "")
            # Reconstruct in the same fixed-column style that DSSAT accepted in
            # the original file. A generic token join can trigger IPEXP errors.
            out.append(
                f" {trno} 1 1 0 Sim{year}                    1  {trno}  0  {trno}  {trno}  {mi}  {mf}  0  0  0  0  0  {trno}"
            )
            changed = True
            continue
        if in_treatments and stripped.startswith("*"):
            in_treatments = False
        out.append(line)
    if not changed:
        raise RuntimeError(f"Could not update treatment pointers for TRNO {trno}")
    return "\n".join(out) + "\n"


def set_management_for_treatment(text: str, trno: int, irrig: str, ferti: str) -> str:
    out = []
    changed = False
    in_management = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
            in_management = True
            out.append(line)
            continue
        if in_management and re.match(rf"^\s*{trno}\s+MA\b", line):
            out.append(f"{trno:2d} MA              R     {irrig}     {ferti}     R     M")
            changed = True
            in_management = False
            continue
        if in_management and (stripped.startswith("@") or stripped.startswith("*")):
            in_management = False
        out.append(line)
    if not changed:
        raise RuntimeError(f"Could not update management line for TRNO {trno}")
    return "\n".join(out) + "\n"


def zero_target_reported_rows(text: str, trno: int) -> str:
    out = []
    in_ir = False
    in_fe = False
    wrote_ir_zero = False
    wrote_fe_zero = False
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            in_ir = True
            in_fe = False
            out.append(line)
            continue
        if line.startswith("@F FDATE"):
            in_fe = True
            in_ir = False
            out.append(line)
            continue
        if in_ir:
            if stripped.startswith("@") or stripped.startswith("*"):
                if not wrote_ir_zero:
                    out.append(f" {trno} 07121 IR001     0")
                    wrote_ir_zero = True
                in_ir = False
                out.append(line)
                continue
            if re.match(rf"^\s*{trno}\s+\d{{5}}\b", line):
                if not wrote_ir_zero:
                    date = line.split()[1]
                    out.append(f" {trno} {date} IR001     0")
                    wrote_ir_zero = True
                continue
        if in_fe:
            if stripped.startswith("@") or stripped.startswith("*"):
                if not wrote_fe_zero:
                    out.append(f" {trno} 07121 FE005 AP002     0     0   -99   -99   -99   -99   -99 null")
                    wrote_fe_zero = True
                in_fe = False
                out.append(line)
                continue
            if re.match(rf"^\s*{trno}\s+\d{{5}}\b", line):
                if not wrote_fe_zero:
                    date = line.split()[1]
                    out.append(f" {trno} {date} FE005 AP002     0     0   -99   -99   -99   -99   -99 null")
                    wrote_fe_zero = True
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_text_for_scenario(source: str, trno: int, scenario: str) -> str:
    if scenario == "recorded":
        return source
    if scenario == "null":
        text = set_treatment_pointers(source, trno, "0", "0")
        text = set_management_for_treatment(text, trno, "N", "N")
        text = zero_target_reported_rows(text, trno)
        return text
    if scenario == "dssat_auto":
        # Keep original MI/MF levels available, but let DSSAT automatic management control water/N.
        text = set_management_for_treatment(source, trno, "A", "A")
        return text
    raise ValueError(scenario)


def prepare_run(year: int, scenario: str) -> Path:
    trno = YEARS[year]
    run_dir = OUT_DIR / "runs" / f"{year}_{scenario}"
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    source = (INPUT_DIR / FILEX_NAME).read_text(encoding="latin-1", errors="ignore")
    text = prepare_text_for_scenario(source, trno, scenario)
    filex = input_dir / f"CNHL{year % 100:02d}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for name in ["MZCER048.CUL", "SOIL.SOL", "CNHL0701.MZA", "CNHL0701.MZT", "CNHL0701.WTH", "CNHL0901.WTH", "CNHL1101.WTH"]:
        shutil.copyfile(INPUT_DIR / name, input_dir / name)
    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"year": year, "trno": trno, "scenario": scenario, "hy0006_line": read_hy0006_line()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 380) -> None:
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


def parse_summary(run_dir: Path) -> dict[str, float | None]:
    path = run_dir / "pdi_tmp_snapshot" / "Summary.OUT"
    df = parse_dssat_table(path)
    row = df.iloc[-1]
    out = {}
    for key in ["HWAM", "CWAM", "ADAT", "MDAT", "IR#M", "IRCM", "NI#M", "NICM", "PRCP", "ETCP"]:
        val = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        out[key] = None if pd.isna(val) else float(val)
    return out


def parse_plantgro(run_dir: Path, year: int, scenario: str) -> pd.DataFrame:
    df = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    out = pd.DataFrame(
        {
            "year": year,
            "scenario": scenario,
            "doy": pd.to_numeric(df.get("DOY"), errors="coerce"),
            "dap": pd.to_numeric(df.get("DAP"), errors="coerce"),
            "wspd": pd.to_numeric(df.get("WSPD"), errors="coerce"),
            "nstd": pd.to_numeric(df.get("NSTD"), errors="coerce"),
            "gwad": pd.to_numeric(df.get("GWAD"), errors="coerce"),
            "cwad": pd.to_numeric(df.get("CWAD"), errors="coerce"),
            "lai": pd.to_numeric(df.get("LAID"), errors="coerce"),
        }
    )
    return out.dropna(subset=["dap"]).drop_duplicates(["year", "scenario", "dap"], keep="last")


def parse_events(run_dir: Path, year: int, scenario: str) -> pd.DataFrame:
    path = run_dir / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit", "raw"])
    pattern = re.compile(
        r"^\s*\d+\s+\w+\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(-?\d+)\s+\w+\s+(.+?)\s+([-+]?\d+(?:\.\d*)?)\s+(\S+)"
    )
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        m = pattern.match(raw)
        if m:
            rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "dap": int(m.group(1)),
                    "operation": m.group(2).strip(),
                    "amount": float(m.group(3)),
                    "unit": m.group(4),
                    "raw": raw.rstrip(),
                }
            )
    return pd.DataFrame(rows)


def parent_run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for year in YEARS:
        for scenario in SCENARIOS:
            run_dir = prepare_run(year, scenario)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child-run-dir", str(run_dir)]
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=180, capture_output=True, text=True)
            (run_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
            (run_dir / "child_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
            statuses.append({"year": year, "scenario": scenario, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1000:]})
            if proc.returncode != 0:
                pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")
                raise RuntimeError(f"{year} {scenario} failed; see {run_dir}")
    pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    daily_frames = []
    event_frames = []
    for year in YEARS:
        for scenario in SCENARIOS:
            run_dir = OUT_DIR / "runs" / f"{year}_{scenario}"
            daily = parse_plantgro(run_dir, year, scenario)
            events = parse_events(run_dir, year, scenario)
            s = parse_summary(run_dir)
            summary_rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "final_dap": float(daily["dap"].max()),
                    "gwad": float(daily.sort_values("dap").iloc[-1]["gwad"]),
                    "cwad": float(daily.sort_values("dap").iloc[-1]["cwad"]),
                    "summary_hwam": s.get("HWAM"),
                    "summary_cwam": s.get("CWAM"),
                    "ir_events": s.get("IR#M"),
                    "ircm": s.get("IRCM"),
                    "n_events": s.get("NI#M"),
                    "nicm": s.get("NICM"),
                    "max_wspd": float(daily["wspd"].max()),
                    "mean_wspd": float(daily["wspd"].mean()),
                    "max_nstd": float(daily["nstd"].max()),
                    "mean_nstd": float(daily["nstd"].mean()),
                }
            )
            daily_frames.append(daily)
            event_frames.append(events)
    summary = pd.DataFrame(summary_rows)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    summary.to_csv(OUT_DIR / "hla_2007_2009_management_space_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT_DIR / "hla_2007_2009_management_space_daily.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(OUT_DIR / "hla_2007_2009_management_space_events.csv", index=False, encoding="utf-8-sig")
    write_report(summary, events_all)
    print(summary.to_string(index=False))


def write_report(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    table = [
        "| year | scenario | HWAM/GWAD | CWAM | IRCM | NICM | max WSPD | max NSTD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        table.append(
            f"| {int(r.year)} | {r.scenario} | {r.gwad:.0f} | {r.cwad:.0f} | {r.ircm:.1f} | {r.nicm:.1f} | {r.max_wspd:.3f} | {r.max_nstd:.3f} |"
        )
    md = [
        "# HLA 2007/2009 management-response space check",
        "",
        "## Scope",
        "",
        "- PDI/gym-DSSAT forward simulations only.",
        "- No PPO training.",
        "- New HY0006 cultivar line from the corrected calibration input package.",
        "- IC=1 is kept through the year-matched IC levels in `CNHL0701_corrected_IC123.MZX`.",
        "- `recorded` is the original field recorded management and can temporarily serve as the expert/reference management.",
        "",
        "## HY0006 line",
        "",
        "```text",
        read_hy0006_line(),
        "```",
        "",
        "## Summary",
        "",
        *table,
        "",
        "## Files",
        "",
        "- `hla_2007_2009_management_space_summary.csv`",
        "- `hla_2007_2009_management_space_daily.csv`",
        "- `hla_2007_2009_management_space_events.csv`",
        "- `runs/*/input/`",
        "- `runs/*/pdi_tmp_snapshot/`",
    ]
    DOC_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")


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

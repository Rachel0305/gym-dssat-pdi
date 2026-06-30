"""Strict same-candidate-IC four-scenario forward comparison for HLA 2004.

No PPO training is performed. Expert and PPO scenarios replay previously saved
008_19 daily actions under the same candidate initial condition.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


YEAR = 2004
SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_null_2004_2023"
    / "runs"
    / str(YEAR)
)
OLD_DAILY = (
    PROJECT_ROOT
    / "Leave_One_experiments"
    / "hla2004_four_scenario_process_plots_008_19"
    / "evaluation"
    / "008_19_hla2004_four_scenario_daily_values_for_plots.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_strict_four_scenario_010_14"
)
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-27_hla2004_candidate_ic_strict_four_scenario_010_14.md"

SCENARIOS = {
    "candidate_null": {
        "label": "Candidate IC null",
        "kind": "zero_action_replay",
        "old_key": None,
        "management": "reported_action",
        "note": "Candidate IC with no irrigation and no nitrogen.",
    },
    "candidate_recorded_expert_replay": {
        "label": "Recorded expert replay",
        "kind": "reported_event_replay",
        "old_key": "expert_reference_recorded",
        "management": "reported_events",
        "note": "008_19 recorded expert daily actions replayed under candidate IC.",
    },
    "candidate_dssat_auto": {
        "label": "DSSAT auto irrigation + auto-N attempt",
        "kind": "zero_action_replay",
        "old_key": None,
        "management": "dssat_auto",
        "note": "DSSAT native IRRIG=A and FERTI=A under candidate IC.",
    },
    "candidate_ppo_action_replay": {
        "label": "PPO action replay",
        "kind": "reported_event_replay",
        "old_key": "ppo_soft_stress_seed0_00815",
        "management": "reported_events",
        "note": "008_19 PPO soft-stress seed0 daily actions replayed under candidate IC; not retrained.",
    },
}

SCENARIO_ORDER = list(SCENARIOS)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "candidate_null": "#464C55",
    "candidate_recorded_expert_replay": "#CC6F47",
    "candidate_dssat_auto": "#5477C4",
    "candidate_ppo_action_replay": "#386411",
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


def set_treatment_mi_mf(text: str, mi: str, mf: str) -> str:
    out = []
    in_treatments = False
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            in_treatments = True
            out.append(line)
            continue
        if in_treatments and re.match(r"^\s*1\s+1\s+1\s+0\s+\S+", line):
            # Keep DSSAT's old fixed-width-ish spacing. Reconstructing this
            # from tokens caused IPEXP crop-input errors.
            out.append(f" 1 1 1 0 Sim2004                    1  1  0  1  1  {mi}  {mf}  0  0  0  0  0  1")
            changed = True
            in_treatments = False
            continue
        if in_treatments and stripped.startswith("*"):
            in_treatments = False
        out.append(line)
    if not changed:
        raise RuntimeError("Could not update MI/MF in treatment line")
    return "\n".join(out) + "\n"


def set_management_line(text: str, irrig: str, ferti: str) -> str:
    out = []
    in_management = False
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
            in_management = True
            out.append(line)
            continue
        if in_management and re.match(r"^\s*1\s+MA\b", line):
            out.append(f" 1 MA              R     {irrig}     {ferti}     R     M")
            changed = True
            in_management = False
            continue
        if in_management and (stripped.startswith("@") or stripped.startswith("*")):
            in_management = False
        out.append(line)
    if not changed:
        raise RuntimeError("Could not update management line")
    return "\n".join(out) + "\n"


def reset_reported_zero_rows(text: str) -> str:
    out: list[str] = []
    pending_i = False
    pending_f = False
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            pending_i = True
            pending_f = False
            out.append(line)
            continue
        if line.startswith("@F FDATE"):
            pending_f = True
            pending_i = False
            out.append(line)
            continue
        if pending_i:
            if re.match(r"^\s*1\s+\d{5}\b", line):
                out.append(" 1 04125 IR001     0")
                pending_i = False
                continue
            pending_i = False
        if pending_f:
            if re.match(r"^\s*1\s+\d{5}\b", line):
                out.append(" 1 04125 FE005 AP002     0     0   -99   -99   -99   -99   -99 2004")
                pending_f = False
                continue
            pending_f = False
        out.append(line)
    return "\n".join(out) + "\n"


def old_action_events(scenario_key: str) -> pd.DataFrame:
    spec = SCENARIOS[scenario_key]
    old_key = spec.get("old_key")
    if old_key is None:
        return pd.DataFrame(columns=["yyddd", "irrigation", "nitrogen"])
    old = pd.read_csv(OLD_DAILY)
    sub = old[old["scenario_key"].eq(old_key)].copy()
    sub["doy"] = pd.to_numeric(sub.get("doy"), errors="coerce")
    sub["real_action_amir"] = pd.to_numeric(sub.get("real_action_amir", 0.0), errors="coerce").fillna(0.0)
    sub["real_action_anfer"] = pd.to_numeric(sub.get("real_action_anfer", 0.0), errors="coerce").fillna(0.0)
    events = sub[(sub["real_action_amir"].abs() > 1e-9) | (sub["real_action_anfer"].abs() > 1e-9)].copy()
    events["yyddd"] = events["doy"].round().astype(int).map(lambda doy: f"{YEAR % 100:02d}{doy:03d}")
    return events[["yyddd", "real_action_amir", "real_action_anfer"]].rename(
        columns={"real_action_amir": "irrigation", "real_action_anfer": "nitrogen"}
    )


def replace_reported_management_rows(text: str, scenario_key: str) -> str:
    events = old_action_events(scenario_key)
    irrigation = events[events["irrigation"].abs() > 1e-9].copy()
    nitrogen = events[events["nitrogen"].abs() > 1e-9].copy()
    out: list[str] = []
    in_irrig_rows = False
    in_fert_rows = False
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            out.append(line)
            if irrigation.empty:
                out.append(" 1 04125 IR001     0")
            else:
                for _, row in irrigation.iterrows():
                    out.append(f" 1 {row['yyddd']} IR001 {float(row['irrigation']):5.1f}")
            in_irrig_rows = True
            continue
        if line.startswith("@F FDATE"):
            out.append(line)
            if nitrogen.empty:
                out.append(" 1 04125 FE005 AP002     0     0   -99   -99   -99   -99   -99 2004")
            else:
                for _, row in nitrogen.iterrows():
                    out.append(
                        f" 1 {row['yyddd']} FE005 AP002     0 {float(row['nitrogen']):5.1f}   -99   -99   -99   -99   -99 2004"
                    )
            in_fert_rows = True
            continue
        if in_irrig_rows:
            if stripped.startswith("@") or stripped.startswith("*"):
                in_irrig_rows = False
                out.append(line)
            elif re.match(r"^\s*1\s+\d{5}\b", line) or stripped == "":
                continue
            else:
                in_irrig_rows = False
                out.append(line)
            continue
        if in_fert_rows:
            if stripped.startswith("@") or stripped.startswith("*"):
                in_fert_rows = False
                out.append(line)
            elif re.match(r"^\s*1\s+\d{5}\b", line) or stripped == "":
                continue
            else:
                in_fert_rows = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_mzx_for_scenario(source_text: str, scenario_key: str) -> str:
    spec = SCENARIOS[scenario_key]
    if scenario_key == "candidate_null":
        # Keep the already validated candidate-null management setup:
        # MI=0, MF=0, IRRIG=N, FERTI=N.
        text = source_text
    elif spec["management"] == "dssat_auto":
        # Keep the same treatment-pointer setup as 010_13.  Changing MI/MF
        # here caused IPIRR errors, while the original pointer setup with only
        # IRRIG/FERTI set to A/A was already validated.
        text = source_text
        text = set_management_line(text, "A", "A")
    else:
        text = set_treatment_mi_mf(source_text, "1", "1")
        text = set_management_line(text, "R", "R")
        text = replace_reported_management_rows(text, scenario_key)
    return text


def load_action_schedule(scenario_key: str) -> dict[int, dict[str, float]]:
    spec = SCENARIOS[scenario_key]
    if spec["kind"] != "action_replay":
        return {}
    old = pd.read_csv(OLD_DAILY)
    sub = old[old["scenario_key"].eq(spec["old_key"])].copy()
    sub["dap"] = pd.to_numeric(sub["dap"], errors="coerce")
    sub["real_action_amir"] = pd.to_numeric(sub.get("real_action_amir", 0.0), errors="coerce").fillna(0.0)
    sub["real_action_anfer"] = pd.to_numeric(sub.get("real_action_anfer", 0.0), errors="coerce").fillna(0.0)
    grouped = sub.groupby("dap", as_index=False)[["real_action_amir", "real_action_anfer"]].sum()
    return {
        int(round(row.dap)): {"amir": float(row.real_action_amir), "anfer": float(row.real_action_anfer)}
        for row in grouped.itertuples()
        if abs(float(row.real_action_amir)) > 1e-9 or abs(float(row.real_action_anfer)) > 1e-9
    }


def prepare_run(scenario_key: str) -> Path:
    run_dir = OUT_DIR / "runs" / scenario_key
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_input = SOURCE_RUN_DIR / "input"
    source_mzx = sorted(source_input.glob("*.MZX"))[0]
    source_text = source_mzx.read_text(encoding="latin-1", errors="ignore")
    text = prepare_mzx_for_scenario(source_text, scenario_key)
    filex = input_dir / f"CNHL04_{scenario_key}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    aux_paths = []
    for src in [*source_input.glob("*.WTH"), *source_input.glob("*.SOL"), *source_input.glob("*.CUL")]:
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux_paths.append(str(dst))

    schedule = load_action_schedule(scenario_key)
    (run_dir / "action_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "scenario_key": scenario_key,
                "scenario": SCENARIOS[scenario_key],
                "source_mzx": str(source_mzx),
                "action_schedule": schedule,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux_paths,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")
    return run_dir


def child_run(run_dir: Path, max_steps: int = 320) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    schedule_raw = json.loads((run_dir / "action_schedule.json").read_text(encoding="utf-8"))
    schedule = {int(k): v for k, v in schedule_raw.items()}
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    rows = []
    action_rows = []
    for step in range(max_steps):
        latest_before = latest_observation_dict(env, obs, info)
        current_dap = scalar(latest_before.get("dap"))
        action_dap = int(round(current_dap + 1)) if np.isfinite(current_dap) else step + 1
        action = {name: 0.0 for name in env.formator.action_names}
        planned = schedule.get(action_dap, {})
        for key, value in planned.items():
            if key in action:
                action[key] = float(value)
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
        obs, reward, terminated, truncated, info = env.step(norm)
        latest = latest_observation_dict(env, obs, info)
        yrdoy = scalar(latest.get("yrdoy"))
        row = {
            "step": step,
            "action_dap": action_dap,
            "yrdoy": yrdoy,
            "year": int(yrdoy // 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
            "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
            "dap": scalar(latest.get("dap")),
            "topwt": scalar(latest.get("topwt")),
            "grnwt": scalar(latest.get("grnwt")),
            "xlai": scalar(latest.get("xlai")),
            "swfac": scalar(latest.get("swfac")),
            "nstres": scalar(latest.get("nstres")),
            "real_action_amir": float(action.get("amir", 0.0)),
            "real_action_anfer": float(action.get("anfer", 0.0)),
            "reward": scalar(reward),
            "done": bool(terminated or truncated),
        }
        rows.append(row)
        if abs(row["real_action_amir"]) > 1e-9 or abs(row["real_action_anfer"]) > 1e-9:
            action_rows.append(row.copy())
        if terminated or truncated:
            break
    pd.DataFrame(rows).to_csv(run_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(action_rows).to_csv(run_dir / "applied_action_rows.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()


def read_weather_rain(input_dir: Path) -> pd.DataFrame:
    wth_files = sorted(input_dir.glob("*.WTH"))
    if not wth_files:
        return pd.DataFrame(columns=["doy", "rain"])
    rows = []
    header = None
    for raw in wth_files[0].read_text(encoding="latin-1", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("@") and "DATE" in stripped and "RAIN" in stripped:
            header = stripped.replace("@", "", 1).split()
            continue
        if not header or not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        row = dict(zip(header, parts))
        date = int(row["DATE"])
        rows.append({"doy": date % 1000, "rain": float(row.get("RAIN", 0))})
    return pd.DataFrame(rows)


def standardize_plantgro(scenario_key: str) -> pd.DataFrame:
    run_dir = OUT_DIR / "runs" / scenario_key
    plantgro = run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT"
    df = parse_dssat_table(plantgro)
    out = pd.DataFrame(
        {
            "scenario_key": scenario_key,
            "scenario_label": SCENARIOS[scenario_key]["label"],
            "year": pd.to_numeric(df.get("YEAR"), errors="coerce"),
            "doy": pd.to_numeric(df.get("DOY"), errors="coerce"),
            "das": pd.to_numeric(df.get("DAS"), errors="coerce"),
            "dap": pd.to_numeric(df.get("DAP"), errors="coerce"),
            "wspd": pd.to_numeric(df.get("WSPD"), errors="coerce"),
            "nstd": pd.to_numeric(df.get("NSTD"), errors="coerce"),
            "gwad": pd.to_numeric(df.get("GWAD"), errors="coerce"),
            "cwad": pd.to_numeric(df.get("CWAD"), errors="coerce"),
            "lai": pd.to_numeric(df.get("LAID"), errors="coerce"),
        }
    )
    rain = read_weather_rain(run_dir / "input")
    out = out.merge(rain, on="doy", how="left") if not rain.empty else out.assign(rain=0.0)
    out["rain"] = out["rain"].fillna(0.0)
    actions = pd.read_csv(run_dir / "gym_post_state_daily.csv")
    actions = actions[["dap", "real_action_amir", "real_action_anfer"]].copy()
    actions["dap"] = pd.to_numeric(actions["dap"], errors="coerce")
    actions = actions.groupby("dap", as_index=False)[["real_action_amir", "real_action_anfer"]].sum()
    out = out.merge(actions, on="dap", how="left")
    out[["real_action_amir", "real_action_anfer"]] = out[["real_action_amir", "real_action_anfer"]].fillna(0.0)
    events = parse_management_events(scenario_key)
    if not events.empty:
        for _, event in events.iterrows():
            dap = pd.to_numeric(event.get("dap"), errors="coerce")
            if pd.isna(dap):
                continue
            amount = float(event.get("amount", 0.0))
            op = str(event.get("operation", "")).lower()
            idx = out["dap"].eq(float(dap))
            if "irrigation" in op:
                out.loc[idx, "real_action_amir"] += amount
            elif "fertil" in op or "nitrogen" in op:
                out.loc[idx, "real_action_anfer"] += amount
    out = out.dropna(subset=["dap"]).drop_duplicates(subset=["scenario_key", "dap"], keep="last")
    return out


def parse_management_events(scenario_key: str) -> pd.DataFrame:
    run_dir = OUT_DIR / "runs" / scenario_key
    path = run_dir / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        events = pd.DataFrame(columns=["scenario_key", "dap", "operation", "amount", "unit", "raw"])
        events.to_csv(run_dir / "management_events.csv", index=False, encoding="utf-8-sig")
        return events
    pattern = re.compile(
        r"^\s*\d+\s+\w+\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(-?\d+)\s+\w+\s+(.+?)\s+([-+]?\d+(?:\.\d*)?)\s+(\S+)"
    )
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        match = pattern.match(raw)
        if match:
            rows.append(
                {
                    "scenario_key": scenario_key,
                    "dap": int(match.group(1)),
                    "operation": match.group(2).strip(),
                    "amount": float(match.group(3)),
                    "unit": match.group(4),
                    "raw": raw.rstrip(),
                }
            )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.drop_duplicates(subset=["scenario_key", "dap", "operation", "amount", "unit"], keep="first")
    events.to_csv(run_dir / "management_events.csv", index=False, encoding="utf-8-sig")
    return events


def parse_summary(scenario_key: str) -> dict[str, float | None]:
    path = OUT_DIR / "runs" / scenario_key / "pdi_tmp_snapshot" / "Summary.OUT"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
    header_line = next((line for line in lines if line.startswith("@")), None)
    data_lines = [line for line in lines if re.match(r"^\s+\d+\s+\d+\s+\d+", line)]
    if not header_line or not data_lines:
        return {}
    columns = header_line.replace("@", "", 1).split()
    parts = data_lines[-1].split()
    if len(columns) - len(parts) == 3 and "SOIL_ID..." in columns:
        soil_idx = columns.index("SOIL_ID...")
        parts = parts[: soil_idx + 1] + ["-99", "-99", "-99"] + parts[soil_idx + 1 :]
    row = dict(zip(columns, parts))
    keys = ["HWAM", "CWAM", "MDAT", "ADAT", "IR#M", "IRCM", "NI#M", "NICM", "PRCP", "ETCP"]
    out = {}
    for key in keys:
        val = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        out[key] = None if pd.isna(val) else float(val)
    return out


def collect_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.concat([standardize_plantgro(key) for key in SCENARIO_ORDER], ignore_index=True)
    daily["scenario_key"] = pd.Categorical(daily["scenario_key"], categories=SCENARIO_ORDER, ordered=True)
    daily = daily.sort_values(["scenario_key", "dap"]).reset_index(drop=True)
    daily.to_csv(OUT_DIR / "hla2004_candidate_ic_strict_four_scenario_daily_values.csv", index=False, encoding="utf-8-sig")

    events = pd.concat([parse_management_events(key) for key in SCENARIO_ORDER], ignore_index=True)
    events.to_csv(OUT_DIR / "hla2004_candidate_ic_strict_four_scenario_management_events.csv", index=False, encoding="utf-8-sig")

    rows = []
    for key in SCENARIO_ORDER:
        sub = daily[daily["scenario_key"].eq(key)].sort_values("dap")
        final = sub.iloc[-1]
        summary = parse_summary(key)
        rows.append(
            {
                "scenario_key": key,
                "scenario_label": SCENARIOS[key]["label"],
                "final_dap": float(final["dap"]),
                "grain_yield_gwad": float(final["gwad"]),
                "biomass_cwad": float(final["cwad"]),
                "summary_hwam": summary.get("HWAM"),
                "summary_cwam": summary.get("CWAM"),
                "total_irrigation_from_actions_or_events": float(sub["real_action_amir"].sum()),
                "total_n_from_actions_or_events": float(sub["real_action_anfer"].sum()),
                "summary_ir_events": summary.get("IR#M"),
                "summary_ircm": summary.get("IRCM"),
                "summary_ni_events": summary.get("NI#M"),
                "summary_nicm": summary.get("NICM"),
                "max_wspd": float(sub["wspd"].max()),
                "mean_wspd": float(sub["wspd"].mean()),
                "max_nstd": float(sub["nstd"].max()),
                "mean_nstd": float(sub["nstd"].mean()),
                "note": SCENARIOS[key]["note"],
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "hla2004_candidate_ic_strict_four_scenario_summary.csv", index=False, encoding="utf-8-sig")
    return daily, events, summary_df


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def plot_process(daily: pd.DataFrame) -> Path:
    use_theme()
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    rain = read_weather_rain(OUT_DIR / "runs" / "candidate_null" / "input")
    if not rain.empty:
        rain["dap"] = rain["doy"] - 125
        rain = rain[(rain["dap"] >= 0) & (rain["dap"] <= daily["dap"].max())]
    else:
        rain = daily[daily["scenario_key"].eq("candidate_null")].sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C5CAD3", edgecolor="#7A828F", linewidth=0.3, width=1.0, label="Rainfall")
    axes[0].set_title("HLA 2004 strict four-scenario process plot", loc="left", fontsize=13, fontweight="semibold")
    axes[0].text(
        0,
        1.16,
        "Same candidate IC for all scenarios; expert and PPO are action replays, not retrained policies.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(frameon=False, loc="upper left")

    for key in SCENARIO_ORDER:
        sub = daily[daily["scenario_key"].eq(key)].sort_values("dap")
        color = COLORS[key]
        label = SCENARIOS[key]["label"]
        axes[1].plot(sub["dap"], sub["wspd"], color=color, linewidth=1.8, label=label)
        axes[2].plot(sub["dap"], sub["nstd"], color=color, linewidth=1.8, label=label)
        ir = sub[sub["real_action_amir"].abs() > 1e-9]
        nf = sub[sub["real_action_anfer"].abs() > 1e-9]
        axes[3].vlines(ir["dap"], 0, ir["real_action_amir"], color=color, linewidth=2.2)
        axes[3].scatter(nf["dap"], nf["real_action_anfer"], color=color, s=34, marker="^", edgecolor="white", linewidth=0.4)
        axes[4].plot(sub["dap"], sub["gwad"], color=color, linewidth=1.8, label=label)
        axes[4].plot(sub["dap"], sub["cwad"], color=color, linewidth=1.1, linestyle="--", alpha=0.72)

    axes[1].set_ylabel("Water\nstress")
    axes[1].set_title("Water stress index; higher means stronger stress in DSSAT output", loc="left", fontsize=10)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_title("Nitrogen stress index; higher means stronger stress in DSSAT output", loc="left", fontsize=10)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain yield, dashed = aboveground biomass", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left", ncol=2)
    axes[4].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    path = OUT_DIR / "hla2004_candidate_ic_strict_four_scenario_process.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_final_bars(summary: pd.DataFrame) -> Path:
    use_theme()
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    x = np.arange(len(summary))
    width = 0.36
    ax.bar(
        x - width / 2,
        summary["grain_yield_gwad"],
        width=width,
        color="#A3BEFA",
        edgecolor="#2E4780",
        label="Grain yield / GWAD",
    )
    ax.bar(
        x + width / 2,
        summary["biomass_cwad"],
        width=width,
        color="#E2E5EA",
        edgecolor="#464C55",
        label="Biomass / CWAD",
    )
    for i, row in summary.iterrows():
        ax.text(i - width / 2, row["grain_yield_gwad"] + 120, f"{row['grain_yield_gwad']:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, row["biomass_cwad"] + 120, f"{row['biomass_cwad']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario_label"], rotation=15, ha="right")
    ax.set_ylabel("kg/ha")
    ax.set_title("HLA 2004 final grain yield and biomass under the same candidate IC", loc="left", fontsize=13, fontweight="semibold", pad=28)
    ax.text(
        0,
        1.015,
        "Expert and PPO are replayed reported schedules; DSSAT auto-N did not trigger in this run.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT_DIR / "hla2004_candidate_ic_strict_four_scenario_final_yield_biomass.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    auto_events = events[events["scenario_key"].eq("candidate_dssat_auto")]
    auto_ir = auto_events[auto_events["operation"].str.contains("Irrigation", case=False, na=False)]
    auto_n = auto_events[auto_events["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)]
    report_cols = [
        "scenario_label",
        "grain_yield_gwad",
        "biomass_cwad",
        "total_irrigation_from_actions_or_events",
        "total_n_from_actions_or_events",
        "summary_ircm",
        "summary_nicm",
        "max_wspd",
        "max_nstd",
    ]
    simple = summary[report_cols].copy()
    simple_md = ["| " + " | ".join(simple.columns) + " |", "| " + " | ".join(["---"] * len(simple.columns)) + " |"]
    for _, row in simple.iterrows():
        vals = []
        for col in simple.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        simple_md.append("| " + " | ".join(vals) + " |")

    md = [
        "# 2026-06-27 HLA 2004 候选 IC 严格四情景 forward 对照",
        "",
        "## 运行性质",
        "",
        "- 本轮不训练 PPO，只做 4 个同输入条件下的 forward simulation。",
        "- 四个情景全部使用 HLA 2004 候选 IC：`0.55 + 0.25N`。",
        "- recorded expert 和 PPO 是旧 008_19 动作表在候选 IC 下的回放，不是重新优化。",
        "",
        "## 结果表",
        "",
        "\n".join(simple_md),
        "",
        "## DSSAT 自动管理触发情况",
        "",
        f"- 自动灌溉事件数：{len(auto_ir)}，合计 {auto_ir['amount'].sum() if not auto_ir.empty else 0:.2f} mm。",
        f"- 自动施肥事件数：{len(auto_n)}，合计 {auto_n['amount'].sum() if not auto_n.empty else 0:.2f}。",
        "",
        "## 初步结论",
        "",
    ]
    if len(auto_ir) > 0 and len(auto_n) == 0:
        md.append("- DSSAT 自动灌溉成功触发，但自动施肥仍未触发，因此第 3 情景不能称为完整自动水氮管理。")
    elif len(auto_ir) > 0 and len(auto_n) > 0:
        md.append("- DSSAT 自动灌溉和自动施肥都触发，可作为原生自动管理候选。")
    else:
        md.append("- DSSAT 自动管理触发不足，需要继续诊断。")
    md.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `hla2004_candidate_ic_strict_four_scenario_daily_values.csv`",
            "- `hla2004_candidate_ic_strict_four_scenario_summary.csv`",
            "- `hla2004_candidate_ic_strict_four_scenario_management_events.csv`",
            "- `hla2004_candidate_ic_strict_four_scenario_process.png`",
            "- `hla2004_candidate_ic_strict_four_scenario_final_yield_biomass.png`",
        ]
    )
    DOC_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")


def parent_run(rerun: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for key in SCENARIO_ORDER:
        run_dir = prepare_run(key)
        plantgro = run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if rerun or not plantgro.exists():
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child-run-dir", str(run_dir)]
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=140, capture_output=True, text=True)
            (run_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
            (run_dir / "child_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
            statuses.append({"scenario_key": key, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1000:]})
            if proc.returncode != 0:
                pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")
                raise RuntimeError(f"{key} failed; see {run_dir / 'child_stderr.txt'}")
        else:
            statuses.append({"scenario_key": key, "returncode": 0, "stderr_tail": "skipped existing"})
    pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")
    daily, events, summary = collect_outputs()
    p1 = plot_process(daily)
    p2 = plot_final_bars(summary)
    write_report(summary, events)
    print(summary.to_string(index=False))
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote outputs to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run-dir", type=Path)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.child_run_dir:
        child_run(args.child_run_dir)
    else:
        parent_run(rerun=args.rerun)


if __name__ == "__main__":
    main()

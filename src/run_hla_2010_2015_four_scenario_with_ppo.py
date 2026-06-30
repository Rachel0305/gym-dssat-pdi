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

from ppo_action_safety import (
    ActionSafetyState,
    apply_action_safety,
    denormalize_action,
    normalize_action,
    update_action_safety_state,
)
from ppo_evaluate import latest_observation_dict, scalar


YEARS = [2010, 2015]
SCENARIOS = ["null", "expert_2007_shifted", "dssat_auto", "ppo_00906_seed0"]
SOURCE_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_new_cultivar_candidate_year_screening"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
NEW_CUL = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "cultivar_calibration_HLA2004_480"
    / "input_corrected_package"
    / "MZCER048.CUL"
)
PPO_MODEL = (
    PROJECT_ROOT
    / "Leave_One_experiments"
    / "hla2004_joint_ppo_i120_cap_n_cost_probe_009_06"
    / "models"
    / "HLA"
    / "ppo_joint_water_n_HLA_2004_seed0.zip"
)

PPO_SAFETY = {
    "enabled": True,
    "daily_irrigation_max": 40.0,
    "daily_n_max": 80.0,
    "season_irrigation_soft_limit": 120.0,
    "season_n_soft_limit": 150.0,
    "min_days_between_irrigation": 7,
    "min_days_between_fertilization": 10,
    "irrigation_allowed_dap_range": [1, 120],
    "fertilization_allowed_dap_range": [1, 90],
}


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
                dap = int(parts[6])
                year = int(parts[3])
                doy = int(parts[4])
            except ValueError:
                continue
            op_tokens = parts[8:]
            if op_tokens and op_tokens[0].isdigit():
                op_tokens = op_tokens[1:]
            operation = " ".join(op_tokens)
            quantity = 0.0
            unit = ""
            qmatch = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(kg(?:\[[A-Za-z]+\])?/ha|mm|kg|%)", operation)
            if qmatch:
                quantity = float(qmatch.group(1))
                unit = qmatch.group(2)
            rows.append({"year": year, "doy": doy, "dap": dap, "operation": operation, "quantity": quantity, "unit": unit})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["year", "doy", "dap", "operation", "quantity", "unit"], keep="last").reset_index(drop=True)
    return out


def source_input_dir(year: int, base: str) -> Path:
    d = SOURCE_ROOT / base / str(year) / "input"
    if not d.exists():
        raise FileNotFoundError(str(d))
    return d


def read_pdate_yyddd(text: str) -> str:
    in_plant = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@P PDATE"):
            in_plant = True
            continue
        if in_plant and re.match(r"^\s*1\s+\d{5}\b", line):
            return line.split()[1]
        if in_plant and (stripped.startswith("*") or stripped.startswith("@")):
            in_plant = False
    raise RuntimeError("PDATE not found")


def yyddd_from_dap(pdate: str, dap: int) -> str:
    yy = int(pdate[:2])
    doy = int(pdate[2:])
    return f"{yy:02d}{doy + int(dap) - 1:03d}"


def set_treatment_mi_mf(text: str, mi: str, mf: str) -> str:
    out = []
    in_treat = False
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            in_treat = True
            out.append(line)
            continue
        if in_treat and re.match(r"^\s*1\s+1\s+1\s+0\s+\S+", line):
            parts = line.split()
            while len(parts) < 17:
                parts.append("0")
            parts[10] = mi
            parts[11] = mf
            out.append(
                f" {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<25} "
                f"{parts[5]:>2} {parts[6]:>2} {parts[7]:>2} {parts[8]:>2} {parts[9]:>2} {parts[10]:>2} {parts[11]:>2} "
                f"{parts[12]:>2} {parts[13]:>2} {parts[14]:>2} {parts[15]:>2} {parts[16]:>2} {parts[17] if len(parts)>17 else '1':>2}"
            )
            changed = True
            continue
        if in_treat and stripped.startswith("*"):
            in_treat = False
        out.append(line)
    if not changed:
        raise RuntimeError("Treatment row not changed")
    return "\n".join(out) + "\n"


def set_management_line(text: str, irrig: str, ferti: str) -> str:
    out = []
    in_mgmt = False
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
            in_mgmt = True
            out.append(line)
            continue
        if in_mgmt and re.match(r"^\s*1\s+MA\b", line):
            out.append(f" 1 MA              R     {irrig}     {ferti}     R     M")
            changed = True
            in_mgmt = False
            continue
        if in_mgmt and (stripped.startswith("@") or stripped.startswith("*")):
            in_mgmt = False
        out.append(line)
    if not changed:
        raise RuntimeError("Management row not changed")
    return "\n".join(out) + "\n"


def replace_irrigation_rows(text: str, pdate: str, expert: bool) -> str:
    rows = [
        f" 1 {yyddd_from_dap(pdate, 50)} IR003    10",
        f" 1 {yyddd_from_dap(pdate, 71)} IR003    10",
        f" 1 {yyddd_from_dap(pdate, 96)} IR003    10",
    ] if expert else [f" 1 {pdate}   -99   -99"]
    out = []
    in_date_rows = False
    wrote = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@I IDATE"):
            in_date_rows = True
            wrote = False
            out.append(line)
            out.extend(rows)
            wrote = True
            continue
        if in_date_rows:
            if stripped.startswith("*") or stripped.startswith("@"):
                in_date_rows = False
                out.append(line)
                continue
            if re.match(r"^\s*1\s+\d{5}\b", line):
                continue
        out.append(line)
    if not wrote:
        raise RuntimeError("Irrigation rows not replaced")
    return "\n".join(out) + "\n"


def replace_fertilizer_rows(text: str, pdate: str, expert: bool) -> str:
    rows = [
        f" 1 {pdate} FE005 AP002     5   138     0   -99   -99   -99   -99 2007shift",
        f" 1 {pdate} FE006 AP002     5    27    30   -99   -99   -99   -99 2007shift",
    ] if expert else [f" 1 {pdate} FE005 AP002     0     0   -99   -99   -99   -99   -99 null"]
    out = []
    in_fert_rows = False
    wrote = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@F FDATE"):
            in_fert_rows = True
            out.append(line)
            out.extend(rows)
            wrote = True
            continue
        if in_fert_rows:
            if stripped.startswith("*") or stripped.startswith("@"):
                in_fert_rows = False
                out.append(line)
                continue
            if re.match(r"^\s*1\s+\d{5}\b", line):
                continue
        out.append(line)
    if not wrote:
        raise RuntimeError("Fertilizer rows not replaced")
    return "\n".join(out) + "\n"


def make_expert_text(null_text: str) -> str:
    pdate = read_pdate_yyddd(null_text)
    text = set_treatment_mi_mf(null_text, "1", "1")
    text = set_management_line(text, "R", "R")
    text = replace_irrigation_rows(text, pdate, expert=True)
    text = replace_fertilizer_rows(text, pdate, expert=True)
    return text


def prepare_case(scenario: str, year: int) -> Path:
    base = "auto_irrig" if scenario == "dssat_auto" else "null"
    src_dir = source_input_dir(year, base)
    case_dir = OUT_DIR / scenario / str(year)
    input_dir = case_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")
    mzx_files = sorted(input_dir.glob("*.MZX"))
    if len(mzx_files) != 1:
        raise RuntimeError(f"Expected one MZX in {input_dir}, found {len(mzx_files)}")
    filex = mzx_files[0]
    if scenario == "expert_2007_shifted":
        text = filex.read_text(encoding="latin1", errors="ignore")
        filex.write_text(make_expert_text(text), encoding="latin1")
    if scenario == "ppo_00906_seed0":
        text = filex.read_text(encoding="latin1", errors="ignore")
        # Keep reported-management sections enabled as an action channel for
        # PDI/gym-DSSAT. Setting IRRIG/FERTI to N logs PPO actions in Python but
        # does not let DSSAT execute them.
        text = set_treatment_mi_mf(text, "1", "1")
        text = set_management_line(text, "R", "R")
        pdate = read_pdate_yyddd(text)
        text = replace_irrigation_rows(text, pdate, expert=False)
        text = replace_fertilizer_rows(text, pdate, expert=False)
        filex.write_text(text, encoding="latin1")
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
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def run_zero_action_env(case_dir: Path, scenario: str, year: int, max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows = []
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
            rows.append({"scenario": scenario, "requested_year": year, "step": step, "yrdoy": yrdoy, "dap": scalar(latest.get("dap")), "reward": scalar(reward), "done": done})
            step += 1
    finally:
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(case_dir / f"{scenario}_{year}_gym_post_state_daily.csv", index=False, encoding="utf-8-sig")


def run_ppo_env(case_dir: Path, scenario: str, year: int, max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper
    from stable_baselines3 import PPO

    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    model = PPO.load(str(PPO_MODEL))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    safety_state = ActionSafetyState()
    rows = []
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step))))
            raw_norm, _ = model.predict(obs, deterministic=True)
            raw_norm = np.asarray(raw_norm, dtype=np.float32).flatten()
            raw_norm = np.clip(raw_norm, -1.0, 1.0)
            raw_real = denormalize_action(env.formator.action_names, env.formator.action_space_dict, raw_norm)
            safety = apply_action_safety(raw_real, dap, safety_state, PPO_SAFETY)
            safe_norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, safety.safe_real_action)
            obs, reward, terminated, truncated, info = env.step(safe_norm)
            done = bool(terminated or truncated)
            update_action_safety_state(safety_state, safety.safe_real_action, dap)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "scenario": scenario,
                    "requested_year": year,
                    "step": step,
                    "yrdoy": yrdoy,
                    "dap": scalar(latest.get("dap")),
                    "raw_action_amir": float(raw_real.get("amir", 0.0)),
                    "raw_action_anfer": float(raw_real.get("anfer", 0.0)),
                    "irrigation_mm": float(safety.safe_real_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safety.safe_real_action.get("anfer", 0.0)),
                    "safety_rule_triggered": safety.safety_rule_triggered,
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
    pd.DataFrame(rows).to_csv(case_dir / f"{scenario}_{year}_ppo_actions_daily.csv", index=False, encoding="utf-8-sig")


def child_run(scenario: str, year: int) -> None:
    case_dir = OUT_DIR / scenario / str(year)
    if scenario == "ppo_00906_seed0":
        run_ppo_env(case_dir, scenario, year)
    else:
        run_zero_action_env(case_dir, scenario, year)


def run_cases(timeout: int) -> pd.DataFrame:
    statuses = []
    for year in YEARS:
        for scenario in SCENARIOS:
            case_dir = prepare_case(scenario, year)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", scenario, str(year)]
            try:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout, capture_output=True, text=True)
                statuses.append({"year": year, "scenario": scenario, "returncode": proc.returncode, "timed_out": False, "stderr_tail": proc.stderr[-1200:], "case_dir": str(case_dir)})
            except subprocess.TimeoutExpired as exc:
                statuses.append({"year": year, "scenario": scenario, "returncode": None, "timed_out": True, "stderr_tail": str(exc)[-1200:], "case_dir": str(case_dir)})
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "hla_2010_2015_four_scenario_run_status.csv", index=False, encoding="utf-8-sig")
    return status


def daily_from_case(scenario: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_dir = OUT_DIR / scenario / str(year)
    raw_dir = case_dir / "pdi_tmp_snapshot"
    plant_path = raw_dir / "PlantGro.OUT"
    if not plant_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    plant = parse_table_out(plant_path).rename(columns={"YEAR": "year", "DOY": "doy", "DAP": "dap", "WSPD": "wspd", "NSTD": "nstd", "CWAD": "cwad", "GWAD": "gwad"})
    weather = parse_table_out(raw_dir / "Weather.OUT").rename(columns={"YEAR": "year", "DOY": "doy", "PRED": "rain"}) if (raw_dir / "Weather.OUT").exists() else pd.DataFrame()
    events = parse_mgmt_events(raw_dir / "MgmtEvent.OUT")
    daily = plant[[c for c in ["year", "doy", "dap", "wspd", "nstd", "cwad", "gwad"] if c in plant.columns]].copy()
    if {"year", "doy", "rain"}.issubset(weather.columns):
        weather_daily = weather[["year", "doy", "rain"]].drop_duplicates(subset=["year", "doy"], keep="last")
        daily = daily.merge(weather_daily, on=["year", "doy"], how="left")
    else:
        daily["rain"] = 0.0
    daily["requested_year"] = year
    daily["scenario"] = scenario
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if scenario == "ppo_00906_seed0":
        action_path = case_dir / f"{scenario}_{year}_ppo_actions_daily.csv"
        if action_path.exists():
            act = pd.read_csv(action_path)
            act = act[["dap", "irrigation_mm", "fertilizer_kg_ha"]].copy()
            act["dap"] = pd.to_numeric(act["dap"], errors="coerce")
            act = act.drop_duplicates(subset=["dap"], keep="last")
            daily = daily.drop(columns=["irrigation_mm", "fertilizer_kg_ha"]).merge(act, on="dap", how="left")
            daily["irrigation_mm"] = daily["irrigation_mm"].fillna(0.0)
            daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha"].fillna(0.0)
    else:
        if not events.empty:
            for _, ev in events.iterrows():
                op = str(ev["operation"])
                dap = int(ev["dap"])
                qty = float(ev["quantity"])
                if "Irrigation" in op:
                    daily.loc[daily["dap"].eq(dap), "irrigation_mm"] += qty
                if "Fertil" in op:
                    daily.loc[daily["dap"].eq(dap), "fertilizer_kg_ha"] += qty
    events["requested_year"] = year
    events["scenario"] = scenario
    return daily, events


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames, event_frames = [], []
    for year in YEARS:
        for scenario in SCENARIOS:
            daily, events = daily_from_case(scenario, year)
            if not daily.empty:
                daily_frames.append(daily)
            if not events.empty:
                event_frames.append(events)
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    daily_all.to_csv(OUT_DIR / "hla_2010_2015_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(OUT_DIR / "hla_2010_2015_four_scenario_events.csv", index=False, encoding="utf-8-sig")
    if daily_all.empty:
        return daily_all, events_all, pd.DataFrame()
    summary = (
        daily_all.sort_values(["requested_year", "scenario", "dap"])
        .groupby(["requested_year", "scenario"], as_index=False)
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
    summary.to_csv(OUT_DIR / "hla_2010_2015_four_scenario_summary.csv", index=False, encoding="utf-8-sig")
    return daily_all, events_all, summary


def plot_year(data: pd.DataFrame, year: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.8), sharex=True)
    colors = {"null": "#2E4780", "expert_2007_shifted": "#386411", "dssat_auto": "#CC6F47", "ppo_00906_seed0": "#8A3A6F"}
    labels = {"null": "Null", "expert_2007_shifted": "Expert 2007 shifted", "dssat_auto": "DSSAT auto", "ppo_00906_seed0": "PPO 009_06 seed0"}
    styles = {"null": "-", "expert_2007_shifted": (0, (4, 2)), "dssat_auto": (0, (2, 2)), "ppo_00906_seed0": (0, (7, 2, 1, 2))}
    year_data = data[data["requested_year"].eq(year)].copy()
    for ax, stress_col, title in [(axes[0], "wspd", "Water stress WSPD"), (axes[1], "nstd", "Nitrogen stress NSTD")]:
        ax2 = ax.twinx()
        rain_source = year_data[year_data["scenario"].eq("null")].sort_values("dap")
        if not rain_source.empty:
            ax2.bar(rain_source["dap"], rain_source["rain"].fillna(0), width=1.0, color="#C5CAD3", edgecolor="#464C55", alpha=0.32, label="Rain")
        for scenario in SCENARIOS:
            sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap")
            if sub.empty:
                continue
            ax.plot(sub["dap"], sub[stress_col], color=colors[scenario], linestyle=styles[scenario], linewidth=2.0, label=labels[scenario])
            if scenario != "null":
                ax2.bar(sub["dap"], sub["irrigation_mm"].fillna(0), width=2.0, color=colors[scenario], alpha=0.18, edgecolor=colors[scenario])
                ax2.bar(sub["dap"], sub["fertilizer_kg_ha"].fillna(0), width=3.0, color=colors[scenario], alpha=0.35, edgecolor=colors[scenario], hatch="//")
        ax.set_ylabel(title)
        ax.set_ylim(-0.03, 1.03)
        ax2.set_ylabel("Rain / irrigation / fertilizer")
        max_amt = max(float(year_data["rain"].max()), float(year_data["irrigation_mm"].max()), float(year_data["fertilizer_kg_ha"].max()), 10.0)
        ax2.set_ylim(0, max_amt * 1.25)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    axes[-1].set_xlabel("DAP")
    line_handles = [plt.Line2D([0], [0], color=colors[s], linestyle=styles[s], linewidth=2.0, label=labels[s]) for s in SCENARIOS]
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C5CAD3", alpha=0.32, label="Rain"),
        plt.Rectangle((0, 0), 1, 1, color="#7A828F", alpha=0.18, label="Management amount, solid=irrigation"),
        plt.Rectangle((0, 0), 1, 1, color="#7A828F", alpha=0.35, hatch="//", label="Management amount, hatched=fertilizer"),
    ]
    fig.suptitle(f"HLA {year} four-scenario forward comparison", fontsize=13)
    fig.legend(line_handles + bar_handles, [h.get_label() for h in line_handles + bar_handles], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"hla_{year}_four_scenario_rain_stress_management.png"
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return out


def plot_yield(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    colors = {"null": "#A3BEFA", "expert_2007_shifted": "#A3D576", "dssat_auto": "#F0986E", "ppo_00906_seed0": "#F390CA"}
    labels = {"null": "Null", "expert_2007_shifted": "Expert 2007 shifted", "dssat_auto": "DSSAT auto", "ppo_00906_seed0": "PPO"}
    for ax, col, title in [(axes[0], "final_gwad", "Grain yield GWAD"), (axes[1], "final_cwad", "Biomass CWAD")]:
        pivot = summary.pivot(index="requested_year", columns="scenario", values=col).reindex(columns=SCENARIOS)
        x = np.arange(len(pivot.index))
        width = 0.18
        for i, sc in enumerate(SCENARIOS):
            vals = pivot[sc].to_numpy()
            xpos = x + (i - 1.5) * width
            ax.bar(xpos, vals, width=width, color=colors[sc], edgecolor="#464C55", linewidth=0.7, label=labels[sc])
            for xi, val in zip(xpos, vals):
                if np.isfinite(val):
                    ax.text(xi, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(y)) for y in pivot.index])
        ax.set_ylabel("kg ha$^{-1}$")
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("HLA 2010/2015 four-scenario yield summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "hla_2010_2015_four_scenario_yield_biomass.png"
    fig.savefig(out, dpi=230)
    plt.close(fig)
    return out


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            d[col] = d[col].astype(str)
    lines = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for row in d.values.tolist():
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)




def plot_year_split(data: pd.DataFrame, year: int) -> list[Path]:
    fig_dir = OUT_DIR / "figures_split"
    fig_dir.mkdir(parents=True, exist_ok=True)
    year_data = data[data["requested_year"].eq(year)].copy()
    colors = {"null": "#2E4780", "expert_2007_shifted": "#386411", "dssat_auto": "#CC6F47", "ppo_00906_seed0": "#8A3A6F"}
    labels = {"null": "Null", "expert_2007_shifted": "Expert 2007 shifted", "dssat_auto": "DSSAT auto", "ppo_00906_seed0": "PPO 009_06 seed0"}
    styles = {"null": "-", "expert_2007_shifted": (0, (4, 2)), "dssat_auto": (0, (2, 2)), "ppo_00906_seed0": (0, (7, 2, 1, 2))}
    outputs: list[Path] = []

    rain = year_data[year_data["scenario"].eq("null")].sort_values("dap")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if not rain.empty:
        ax.bar(rain["dap"], rain["rain"].fillna(0), width=1.0, color="#C5CAD3", edgecolor="#464C55", alpha=0.75)
    ax.set_title(f"HLA {year}: rainfall", loc="left", fontsize=12)
    ax.set_xlabel("DAP")
    ax.set_ylabel("Rain (mm)")
    ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    out = fig_dir / f"hla_{year}_split_01_rainfall.png"
    fig.tight_layout(); fig.savefig(out, dpi=230); plt.close(fig); outputs.append(out)

    for stress_col, title, ylabel, stem in [
        ("wspd", "water stress WSPD", "WSPD", "02_wspd"),
        ("nstd", "nitrogen stress NSTD", "NSTD", "03_nstd"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4.1))
        for scenario in SCENARIOS:
            sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap")
            if not sub.empty:
                ax.plot(sub["dap"], sub[stress_col], color=colors[scenario], linestyle=styles[scenario], linewidth=2.1, label=labels[scenario])
        ax.set_title(f"HLA {year}: {title}", loc="left", fontsize=12)
        ax.set_xlabel("DAP"); ax.set_ylabel(ylabel); ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        out = fig_dir / f"hla_{year}_split_{stem}.png"
        fig.tight_layout(); fig.savefig(out, dpi=230); plt.close(fig); outputs.append(out)

    for amount_col, title, ylabel, stem in [
        ("irrigation_mm", "irrigation events", "Irrigation (mm)", "04_irrigation"),
        ("fertilizer_kg_ha", "fertilizer events", "Fertilizer N (kg ha$^{-1}$)", "05_fertilizer"),
    ]:
        fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 7.2), sharex=True, sharey=True)
        for ax, scenario in zip(axes, SCENARIOS):
            sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap")
            ax.bar(sub["dap"], sub[amount_col].fillna(0), width=2.2, color=colors[scenario], edgecolor="#464C55", alpha=0.78)
            ax.set_ylabel(labels[scenario], rotation=0, ha="right", va="center", labelpad=75, fontsize=9)
            ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_title(f"HLA {year}: {title}", loc="left", fontsize=12)
        axes[-1].set_xlabel("DAP")
        fig.text(0.015, 0.5, ylabel, rotation="vertical", va="center", fontsize=10)
        out = fig_dir / f"hla_{year}_split_{stem}.png"
        fig.tight_layout(rect=(0.06, 0, 1, 1)); fig.savefig(out, dpi=230); plt.close(fig); outputs.append(out)
    return outputs


def plot_yield_split(summary: pd.DataFrame) -> Path:
    fig_dir = OUT_DIR / "figures_split"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"null": "#A3BEFA", "expert_2007_shifted": "#A3D576", "dssat_auto": "#F0986E", "ppo_00906_seed0": "#F390CA"}
    labels = {"null": "Null", "expert_2007_shifted": "Expert 2007 shifted", "dssat_auto": "DSSAT auto", "ppo_00906_seed0": "PPO"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    for ax, col, title in [(axes[0], "final_gwad", "Grain yield GWAD"), (axes[1], "final_cwad", "Biomass CWAD")]:
        pivot = summary.pivot(index="requested_year", columns="scenario", values=col).reindex(columns=SCENARIOS)
        x = np.arange(len(pivot.index))
        width = 0.18
        for i, sc in enumerate(SCENARIOS):
            vals = pivot[sc].to_numpy()
            xpos = x + (i - 1.5) * width
            ax.bar(xpos, vals, width=width, color=colors[sc], edgecolor="#464C55", linewidth=0.7, label=labels[sc])
            for xi, val in zip(xpos, vals):
                if np.isfinite(val):
                    ax.text(xi, val, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel("kg ha$^{-1}$")
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(np.arange(len(pivot.index)))
    axes[-1].set_xticklabels([str(int(y)) for y in pivot.index])
    axes[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.suptitle("HLA 2010/2015 four-scenario yield and biomass", fontsize=13)
    out = fig_dir / "hla_2010_2015_split_yield_biomass.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=230); plt.close(fig)
    return out

def write_report(status: pd.DataFrame, summary: pd.DataFrame, figures: list[Path]) -> Path:
    report = OUT_DIR / "README.md"
    lines = [
        "# HLA 2010/2015 四情景 forward 对比",
        "",
        "本实验不训练 PPO。PPO 情景加载既有 `009_06 seed0` 模型，在 2010/2015 环境中实时 `model.predict()` 决策；专家情景为 2007 记录管理按 DAP 平移。",
        "",
        "## 运行状态",
        "",
        df_to_md(status),
        "",
        "## 汇总结果",
        "",
        df_to_md(summary.round(3) if not summary.empty else summary),
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
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.child:
        child_run(args.child[0], int(args.child[1]))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "hla_2010_2015_four_scenario_run_status.csv"
    if args.collect_only and status_path.exists():
        status = pd.read_csv(status_path, keep_default_na=False)
    else:
        status = run_cases(args.timeout)
    daily, events, summary = collect()
    figures = []
    if not daily.empty:
        figures.extend(plot_year(daily, year) for year in YEARS)
    if not summary.empty:
        figures.append(plot_yield(summary))
    if not daily.empty:
        for year in YEARS:
            figures.extend(plot_year_split(daily, year))
    if not summary.empty:
        figures.append(plot_yield_split(summary))
    report = write_report(status, summary, figures)
    manifest = {
        "out_dir": str(OUT_DIR),
        "status_csv": str(status_path),
        "daily_csv": str(OUT_DIR / "hla_2010_2015_four_scenario_daily.csv"),
        "events_csv": str(OUT_DIR / "hla_2010_2015_four_scenario_events.csv"),
        "summary_csv": str(OUT_DIR / "hla_2010_2015_four_scenario_summary.csv"),
        "report": str(report),
        "figures": [str(p) for p in figures],
        "ppo_model": str(PPO_MODEL),
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

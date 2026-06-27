from __future__ import annotations

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


SCENARIOS = {
    "CNHL0408_IC1_auto_irrig": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0408_DSSAT480_2004",
    "CNHL0409_IC1_null": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0409_DSSAT480_2004",
}
SOURCE_AUX = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "analysis_hla0408_0409_windows480_vs_pdi_ic1"


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


def prepare_scenario(name: str, run_dir: Path) -> Path:
    scenario_dir = OUT_DIR / name
    input_dir = scenario_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    mzx_files = sorted(run_dir.glob("*.MZX"))
    if not mzx_files:
        raise FileNotFoundError(f"No MZX in {run_dir}")
    filex = input_dir / mzx_files[0].name
    shutil.copyfile(mzx_files[0], filex)

    aux_sources = [
        SOURCE_AUX / "CNHL0401.WTH",
        SOURCE_AUX / "SOIL.SOL",
        PROJECT_ROOT / "my_data" / "MZCER048.CUL",
    ]
    aux_paths = []
    for src in aux_sources:
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux_paths.append(str(dst))

    env_args = {
        "log_saving_path": str(scenario_dir / f"{name}_pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux_paths,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (scenario_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return scenario_dir


def run_child(name: str, max_steps: int = 300) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    scenario_dir = OUT_DIR / name
    env_args = json.loads((scenario_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = scenario_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = {action_name: 0.0 for action_name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            records.append(
                {
                    "scenario": name,
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
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            step += 1
    finally:
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(records).to_csv(scenario_dir / f"{name}_gym_post_state_daily.csv", index=False, encoding="utf-8-sig")


def run_pdi_gym() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for name, run_dir in SCENARIOS.items():
        scenario_dir = prepare_scenario(name, run_dir)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", name]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=180, capture_output=True, text=True)
            statuses.append(
                {
                    "scenario": name,
                    "returncode": proc.returncode,
                    "timed_out": False,
                    "stdout_tail": proc.stdout[-1000:],
                    "stderr_tail": proc.stderr[-1000:],
                    "scenario_dir": str(scenario_dir),
                }
            )
        except subprocess.TimeoutExpired as exc:
            statuses.append(
                {
                    "scenario": name,
                    "returncode": None,
                    "timed_out": True,
                    "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else str(exc)[-1000:],
                    "scenario_dir": str(scenario_dir),
                }
            )
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "hla0408_0409_pdi_gym_run_status.csv", index=False, encoding="utf-8-sig")
    return status


def daily_from_raw(raw_dir: Path, prefix: str) -> pd.DataFrame:
    plant = parse_table_out(raw_dir / "PlantGro.OUT")
    weather = parse_table_out(raw_dir / "Weather.OUT")
    events = parse_mgmt_events(raw_dir / "MgmtEvent.OUT")
    plant = plant.rename(columns={"YEAR": "year", "DOY": "doy", "DAP": "dap", "WSPD": "wspd", "NSTD": "nstd", "CWAD": "cwad", "GWAD": "gwad"})
    weather = weather.rename(columns={"YEAR": "year", "DOY": "doy", "PRED": "rain"})
    cols = [c for c in ["year", "doy", "dap", "wspd", "nstd", "cwad", "gwad"] if c in plant.columns]
    daily = plant[cols].copy()
    if {"year", "doy", "rain"}.issubset(weather.columns):
        daily = daily.merge(weather[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    else:
        daily["rain"] = 0.0
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if not events.empty and "dap" in daily.columns:
        for _, ev in events.iterrows():
            op = str(ev.get("operation", ""))
            dap = int(ev["dap"])
            qty = float(ev.get("quantity", 0.0))
            if "Irrigation" in op:
                daily.loc[daily["dap"].eq(dap), "irrigation_mm"] += qty
            if "Fertil" in op:
                daily.loc[daily["dap"].eq(dap), "fertilizer_kg_ha"] += qty
    daily = daily.add_prefix(f"{prefix}_")
    return daily


def collect_compare() -> pd.DataFrame:
    frames = []
    summaries = []
    for scenario, win_dir in SCENARIOS.items():
        pdi_dir = OUT_DIR / scenario / "pdi_tmp_snapshot"
        if not (pdi_dir / "PlantGro.OUT").exists():
            continue
        win = daily_from_raw(win_dir, "windows")
        pdi = daily_from_raw(pdi_dir, "pdi")
        merged = win.merge(
            pdi,
            left_on=["windows_year", "windows_doy", "windows_dap"],
            right_on=["pdi_year", "pdi_doy", "pdi_dap"],
            how="outer",
        )
        merged.insert(0, "scenario", scenario)
        for var in ["wspd", "nstd", "cwad", "gwad", "rain", "irrigation_mm", "fertilizer_kg_ha"]:
            wc = f"windows_{var}"
            pc = f"pdi_{var}"
            if wc in merged.columns and pc in merged.columns:
                merged[f"diff_{var}"] = pd.to_numeric(merged[pc], errors="coerce") - pd.to_numeric(merged[wc], errors="coerce")
        frames.append(merged)
        summary = {"scenario": scenario}
        for var in ["wspd", "nstd", "cwad", "gwad", "irrigation_mm", "fertilizer_kg_ha"]:
            dc = f"diff_{var}"
            if dc in merged.columns:
                summary[f"max_abs_diff_{var}"] = float(pd.to_numeric(merged[dc], errors="coerce").abs().max())
        summary["windows_final_gwad"] = float(pd.to_numeric(merged["windows_gwad"], errors="coerce").dropna().iloc[-1]) if "windows_gwad" in merged else np.nan
        summary["pdi_final_gwad"] = float(pd.to_numeric(merged["pdi_gwad"], errors="coerce").dropna().iloc[-1]) if "pdi_gwad" in merged else np.nan
        summary["windows_irrig_total"] = float(pd.to_numeric(merged.get("windows_irrigation_mm", pd.Series(dtype=float)), errors="coerce").sum())
        summary["pdi_irrig_total"] = float(pd.to_numeric(merged.get("pdi_irrigation_mm", pd.Series(dtype=float)), errors="coerce").sum())
        summaries.append(summary)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(OUT_DIR / "daily_windows480_vs_pdi_gym_hla0408_0409_ic1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary_windows480_vs_pdi_gym_hla0408_0409_ic1.csv", index=False, encoding="utf-8-sig")
    return out


def plot_scenario(data: pd.DataFrame, scenario: str) -> Path:
    sub = data[data["scenario"].eq(scenario)].copy()
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax_stress = plt.subplots(figsize=(14, 6))
    ax_amt = ax_stress.twinx()
    x = pd.to_numeric(sub["windows_dap"], errors="coerce").fillna(pd.to_numeric(sub["pdi_dap"], errors="coerce"))

    ax_amt.bar(x, pd.to_numeric(sub.get("windows_rain", 0), errors="coerce").fillna(0), width=1.0, color="#BFC5D2", edgecolor="#69707D", alpha=0.45, label="Windows rainfall")
    ax_amt.bar(x, pd.to_numeric(sub.get("windows_irrigation_mm", 0), errors="coerce").fillna(0), width=2.4, color="#1F77B4", edgecolor="#0B3D70", alpha=0.80, label="Windows irrigation")
    ax_amt.bar(x, pd.to_numeric(sub.get("windows_fertilizer_kg_ha", 0), errors="coerce").fillna(0), width=3.0, color="#FF7F0E", edgecolor="#9A4B00", alpha=0.75, label="Windows fertilizer")

    lines = []
    lines += ax_stress.plot(x, pd.to_numeric(sub["windows_wspd"], errors="coerce"), color="#D62728", linewidth=2.4, label="Windows WSPD")
    lines += ax_stress.plot(x, pd.to_numeric(sub["pdi_wspd"], errors="coerce"), color="#D62728", linewidth=2.0, linestyle=(0, (2, 2)), label="PDI/gym WSPD")
    lines += ax_stress.plot(x, pd.to_numeric(sub["windows_nstd"], errors="coerce"), color="#2CA02C", linewidth=2.4, label="Windows NSTD")
    lines += ax_stress.plot(x, pd.to_numeric(sub["pdi_nstd"], errors="coerce"), color="#2CA02C", linewidth=2.0, linestyle=(0, (2, 2)), label="PDI/gym NSTD")

    ax_stress.set_title(f"{scenario}: Windows DSSAT 4.8.0 vs PDI/gym DSSAT 4.8.0")
    ax_stress.set_xlabel("DAP")
    ax_stress.set_ylabel("Stress index (larger = stronger stress in this output)")
    ax_amt.set_ylabel("Rain / irrigation / fertilizer amount")
    ax_stress.set_ylim(-0.03, 1.03)
    max_amt = 10.0
    for col in ["windows_rain", "windows_irrigation_mm", "windows_fertilizer_kg_ha"]:
        if col in sub:
            max_amt = max(max_amt, float(pd.to_numeric(sub[col], errors="coerce").max()))
    ax_amt.set_ylim(0, max_amt * 1.25)
    ax_stress.grid(True, axis="y", color="#E4E7EF", linewidth=0.8)
    ax_stress.spines["top"].set_visible(False)
    ax_amt.spines["top"].set_visible(False)
    handles1, labels1 = ax_stress.get_legend_handles_labels()
    handles2, labels2 = ax_amt.get_legend_handles_labels()
    ax_stress.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = fig_dir / f"{scenario}_windows480_vs_pdi_gym_rain_irrig_fert_wspd_nstd.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    status = run_pdi_gym()
    compare = collect_compare()
    figures = []
    if not compare.empty:
        for scenario in SCENARIOS:
            figures.append(str(plot_scenario(compare, scenario)))
    result = {
        "status_csv": str(OUT_DIR / "hla0408_0409_pdi_gym_run_status.csv"),
        "daily_compare_csv": str(OUT_DIR / "daily_windows480_vs_pdi_gym_hla0408_0409_ic1.csv"),
        "summary_csv": str(OUT_DIR / "summary_windows480_vs_pdi_gym_hla0408_0409_ic1.csv"),
        "figures": figures,
        "run_status": status.to_dict(orient="records"),
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        run_child(sys.argv[2])
    else:
        main()

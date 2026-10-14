from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

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
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    SITE,
    STATION,
    prepare_run_dir,
    make_raw_env,
    parse_events,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_fixed_water_nitrogen_scan_015_23"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_015_23_fq2016_fixed_water_nitrogen_scan_record.md"
YEAR = 2016

IRR_SCHEDULES = {
    "I0": {},
    "I60": {35: 30.0, 55: 30.0},
    "I120": {35: 30.0, 55: 30.0, 75: 30.0, 95: 30.0},
}

N_SCHEDULES = {
    "N0": {},
    "N100": {10: 100.0},
    "N200": {10: 100.0, 45: 100.0},
    "N300": {10: 100.0, 45: 100.0, 65: 100.0},
}


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def run_case(irrig_label: str, fert_label: str, irrig_schedule: dict[int, float], fert_schedule: dict[int, float]) -> dict:
    case_label = f"{irrig_label}_{fert_label}"
    case_dir = OUT_DIR / case_label
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    run_dir = prepare_run_dir(YEAR, "dqn_linked_free_daily", seed=0)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    env = make_raw_env(env_args)
    rows = []
    used_i = 0.0
    used_n = 0.0
    try:
        obs, info = env.reset()
        for step in range(380):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step)) or 0))
            real = {
                "amir": float(irrig_schedule.get(dap_before, 0.0)),
                "anfer": float(fert_schedule.get(dap_before, 0.0)),
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            used_i += real["amir"]
            used_n += real["anfer"]
            latest = latest_observation_dict(env, obs, info)
            done = bool(terminated or truncated)
            rows.append(
                {
                    "case": case_label,
                    "step": step,
                    "dap_before": dap_before,
                    "dap": scalar(latest.get("dap")),
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "used_irrigation": used_i,
                    "used_nitrogen": used_n,
                    "reward": reward,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / "daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(case_dir, case_label, snapshot_name="pdi_tmp_snapshot")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": YEAR,
        "case": case_label,
        "irrigation_label": irrig_label,
        "nitrogen_label": fert_label,
        "planned_irrigation_total": float(sum(irrig_schedule.values())),
        "planned_fertilizer_total": float(sum(fert_schedule.values())),
        "mgmtevent_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmtevent_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "mgmtevent_irrigation_events": int(events.loc[events["unit"].eq("mm")].shape[0]) if not events.empty else 0,
        "mgmtevent_fertilizer_events": int(events.loc[events["unit"].str.contains("kg", na=False)].shape[0]) if not events.empty else 0,
        "harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else None,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else None,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else None,
        "mean_swfac": float(daily["swfac"].mean()) if not daily.empty else None,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else None,
        "mean_nstres": float(daily["nstres"].mean()) if not daily.empty else None,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else None,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    color_map = {"I0": "#444444", "I60": "#2E86AB", "I120": "#C0392B"}
    x_map = {"N0": 0, "N100": 100, "N200": 200, "N300": 300}
    for irrig_label, sub in df.groupby("irrigation_label"):
        sub = sub.copy()
        sub["x"] = sub["nitrogen_label"].map(x_map)
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["final_gwad"], marker="o", linewidth=2, color=color_map.get(irrig_label, None), label=irrig_label)
    ax.set_xlabel("Nitrogen total (kg/ha)")
    ax.set_ylabel("Final GWAD (kg/ha)")
    ax.set_title("FQ2016 fixed water-nitrogen scan")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, title="Irrigation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_doc(df: pd.DataFrame) -> None:
    headers = list(df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if pd.notna(val) else "")
            else:
                vals.append("" if pd.isna(val) else str(val))
        table_lines.append("| " + " | ".join(vals) + " |")
    lines = [
        "# 015_23 FQ2016 固定灌溉/施氮组合扫描",
        "",
        "## 目的",
        "",
        "- 在不训练 RL 的前提下，检查 FQ2016 的真实水氮边际响应。",
        "- 判断当前 DQN 很快用满 N300，是否真的有 agronomic 产量支持。",
        "",
        "## 组合设计",
        "",
        "- 灌溉：I0 / I60 / I120",
        "- 施氮：N0 / N100 / N200 / N300",
        "- 共 12 组",
        "",
        "## 汇总结果",
        "",
        *table_lines,
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for irrig_label, irrig_schedule in IRR_SCHEDULES.items():
        for fert_label, fert_schedule in N_SCHEDULES.items():
            print(f"running {irrig_label}_{fert_label}", flush=True)
            summaries.append(run_case(irrig_label, fert_label, irrig_schedule, fert_schedule))
    df = pd.DataFrame(summaries)
    df.to_csv(OUT_DIR / "fq2016_fixed_water_nitrogen_scan_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(df, OUT_DIR / "figures" / "fq2016_fixed_water_nitrogen_scan.png")
    write_doc(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

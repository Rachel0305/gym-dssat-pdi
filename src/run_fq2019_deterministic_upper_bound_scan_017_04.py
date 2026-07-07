from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
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
    make_raw_env,
    parse_events,
    parse_weather,
    prepare_run_dir,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


YEAR = 2019
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2019_deterministic_upper_bound_scan_017_04"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_04_fq2019_deterministic_upper_bound_scan_record.md"

BASE_SUMMARY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq2016_seed1_checkpoint30000_all_year_transfer_017_03"
    / "fq2016_seed1_ckpt30000_transfer_success_by_year.csv"
)

IRR_SCHEDULES: dict[str, dict[int, float]] = {
    "I0": {},
    "I60_early": {35: 30.0, 55: 30.0},
    "I60_late": {75: 30.0, 95: 30.0},
    "I90_mid": {35: 30.0, 55: 30.0, 75: 30.0},
    "I120_even": {35: 30.0, 55: 30.0, 75: 30.0, 95: 30.0},
    "I120_late": {65: 30.0, 80: 30.0, 95: 30.0, 110: 30.0},
    "I180_even_ref": {30: 30.0, 45: 30.0, 60: 30.0, 75: 30.0, 90: 30.0, 105: 30.0},
}

N_SCHEDULES: dict[str, dict[int, float]] = {
    "N0": {},
    "N100_early": {10: 100.0},
    "N200_split": {10: 100.0, 45: 100.0},
    "N300_split": {10: 100.0, 45: 100.0, 65: 100.0},
}

REF_ONLY = {"I180_even_ref"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values: list[float] = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def add_rain(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["doy"] = pd.to_numeric(daily["doy"], errors="coerce")
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    missing = daily["doy"].isna() & daily["dap"].notna()
    daily.loc[missing, "doy"] = 162 + daily.loc[missing, "dap"].round().astype(int)
    weather = parse_weather(YEAR).rename(columns={"rain": "rain_weather"})
    daily = daily.drop(columns=["rain"], errors="ignore")
    daily = daily.merge(weather[["doy", "rain_weather"]], on="doy", how="left")
    daily["rain"] = daily["rain_weather"].fillna(0.0)
    return daily.drop(columns=["rain_weather"])


def run_case(irrig_label: str, fert_label: str, irrig_schedule: dict[int, float], fert_schedule: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    case_label = f"{irrig_label}_{fert_label}"
    case_dir = OUT_DIR / "runs" / case_label
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    source_run = prepare_run_dir(YEAR, "dqn_linked_free_daily", seed=0)
    env_args = json.loads((source_run / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"

    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    used_i = 0.0
    used_n = 0.0
    try:
        obs, info = env.reset()
        for step in range(380):
            before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(before.get("dap", step)) or 0))
            real = {
                "amir": float(irrig_schedule.get(dap_before, 0.0)),
                "anfer": float(fert_schedule.get(dap_before, 0.0)),
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            used_i += real["amir"]
            used_n += real["anfer"]
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": YEAR,
                    "case": case_label,
                    "irrigation_label": irrig_label,
                    "nitrogen_label": fert_label,
                    "within_current_budget": irrig_label not in REF_ONLY,
                    "step": step,
                    "dap_before": dap_before,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "used_irrigation": used_i,
                    "used_nitrogen": used_n,
                    "reward": float(reward),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
        shutil.rmtree(source_run, ignore_errors=True)

    daily = add_rain(pd.DataFrame(rows))
    events = parse_events(case_dir, case_label, snapshot_name="pdi_tmp_snapshot")
    if not events.empty:
        events = events.assign(site=SITE, station=STATION, year=YEAR, case=case_label)
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": YEAR,
        "case": case_label,
        "irrigation_label": irrig_label,
        "nitrogen_label": fert_label,
        "within_current_budget": irrig_label not in REF_ONLY,
        "planned_irrigation_total": float(sum(irrig_schedule.values())),
        "planned_fertilizer_total": float(sum(fert_schedule.values())),
        "mgmtevent_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmtevent_fertilizer_total": float(events.loc[events["unit"].astype(str).str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "harvest_yield_kg_ha": hvals[-1] if hvals else np.nan,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else np.nan,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "mean_swfac": float(daily["swfac"].mean()) if not daily.empty else np.nan,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "mean_nstres": float(daily["nstres"].mean()) if not daily.empty else np.nan,
        "rain_in_season": float(daily[["dap", "rain"]].drop_duplicates("dap")["rain"].sum()) if not daily.empty else np.nan,
        "run_dir": str(case_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, events, summary


def reference_values() -> dict[str, float]:
    if not BASE_SUMMARY.exists():
        return {}
    df = pd.read_csv(BASE_SUMMARY)
    row = df[df["year"].eq(YEAR)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "null": float(r["null_zero"]),
        "recorded": float(r["recorded_shifted"]),
        "auto": float(r["dssat_auto"]),
        "dqn_transfer": float(r["dqn_seed1_ckpt30000_transfer"]),
    }


def plot_yield_scan(summary: pd.DataFrame, refs: dict[str, float]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    colors = {
        "I0": "#333333",
        "I60_early": "#4C78A8",
        "I60_late": "#72B7B2",
        "I90_mid": "#F58518",
        "I120_even": "#54A24B",
        "I120_late": "#B279A2",
        "I180_even_ref": "#E45756",
    }
    x_order = {"N0": 0, "N100_early": 100, "N200_split": 200, "N300_split": 300}
    for irrig_label, sub in summary.groupby("irrigation_label", sort=False):
        sub = sub.copy()
        sub["x"] = sub["nitrogen_label"].map(x_order)
        sub = sub.sort_values("x")
        ls = "--" if irrig_label in REF_ONLY else "-"
        ax.plot(sub["x"], sub["final_gwad"], marker="o", lw=2.0, ls=ls, color=colors.get(irrig_label), label=irrig_label)
    ref_colors = {"null": "#333333", "recorded": "#C73E3A", "auto": "#B8860B", "dqn_transfer": "#2E8B57"}
    for name, value in refs.items():
        ax.axhline(value, color=ref_colors.get(name, "#777777"), lw=1.2, ls=":", label=f"{name}: {value:.0f}")
    ax.set_title("FQ2019 deterministic water-nitrogen upper-bound scan", loc="left", fontweight="bold")
    ax.set_xlabel("Nitrogen total (kg/ha)")
    ax.set_ylabel("Final grain yield GWAD (kg/ha)")
    ax.grid(True, color="#E6E8F0", alpha=0.9)
    ax.legend(ncol=2, fontsize=8)
    stem = FIG_DIR / "fq2019_deterministic_upper_bound_yield_scan"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stress_for_top_cases(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    top_cases = summary.sort_values("final_gwad", ascending=False).head(6)["case"].tolist()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True, gridspec_kw={"hspace": 0.18})
    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C9CED8", edgecolor="#AEB6C2", width=0.9, label="Rainfall")
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(loc="upper left")
    palette = plt.get_cmap("tab10")
    for i, case in enumerate(top_cases):
        sub = daily[daily["case"].eq(case)].sort_values("dap")
        color = palette(i % 10)
        axes[1].plot(sub["dap"], sub["swfac"], lw=1.9, color=color, label=case)
        axes[2].plot(sub["dap"], sub["nstres"], lw=1.9, color=color, label=case)
    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_xlabel("DAP")
    axes[1].legend(ncol=2, fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(True, color="#E6E8F0", alpha=0.9)
    stem = FIG_DIR / "fq2019_top_cases_rain_stress"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.3f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(summary: pd.DataFrame, refs: dict[str, float]) -> None:
    within = summary[summary["within_current_budget"]].copy()
    best_within = within.sort_values("final_gwad", ascending=False).head(5)
    best_all = summary.sort_values("final_gwad", ascending=False).head(8)
    refs_text = "\n".join([f"- {k}: {v:.1f} kg/ha" for k, v in refs.items()])
    lines = [
        "# 017_04 FQ2019 确定性上界扫描记录",
        "",
        "## 目的",
        "",
        "FQ2016 seed1 checkpoint30000 迁移到 FQ2019 后只达到 7759 kg/ha，低于 recorded 和 DSSAT auto。本实验不训练模型，只用人工调度组合检查当前约束下是否存在更高产的确定性方案。",
        "",
        "## 参考值",
        "",
        refs_text,
        "",
        "## 当前 I≤120/N≤300 约束内前 5 名",
        "",
        markdown_table(best_within[["case", "final_gwad", "final_cwad", "planned_irrigation_total", "planned_fertilizer_total", "max_swfac", "max_nstres"]]),
        "",
        "## 全部扫描前 8 名（含 I180 参考）",
        "",
        markdown_table(best_all[["case", "within_current_budget", "final_gwad", "final_cwad", "planned_irrigation_total", "planned_fertilizer_total", "max_swfac", "max_nstres"]]),
        "",
        "## 初步判读",
        "",
        "- 如果约束内最优仍明显低于 DSSAT auto，说明当前 DQN 预算可能限制了追上 auto 的空间。",
        "- 如果约束内最优高于 DQN transfer 很多，说明 FQ2019 并非不能优化，而是 FQ2016 模型迁移没有学到合适时机。",
        "- I180 仅作为水预算诊断参考，不作为当前 DQN 可行策略。",
        "",
        "## 输出",
        "",
        f"- summary: `{(OUT_DIR / 'fq2019_deterministic_upper_bound_scan_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- daily: `{(OUT_DIR / 'fq2019_deterministic_upper_bound_scan_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- events: `{(OUT_DIR / 'fq2019_deterministic_upper_bound_scan_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- figures: `{FIG_DIR.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for irrig_label, irrig_schedule in IRR_SCHEDULES.items():
        for n_label, n_schedule in N_SCHEDULES.items():
            if irrig_label in REF_ONLY and n_label not in {"N0", "N200_split", "N300_split"}:
                continue
            print(f"[017_04] {irrig_label}_{n_label}", flush=True)
            daily, events, summary = run_case(irrig_label, n_label, irrig_schedule, n_schedule)
            all_daily.append(daily)
            if not events.empty:
                all_events.append(events)
            summaries.append(summary)
    daily_df = pd.concat(all_daily, ignore_index=True, sort=False)
    events_df = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    refs = reference_values()
    for k, v in refs.items():
        summary_df[f"diff_vs_{k}"] = summary_df["final_gwad"] - v
    daily_df.to_csv(OUT_DIR / "fq2019_deterministic_upper_bound_scan_daily.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(OUT_DIR / "fq2019_deterministic_upper_bound_scan_events.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "fq2019_deterministic_upper_bound_scan_summary.csv", index=False, encoding="utf-8-sig")
    plot_yield_scan(summary_df, refs)
    plot_stress_for_top_cases(daily_df, summary_df)
    write_doc(summary_df, refs)
    print(summary_df.sort_values("final_gwad", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


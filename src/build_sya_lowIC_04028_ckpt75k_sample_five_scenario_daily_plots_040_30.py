from __future__ import annotations

import math
import shutil
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_sya_lowIC_04013_ckpt25k_five_scenario_daily_plots_040_14 as baseplot


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_30_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION_CODE = "SYA"
SITE = "SY"
STATION_NAME = "Shenyang"
SEED = 0
CHECKPOINT = 75_000
YEARS = [2014, 2017, 2022]
PLOT_TAG = "040_28 MaskablePPO ckpt75k"
FILE_TAG = "040_30_lowIC_04028_ckpt75k"
TABLE_TAG = "040_30_sya_2014_2017_2022_lowIC_04028_ckpt75k"

PPO04028_EVAL = (
    ROOT
    / "benchmark_results"
    / "040_28_sya_lowIC_ppo_i240_swfac_guardrail_reward"
    / "evaluation"
    / "040_28_checkpoint_validation_summary.csv"
)

SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "040_28 MaskablePPO ckpt75k",
}

# 040_28 training reward constants.
YIELD_COEF = 0.158
WATER_COST = 1.1
N_COST = 1.58
WATER_RELIEF_COEF = 10.0
N_RELIEF_COEF = 5.0
SWFAC_PENALTY_THRESHOLD = 0.05
SWFAC_PENALTY_COEF = 50.0
REWARD_SCALE = 0.001


def configure_baseplot_globals() -> None:
    """Point the reusable 040_14 baseline helpers at the 040_30 output tree."""

    baseplot.TASK = TASK
    baseplot.OUT = OUT
    baseplot.FIG_DIR = FIG_DIR
    baseplot.TAB_DIR = TAB_DIR
    baseplot.SNAP_DIR = SNAP_DIR
    baseplot.DOC = DOC
    baseplot.PROMPT = PROMPT
    baseplot.CHECKPOINT = CHECKPOINT
    baseplot.YEARS = YEARS
    baseplot.STATION_CODE = STATION_CODE
    baseplot.SITE = SITE
    baseplot.STATION_NAME = STATION_NAME
    baseplot.SEED = SEED


def ensure_dirs() -> None:
    for path in [FIG_DIR, TAB_DIR, SNAP_DIR, OUT / "configs", OUT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def ppo_daily_from_04028(year: int) -> tuple[pd.DataFrame, dict]:
    if not PPO04028_EVAL.exists():
        raise FileNotFoundError(PPO04028_EVAL)
    eval_df = pd.read_csv(PPO04028_EVAL, keep_default_na=False)
    sub = eval_df[
        eval_df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(eval_df["year"], errors="coerce").eq(int(year))
        & pd.to_numeric(eval_df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
    ]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one 040_28 eval row for {STATION_CODE}{year} ckpt{CHECKPOINT}, found {len(sub)}")
    row = sub.iloc[0]
    daily_path = ROOT / str(row["daily_csv_path"])
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    source = pd.read_csv(daily_path, keep_default_na=False)
    daily = pd.DataFrame(
        {
            "site": SITE,
            "station": STATION_NAME,
            "requested_year": int(year),
            "algorithm": "MaskablePPO",
            "seed": SEED,
            "checkpoint": CHECKPOINT,
            "scenario": "rl_candidate",
            "year_out": int(year),
            "doy": pd.to_numeric(source["doy"], errors="coerce"),
            "das": pd.to_numeric(source["dap"], errors="coerce"),
            "dap": pd.to_numeric(source["dap"], errors="coerce"),
            "rainfall_mm": pd.to_numeric(source["rain"], errors="coerce").fillna(0.0),
            "tmax_c": pd.to_numeric(source["tmax"], errors="coerce"),
            "tmin_c": pd.to_numeric(source["tmin"], errors="coerce"),
            "grain_yield_kg_ha": pd.to_numeric(source["grnwt"], errors="coerce").ffill().fillna(0.0),
            "biomass_kg_ha": pd.to_numeric(source["topwt"], errors="coerce").ffill().fillna(0.0),
            "water_stress_index_wspd": pd.to_numeric(source["swfac"], errors="coerce").ffill().fillna(0.0),
            "nitrogen_stress_index_nstd": pd.to_numeric(source["nstres"], errors="coerce").ffill().fillna(0.0),
            "soil_water_mm": np.nan,
            "irrigation_executed_mm": pd.to_numeric(source["safe_action_amir"], errors="coerce").fillna(0.0),
            "nitrogen_executed_kg_ha": pd.to_numeric(source["safe_action_anfer"], errors="coerce").fillna(0.0),
            "date": pd.to_datetime(source["date"], errors="coerce"),
        }
    )
    temp_ok = (
        daily["tmax_c"].between(-60.0, 60.0, inclusive="both")
        & daily["tmin_c"].between(-70.0, 50.0, inclusive="both")
        & daily["tmax_c"].ge(daily["tmin_c"])
    )
    daily["temperature_source_qc"] = np.where(temp_ok, "pass", "source_anomaly_preserved_not_imputed")
    final_y = float(pd.to_numeric(row["final_grnwt"], errors="coerce"))
    total_i = float(pd.to_numeric(row["total_irrigation"], errors="coerce"))
    total_n = float(pd.to_numeric(row["total_n"], errors="coerce"))
    summary = {
        "site": SITE,
        "station": STATION_NAME,
        "station_code": STATION_CODE,
        "year": int(year),
        "algorithm": "MaskablePPO",
        "seed": SEED,
        "checkpoint": CHECKPOINT,
        "scenario": "rl_candidate",
        "final_grain_kg_ha": final_y,
        "final_biomass_kg_ha": float(daily["biomass_kg_ha"].dropna().iloc[-1]),
        "rain_total_mm": float(daily["rainfall_mm"].sum()),
        "irrigation_event_total_mm": total_i,
        "nitrogen_event_total_kg_ha": total_n,
        "max_water_stress_wspd": float(daily["water_stress_index_wspd"].max()),
        "max_nitrogen_stress_nstd": float(daily["nitrogen_stress_index_nstd"].max()),
        "snapshot_path": "",
        "etcp_mm": math.nan,
        "WP_ET_kg_m3": math.nan,
        "PFP_N_kg_kg": final_y / total_n if total_n > 0 else math.nan,
        "source_status": "authoritative_040_28_validation_daily_csv_no_swtd",
        "source_daily_csv": daily_path.relative_to(ROOT).as_posix(),
    }
    return daily, summary


def recompute_04028_plot_reward(daily: pd.DataFrame) -> pd.DataFrame:
    """Comparable diagnostic reward for all five scenarios.

    This is not a retraining signal.  It is a plotting-only, unified 040_28-style
    reward so the cumulative-reward panel does not mix old and new reward
    definitions.
    """

    frames: list[pd.DataFrame] = []
    for (_year, scenario), sub in daily.groupby(["requested_year", "scenario"], sort=False):
        sub = sub.sort_values("dap").copy()
        i = pd.to_numeric(sub["irrigation_executed_mm"], errors="coerce").fillna(0.0)
        n = pd.to_numeric(sub["nitrogen_executed_kg_ha"], errors="coerce").fillna(0.0)
        wspd = pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").fillna(0.0)
        nstd = pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").fillna(0.0)
        prev_wspd = wspd.shift(1).fillna(wspd.iloc[0] if len(wspd) else 0.0)
        prev_nstd = nstd.shift(1).fillna(nstd.iloc[0] if len(nstd) else 0.0)
        relief = WATER_RELIEF_COEF * i * np.maximum(prev_wspd - wspd, 0.0) + N_RELIEF_COEF * n * np.maximum(prev_nstd - nstd, 0.0)
        penalty = SWFAC_PENALTY_COEF * np.maximum(wspd - SWFAC_PENALTY_THRESHOLD, 0.0)
        step_reward = -WATER_COST * i - N_COST * n + relief - penalty
        if len(sub):
            final_y = float(pd.to_numeric(sub["grain_yield_kg_ha"], errors="coerce").ffill().fillna(0.0).iloc[-1])
            step_reward.iloc[-1] += YIELD_COEF * final_y
        sub["plot_reward_04028_unscaled"] = step_reward
        sub["plot_reward_04028_scaled"] = step_reward * REWARD_SCALE
        sub["cumulative_common_reward"] = sub["plot_reward_04028_scaled"].cumsum()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def plot_daily(daily: pd.DataFrame, year: int) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    # Use the longest complete weather series as the weather background.
    # In severe lowIC years the null scenario can terminate early, so using
    # null as the weather source truncates the apparent season even though the
    # weather input itself is complete.  Weather is scenario-invariant, so the
    # longest available scenario is the safest plotting source.
    weather_candidates = []
    for scenario, sub in daily.groupby("scenario", sort=False):
        work = sub.sort_values("dap")
        non_missing_weather = int(
            work[["rainfall_mm", "tmax_c", "tmin_c"]]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
            .sum()
        )
        weather_candidates.append((non_missing_weather, float(pd.to_numeric(work["dap"], errors="coerce").max()), scenario, work))
    weather_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    weather = weather_candidates[0][3]
    ax = axes[0, 0]
    ax.bar(weather["dap"], weather["rainfall_mm"], color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], weather["tmax_c"].where(weather["temperature_source_qc"].eq("pass")), color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], weather["tmin_c"].where(weather["temperature_source_qc"].eq("pass")), color="#686868", lw=1.3, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=3, fontsize=8, loc="upper right")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        label = LABELS[scenario]
        color = baseplot.COLORS.get(scenario, "#18864B")
        style = baseplot.STYLES.get(scenario, "-")
        sw = pd.to_numeric(sub["soil_water_mm"], errors="coerce")
        if sw.notna().any():
            axes[0, 1].plot(sub["dap"], sw, color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        for ev_ax, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            ev = sub[pd.to_numeric(sub[col], errors="coerce").gt(0)]
            marker = "D" if scenario == "rl_candidate" else "o"
            ev_ax.vlines(ev["dap"], 0, ev[col], color=color, lw=2, alpha=0.88)
            ev_ax.scatter(ev["dap"], ev[col], color=color, marker=marker, s=24, label=label)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.4, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.9, alpha=0.42)
        axes[3, 1].plot(sub["dap"], sub["cumulative_common_reward"], color=color, ls=style, lw=1.4, label=label)

    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 1], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Cumulative reward (040_28 scaled)", "scaled reward"),
    ]
    for ax, title, ylabel in titles:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E8E8E8", linewidth=0.65)
    for ax in axes[2, :]:
        handles, names = ax.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[0, 1].text(
        0.01,
        0.03,
        "PPO SWTD not plotted when no SoilWat snapshot is available.",
        transform=axes[0, 1].transAxes,
        fontsize=7,
        color="#555555",
    )
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"SY{year} lowIC {PLOT_TAG} five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    base = FIG_DIR / f"{FILE_TAG}_sy{year}_five_scenario_daily"
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def main() -> int:
    configure_baseplot_globals()
    ensure_dirs()
    failures: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    figures: list[Path] = []

    rec_config, rec_env_config, recorded_templates = baseplot.configure_recorded_runner()
    for year in YEARS:
        try:
            snapshots = {
                "null": baseplot.load_039_lowic_snapshot(year, "null"),
                "dssat_auto": baseplot.load_039_lowic_snapshot(year, "dssat_auto"),
                "official_extension_expert": baseplot.load_039_lowic_snapshot(year, "official_extension_expert"),
            }
            year_daily: list[pd.DataFrame] = []
            year_summary: list[dict] = []
            null_daily, null_summary = baseplot.scenario_from_snapshot(year, "null", snapshots["null"], null_yield=None)
            year_daily.append(null_daily)
            year_summary.append(null_summary)
            rec_daily, rec_summary = baseplot.run_recorded_farmer_template(year, rec_config, rec_env_config, recorded_templates)
            year_daily.append(rec_daily)
            year_summary.append(rec_summary)
            for scenario in ["dssat_auto", "official_extension_expert"]:
                daily, summary = baseplot.scenario_from_snapshot(year, scenario, snapshots[scenario], null_yield=float(null_summary["final_grain_kg_ha"]))
                year_daily.append(daily)
                year_summary.append(summary)
            ppo_daily, ppo_summary = ppo_daily_from_04028(year)
            year_daily.append(ppo_daily)
            year_summary.append(ppo_summary)
            merged_daily = recompute_04028_plot_reward(pd.concat(year_daily, ignore_index=True))
            figures.extend(plot_daily(merged_daily, year))
            daily_frames.append(merged_daily)
            summary_rows.extend(year_summary)
        except Exception:
            failures.append({"station_code": STATION_CODE, "year": year, "traceback": traceback.format_exc()[-4000:]})
            print(f"[040_30] failed {STATION_CODE}{year}")
            print(traceback.format_exc()[-1200:])

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failures)
    daily_path = TAB_DIR / f"{TABLE_TAG}_five_scenario_daily.csv"
    summary_path = TAB_DIR / f"{TABLE_TAG}_five_scenario_summary.csv"
    failures_path = TAB_DIR / "040_30_failures.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")

    action_cols = ["requested_year", "scenario", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]
    actions = daily_all[
        daily_all["scenario"].eq("rl_candidate")
        & (
            pd.to_numeric(daily_all["irrigation_executed_mm"], errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(daily_all["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).gt(0)
        )
    ][action_cols].copy() if not daily_all.empty else pd.DataFrame(columns=action_cols)
    actions_path = TAB_DIR / f"{TABLE_TAG}_ppo_management_events.csv"
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 040_30 SYA lowIC 040_28 checkpoint75k 代表年份五情景日过程图记录",
        "",
        "## 结论先说",
        "",
        f"- 成功绘图年份：{len(set(daily_all['requested_year'])) if not daily_all.empty else 0} / {len(YEARS)}。",
        f"- 失败年份：{len(failures_df)}。",
        "- 本任务没有训练、没有改 checkpoint、没有改输入；只把 040_28 seed0 checkpoint75000 的冻结评估结果画出来。",
        "- PPO 的管理措施、WSPD/NSTD、产量/生物量来自 040_28 validation daily CSV。",
        "- 四基线来自 039_02 lowIC 快照；recorded farmer 使用既有模板复用路径。",
        "- 040_28 PPO daily CSV 不含 SoilWat SWTD，因此土壤水面板不伪造 PPO 线。",
        "- 累计奖励面板使用统一 040_28 风格诊断奖励，避免旧 common reward 口径与本轮训练 reward 混用。",
        "",
        "## PPO 管理事件",
        "",
        md_table(actions, max_rows=120),
        "",
        "## 五情景终值摘要",
        "",
        md_table(summary, max_rows=80),
        "",
        "## 输出文件",
        "",
        f"- 日值表：`{daily_path.relative_to(ROOT).as_posix()}`",
        f"- 摘要表：`{summary_path.relative_to(ROOT).as_posix()}`",
        f"- 管理事件表：`{actions_path.relative_to(ROOT).as_posix()}`",
        f"- 失败表：`{failures_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 图件",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures if p.suffix.lower() == ".png"],
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "task": TASK,
        "years": YEARS,
        "checkpoint": CHECKPOINT,
        "figures": [p.relative_to(ROOT).as_posix() for p in figures],
        "daily_table": daily_path.relative_to(ROOT).as_posix(),
        "summary_table": summary_path.relative_to(ROOT).as_posix(),
        "actions_table": actions_path.relative_to(ROOT).as_posix(),
        "failures": int(len(failures_df)),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_30_result.json").write_text(pd.Series(result).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

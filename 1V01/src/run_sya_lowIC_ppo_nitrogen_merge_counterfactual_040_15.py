from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_static_level1_four_baseline_rebuild_037_07 as baseline037
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import check_rendered_input


TASK = "040_15_sya_lowIC_ppo_nitrogen_merge_counterfactual"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
PPO_DAILY = (
    ROOT
    / "benchmark_results"
    / "040_14_sya_lowIC_04013_ckpt25k_five_scenario_daily_plots"
    / "tables"
    / "040_14_sya_2014_2023_lowIC_04013_ckpt25k_five_scenario_daily.csv"
)
PPO_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_14_sya_lowIC_04013_ckpt25k_five_scenario_daily_plots"
    / "tables"
    / "040_14_sya_2014_2023_lowIC_04013_ckpt25k_five_scenario_summary.csv"
)

STATION_CODE = "SYA"
SITE = "SY"
SEED = 0
MAX_STEPS = 260
YEARS = list(range(2014, 2024))

PPO_IRRIGATION = {1: 45.0, 8: 45.0, 31: 30.0, 38: 30.0}
PPO_N = {1: 40.0, 8: 40.0, 15: 40.0, 22: 40.0, 29: 40.0, 36: 40.0}
N_VARIANTS: dict[str, dict[int, float]] = {
    "ppo_static_original": PPO_N,
    "n_merge_3_early": {1: 80.0, 15: 80.0, 29: 80.0},
    "n_merge_2_early": {1: 120.0, 22: 120.0},
    "n_merge_3_mid": {1: 80.0, 30: 80.0, 50: 80.0},
}

COLORS = {
    "ppo_original": "#18864B",
    "ppo_static_original": "#444444",
    "n_merge_3_early": "#D55E00",
    "n_merge_2_early": "#0072B2",
    "n_merge_3_mid": "#CC79A7",
}
LABELS = {
    "ppo_original": "PPO original N40x6",
    "ppo_static_original": "Static replay of PPO N40x6",
    "n_merge_3_early": "Merge N: 80@1/15/29",
    "n_merge_2_early": "Merge N: 120@1/22",
    "n_merge_3_mid": "Merge N: 80@1/30/50",
}


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TAB_DIR, SNAP_DIR, OUT / "configs", OUT / "logs"]:
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


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline034.TASK_ID = "040_15"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.MAX_STEPS = MAX_STEPS
    baseline034.replace_static_application_rows = baseline037.replace_static_application_rows_level1  # type: ignore[assignment]
    run_config = direct_ppo.load_yaml(baseline034.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS

    split = pd.read_csv(baseline034.SPLIT_CSV, keep_default_na=False)
    split = split[split["station_code"].astype(str).eq(STATION_CODE) & split["year"].astype(int).isin(YEARS)].copy()
    split["selected_for_train"] = False
    split["selected_for_eval"] = True
    split["selection_reason"] = TASK
    env_config = baseline034.build_env_config(run_config, split)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    split.to_csv(OUT / "configs" / "040_15_selected_years.csv", index=False, encoding="utf-8-sig")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_15_resolved_env_config.yaml")
    return run_config, env_config


def static_schedule(irrigation: dict[int, float], nitrogen: dict[int, float]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for dap in sorted(set(irrigation) | set(nitrogen)):
        i = float(irrigation.get(dap, 0.0))
        n = float(nitrogen.get(dap, 0.0))
        if i > 0 or n > 0:
            out[int(dap)] = {"amir": i, "anfer": n}
    return out


def zero_action(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def weather_row(weather: pd.DataFrame, station: str, date: pd.Timestamp) -> dict[str, Any]:
    sub = weather[(weather["station_code"].astype(str).eq(station)) & (weather["date"].eq(date))]
    return sub.iloc[0].to_dict() if len(sub) else {}


def metrics_from_env_snapshot(env: Any, snapshot: Path, final_yield: float) -> dict[str, Any]:
    tmp = siteppo.snapshot_from_env(env)
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(tmp, snapshot)
    return baseline034.metrics_from_snapshot(snapshot, final_yield)


def read_snapshot_soil_water(snapshot: Path) -> pd.DataFrame:
    try:
        import build_sya_lowIC_04013_ckpt25k_five_scenario_daily_plots_040_14 as plot04014

        table = plot04014.parse_table(snapshot / "SoilWat.OUT")
        if table.empty:
            return pd.DataFrame(columns=["doy", "das", "soil_water_mm"])
        cols = {c: c for c in table.columns}
        work = table[[c for c in ["DOY", "DAS", "SWTD"] if c in table.columns]].copy()
        work = work.rename(columns={"DOY": "doy", "DAS": "das", "SWTD": "soil_water_mm"})
        work["doy"] = pd.to_numeric(work["doy"], errors="coerce")
        work["das"] = pd.to_numeric(work["das"], errors="coerce")
        work["soil_water_mm"] = pd.to_numeric(work["soil_water_mm"], errors="coerce")
        work = work.dropna(subset=["doy", "das"]).drop_duplicates(["doy", "das"], keep="last")
        return work
    except Exception:
        return pd.DataFrame(columns=["doy", "das", "soil_water_mm"])


def run_static_variant(
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    scenario: str,
    schedule: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    env = None
    try:
        env, env_args = baseline034.make_base_env(
            env_config,
            STATION_CODE,
            int(year),
            f"{STATION_CODE}_{year}_{TASK}_{scenario}",
            auto_management=False,
            static_schedule=schedule,
        )
        render_checks = check_rendered_input(Path(env_args["fileX_template_path"]))
        weather = direct_ppo.weather_for_daily(run_config)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION_CODE, int(year))["planting_date"])
        rows: list[dict[str, Any]] = []
        while not done and step_count < MAX_STEPS:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(float(dap_raw))) if np.isfinite(float(dap_raw)) and float(dap_raw) > 0 else step_count + 1
            requested = schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(zero_action(env))
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = weather_row(weather, STATION_CODE, date)
            rows.append(
                {
                    "site": SITE,
                    "station": "Shenyang",
                    "station_code": STATION_CODE,
                    "requested_year": int(year),
                    "scenario": scenario,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "das": int(step_count + 1),
                    "dap": int(dap),
                    "rainfall_mm": scalar(wrow.get("rain"), np.nan),
                    "tmax_c": scalar(wrow.get("tmax"), np.nan),
                    "tmin_c": scalar(wrow.get("tmin"), np.nan),
                    "grain_yield_kg_ha": scalar(latest.get("grnwt")),
                    "biomass_kg_ha": scalar(latest.get("topwt")),
                    "water_stress_index_wspd": scalar(latest.get("swfac")),
                    "nitrogen_stress_index_nstd": scalar(latest.get("nstres")),
                    "soil_water_mm": np.nan,
                    "irrigation_executed_mm": float(requested.get("amir", 0.0)),
                    "nitrogen_executed_kg_ha": float(requested.get("anfer", 0.0)),
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{STATION_CODE}{year} {scenario} 未在 {MAX_STEPS} 步内结束")
        final_y = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        snapshot = SNAP_DIR / STATION_CODE / str(year) / scenario
        metrics = metrics_from_env_snapshot(env, snapshot, final_y)
        sw = read_snapshot_soil_water(snapshot)
        if not sw.empty:
            daily = daily.drop(columns=["soil_water_mm"]).merge(sw, on=["doy", "das"], how="left")
            daily["soil_water_mm"] = pd.to_numeric(daily["soil_water_mm"], errors="coerce").ffill().fillna(0.0)
        events = parse_management_events(snapshot / "MgmtEvent.OUT")
        summary = {
            "site": SITE,
            "station": "Shenyang",
            "station_code": STATION_CODE,
            "year": int(year),
            "scenario": scenario,
            "final_grain_kg_ha": final_y,
            "final_biomass_kg_ha": float(pd.to_numeric(daily["biomass_kg_ha"], errors="coerce").iloc[-1]),
            "rain_total_mm": float(pd.to_numeric(daily["rainfall_mm"], errors="coerce").fillna(0).sum()),
            "irrigation_event_total_mm": float(metrics["actual_irrigation_mm"]),
            "nitrogen_event_total_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "irrigation_event_count": int(sum(1 for item in schedule.values() if float(item.get("amir", 0.0)) > 0)),
            "nitrogen_event_count": int(sum(1 for item in schedule.values() if float(item.get("anfer", 0.0)) > 0)),
            "max_water_stress_wspd": float(pd.to_numeric(daily["water_stress_index_wspd"], errors="coerce").max()),
            "max_nitrogen_stress_nstd": float(pd.to_numeric(daily["nitrogen_stress_index_nstd"], errors="coerce").max()),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "render_issue_count": int(sum(1 for item in render_checks if item.get("status") != "ok")),
            "mgmt_event_count": int(len(events)),
            **metrics,
        }
        return daily, summary, render_checks
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def ppo_original_daily_and_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PPO_DAILY.exists() or not PPO_SUMMARY.exists():
        raise FileNotFoundError("缺少 040_14 PPO 原始日值或汇总表，请先运行 040_14。")
    daily = pd.read_csv(PPO_DAILY, keep_default_na=False)
    daily = daily[daily["scenario"].astype(str).eq("rl_candidate") & daily["requested_year"].astype(int).isin(YEARS)].copy()
    daily["scenario"] = "ppo_original"
    summary = pd.read_csv(PPO_SUMMARY, keep_default_na=False)
    summary = summary[summary["scenario"].astype(str).eq("rl_candidate") & summary["year"].astype(int).isin(YEARS)].copy()
    summary["scenario"] = "ppo_original"
    for col in ["irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "max_water_stress_wspd", "max_nitrogen_stress_nstd", "final_grain_kg_ha", "final_biomass_kg_ha"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["irrigation_event_count"] = 4
    summary["nitrogen_event_count"] = 6
    return daily, summary


def recompute_common_reward(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (year, scenario), group in daily.groupby(["requested_year", "scenario"], sort=True):
        sub = group.sort_values("dap").copy()
        final_y = float(pd.to_numeric(sub["grain_yield_kg_ha"], errors="coerce").dropna().iloc[-1])
        # 这里只做合并方案内部比较，使用投入成本 + 终端产量的共同展示奖励。
        sub["common_reward_step"] = -(
            pd.to_numeric(sub["irrigation_executed_mm"], errors="coerce").fillna(0.0)
            + 5.0 * pd.to_numeric(sub["nitrogen_executed_kg_ha"], errors="coerce").fillna(0.0)
        )
        sub.loc[sub.index[-1], "common_reward_step"] += final_y
        sub["cumulative_common_reward"] = sub["common_reward_step"].cumsum()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in summary.groupby("year", sort=True):
        base = group[group["scenario"].eq("ppo_original")]
        static_base = group[group["scenario"].eq("ppo_static_original")]
        if base.empty:
            continue
        b = base.iloc[0]
        sb = static_base.iloc[0] if not static_base.empty else None
        for _, row in group.iterrows():
            if row["scenario"] == "ppo_original":
                continue
            out = {
                "year": int(year),
                "scenario": row["scenario"],
                "yield_delta_vs_ppo_online_kg_ha": float(row["final_grain_kg_ha"]) - float(b["final_grain_kg_ha"]),
                "biomass_delta_vs_ppo_online_kg_ha": float(row["final_biomass_kg_ha"]) - float(b["final_biomass_kg_ha"]),
                "max_wspd_delta_vs_ppo_online": float(row["max_water_stress_wspd"]) - float(b["max_water_stress_wspd"]),
                "max_nstd_delta_vs_ppo_online": float(row["max_nitrogen_stress_nstd"]) - float(b["max_nitrogen_stress_nstd"]),
                "pfp_n_delta_vs_ppo_online": float(row.get("PFP_N_kg_kg", math.nan)) - float(b.get("pfp_n_kg_kg", b.get("PFP_N_kg_kg", math.nan))),
                "n_event_count_delta_vs_ppo_online": int(row["nitrogen_event_count"]) - int(b["nitrogen_event_count"]),
                "i_total_delta_mm": float(row["irrigation_event_total_mm"]) - float(b["irrigation_event_total_mm"]),
                "n_total_delta_kg_ha": float(row["nitrogen_event_total_kg_ha"]) - float(b["nitrogen_event_total_kg_ha"]),
            }
            if sb is not None:
                out.update(
                    {
                        "yield_delta_vs_static_original_kg_ha": float(row["final_grain_kg_ha"]) - float(sb["final_grain_kg_ha"]),
                        "max_nstd_delta_vs_static_original": float(row["max_nitrogen_stress_nstd"]) - float(sb["max_nitrogen_stress_nstd"]),
                    }
                )
            rows.append(out)
    return pd.DataFrame(rows)


def event_table_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    ev = daily[
        (pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").fillna(0) > 0)
    ].copy()
    return ev[["requested_year", "scenario", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]].sort_values(
        ["requested_year", "scenario", "dap"]
    )


def plot_year(daily: pd.DataFrame, year: int) -> Path:
    sub_all = daily[daily["requested_year"].astype(int).eq(int(year))].copy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    weather = sub_all[sub_all["scenario"].eq("ppo_original")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], pd.to_numeric(weather["rainfall_mm"], errors="coerce"), color="#3977A8", alpha=0.55, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmax_c"], errors="coerce"), color="#C23B32", lw=1.2, label="Tmax")
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmin_c"], errors="coerce"), color="#686868", lw=1.2, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, ncol=3, loc="upper right")

    for scenario in ["ppo_original", *N_VARIANTS.keys()]:
        sub = sub_all[sub_all["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        label = LABELS[scenario]
        axes[0, 1].plot(sub["dap"], pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce"), color=color, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce"), color=color, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], pd.to_numeric(sub["grain_yield_kg_ha"], errors="coerce"), color=color, lw=1.45, label=f"{label} grain")
        axes[1, 1].plot(sub["dap"], pd.to_numeric(sub["biomass_kg_ha"], errors="coerce"), color=color, lw=0.9, alpha=0.35)
        axes[2, 1].plot(sub["dap"], pd.to_numeric(sub["cumulative_common_reward"], errors="coerce"), color=color, lw=1.35, label=label)
        ev = sub[pd.to_numeric(sub["nitrogen_executed_kg_ha"], errors="coerce").fillna(0) > 0]
        axes[2, 0].vlines(ev["dap"], 0, pd.to_numeric(ev["nitrogen_executed_kg_ha"], errors="coerce"), color=color, lw=2, alpha=0.85)
        axes[2, 0].scatter(ev["dap"], pd.to_numeric(ev["nitrogen_executed_kg_ha"], errors="coerce"), color=color, s=24, label=label)

    titles = [
        (axes[0, 1], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 0], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[1, 1], "Grain and biomass", "kg/ha"),
        (axes[2, 0], "Nitrogen application events", "kg/ha/event"),
        (axes[2, 1], "Cumulative common reward", "kg/ha-equivalent"),
    ]
    for ax, title, ylabel in titles:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E8E8E8", linewidth=0.65)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
        handles, names = ax.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
    axes[2, 0].set_xlabel("DAP")
    axes[2, 1].set_xlabel("DAP")
    fig.suptitle(f"SY{year} lowIC PPO nitrogen-merge counterfactual (040_15)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / f"040_15_sy{year}_ppo_nitrogen_merge_counterfactual.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_record(summary: pd.DataFrame, delta: pd.DataFrame, events: pd.DataFrame, failures: pd.DataFrame, figures: list[Path], elapsed: float) -> None:
    variant_mean = (
        delta.groupby("scenario", as_index=False)
        .agg(
            n=("year", "count"),
            mean_yield_delta_vs_ppo_online=("yield_delta_vs_ppo_online_kg_ha", "mean"),
            min_yield_delta_vs_ppo_online=("yield_delta_vs_ppo_online_kg_ha", "min"),
            max_yield_delta_vs_ppo_online=("yield_delta_vs_ppo_online_kg_ha", "max"),
            mean_yield_delta_vs_static_original=("yield_delta_vs_static_original_kg_ha", "mean"),
            mean_max_nstd_delta_vs_ppo_online=("max_nstd_delta_vs_ppo_online", "mean"),
            mean_n_event_count_delta_vs_ppo_online=("n_event_count_delta_vs_ppo_online", "mean"),
        )
        if not delta.empty
        else pd.DataFrame()
    )
    lines = [
        "# 040_15 SYA lowIC PPO 施氮合并反事实审计记录",
        "",
        "## 结论先留空",
        "",
        "本任务只做反事实模拟和表格整理，不在脚本内替代科研判断。请根据下表判断频繁 N40 是否必要。",
        "",
        "## 固定边界",
        "",
        "- 不训练 PPO/DQN。",
        "- 原 PPO 使用 040_13 ckpt25k、040_14 权威 daily CSV。",
        "- 合并方案保持 PPO 原灌溉日期/总量不变，只改变施氮次数和日期。",
        "- 所有合并方案总施氮量均为 N240。",
        "- 使用 lowIC 输入目录。",
        "",
        "## 方案均值差异",
        "",
        md_table(variant_mean, 20),
        "",
        "## 逐年差异",
        "",
        md_table(delta, 120),
        "",
        "## 终值表",
        "",
        md_table(summary[["year", "scenario", "final_grain_kg_ha", "final_biomass_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "irrigation_event_count", "nitrogen_event_count", "WP_ET_kg_m3", "PFP_N_kg_kg", "max_water_stress_wspd", "max_nitrogen_stress_nstd"]] if not summary.empty else summary, 160),
        "",
        "## 管理事件",
        "",
        md_table(events, 160),
        "",
        "## 失败记录",
        "",
        md_table(failures, 50),
        "",
        "## 图件",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures],
        "",
        "## 输出表",
        "",
        f"- `{(TAB_DIR / '040_15_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(TAB_DIR / '040_15_delta_vs_ppo_original.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(TAB_DIR / '040_15_daily.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(TAB_DIR / '040_15_management_events.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(TAB_DIR / '040_15_failures.csv').relative_to(ROOT).as_posix()}`",
        "",
        f"耗时：{elapsed:.1f} 秒。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run() -> dict[str, Any]:
    return {
        "task": TASK,
        "mode": "dry_run",
        "lowIC_input_root": str(LOWIC_INPUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "ppo_daily_exists": PPO_DAILY.exists(),
        "ppo_summary_exists": PPO_SUMMARY.exists(),
        "years": YEARS,
        "ppo_irrigation": PPO_IRRIGATION,
        "ppo_original_n": PPO_N,
        "n_variants": N_VARIANTS,
        "variant_total_n": {k: sum(v.values()) for k, v in N_VARIANTS.items()},
        "next_step_allowed": LOWIC_INPUT_ROOT.exists() and PPO_DAILY.exists() and PPO_SUMMARY.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2, ensure_ascii=False))
        return

    start = time.time()
    ensure_dirs()
    run_config, env_config = load_config_and_env_config()
    ppo_daily, ppo_summary = ppo_original_daily_and_summary()
    ppo_daily["scenario"] = "ppo_original"
    all_daily = [ppo_daily]
    all_summary = [ppo_summary]
    failures: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for year in YEARS:
        for scenario, nitrogen in N_VARIANTS.items():
            try:
                daily, summary, render_checks = run_static_variant(
                    run_config,
                    env_config,
                    int(year),
                    scenario,
                    static_schedule(PPO_IRRIGATION, nitrogen),
                )
                all_daily.append(daily)
                all_summary.append(pd.DataFrame([summary]))
                for item in render_checks:
                    out = dict(item)
                    out.update({"year": int(year), "scenario": scenario})
                    render_rows.append(out)
            except Exception:
                failures.append({"year": int(year), "scenario": scenario, "traceback": traceback.format_exc()[-5000:]})
                print(f"[040_15] failed SYA{year} {scenario}", file=sys.stderr, flush=True)
                print(traceback.format_exc()[-1200:], file=sys.stderr, flush=True)

    daily_all = recompute_common_reward(pd.concat(all_daily, ignore_index=True, sort=False))
    summary = pd.concat(all_summary, ignore_index=True, sort=False)
    # 统一字段大小写，便于下游读取。
    if "pfp_n_kg_kg" in summary.columns and "PFP_N_kg_kg" not in summary.columns:
        summary["PFP_N_kg_kg"] = summary["pfp_n_kg_kg"]
    if "wp_et_kg_m3" in summary.columns and "WP_ET_kg_m3" not in summary.columns:
        summary["WP_ET_kg_m3"] = summary["wp_et_kg_m3"]
    delta = build_delta_table(summary)
    events = event_table_from_daily(daily_all)
    failures_df = pd.DataFrame(failures)
    render_df = pd.DataFrame(render_rows)

    figures = []
    for year in YEARS:
        figures.append(plot_year(daily_all, int(year)))

    summary.to_csv(TAB_DIR / "040_15_summary.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(TAB_DIR / "040_15_delta_vs_ppo_original.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(TAB_DIR / "040_15_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TAB_DIR / "040_15_management_events.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(TAB_DIR / "040_15_failures.csv", index=False, encoding="utf-8-sig")
    render_df.to_csv(TAB_DIR / "040_15_render_checks.csv", index=False, encoding="utf-8-sig")
    write_record(summary, delta, events, failures_df, figures, time.time() - start)

    result = {
        "task": TASK,
        "branch": "complete" if failures_df.empty else "partial_failures",
        "years": YEARS,
        "variants": ["ppo_original", *N_VARIANTS.keys()],
        "failed_runs": int(len(failures_df)),
        "summary_csv": str((TAB_DIR / "040_15_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "delta_csv": str((TAB_DIR / "040_15_delta_vs_ppo_original.csv").relative_to(ROOT)).replace("\\", "/"),
        "daily_csv": str((TAB_DIR / "040_15_daily.csv").relative_to(ROOT)).replace("\\", "/"),
        "events_csv": str((TAB_DIR / "040_15_management_events.csv").relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "figures": [str(p.relative_to(ROOT)).replace("\\", "/") for p in figures],
    }
    (OUT / "040_15_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

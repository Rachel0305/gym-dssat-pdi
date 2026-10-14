from __future__ import annotations

import json
import shutil
import sys
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
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_irrigation_reallocation_counterfactual_040_07 as replay04007
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as ppo04036
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


TASK = "040_38_sya_lowIC_04036_ckpt100k_validation_five_scenario_metric_bars"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DAILY_DIR = OUT / "daily_outputs"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = "SYA"
SITE = "SY"
YEARS = list(range(2014, 2024))
CHECKPOINT = 100_000
LOWIC_INPUT_ROOT = ppo04036.LOWIC_INPUT_ROOT
PPO_EVAL = (
    ROOT
    / "benchmark_results"
    / "040_36_sya_lowIC_ppo_late_irrigation_reserve_mask"
    / "evaluation"
    / "040_36_checkpoint_validation_summary.csv"
)
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_21_sya_lowIC_four_baseline_rebuild"
    / "evaluation"
    / "040_21_baseline_summary.csv"
)

SCENARIO_LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded farmer template",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO 040_36 ckpt100k",
}
SCENARIO_ORDER = [
    "null",
    "recorded_farmer_template",
    "dssat_auto",
    "official_extension_expert",
    "rl_candidate",
]
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "rl_candidate": "#2A9D55",
}


def ensure_dirs() -> None:
    for path in [FIG_DIR, TAB_DIR, SNAP_DIR, DAILY_DIR, OUT / "configs", OUT / "logs"]:
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
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config = ppo04036.load_config()
    split = base03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = base03222.build_selection(split)
    config["runtime"]["smoke_station"] = STATION
    config["paths"]["output_root"] = OUT.relative_to(ROOT).as_posix()

    old_out = direct_ppo.OUTPUT_ROOT
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        direct_ppo.OUTPUT_ROOT = OUT
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
        env_config = direct_ppo.build_env_config(config, selection)
    finally:
        direct_ppo.OUTPUT_ROOT = old_out
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input
    return config, env_config, selection


def ppo_eval_rows() -> pd.DataFrame:
    df = pd.read_csv(PPO_EVAL, keep_default_na=False)
    rows = df[
        df["station_code"].astype(str).eq(STATION)
        & pd.to_numeric(df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
        & pd.to_numeric(df["year"], errors="coerce").isin(YEARS)
    ].copy()
    if len(rows) != len(YEARS):
        raise RuntimeError(f"040_36 PPO rows incomplete: expected {len(YEARS)}, got {len(rows)}")
    return rows.sort_values("year").reset_index(drop=True)


def action_index_for(config: dict[str, Any], irrigation: float, nitrogen: float) -> int:
    return replay04007.action_index_for(config, float(irrigation), float(nitrogen))


def scalar(x: Any, default: float = np.nan) -> float:
    return base032.scalar(x, default)


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def replay_ppo_actions_with_snapshot(
    config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    actions_by_dap: dict[int, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_out = direct_ppo.OUTPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    direct_ppo.OUTPUT_ROOT = OUT
    env = None
    try:
        env = base032.make_env(config, env_config, STATION, int(year), 0, f"{STATION}_{year}_040_38_ppo_replay", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            i, n = actions_by_dap.get(dap, (0.0, 0.0))
            action = action_index_for(config, i, n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base032.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "scenario": "rl_candidate",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "safe_action_amir": float(i),
                    "safe_action_anfer": float(n),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1

        daily = pd.DataFrame(records)
        daily_path = DAILY_DIR / f"SYA_{year}_04036_ckpt100k_replay_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").ffill().iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
        snapshot_src = siteppo.snapshot_from_env(env)
        snapshot = SNAP_DIR / STATION / str(year) / "rl_candidate_04036_ckpt100k"
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_src, snapshot)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        summary = {
            "station_code": STATION,
            "site": SITE,
            "year": int(year),
            "scenario": "rl_candidate",
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": float(pd.to_numeric(daily["topwt"], errors="coerce").ffill().iloc[-1]),
            "actual_irrigation_mm": total_i,
            "actual_nitrogen_kg_ha": total_n,
            "requested_irrigation_mm": total_i,
            "requested_nitrogen_kg_ha": total_n,
            "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
            "source_status": "040_38_fixed_replay_of_040_36_daily_actions",
            "recorded_farmer_status": "",
            "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
            **metrics,
            **stress_summary(daily),
        }
        return daily, summary
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input
        direct_ppo.OUTPUT_ROOT = old_out


def build_ppo_metric_rows(config: dict[str, Any], env_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in ppo_eval_rows().iterrows():
        year = int(row["year"])
        actions = replay04007.original_actions_by_dap(str(row["daily_csv_path"]))
        daily, summary = replay_ppo_actions_with_snapshot(config, env_config, year, actions)
        daily_frames.append(daily)
        summary_rows.append(summary)
    return pd.concat(daily_frames, ignore_index=True), pd.DataFrame(summary_rows)


def load_baseline_rows() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    rows = df[
        df["station_code"].astype(str).eq(STATION)
        & pd.to_numeric(df["year"], errors="coerce").isin(YEARS)
        & df["scenario"].astype(str).isin(SCENARIO_ORDER[:-1])
    ].copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype(int)
    rows["grain_yield_kg_ha"] = pd.to_numeric(rows["grain_yield_kg_ha"], errors="coerce")
    rows["actual_irrigation_mm"] = pd.to_numeric(rows["actual_irrigation_mm"], errors="coerce")
    rows["actual_nitrogen_kg_ha"] = pd.to_numeric(rows["actual_nitrogen_kg_ha"], errors="coerce")
    rows["WP_ET_kg_m3"] = pd.to_numeric(rows["WP_ET_kg_m3"], errors="coerce")
    rows["PFP_N_kg_kg"] = pd.to_numeric(rows["PFP_N_kg_kg"], errors="coerce")
    return rows


def compute_gaps(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in combined.groupby("year"):
        ppo = group[group["scenario"].eq("rl_candidate")]
        if ppo.empty:
            continue
        ppo = ppo.iloc[0]
        bases = group[~group["scenario"].eq("rl_candidate")]
        row: dict[str, Any] = {"station_code": STATION, "site": SITE, "year": int(year)}
        for label, col in [
            ("yield", "grain_yield_kg_ha"),
            ("wp_et", "WP_ET_kg_m3"),
            ("pfp_n", "PFP_N_kg_kg"),
        ]:
            vals = pd.to_numeric(bases[col], errors="coerce")
            best = float(vals.max()) if vals.notna().any() else np.nan
            best_s = str(bases.loc[vals.idxmax(), "scenario"]) if vals.notna().any() else ""
            ppo_v = pd.to_numeric(pd.Series([ppo[col]]), errors="coerce").iloc[0]
            row[f"ppo_{label}"] = float(ppo_v) if pd.notna(ppo_v) else np.nan
            row[f"four_max_{label}"] = best
            row[f"four_max_{label}_scenario"] = best_s
            row[f"gap_{label}_vs_four_max"] = (float(ppo_v) - best) if pd.notna(ppo_v) and np.isfinite(best) else np.nan
            row[f"win_{label}_vs_four_max"] = bool(pd.notna(ppo_v) and np.isfinite(best) and float(ppo_v) > best)
        row["any_metric_win_four"] = bool(
            row["win_yield_vs_four_max"] or row["win_wp_et_vs_four_max"] or row["win_pfp_n_vs_four_max"]
        )
        row["ppo_irrigation"] = float(ppo["actual_irrigation_mm"])
        row["ppo_nitrogen"] = float(ppo["actual_nitrogen_kg_ha"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_bars(combined: pd.DataFrame) -> list[Path]:
    metrics = [
        ("grain_yield_kg_ha", "Grain yield (kg/ha)", "Yield"),
        ("WP_ET_kg_m3", "WP_ET (kg/m³)", "Water productivity"),
        ("PFP_N_kg_kg", "PFP_N (kg/kg)", "Nitrogen productivity"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    years = YEARS
    x = np.arange(len(years))
    width = 0.16
    offsets = np.linspace(-2, 2, len(SCENARIO_ORDER)) * width
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for scenario, offset in zip(SCENARIO_ORDER, offsets):
            vals = []
            for year in years:
                sub = combined[(combined["year"].eq(year)) & (combined["scenario"].eq(scenario))]
                vals.append(pd.to_numeric(sub[col], errors="coerce").iloc[0] if len(sub) else np.nan)
            ax.bar(x + offset, vals, width, label=SCENARIO_LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(y) for y in years], rotation=0)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.28))
    fig.suptitle("SYA lowIC validation years: five-scenario metric comparison (PPO 040_36 ckpt100k)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = FIG_DIR / "040_38_sya_lowIC_04036_ckpt100k_validation_five_scenario_metrics.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def plot_resource_bars(combined: pd.DataFrame) -> list[Path]:
    metrics = [
        ("actual_irrigation_mm", "Irrigation (mm)", "Total irrigation"),
        ("actual_nitrogen_kg_ha", "Nitrogen (kg/ha)", "Total nitrogen"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    years = YEARS
    x = np.arange(len(years))
    width = 0.16
    offsets = np.linspace(-2, 2, len(SCENARIO_ORDER)) * width
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for scenario, offset in zip(SCENARIO_ORDER, offsets):
            vals = []
            for year in years:
                sub = combined[(combined["year"].eq(year)) & (combined["scenario"].eq(scenario))]
                vals.append(pd.to_numeric(sub[col], errors="coerce").iloc[0] if len(sub) else np.nan)
            ax.bar(x + offset, vals, width, label=SCENARIO_LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(y) for y in years], rotation=0)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.25))
    fig.suptitle("SYA lowIC validation years: five-scenario resource input comparison", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = FIG_DIR / "040_38_sya_lowIC_04036_ckpt100k_validation_five_scenario_resources.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def write_record(combined: pd.DataFrame, gaps: pd.DataFrame, figures: list[Path], failures: list[dict[str, Any]]) -> None:
    win_counts = {
        "years": int(len(gaps)),
        "any_metric_win": int(gaps["any_metric_win_four"].fillna(False).sum()) if "any_metric_win_four" in gaps else 0,
        "yield_win": int(gaps["win_yield_vs_four_max"].fillna(False).sum()) if "win_yield_vs_four_max" in gaps else 0,
        "wp_et_win": int(gaps["win_wp_et_vs_four_max"].fillna(False).sum()) if "win_wp_et_vs_four_max" in gaps else 0,
        "pfp_n_win": int(gaps["win_pfp_n_vs_four_max"].fillna(False).sum()) if "win_pfp_n_vs_four_max" in gaps else 0,
    }
    lines = [
        "# 040_38 SYA lowIC 040_36 checkpoint100k 验证年份五情景指标柱状图记录",
        "",
        "## 结论先说",
        "",
        f"- 验证年份：{win_counts['years']} 年（2014–2023）。",
        f"- PPO 至少一项指标超过四基线最高值：{win_counts['any_metric_win']} / {win_counts['years']} 年。",
        f"- 产量超过四基线最高值：{win_counts['yield_win']} / {win_counts['years']} 年。",
        f"- WP_ET 超过四基线最高值：{win_counts['wp_et_win']} / {win_counts['years']} 年。",
        f"- PFP_N 超过四基线最高值：{win_counts['pfp_n_win']} / {win_counts['years']} 年。",
        "- 本任务不训练；PPO 指标由 040_36 checkpoint100000 动作序列固定重放 DSSAT 后从 Summary.OUT 补齐。",
        "",
        "## 指标差距表",
        "",
        md_table(gaps, 40),
        "",
        "## 五情景终值表",
        "",
        md_table(combined[[
            "year",
            "scenario",
            "grain_yield_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "source_status",
        ]], 80),
        "",
        "## 输出图",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures if p.suffix.lower() == ".png"],
    ]
    if failures:
        lines.extend(["", "## 失败记录", "", md_table(pd.DataFrame(failures), 20)])
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not PPO_EVAL.exists():
        raise FileNotFoundError(PPO_EVAL)
    if not BASELINE_SUMMARY.exists():
        raise FileNotFoundError(BASELINE_SUMMARY)

    config, env_config, selection = load_config_and_env_config()
    selection.to_csv(OUT / "configs" / "040_38_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_38_resolved_env_config.yaml")

    failures: list[dict[str, Any]] = []
    try:
        ppo_daily, ppo_summary = build_ppo_metric_rows(config, env_config)
    except Exception:
        failures.append({"stage": "ppo_fixed_replay", "traceback": traceback.format_exc()[-4000:]})
        raise
    baseline = load_baseline_rows()
    combined = pd.concat([baseline, ppo_summary], ignore_index=True, sort=False)
    combined["scenario"] = pd.Categorical(combined["scenario"], categories=SCENARIO_ORDER, ordered=True)
    combined = combined.sort_values(["year", "scenario"]).reset_index(drop=True)
    gaps = compute_gaps(combined)

    ppo_daily.to_csv(TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_ppo_fixed_replay_daily.csv", index=False, encoding="utf-8-sig")
    ppo_summary.to_csv(TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_ppo_fixed_replay_summary.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_five_scenario_metric_summary.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_metric_gaps_vs_four_max.csv", index=False, encoding="utf-8-sig")

    figures = []
    figures.extend(plot_metric_bars(combined))
    figures.extend(plot_resource_bars(combined))
    write_record(combined, gaps, figures, failures)

    result = {
        "task": TASK,
        "years": YEARS,
        "checkpoint": CHECKPOINT,
        "combined_summary": (TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_five_scenario_metric_summary.csv").relative_to(ROOT).as_posix(),
        "gaps": (TAB_DIR / "040_38_sya_lowIC_04036_ckpt100k_metric_gaps_vs_four_max.csv").relative_to(ROOT).as_posix(),
        "figures": [p.relative_to(ROOT).as_posix() for p in figures],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_38_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

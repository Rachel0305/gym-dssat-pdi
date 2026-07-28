from __future__ import annotations

import json
import math
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
import run_lc_multiyear_75k_future_year_transfer_032_12 as transfer_03212
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as train_03210
import run_missing_dssat_auto_completion_for_03134_031_36 as auto_03136
import run_missing_four_baseline_completion_for_03134_031_35 as four_03135
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from build_relaxed_success_five_scenario_daily_evidence_027_05 import (
    COLORS,
    LABELS,
    SCENARIOS,
    STYLES,
    Case,
    build_case,
)
from ppo_action_safety import normalize_action


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild.md"
MODEL = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "models" / "LCA" / "LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
OUT = ROOT / "benchmark_results" / "032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild"
FIG = OUT / "figures"
TABLES = OUT / "tables"
EVAL = OUT / "evaluation"
SNAP = OUT / "snapshots" / "LCA"
LOCAL_LC2010_BASELINE = SNAP / "2010"
DOC = ROOT / "docs" / "032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild_record.md"

STATION = "LCA"
SITE = "LC"
SEED = 0
CHECKPOINT = 75000
YEARS = list(range(2005, 2021))

BASELINE_03135 = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "snapshots" / "LCA"
AUTO_03136 = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "snapshots" / "LCA"
AUTO_03218 = ROOT / "benchmark_results" / "032_18_lc_missing_dssat_auto_daily_completion" / "snapshots" / "LCA"
LC2010_02705 = {
    "null": ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11" / "runs" / "2010" / "null" / "pdi_tmp_snapshot",
    "recorded_farmer": ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11" / "runs" / "2010" / "recorded" / "pdi_tmp_snapshot",
    "dssat_auto": ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11" / "runs" / "2010" / "dssat_auto" / "pdi_tmp_snapshot",
    "official_extension_expert": ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "LC2010" / "extension_expert_fixed_dap" / "pdi_tmp_snapshot_eval",
}


@dataclass(frozen=True)
class SnapshotSource:
    scenario: str
    path: Path
    source: str


def ensure_dirs() -> None:
    for path in [FIG, TABLES, EVAL, SNAP, OUT / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
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


def copy_snapshot(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def load_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = YEARS[0]
    pool = pd.read_csv(POOL)
    selection = pool[(pool["station_code"].eq(STATION)) & (pool["year"].astype(int).isin(YEARS))].copy()
    found = sorted(selection["year"].astype(int).unique().tolist())
    if found != YEARS:
        raise RuntimeError(f"LC 年份不完整：期望 {YEARS}，实际 {found}")
    selection["selected_for_train"] = selection["year"].astype(int).isin(range(2005, 2011))
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "032_20_consistent_snapshot_rebuild_only"
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(cfg, selection)
    return cfg, env_config, selection.sort_values(["station_code", "year"]).reset_index(drop=True)


def ppo_snapshot_path(year: int) -> Path:
    return SNAP / str(year) / "rl_candidate"


def evaluate_ppo_snapshot(config: dict[str, Any], env_config: dict[str, Any], year: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    dst = ppo_snapshot_path(year)
    required = [dst / name for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]]
    if all(path.exists() for path in required):
        return {"year": year, "scenario": "rl_candidate", "status": "ok_existing", "snapshot_path": dst.relative_to(ROOT).as_posix()}
    if not MODEL.exists():
        return {"year": year, "scenario": "rl_candidate", "status": "missing_model", "snapshot_path": ""}

    env = None
    try:
        model = MaskablePPO.load(str(MODEL), device="cpu")
        env = base.make_env(config, env_config, STATION, year, SEED, f"{STATION}_{year}_032_20_frozen75k_snapshot_eval", evaluation=True)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < int(config["runtime"]["max_steps"]):
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        snapshot = siteppo.snapshot_from_env(env)
        copy_snapshot(snapshot, dst)
        return {
            "year": year,
            "scenario": "rl_candidate",
            "status": "ok",
            "steps": steps,
            "snapshot_path": dst.relative_to(ROOT).as_posix(),
        }
    except Exception:
        return {
            "year": year,
            "scenario": "rl_candidate",
            "status": "failed",
            "error": traceback.format_exc()[-4000:],
            "snapshot_path": dst.relative_to(ROOT).as_posix(),
        }
    finally:
        if env is not None:
            env.close()


def load_four_baseline_meta() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(ROOT / "experiments" / "ppo_observed_years" / "config_031_35_missing_four_baseline_completion_for_03134.yaml")
    return json.loads(json.dumps(cfg))


def evaluate_fixed_baseline_snapshot(
    config: dict[str, Any],
    env_config: dict[str, Any],
    meta: dict[str, Any],
    scenario: str,
    schedule: dict[int, dict[str, float]],
) -> dict[str, Any]:
    dst = LOCAL_LC2010_BASELINE / scenario
    required = [dst / name for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]]
    if all(path.exists() for path in required):
        return {"year": 2010, "scenario": scenario, "status": "ok_existing", "snapshot_path": dst.relative_to(ROOT).as_posix()}

    env = None
    try:
        env = direct_ppo.make_base_env(env_config, STATION, 2010, SEED, f"{STATION}_2010_032_20_{scenario}", evaluation=True)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < int(meta.get("max_steps", config["runtime"]["max_steps"])):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = base.scalar(latest.get("dap", steps + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else steps + 1
            action_real = schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            action = normalize_action(
                env.formator.action_names,
                env.formator.action_space_dict,
                {"amir": float(action_real.get("amir", 0.0)), "anfer": float(action_real.get("anfer", 0.0))},
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        snapshot = siteppo.snapshot_from_env(env)
        copy_snapshot(snapshot, dst)
        return {"year": 2010, "scenario": scenario, "status": "ok", "steps": steps, "snapshot_path": dst.relative_to(ROOT).as_posix()}
    except Exception:
        return {
            "year": 2010,
            "scenario": scenario,
            "status": "failed",
            "error": traceback.format_exc()[-4000:],
            "snapshot_path": dst.relative_to(ROOT).as_posix(),
        }
    finally:
        if env is not None:
            env.close()


def evaluate_dssat_auto_snapshot(config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    scenario = "dssat_auto"
    dst = LOCAL_LC2010_BASELINE / scenario
    required = [dst / name for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]]
    if all(path.exists() for path in required):
        return {"year": 2010, "scenario": scenario, "status": "ok_existing", "snapshot_path": dst.relative_to(ROOT).as_posix()}

    env = None
    try:
        meta = {
            "seed": SEED,
            "max_steps": int(config["runtime"]["max_steps"]),
        }
        env = auto_03136.make_auto_env(env_config, STATION, 2010, SEED)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < int(meta["max_steps"]):
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        snapshot = siteppo.snapshot_from_env(env)
        copy_snapshot(snapshot, dst)
        return {"year": 2010, "scenario": scenario, "status": "ok", "steps": steps, "snapshot_path": dst.relative_to(ROOT).as_posix()}
    except Exception:
        return {
            "year": 2010,
            "scenario": scenario,
            "status": "failed",
            "error": traceback.format_exc()[-4000:],
            "snapshot_path": dst.relative_to(ROOT).as_posix(),
        }
    finally:
        if env is not None:
            env.close()


def ensure_lc2010_local_baseline_snapshots(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    meta = load_four_baseline_meta()
    templates = four_03135.recorded_templates(meta)
    schedule_map = {
        "null": {},
        "recorded_farmer": templates.get(SITE, {}),
        "official_extension_expert": four_03135.expert_schedule(SITE, meta),
    }
    rows = [
        evaluate_fixed_baseline_snapshot(config, env_config, meta, scenario, schedule)
        for scenario, schedule in schedule_map.items()
    ]
    rows.append(evaluate_dssat_auto_snapshot(config, env_config))
    return pd.DataFrame(rows)


def scenario_source(year: int, scenario: str) -> SnapshotSource:
    if year == 2010 and scenario != "rl_candidate":
        local = LOCAL_LC2010_BASELINE / scenario
        if local.exists():
            return SnapshotSource(scenario, local, "032_20_local_lc2010_regenerated_baseline_snapshot")
        if scenario in LC2010_02705:
            return SnapshotSource(scenario, LC2010_02705[scenario], "027_05_lc2010_archived_snapshot_fallback")
    if scenario == "rl_candidate":
        return SnapshotSource(scenario, ppo_snapshot_path(year), "032_20_frozen_ppo_reevaluation_snapshot")
    if scenario == "dssat_auto":
        p = AUTO_03218 / str(year) / "dssat_auto"
        if p.exists():
            return SnapshotSource(scenario, p, "032_18_completed_dssat_auto_snapshot")
        p = AUTO_03136 / str(year) / "dssat_auto"
        if p.exists():
            return SnapshotSource(scenario, p, "031_36_completed_dssat_auto_snapshot")
        p = BASELINE_03135 / str(year) / "dssat_auto"
        return SnapshotSource(scenario, p, "031_35_dssat_auto_snapshot_fallback")
    if scenario == "recorded_farmer":
        return SnapshotSource(scenario, BASELINE_03135 / str(year) / "recorded_farmer_template_02705", "031_35_recorded_farmer_template_02705_snapshot")
    return SnapshotSource(scenario, BASELINE_03135 / str(year) / scenario, f"031_35_{scenario}_snapshot")


def build_year_case(year: int) -> tuple[Case, list[dict[str, Any]]]:
    sources = {scenario: scenario_source(year, scenario) for scenario in SCENARIOS}
    rows = []
    for scenario, src in sources.items():
        required = [src.path / name for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]]
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "source": src.source,
                "snapshot_path": src.path.relative_to(ROOT).as_posix() if src.path.exists() else str(src.path),
                "snapshot_exists": bool(src.path.exists()),
                "required_files_complete": bool(all(path.exists() for path in required)),
            }
        )
    case = Case(
        SITE,
        STATION,
        year,
        SEED,
        CHECKPOINT,
        MODEL,
        {scenario: src.path for scenario, src in sources.items()},
        PROMPT,
        "LC 75k stress-aware free-timing MaskablePPO frozen checkpoint; daily process rebuilt from DSSAT snapshots.",
    )
    return case, rows


def plot_daily_03220(daily: pd.DataFrame, year: int) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], weather["rainfall_mm"], color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    tmax = weather["tmax_c"].where(weather["temperature_source_qc"].eq("pass"))
    tmin = weather["tmin_c"].where(weather["temperature_source_qc"].eq("pass"))
    ax2.plot(weather["dap"], tmax, color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], tmin, color="#686868", lw=1.3, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=3, fontsize=8, loc="upper right")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        label = LABELS[scenario] if scenario != "rl_candidate" else "MaskablePPO candidate"
        color = COLORS[scenario]
        style = STYLES[scenario]
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        for event_ax, column in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            ev = sub[pd.to_numeric(sub[column], errors="coerce").gt(0)]
            marker = "D" if scenario == "rl_candidate" else "o"
            event_ax.vlines(ev["dap"], 0, ev[column], color=color, lw=2, alpha=0.88)
            event_ax.scatter(ev["dap"], ev[column], color=color, marker=marker, s=24, label=label)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.4)
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.9, alpha=0.42)
        axes[3, 1].plot(sub["dap"], sub["cumulative_common_reward"], color=color, ls=style, lw=1.4, label=label)

    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 1], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Cumulative common reward", "kg/ha-equivalent"),
    ]
    for sub_ax, title, ylabel in titles:
        sub_ax.set_title(title, loc="left", fontweight="bold")
        sub_ax.set_ylabel(ylabel)
        sub_ax.grid(color="#E8E8E8", linewidth=0.65)
    for sub_ax in axes[2, :]:
        handles, names = sub_ax.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        sub_ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"LC{year} MaskablePPO five-scenario daily process (snapshot-derived)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    base_path = FIG / f"032_20_lc{year}_75k_ppo_five_scenario_daily_snapshot_derived"
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def qa_year(daily: pd.DataFrame, summary: pd.DataFrame, year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dup = daily.duplicated(["scenario", "date", "dap"], keep=False)
    rows.append({"year": year, "check": "no_duplicate_scenario_date_dap", "passed": bool(not dup.any()), "value": int(dup.sum())})
    rain = summary.groupby("scenario")["rain_total_mm"].first()
    spread = float(rain.max() - rain.min()) if len(rain) else math.nan
    rows.append({"year": year, "check": "five_scenario_rain_total_invariant", "passed": bool(abs(spread) <= 1e-6), "value": spread})
    for scenario, group in daily.groupby("scenario"):
        srow = summary[summary["scenario"].eq(scenario)].iloc[0]
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "check": "daily_irrigation_sum_matches_summary",
                "passed": bool(abs(float(group["irrigation_executed_mm"].sum()) - float(srow["irrigation_event_total_mm"])) <= 1e-9),
                "value": float(group["irrigation_executed_mm"].sum()),
            }
        )
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "check": "daily_nitrogen_sum_matches_summary",
                "passed": bool(abs(float(group["nitrogen_executed_kg_ha"].sum()) - float(srow["nitrogen_event_total_kg_ha"])) <= 1e-9),
                "value": float(group["nitrogen_executed_kg_ha"].sum()),
            }
        )
    return rows


def write_record(
    eval_df: pd.DataFrame,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    qa: pd.DataFrame,
    fig_index: pd.DataFrame,
) -> None:
    failed_qa = qa[~qa["passed"].fillna(False)].copy()
    complete_years = (
        manifest.groupby("year")["required_files_complete"].all().reset_index(name="complete")
    )
    endpoint_cols = [
        "year",
        "scenario",
        "final_grain_kg_ha",
        "irrigation_event_total_mm",
        "nitrogen_event_total_kg_ha",
        "wp_et_kg_m3",
        "pfp_n_kg_kg",
        "max_water_stress_wspd",
        "max_nitrogen_stress_nstd",
        "common_reward_total",
    ]
    lines = [
        "# 032_20 LC 75k PPO 一致 DSSAT snapshot 日过程重建记录",
        "",
        "## 结论先说",
        "",
        f"- 本轮没有训练、没有重新选择 checkpoint；只对已冻结 `{CHECKPOINT}` 步模型做确定性评估并保存 snapshot。",
        f"- LC2005-LC2020 共 `{len(YEARS)}` 年；五情景来源完整年份数：`{int(complete_years['complete'].sum())}/{len(YEARS)}`。",
        f"- PPO 候选 snapshot 生成/复用状态：`{eval_df['status'].value_counts().to_dict()}`。",
        f"- QA 未通过条目数：`{len(failed_qa)}`。",
        "",
        "## 为什么做这一轮",
        "",
        "032_19 发现旧日值表存在 PPO 候选重复 date-DAP 行和五情景降雨总量不一致问题，因此旧图中的 WSPD/NSTD 过程线不能直接用于解释 PPO 决策。本轮从 DSSAT 原始输出重新解析日过程，目的是先修正证据口径，而不是改善模型结果。",
        "",
        "## 模型与边界",
        "",
        f"- 模型：`{MODEL.relative_to(ROOT).as_posix()}`。",
        "- 算法：LC 多年自由时序 stress-aware MaskablePPO。",
        "- 训练年份：LC2005-LC2010；本轮只做冻结评估。",
        "- 验证/绘图年份：LC2005-LC2020。",
        "- 不调用 `learn()`；不修改原始输入；不覆盖 032_17。",
        "",
        "## Snapshot 来源清单",
        "",
        md_table(manifest[["year", "scenario", "source", "required_files_complete", "snapshot_path"]], max_rows=120),
        "",
        "## LC2010 特别说明",
        "",
        "第一次重建时，LC2010 使用 027_05 历史 snapshot 造成五情景降雨总量不一致，因此本轮在 032_20 输出目录内重新生成 LC2010 的 null、recorded_farmer、dssat_auto、official_extension_expert 四条基线 snapshot。重跑过程中历史 recorded/expert 的部分单次灌水量超过当前 gym-DSSAT action 上限并被裁剪；因此 LC2010 图表反映的是本轮当前环境约束下 DSSAT 实际执行后的基线过程。这个边界只影响 LC2010 的本轮过程图解释，不回写或覆盖旧 027/031 结果。",
        "",
        "## 终值摘要",
        "",
        md_table(summary[[c for c in endpoint_cols if c in summary.columns]], max_rows=120),
        "",
        "## QA 摘要",
        "",
        md_table(qa.groupby("check", as_index=False).agg(total=("passed", "size"), passed=("passed", "sum")), max_rows=40),
        "",
        "## QA 未通过项",
        "",
        md_table(failed_qa, max_rows=80),
        "",
        "## 图件索引",
        "",
        md_table(fig_index, max_rows=80),
        "",
        "## 解释边界",
        "",
        "- 如果某个年份 null 情景没有 WSPD，并不自动说明图错；DSSAT 的 WSPD 是由土壤水、根系、蒸散需求等共同决定，不等同于降雨少。",
        "- 本轮修复的是日值来源一致性。若重建图里仍出现 PPO 独有胁迫，需要再做动作删除/降档反事实审计，不能仅凭图直接下因果结论。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    config, env_config, selection = load_config()
    selection.to_csv(OUT / "configs" / "032_20_lc_year_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(env_config.get("observed_years", {}).get(STATION, [])).to_csv(
        OUT / "configs" / "032_20_resolved_lc_observed_years.csv", index=False, encoding="utf-8-sig"
    )

    eval_rows = [evaluate_ppo_snapshot(config, env_config, year) for year in YEARS]
    lc2010_rows = ensure_lc2010_local_baseline_snapshots(config, env_config)
    if not lc2010_rows.empty:
        lc2010_rows.to_csv(EVAL / "032_20_lc2010_local_baseline_snapshot_eval_status.csv", index=False, encoding="utf-8-sig")
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(EVAL / "032_20_frozen_ppo_snapshot_eval_status.csv", index=False, encoding="utf-8-sig")

    all_daily: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    fig_rows: list[dict[str, Any]] = []
    for year in YEARS:
        case, source_rows = build_year_case(year)
        manifest_rows.extend(source_rows)
        if not all(row["required_files_complete"] for row in source_rows):
            continue
        daily, summary, checks = build_case(case, algorithm="MaskablePPO")
        daily.to_csv(TABLES / f"032_20_lc{year}_five_scenario_daily_snapshot_derived.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(TABLES / f"032_20_lc{year}_five_scenario_summary_snapshot_derived.csv", index=False, encoding="utf-8-sig")
        paths = plot_daily_03220(daily, year)
        all_daily.append(daily)
        all_summary.append(summary)
        qa_rows.extend(qa_year(daily, summary, year))
        qa_rows.extend({"year": year, **row} for row in checks)
        fig_rows.append(
            {
                "year": year,
                "png": paths[0].relative_to(ROOT).as_posix(),
                "svg": paths[1].relative_to(ROOT).as_posix(),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    qa = pd.DataFrame(qa_rows)
    fig_index = pd.DataFrame(fig_rows)
    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    daily_all = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    manifest.to_csv(OUT / "032_20_snapshot_source_manifest.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(OUT / "032_20_daily_rebuild_qa.csv", index=False, encoding="utf-8-sig")
    fig_index.to_csv(OUT / "032_20_figure_index.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT / "032_20_lc2005_2020_five_scenario_summary_snapshot_derived.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT / "032_20_lc2005_2020_five_scenario_daily_snapshot_derived.csv", index=False, encoding="utf-8-sig")
    write_record(eval_df, manifest, summary_all, qa, fig_index)
    result = {
        "task": "032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild",
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "figures": len(fig_index),
        "qa_failed": int((~qa["passed"].fillna(False)).sum()) if not qa.empty else None,
        "summary_csv": (OUT / "032_20_lc2005_2020_five_scenario_summary_snapshot_derived.csv").relative_to(ROOT).as_posix(),
        "daily_csv": (OUT / "032_20_lc2005_2020_five_scenario_daily_snapshot_derived.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / "032_20_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

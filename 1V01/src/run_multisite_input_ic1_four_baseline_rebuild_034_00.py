from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_extension_expert_baseline_018_03 as extension
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as stress_ppo
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import (
    MULTISITE_INPUT_ROOT,
    build_env_args,
    check_rendered_input,
    ensure_project_on_path,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import set_management_for_treatment


TASK_ID = "034_00"
OUT = ROOT / "benchmark_results" / "034_00_multisite_input_ic1_four_baseline_rebuild"
DOC = ROOT / "docs" / "034_00_multisite_input_ic1_four_baseline_rebuild_record.md"
PROMPT = ROOT / "prompts" / "034_00_multisite_input_ic1_four_baseline_rebuild.md"
BASE_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
SPLIT_CSV = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "033_04_available_weather_half_split_years.csv"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
RECORDED_TEMPLATE_DAILY = ROOT / "benchmark_results" / "027_05" / "027_05_daily_values.csv"

STATIONS = ["FQA", "HLA", "LCA", "SYA", "YCA"]
STATION_TO_SITE = {"FQA": "FQ", "HLA": "HLA", "LCA": "LC", "SYA": "SY", "YCA": "YC"}
SITE_TO_REGION = {
    "HLA": "northeast_greatwall_spring_maize",
    "SY": "northeast_greatwall_spring_maize",
    "FQ": "huanghuai_fenwei_summer_maize",
    "LC": "huanghuai_fenwei_summer_maize",
    "YC": "huanghuai_fenwei_summer_maize",
}
SCENARIOS = ["null", "recorded_farmer_template", "official_extension_expert", "dssat_auto"]
SEED = 0
MAX_STEPS = 260


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "daily_outputs", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


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


def load_selection(mode: str) -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"缺少 033_04 年份清单：{SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV, keep_default_na=False)
    df = df[df["station_code"].isin(STATIONS)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.sort_values(["station_code", "year"]).reset_index(drop=True)
    if mode == "smoke":
        rows = []
        for station, group in df.groupby("station_code", sort=True):
            rows.append(group.sort_values("year").iloc[0])
        df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def build_env_config(run_config: dict[str, Any], selected_years: pd.DataFrame) -> dict[str, Any]:
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool.merge(selected_years[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selected["selected_for_train"] = False
    selected["selected_for_eval"] = True
    selected["selection_reason"] = "034_00_multisite_input_ic1_four_baseline_rebuild"
    env_config = direct_ppo.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["seed"] = SEED
    env_config["runtime"]["max_steps"] = MAX_STEPS
    return env_config


def recorded_template_schedules() -> dict[str, dict[int, dict[str, float]]]:
    if not RECORDED_TEMPLATE_DAILY.exists():
        return {}
    df = pd.read_csv(RECORDED_TEMPLATE_DAILY, keep_default_na=False)
    if "scenario" in df.columns:
        df = df[df["scenario"].astype(str).eq("recorded_farmer")].copy()
    out: dict[str, dict[int, dict[str, float]]] = {}
    for site, group in df.groupby("site"):
        schedule: dict[int, dict[str, float]] = {}
        for _, row in group.iterrows():
            dap = pd.to_numeric(row.get("dap", np.nan), errors="coerce")
            if pd.isna(dap):
                continue
            irrigation = pd.to_numeric(row.get("irrigation_executed_mm", 0.0), errors="coerce")
            nitrogen = pd.to_numeric(row.get("nitrogen_executed_kg_ha", 0.0), errors="coerce")
            i = 0.0 if pd.isna(irrigation) else float(irrigation)
            n = 0.0 if pd.isna(nitrogen) else float(nitrogen)
            if abs(i) < 1e-9 and abs(n) < 1e-9:
                continue
            key = max(1, int(round(float(dap))))
            schedule.setdefault(key, {"amir": 0.0, "anfer": 0.0})
            schedule[key]["amir"] += i
            schedule[key]["anfer"] += n
        out[str(site)] = schedule
    return out


def expert_schedule(site: str) -> dict[int, dict[str, float]]:
    region = SITE_TO_REGION[site]
    sched = extension.build_region_schedule()
    sub = sched[sched["region"].eq(region)].copy()
    sub.insert(0, "site", site)
    sub.insert(1, "station", site)
    sub.insert(2, "year", 0)
    return extension.split_irrigation_events(sub)


def schedule_for(scenario: str, site: str, recorded_templates: dict[str, dict[int, dict[str, float]]]) -> dict[int, dict[str, float]]:
    if scenario == "null":
        return {}
    if scenario == "recorded_farmer_template":
        return recorded_templates.get(site, {})
    if scenario == "official_extension_expert":
        return expert_schedule(site)
    raise ValueError(f"不是固定动作情景：{scenario}")


def automatic_management_block(year: int) -> str:
    yy001 = f"{int(year) % 100:02d}001"
    return (
        "\n@  AUTOMATIC MANAGEMENT\n"
        "@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n"
        f" 1 PL          {yy001} {yy001}    40   100    30    40    10\n"
        "@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n"
        " 1 IR             30    50   100 GS000 IR001    10     1\n"
        "@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n"
        " 1 NI             30    50    25 FE001 GS000\n"
        "@N RESIDUES    RIPCN RTIME RIDEP\n"
        " 1 RE            100     1    20\n"
        "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
        " 1 HA              0 01001   100     0\n"
    )


def ensure_automatic_management(text: str, year: int) -> str:
    if "@  AUTOMATIC MANAGEMENT" in text:
        return text
    block = automatic_management_block(year)
    marker = "@N OUTPUTS"
    idx = text.find(marker)
    if idx < 0:
        return text.rstrip() + "\n" + block
    next_star = text.find("\n*", idx + 1)
    if next_star < 0:
        return text.rstrip() + "\n" + block
    return text[:next_star].rstrip() + "\n" + block + "\n" + text[next_star:]


def set_auto_treatment_one(text: str, year: int) -> str:
    text = ensure_automatic_management(text, year)
    try:
        return set_management_for_treatment(text, 1, "A", "A")
    except Exception:
        lines: list[str] = []
        changed = False
        in_management = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
                in_management = True
                lines.append(line)
                continue
            if in_management and re.match(r"^\s*1\s+MA\b", line):
                parts = line.split()
                if len(parts) >= 7:
                    parts[4] = "A"
                    parts[5] = "A"
                    lines.append(f"{int(parts[0]):2d} MA              R     {parts[4]}     {parts[5]}     R     M")
                    changed = True
                    continue
            if in_management and (stripped.startswith("@") or stripped.startswith("*")):
                in_management = False
            lines.append(line)
        if not changed:
            raise
        return "\n".join(lines) + "\n"


def yyddd_from_dap(planting_date: str, dap: int) -> str:
    planting = pd.Timestamp(planting_date)
    date = planting + pd.Timedelta(days=max(int(dap) - 1, 0))
    return f"{date.year % 100:02d}{date.dayofyear:03d}"


def replace_static_application_rows(text: str, schedule: dict[int, dict[str, float]], planting_date: str, year: int) -> str:
    irrigation_rows: list[str] = []
    fertilizer_rows: list[str] = []
    i_idx = 1
    f_idx = 1
    for dap, action in sorted(schedule.items()):
        irrigation = float(action.get("amir", 0.0))
        nitrogen = float(action.get("anfer", 0.0))
        date = yyddd_from_dap(planting_date, int(dap))
        if irrigation > 1e-9:
            irrigation_rows.append(f"{i_idx:2d} {date} IR001 {irrigation:5.1f}")
            i_idx += 1
        if nitrogen > 1e-9:
            fertilizer_rows.append(f"{f_idx:2d} {date} FE005 AP002     5 {nitrogen:6.1f}   -99   -99   -99   -99   -99 {year}")
            f_idx += 1
    if not irrigation_rows:
        irrigation_rows = [f" 1 {yyddd_from_dap(planting_date, 1)} IR001     0"]
    if not fertilizer_rows:
        fertilizer_rows = [f" 1 {yyddd_from_dap(planting_date, 1)} FE005 AP002     5     0   -99   -99   -99   -99   -99 {year}"]

    lines = text.splitlines()
    output: list[str] = []
    mode: str | None = None
    inserted_i = False
    inserted_f = False
    for line in lines:
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            output.append(line)
            output.extend(irrigation_rows)
            mode = "I"
            inserted_i = True
            continue
        if line.startswith("@F FDATE"):
            output.append(line)
            output.extend(fertilizer_rows)
            mode = "F"
            inserted_f = True
            continue
        if mode is not None:
            if stripped.startswith("@") or stripped.startswith("*"):
                mode = None
            elif stripped == "":
                continue
            elif re.match(r"^\s*\d+\s+\d{5}\b", line):
                continue
        output.append(line)
    if not inserted_i:
        raise ValueError("渲染模板中缺少 @I IDATE 行，无法写入静态灌溉表")
    if not inserted_f:
        raise ValueError("渲染模板中缺少 @F FDATE 行，无法写入静态施肥表")
    return "\n".join(output) + "\n"


def make_base_env(
    env_config: dict[str, Any],
    station: str,
    year: int,
    run_tag: str,
    auto_management: bool = False,
    static_schedule: dict[int, dict[str, float]] | None = None,
):
    ensure_project_on_path()
    import gym
    from sb3_wrapper import GymDssatWrapper

    year_info = direct_ppo.find_year(env_config, station, int(year))
    env_args = build_env_args(
        station=station,
        year=int(year),
        planting_date=year_info["planting_date"],
        seed=SEED,
        config=env_config,
        run_tag=run_tag,
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    if auto_management:
        text = template.read_text(encoding="utf-8", errors="replace")
        template.write_text(set_auto_treatment_one(text, int(year)), encoding="utf-8")
    elif static_schedule is not None:
        text = template.read_text(encoding="utf-8", errors="replace")
        template.write_text(replace_static_application_rows(text, static_schedule, year_info["planting_date"], int(year)), encoding="utf-8")
    return GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped), env_args


def metrics_from_snapshot(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise ValueError(f"Summary.OUT 无可用 HWAM 行：{snapshot}")
    _, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    ircm = float(siteppo.num(row, "IRCM") or 0.0)
    nicm = float(siteppo.num(row, "NICM") or 0.0)
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    if etcp is None or float(etcp) <= 0:
        raise ValueError(f"ETCP 无效：{snapshot} ETCP={etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "actual_irrigation_mm": ircm,
        "actual_nitrogen_kg_ha": nicm,
        "etcp_mm": float(etcp),
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "summary_match_score": abs(float(siteppo.num(row, "HWAM") or final_yield) - float(final_yield)),
        "summary_row_index": int(-neg_idx),
    }


def weather_row(weather: pd.DataFrame, station: str, date: pd.Timestamp) -> dict[str, Any]:
    sub = weather[(weather["station_code"].astype(str).eq(station)) & (weather["date"].eq(date))]
    return sub.iloc[0].to_dict() if len(sub) else {}


def step_zero_action(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def step_scheduled_action(env: Any, action_real: dict[str, float]) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)


def evaluate_scenario(
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    station: str,
    year: int,
    scenario: str,
    recorded_templates: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    site = STATION_TO_SITE[station]
    auto = scenario == "dssat_auto"
    requested_schedule = {} if auto else schedule_for(scenario, site, recorded_templates)
    env, env_args = make_base_env(
        env_config,
        station,
        year,
        f"{station}_{year}_{TASK_ID}_{scenario}",
        auto_management=auto,
        static_schedule=None if auto else requested_schedule,
    )
    weather = direct_ppo.weather_for_daily(run_config)
    rendered_checks = check_rendered_input(Path(env_args["fileX_template_path"]))
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(year))["planting_date"])
        while not done and step_count < MAX_STEPS:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(float(dap_raw))) if np.isfinite(float(dap_raw)) and float(dap_raw) > 0 else step_count + 1
            action_real = {"amir": 0.0, "anfer": 0.0} if auto else requested_schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            # 固定四情景通过静态 MZX 管理表执行；step 阶段只发送 no-op，避免外部 action 通道混淆。
            action = step_zero_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = weather_row(weather, station, date)
            rows.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": int(year),
                    "scenario": scenario,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": int(dap),
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_requested_mm": float(action_real.get("amir", 0.0)),
                    "nitrogen_requested_kg_ha": float(action_real.get("anfer", 0.0)),
                    "external_action_note": "zero_external_action_for_dssat_auto" if auto else "static_mzx_schedule_zero_step_action",
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{station}{year} {scenario} 未在 {MAX_STEPS} 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = siteppo.snapshot_from_env(env)
        snapshot = OUT / "snapshots" / station / str(year) / scenario
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = metrics_from_snapshot(snapshot, final_y)
        summary = {
            "station_code": station,
            "site": site,
            "year": int(year),
            "scenario": scenario,
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": final_b,
            "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "actual_nitrogen_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "requested_irrigation_mm": float(pd.to_numeric(daily["irrigation_requested_mm"], errors="coerce").fillna(0).sum()),
            "requested_nitrogen_kg_ha": float(pd.to_numeric(daily["nitrogen_requested_kg_ha"], errors="coerce").fillna(0).sum()),
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "source_input_root": str(MULTISITE_INPUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
            "source_status": "generated_034_00_multisite_ic1",
            "recorded_farmer_status": "template_reused_not_year_specific" if scenario == "recorded_farmer_template" else "",
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            **metrics,
        }
        return daily, summary, rendered_checks
    finally:
        env.close()


def write_record(mode: str, selected: pd.DataFrame, summary: pd.DataFrame, manifest: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    coverage = manifest.groupby(["scenario", "status"]).size().reset_index(name="n") if not manifest.empty else pd.DataFrame()
    summary_for_agg = summary.copy()
    for col in [
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_water_stress",
        "max_nitrogen_stress",
    ]:
        if col in summary_for_agg.columns:
            summary_for_agg[col] = pd.to_numeric(summary_for_agg[col], errors="coerce")
    by_station = summary_for_agg.groupby(["station_code", "scenario"]).agg(
        n=("year", "count"),
        mean_yield=("grain_yield_kg_ha", "mean"),
        mean_irrigation=("actual_irrigation_mm", "mean"),
        mean_nitrogen=("actual_nitrogen_kg_ha", "mean"),
        mean_wp_et=("WP_ET_kg_m3", "mean"),
        mean_pfp_n=("PFP_N_kg_kg", "mean"),
        max_water_stress=("max_water_stress", "max"),
        max_nitrogen_stress=("max_nitrogen_stress", "max"),
    ).reset_index() if not summary.empty else pd.DataFrame()
    lines = [
        "# 034_00 multisite 输入链 IC=1 四情景基线重建记录",
        "",
        f"执行模式：`{mode}`",
        f"耗时：{elapsed:.1f} 秒",
        "",
        "## 本轮回答的问题",
        "",
        "在与 033_04 PPO 相同的 multisite 输入源和 IC=1 渲染条件下，重建 null、recorded farmer 模板、official expert、DSSAT auto 四情景基线。",
        "",
        "## 固定边界",
        "",
        f"- 输入源：`{MULTISITE_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        "- 年份清单：读取 `benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/configs/033_04_available_weather_half_split_years.csv`",
        "- 固定 `recorded_farmer_template` / `official_extension_expert` 通过渲染后 `.MZX` 静态管理表（`@I`/`@F` 行）执行；step 阶段只发送 no-op。",
        "- `dssat_auto` 只启用 DSSAT 自动管理块，不额外写入固定灌溉/施氮表。",
        "- 不训练 PPO/DQN。",
        "- 不混用旧输入链基线。",
        "- `recorded_farmer_template` 是模板复用比较项，不冒充逐年真实 recorded farmer。",
        "",
        "## 站点年份范围",
        "",
        md_table(selected.groupby(["station_code", "site", "split"]).size().reset_index(name="n"), 80),
        "",
        "## 覆盖状态",
        "",
        md_table(coverage, 80),
        "",
        "## 失败记录",
        "",
        md_table(failures[["station_code", "year", "scenario", "status", "details"]] if not failures.empty else failures, 80),
        "",
        "## 指标预览（按站点和情景均值）",
        "",
        md_table(by_station, 120),
        "",
        "## 输出文件",
        "",
        f"- summary: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_{mode}_baseline_summary.csv`",
        f"- daily: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_{mode}_baseline_daily.csv`",
        f"- manifest: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_{mode}_coverage_manifest.csv`",
        f"- failures: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_{mode}_failures.csv`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    start = time.time()
    ensure_dirs()
    run_config = direct_ppo.load_yaml(BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS
    shutil.copyfile(BASE_CONFIG, OUT / "configs" / BASE_CONFIG.name)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)

    selected = load_selection(args.mode)
    selected.to_csv(OUT / "configs" / f"034_00_{args.mode}_selected_station_years.csv", index=False, encoding="utf-8-sig")
    env_config = build_env_config(run_config, selected)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"034_00_{args.mode}_resolved_env_config.yaml")
    recorded_templates = recorded_template_schedules()

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        station = str(row.station_code)
        year = int(row.year)
        for scenario in SCENARIOS:
            print(f"[034_00] {args.mode} {station}{year} {scenario}", flush=True)
            try:
                daily, summary, checks = evaluate_scenario(run_config, env_config, station, year, scenario, recorded_templates)
                daily_frames.append(daily)
                summary_rows.append(summary)
                for check in checks:
                    render_rows.append({"station_code": station, "year": year, "scenario": scenario, **check})
                manifest_rows.append({"station_code": station, "site": STATION_TO_SITE[station], "year": year, "scenario": scenario, "status": "ok", "details": ""})
                pd.DataFrame(summary_rows).to_csv(OUT / "evaluation" / f"034_00_{args.mode}_baseline_summary_partial.csv", index=False, encoding="utf-8-sig")
            except Exception as exc:
                manifest_rows.append(
                    {
                        "station_code": station,
                        "site": STATION_TO_SITE.get(station, ""),
                        "year": year,
                        "scenario": scenario,
                        "status": "failed",
                        "details": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    failures = manifest[manifest["status"].ne("ok")].copy() if not manifest.empty else pd.DataFrame()
    render_check = pd.DataFrame(render_rows)

    summary.to_csv(OUT / "evaluation" / f"034_00_{args.mode}_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT / "evaluation" / f"034_00_{args.mode}_baseline_daily.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "evaluation" / f"034_00_{args.mode}_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(OUT / "evaluation" / f"034_00_{args.mode}_failures.csv", index=False, encoding="utf-8-sig")
    render_check.to_csv(OUT / "evaluation" / f"034_00_{args.mode}_render_check.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    write_record(args.mode, selected, summary, manifest, failures, elapsed)
    result = {
        "task": TASK_ID,
        "mode": args.mode,
        "selected_station_years": int(len(selected)),
        "expected_runs": int(len(selected) * len(SCENARIOS)),
        "successful_runs": int(len(summary)),
        "failed_runs": int(len(failures)),
        "output_root": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_seconds": elapsed,
    }
    (OUT / f"034_00_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not failures.empty:
        print(failures[["station_code", "year", "scenario", "details"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

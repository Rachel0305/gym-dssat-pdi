from __future__ import annotations

import json
import re
import shutil
import sys
import time
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
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_safe_rendering import (
    build_env_args,
    source_cultivar_path,
    source_soil_path,
    source_template_path,
    source_weather_path,
    target_weather_stem,
)


TASK_ID = "036_00"
OUT = ROOT / "benchmark_results" / "036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness"
DOC = ROOT / "docs" / "036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness_record.md"
PROMPT = ROOT / "prompts" / "036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness.md"
FACTORS = ["CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM"]
EXPECTED_SOURCE_FRAGMENT = "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "daily_outputs", "rendered_inputs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


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


def parse_treatment_one(text: str) -> dict[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("@N R O C TNAME"):
            for row in lines[idx + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("@") or stripped.startswith("*"):
                    continue
                parts = stripped.split()
                if parts and parts[0] == "1":
                    return {"treatment_row": stripped, **dict(zip(FACTORS, parts[5 : 5 + len(FACTORS)]))}
    return {"treatment_row": ""}


def parse_ic_ids(text: str) -> set[str]:
    ids: set[str] = set()
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*INITIAL CONDITIONS"):
            in_section = True
            continue
        if in_section and stripped.startswith("*"):
            break
        if not in_section or not stripped or stripped.startswith("@"):
            continue
        parts = stripped.split()
        if parts and re.fullmatch(r"\d+", parts[0]):
            ids.add(parts[0])
    return ids


def overview_modes(snapshot: Path) -> str:
    overview = snapshot / "OVERVIEW.OUT"
    if not overview.exists():
        return ""
    lines = [line.strip() for line in overview.read_text(encoding="utf-8", errors="ignore").splitlines() if "MANAGEMENT OPT" in line]
    return " | ".join(lines[:3])


def prepare_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = batch03222.load_config()
    config = json.loads(json.dumps(config))
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = "FQA"
    config["runtime"]["smoke_year"] = 2014
    split = batch03222.load_split()
    selection = batch03222.build_selection(split)
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(config, OUT / "configs" / "036_00_original_config_snapshot.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "036_00_resolved_env_config.yaml")
    split.to_csv(OUT / "configs" / "036_00_half_split_years.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "configs" / "036_00_selection.csv", index=False, encoding="utf-8-sig")
    return config, env_config, split, selection


def config_check(config: dict[str, Any]) -> pd.DataFrame:
    checks = [
        ("total_timesteps", config.get("total_timesteps"), 100000),
        ("checkpoint_steps", ",".join(map(str, batch03222.CHECKPOINT_STEPS)), "25000,50000,75000,100000"),
        ("irrigation_levels", ",".join(map(str, config["discrete_actions"]["irrigation_levels"])), "0.0,15.0,30.0,45.0"),
        ("nitrogen_levels", ",".join(map(str, config["discrete_actions"]["nitrogen_levels"])), "0.0,40.0,80.0,120.0"),
        ("season_irrigation_soft_limit", config["action_safety"]["season_irrigation_soft_limit"], 160.0),
        ("season_n_soft_limit", config["action_safety"]["season_n_soft_limit"], 250.0),
        ("min_days_between_irrigation", config["action_safety"]["min_days_between_irrigation"], 7),
        ("min_days_between_fertilization", config["action_safety"]["min_days_between_fertilization"], 7),
        ("irrigation_allowed_dap_range", ",".join(map(str, config["action_safety"]["irrigation_allowed_dap_range"])), "1,120"),
        ("fertilization_allowed_dap_range", ",".join(map(str, config["action_safety"]["fertilization_allowed_dap_range"])), "1,90"),
        ("reward_type", config["reward"]["reward_type"], "harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001"),
        ("yield_coef", config["reward"]["yield_coef"], 0.158),
        ("water_cost", config["reward"]["water_cost"], 1.1),
        ("nitrogen_cost", config["reward"]["nitrogen_cost"], 1.58),
        ("water_stress_relief_coef", config["reward"]["water_stress_relief_coef"], 10.0),
        ("nitrogen_stress_relief_coef", config["reward"]["nitrogen_stress_relief_coef"], 5.0),
        ("reward_scale", config["reward"]["reward_scale"], 0.001),
        ("ppo_learning_rate", config["ppo"]["learning_rate"], 0.0003),
        ("ppo_gamma", config["ppo"]["gamma"], 1.0),
        ("ppo_gae_lambda", config["ppo"]["gae_lambda"], 1.0),
        ("ppo_n_steps", config["ppo"]["n_steps"], 144),
        ("ppo_batch_size", config["ppo"]["batch_size"], 144),
        ("ppo_n_epochs", config["ppo"]["n_epochs"], 5),
        ("ppo_ent_coef", config["ppo"]["ent_coef"], 0.01),
        ("ppo_clip_range", config["ppo"]["clip_range"], 0.2),
    ]
    rows = []
    for name, actual, expected in checks:
        rows.append({"check": name, "actual": str(actual), "expected": str(expected), "pass": str(actual) == str(expected)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evaluation" / "036_00_config_check.csv", index=False, encoding="utf-8-sig")
    return df


def ic_source_audit(config: dict[str, Any], env_config: dict[str, Any], split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in split.sort_values(["station_code", "year"]).iterrows():
        station = str(item["station_code"])
        year = int(item["year"])
        year_info = direct_ppo.find_year(env_config, station, year)
        args = build_env_args(
            station=station,
            year=year,
            planting_date=year_info["planting_date"],
            seed=0,
            config=config,
            run_tag=f"{station}_{year}_{TASK_ID}_ic_source",
            evaluation=True,
            mode=config.get("runtime", {}).get("mode", "all"),
            linked_management=True,
        )
        template = Path(args["fileX_template_path"])
        text = template.read_text(encoding="utf-8", errors="replace")
        tr = parse_treatment_one(text)
        ic_ids = parse_ic_ids(text)
        expected_wsta = target_weather_stem(station, year)
        wsta_tokens = sorted(set(re.findall(r"\bCN[A-Z]{2}(?:\d{2}01|20\d{2})\b", text)))
        rendered_wth = sorted(p.name for p in template.parent.glob("*.WTH"))
        source_paths = {
            "source_template": source_template_path(station),
            "source_weather": source_weather_path(station, year),
            "source_soil": source_soil_path(station),
            "source_cultivar": source_cultivar_path(station),
        }
        row = {
            "station_code": station,
            "year": year,
            "split": item.get("split", ""),
            "IC": tr.get("IC", ""),
            "MI": tr.get("MI", ""),
            "MF": tr.get("MF", ""),
            "initial_condition_ids": ";".join(sorted(ic_ids, key=lambda x: int(x))),
            "ic_id_valid": tr.get("IC", "") in ic_ids,
            "expected_wsta": expected_wsta,
            "wsta_tokens": ";".join(wsta_tokens),
            "wsta_all_expected": wsta_tokens == [expected_wsta],
            "rendered_wth": ";".join(rendered_wth),
            "rendered_wth_ok": rendered_wth == [f"{expected_wsta}.WTH"],
            "rendered_template": str(template.relative_to(ROOT)).replace("\\", "/"),
            "treatment_row": tr.get("treatment_row", ""),
        }
        for key, path in source_paths.items():
            row[key] = str(path.relative_to(ROOT)).replace("\\", "/")
        row["source_all_multisite_013"] = all(EXPECTED_SOURCE_FRAGMENT in str(row[key]) for key in source_paths)
        row["pass"] = bool(
            row["IC"] == "1"
            and row["ic_id_valid"]
            and row["wsta_all_expected"]
            and row["rendered_wth_ok"]
            and row["source_all_multisite_013"]
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evaluation" / "036_00_ic_source_audit.csv", index=False, encoding="utf-8-sig")
    df[~df["pass"]].to_csv(OUT / "evaluation" / "036_00_ic_source_audit_failures.csv", index=False, encoding="utf-8-sig")
    return df


def grid_action_index(config: dict[str, Any], amir: float, anfer: float) -> int:
    grid = base032.action_grid(config)
    distances = [abs(float(item["amir"]) - float(amir)) + abs(float(item["anfer"]) - float(anfer)) for item in grid]
    return int(np.argmin(distances))


def forced_action_run(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, scenario: str) -> dict[str, Any]:
    env = base032.make_env(config, env_config, station, year, 0, f"{station}_{year}_{TASK_ID}_{scenario}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            if scenario == "forced_i45_n80" and dap == 1:
                action_idx = grid_action_index(config, 45.0, 80.0)
            else:
                action_idx = grid_action_index(config, 0.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = bool(terminated or truncated)
            latest_after = base032.latest_observation_dict(env, obs, info)
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "scenario": scenario,
                    "dap": dap,
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily_path = OUT / "daily_outputs" / f"{station}_{year}_{scenario}_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]) if len(daily) else np.nan
        snapshot = OUT / "snapshots" / station / str(year) / scenario
        tmp = siteppo.snapshot_from_env(env)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(tmp, snapshot)
        metrics = baseline03400.metrics_from_snapshot(snapshot, final_y)
        irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
        n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
        modes = overview_modes(snapshot)
        safe_i = float(irr.sum())
        safe_n = float(n.sum())
        summary_i = float(metrics["actual_irrigation_mm"])
        summary_n = float(metrics["actual_nitrogen_kg_ha"])
        return {
            "station_code": station,
            "year": year,
            "scenario": scenario,
            "episode_length": int(len(daily)),
            "grain_yield_kg_ha": final_y,
            "safe_i": safe_i,
            "safe_n": safe_n,
            "summary_i": summary_i,
            "summary_n": summary_n,
            "safe_summary_i_match": abs(safe_i - summary_i) < 1e-6,
            "safe_summary_n_match": abs(safe_n - summary_n) < 1e-6,
            "overview_is_linked": "IRRIG   :L" in modes and "FERT :L" in modes,
            "interface_pass": "IRRIG   :L" in modes and "FERT :L" in modes and abs(safe_i - summary_i) < 1e-6 and abs(safe_n - summary_n) < 1e-6,
            "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "overview_management_opt": modes,
        }
    finally:
        env.close()


def linked_forced_action_smoke(config: dict[str, Any], env_config: dict[str, Any], selection: pd.DataFrame) -> pd.DataFrame:
    cases = []
    for station, group in selection.groupby("station_code"):
        train = group[group["selected_for_train"].astype(bool)].sort_values("year")
        row = train.iloc[0] if len(train) else group.sort_values("year").iloc[0]
        cases.append((str(station), int(row["year"])))
    rows: list[dict[str, Any]] = []
    for station, year in cases:
        rows.append(forced_action_run(config, env_config, station, year, "null_noop"))
        rows.append(forced_action_run(config, env_config, station, year, "forced_i45_n80"))
    df = pd.DataFrame(rows)
    comp_rows = []
    for (station, year), group in df.groupby(["station_code", "year"]):
        wide = {row["scenario"]: row for _, row in group.iterrows()}
        null = wide.get("null_noop")
        forced = wide.get("forced_i45_n80")
        if null is None or forced is None:
            continue
        comp_rows.append(
            {
                "station_code": station,
                "year": int(year),
                "null_summary_i": float(null["summary_i"]),
                "null_summary_n": float(null["summary_n"]),
                "forced_safe_i": float(forced["safe_i"]),
                "forced_safe_n": float(forced["safe_n"]),
                "forced_summary_i": float(forced["summary_i"]),
                "forced_summary_n": float(forced["summary_n"]),
                "forced_overview_is_linked": bool(forced["overview_is_linked"]),
                "pass": bool(
                    abs(float(null["summary_i"])) < 1e-6
                    and abs(float(null["summary_n"])) < 1e-6
                    and abs(float(forced["safe_i"]) - 45.0) < 1e-6
                    and abs(float(forced["safe_n"]) - 80.0) < 1e-6
                    and abs(float(forced["summary_i"]) - 45.0) < 1e-6
                    and abs(float(forced["summary_n"]) - 80.0) < 1e-6
                    and bool(forced["overview_is_linked"])
                ),
            }
        )
    comp = pd.DataFrame(comp_rows)
    df.to_csv(OUT / "evaluation" / "036_00_linked_forced_action_smoke_raw.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "evaluation" / "036_00_linked_forced_action_smoke.csv", index=False, encoding="utf-8-sig")
    return comp


def write_record(config_df: pd.DataFrame, ic_df: pd.DataFrame, smoke_df: pd.DataFrame, elapsed: float) -> None:
    config_pass = int(config_df["pass"].sum())
    ic_pass = int(ic_df["pass"].sum())
    smoke_pass = int(smoke_df["pass"].sum())
    all_pass = config_pass == len(config_df) and ic_pass == len(ic_df) and smoke_pass == len(smoke_df)
    by_station = ic_df.groupby("station_code", as_index=False).agg(rows=("year", "count"), passed=("pass", "sum"))
    lines = [
        "# 036_00 原自由时序 MaskablePPO：IC=1 + linked 动作接口正式重跑前审计记录",
        "",
        "## 结论先说",
        "",
        f"- 配置检查通过：{config_pass}/{len(config_df)}。",
        f"- IC/输入源检查通过：{ic_pass}/{len(ic_df)}。",
        f"- linked 强制动作 smoke 通过：{smoke_pass}/{len(smoke_df)}。",
        f"- 总判定：{'通过，可以进入 036_01 正式重跑' if all_pass else '不通过，不能启动正式训练'}。",
        "",
        "## 关键边界",
        "",
        "- 036 主线恢复 032_22 原 stress-aware reward，不采用 035_04/035_06 的 reward 改动。",
        "- 036 主线只承认两个底层修复：IC=1 输入链、DSSAT linked 管理模式。",
        "- 本任务不训练 PPO，不评价最终农学优劣，只检查正式重跑前的地基是否干净。",
        "",
        "## 配置检查",
        "",
        md_table(config_df, 80),
        "",
        "## IC/输入源按站点汇总",
        "",
        md_table(by_station, 20),
        "",
        "## linked 强制动作 smoke",
        "",
        md_table(smoke_df, 20),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_config_check.csv`",
        "- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_ic_source_audit.csv`",
        "- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_linked_forced_action_smoke.csv`",
        "",
        f"耗时：{elapsed:.1f} 秒",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "task": TASK_ID,
        "config_pass": config_pass,
        "config_total": int(len(config_df)),
        "ic_pass": ic_pass,
        "ic_total": int(len(ic_df)),
        "smoke_pass": smoke_pass,
        "smoke_total": int(len(smoke_df)),
        "all_pass": bool(all_pass),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "evaluation" / "036_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config, split, selection = prepare_config()
    config_df = config_check(config)
    ic_df = ic_source_audit(config, env_config, split)
    smoke_df = linked_forced_action_smoke(config, env_config, selection)
    write_record(config_df, ic_df, smoke_df, time.time() - start)
    if not ((config_df["pass"].all()) and (ic_df["pass"].all()) and (smoke_df["pass"].all())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

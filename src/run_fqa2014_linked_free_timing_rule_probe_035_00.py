from __future__ import annotations

import json
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
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
import run_linked_free_timing_ppo_dqn_train_smoke_034_04 as linked03404
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400


TASK_ID = "035_00"
STATION = "FQA"
YEAR = 2014
SEED = 0
OUT = ROOT / "benchmark_results" / "035_00_fqa2014_linked_free_timing_rule_probe"
DOC = ROOT / "docs" / "035_00_fqa2014_linked_free_timing_rule_probe_record.md"
PROMPT = ROOT / "prompts" / "035_00_fqa2014_linked_free_timing_rule_probe.md"
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "034_00_multisite_input_ic1_four_baseline_rebuild"
    / "evaluation"
    / "034_00_full_baseline_summary.csv"
)
PPO_03405_COMPARISON = (
    ROOT
    / "benchmark_results"
    / "034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison"
    / "evaluation"
    / "034_05_scenario_comparison.csv"
)


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/FQA", "snapshots/FQA/2014", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    config, env_config = linked03404.load_config_and_env()
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = YEAR
    config["seed"] = SEED
    direct_ppo.write_yaml(config, OUT / "configs" / "035_00_rule_probe_config.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "035_00_resolved_env_config.yaml")
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    return config, env_config


def grid_action_index(config: dict[str, Any], amir: float, anfer: float) -> int:
    grid = base.action_grid(config)
    distances = [
        abs(float(item["amir"]) - float(amir)) + abs(float(item["anfer"]) - float(anfer))
        for item in grid
    ]
    return int(np.argmin(distances))


def rule_action(rule: str, dap: int, latest: dict[str, Any]) -> tuple[float, float]:
    schedules: dict[str, dict[int, tuple[float, float]]] = {
        "noop": {},
        "early_dump_cap": {
            1: (45.0, 120.0),
            8: (45.0, 120.0),
            15: (45.0, 0.0),
            22: (15.0, 0.0),
        },
        "ppo_03405_replay": {
            1: (45.0, 40.0),
            8: (45.0, 120.0),
            15: (15.0, 0.0),
            16: (0.0, 40.0),
            22: (15.0, 0.0),
            23: (0.0, 40.0),
            29: (15.0, 0.0),
            36: (15.0, 0.0),
        },
        "split_moderate_n160": {
            1: (15.0, 40.0),
            30: (15.0, 40.0),
            50: (15.0, 40.0),
            65: (15.0, 40.0),
            85: (15.0, 0.0),
        },
        "critical_i90_n200": {
            1: (15.0, 40.0),
            30: (15.0, 40.0),
            50: (30.0, 80.0),
            65: (30.0, 40.0),
            85: (0.0, 0.0),
        },
        "water_saving_n160": {
            1: (15.0, 40.0),
            30: (0.0, 40.0),
            50: (15.0, 40.0),
            65: (15.0, 40.0),
        },
        "delayed_late": {
            50: (30.0, 80.0),
            65: (30.0, 80.0),
            80: (30.0, 80.0),
            95: (30.0, 0.0),
        },
    }
    if rule == "stress_triggered":
        swfac = scalar(latest.get("swfac"), 0.0)
        nstres = scalar(latest.get("nstres"), 0.0)
        amir = 30.0 if swfac > 0.05 else 0.0
        anfer = 80.0 if nstres > 0.05 and dap <= 90 else 0.0
        return amir, anfer
    if rule not in schedules:
        raise ValueError(f"未知规则：{rule}")
    return schedules[rule].get(dap, (0.0, 0.0))


def overview_modes(snapshot: Path) -> str:
    overview = snapshot / "OVERVIEW.OUT"
    if not overview.exists():
        return ""
    lines = [
        line.strip()
        for line in overview.read_text(encoding="utf-8", errors="ignore").splitlines()
        if "MANAGEMENT OPT" in line
    ]
    return " | ".join(lines)


def summarize_daily(rule: str, daily: pd.DataFrame, daily_path: Path, snapshot: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )
    safe_i = float(irr.sum())
    safe_n = float(n.sum())
    summary_i = float(metrics["actual_irrigation_mm"])
    summary_n = float(metrics["actual_nitrogen_kg_ha"])
    return {
        "rule": rule,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "grain_yield_kg_ha": final_y,
        "safe_irrigation_mm": safe_i,
        "safe_nitrogen_kg_ha": safe_n,
        "summary_irrigation_mm": summary_i,
        "summary_nitrogen_kg_ha": summary_n,
        "safe_summary_i_match": abs(safe_i - summary_i) < 1e-6,
        "safe_summary_n_match": abs(safe_n - summary_n) < 1e-6,
        "etcp_mm": float(metrics["etcp_mm"]),
        "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
        "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) else np.nan,
        "simple_profit": final_y - 1.1 * summary_i - 1.58 * summary_n,
        "reward_sum": float(pd.to_numeric(daily.get("reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "stress_relief_bonus_sum_unscaled": float(pd.to_numeric(daily.get("stress_relief_bonus", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(daily) else np.nan,
        "irrigation_event_count": int((irr > 0).sum()),
        "nitrogen_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "overview_is_linked": "IRRIG   :L" in overview_modes(snapshot) and "FERT :L" in overview_modes(snapshot),
        "interface_pass": "IRRIG   :L" in overview_modes(snapshot)
        and "FERT :L" in overview_modes(snapshot)
        and abs(safe_i - summary_i) < 1e-6
        and abs(safe_n - summary_n) < 1e-6,
        "action_sequence": action_sequence,
        "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
    }


def run_rule(rule: str, config: dict[str, Any], env_config: dict[str, Any], weather: pd.DataFrame) -> dict[str, Any]:
    env = base.make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_{rule}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, YEAR)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            amir, anfer = rule_action(rule, dap, latest)
            action_index = grid_action_index(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(action_index)
            done = bool(terminated or truncated)
            latest_after = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "rule": rule,
                    "station_code": STATION,
                    "year": YEAR,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "xlai": scalar(latest_after.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily_path = OUT / "daily_outputs" / STATION / f"{YEAR}_{rule}_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]) if len(daily) else np.nan
        snapshot = OUT / "snapshots" / STATION / str(YEAR) / rule
        tmp = linked03404.siteppo.snapshot_from_env(env)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(tmp, snapshot)
        metrics = baseline03400.metrics_from_snapshot(snapshot, final_y)
        return summarize_daily(rule, daily, daily_path, snapshot, metrics)
    finally:
        env.close()


def comparison_table(rule_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    baseline["year"] = pd.to_numeric(baseline["year"], errors="coerce").astype(int)
    base_rows = baseline[(baseline["station_code"].eq(STATION)) & (baseline["year"].eq(YEAR))].copy()
    base_rows["source"] = "baseline_03400"
    base_rows["name"] = base_rows["scenario"]
    for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        base_rows[col] = pd.to_numeric(base_rows[col], errors="coerce")
    base_rows["simple_profit"] = (
        base_rows["grain_yield_kg_ha"]
        - 1.1 * base_rows["actual_irrigation_mm"]
        - 1.58 * base_rows["actual_nitrogen_kg_ha"]
    )
    base_comp = base_rows[
        [
            "source",
            "name",
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "etcp_mm",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
        ]
    ].copy()

    rules = rule_summary.rename(
        columns={
            "rule": "name",
            "summary_irrigation_mm": "actual_irrigation_mm",
            "summary_nitrogen_kg_ha": "actual_nitrogen_kg_ha",
        }
    ).copy()
    rules["source"] = "linked_rule_03500"
    rule_comp = rules[
        [
            "source",
            "name",
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "etcp_mm",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
        ]
    ].copy()

    ppo_comp = pd.DataFrame()
    if PPO_03405_COMPARISON.exists():
        old = pd.read_csv(PPO_03405_COMPARISON)
        ppo = old[old["scenario"].astype(str).eq("linked_free_timing_maskableppo_50k")].copy()
        if len(ppo):
            ppo["source"] = "rl_03405"
            ppo["name"] = ppo["scenario"]
            ppo = ppo.rename(columns={"actual_irrigation_mm": "actual_irrigation_mm", "actual_nitrogen_kg_ha": "actual_nitrogen_kg_ha"})
            ppo_comp = ppo[
                [
                    "source",
                    "name",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "etcp_mm",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                ]
            ].copy()
    comp = pd.concat([base_comp, ppo_comp, rule_comp], ignore_index=True, sort=False)
    baseline_only = comp[comp["source"].eq("baseline_03400")]
    for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit"]:
        comp[f"delta_vs_best_baseline_{metric}"] = (
            pd.to_numeric(comp[metric], errors="coerce")
            - pd.to_numeric(baseline_only[metric], errors="coerce").max()
        )
    return comp.sort_values(["source", "name"]).reset_index(drop=True)


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


def write_record(summary: pd.DataFrame, comp: pd.DataFrame, elapsed: float) -> None:
    pass_count = int(summary["interface_pass"].sum()) if "interface_pass" in summary else 0
    best_rule = summary.sort_values("simple_profit", ascending=False).iloc[0].to_dict() if len(summary) else {}
    lines = [
        "# 035_00 FQA2014 linked 自由时序规则探针记录",
        "",
        "## 结论先说",
        "",
        f"- 规则策略接口闭环通过：{pass_count}/{len(summary)}。",
        "- 本任务不训练 PPO/DQN，只在 linked 修复后的真实动作链路下测试固定规则策略。",
        f"- 当前 simple_profit 最高规则：{best_rule.get('rule', 'NA')}。",
        "- 该结果用于指导下一轮 RL 训练目标，不是最终管理方案。",
        "",
        "## 规则策略 summary",
        "",
        md_table(
            summary[
                [
                    "rule",
                    "grain_yield_kg_ha",
                    "summary_irrigation_mm",
                    "summary_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "reward_sum",
                    "interface_pass",
                    "action_sequence",
                ]
            ],
            40,
        ),
        "",
        "## 与四情景基线及 034_05 PPO 50K 对比",
        "",
        md_table(
            comp[
                [
                    "source",
                    "name",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "delta_vs_best_baseline_grain_yield_kg_ha",
                    "delta_vs_best_baseline_WP_ET_kg_m3",
                    "delta_vs_best_baseline_PFP_N_kg_kg",
                    "delta_vs_best_baseline_simple_profit",
                ]
            ],
            80,
        ),
        "",
        "## 解释边界",
        "",
        "- 若某个规则优于 PPO 50K，说明当前 PPO 尚未学到该可达行为，不等于规则是最终答案。",
        "- 若某个规则超过 expert，说明 linked 自由时序动作空间存在可达优质策略，下一步应训练 PPO/DQN 学到它。",
        "- 若所有规则都不超过 expert，也不能直接判死 RL，只能说明本轮规则集合未发现足够好候选。",
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_rule_summary.csv`",
        "- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_scenario_comparison.csv`",
        "- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/daily_outputs/FQA/*.csv`",
        "",
        f"耗时：{elapsed:.1f} 秒。",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    weather = direct_ppo.weather_for_daily(config)
    rules = [
        "noop",
        "early_dump_cap",
        "ppo_03405_replay",
        "split_moderate_n160",
        "critical_i90_n200",
        "water_saving_n160",
        "delayed_late",
        "stress_triggered",
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rule in rules:
        try:
            rows.append(run_rule(rule, config, env_config, weather))
        except Exception as exc:
            failures.append({"rule": rule, "error": repr(exc)})
    summary = pd.DataFrame(rows)
    comp = comparison_table(summary) if len(summary) else pd.DataFrame()
    summary.to_csv(OUT / "evaluation" / "035_00_rule_summary.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "evaluation" / "035_00_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(OUT / "evaluation" / "035_00_failures.csv", index=False, encoding="utf-8-sig")
    write_record(summary, comp, time.time() - start)
    result = {
        "task": TASK_ID,
        "rules": int(len(summary)),
        "failures": int(len(failures)),
        "interface_pass": int(summary["interface_pass"].sum()) if len(summary) and "interface_pass" in summary else 0,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "comparison_csv": str((OUT / "evaluation" / "035_00_scenario_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "035_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if len(comp):
        print(
            comp[
                [
                    "source",
                    "name",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

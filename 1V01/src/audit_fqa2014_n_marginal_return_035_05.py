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
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_linked_free_timing_ppo_dqn_train_smoke_034_04 as linked03404
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400


TASK_ID = "035_05"
STATION = "FQA"
YEAR = 2014
SEED = 0
OUT = ROOT / "benchmark_results" / "035_05_fqa2014_n_marginal_return_audit"
DOC = ROOT / "docs" / "035_05_fqa2014_n_marginal_return_audit_record.md"
PROMPT = ROOT / "prompts" / "035_05_fqa2014_n_marginal_return_audit.md"


N_SCHEDULES: dict[str, dict[int, float]] = {
    "i45_n0": {},
    "i45_n80": {1: 40.0, 30: 40.0},
    "i45_n120": {1: 40.0, 30: 40.0, 50: 40.0},
    "i45_n160": {1: 40.0, 30: 40.0, 50: 40.0, 65: 40.0},
    "i45_n200": {1: 40.0, 30: 40.0, 50: 80.0, 65: 40.0},
    "i45_n240": {1: 80.0, 30: 40.0, 50: 80.0, 65: 40.0},
}
IRRIGATION_SCHEDULE = {1: 15.0, 50: 15.0, 65: 15.0}
PROJECT_WATER_COST = 1.1
PROJECT_N_COST = 1.58


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/FQA", "snapshots/FQA/2014", "evaluation"]:
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


def configure_modules() -> None:
    base032.OUT = OUT
    linked03404.OUT = OUT
    linked03404.TASK_ID = TASK_ID
    linked03404.DOC = DOC
    linked03404.PROMPT = PROMPT
    direct_ppo.OUTPUT_ROOT = OUT


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    configure_modules()
    config, env_config = linked03404.load_config_and_env()
    config["seed"] = SEED
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = YEAR
    direct_ppo.write_yaml(config, OUT / "configs" / "035_05_config.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "035_05_resolved_env_config.yaml")
    return config, env_config


def grid_action_index(config: dict[str, Any], amir: float, anfer: float) -> int:
    grid = base032.action_grid(config)
    distances = [abs(float(item["amir"]) - float(amir)) + abs(float(item["anfer"]) - float(anfer)) for item in grid]
    return int(np.argmin(distances))


def overview_modes(snapshot: Path) -> str:
    overview = snapshot / "OVERVIEW.OUT"
    if not overview.exists():
        return ""
    lines = [line.strip() for line in overview.read_text(encoding="utf-8", errors="ignore").splitlines() if "MANAGEMENT OPT" in line]
    return " | ".join(lines)


def rule_action(rule: str, dap: int) -> tuple[float, float]:
    return IRRIGATION_SCHEDULE.get(dap, 0.0), N_SCHEDULES[rule].get(dap, 0.0)


def summarize(rule: str, daily: pd.DataFrame, daily_path: Path, snapshot: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    safe_i = float(irr.sum())
    safe_n = float(n.sum())
    summary_i = float(metrics["actual_irrigation_mm"])
    summary_n = float(metrics["actual_nitrogen_kg_ha"])
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )
    modes = overview_modes(snapshot)
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
        "project_simple_profit": final_y - PROJECT_WATER_COST * summary_i - PROJECT_N_COST * summary_n,
        "irrigation_event_count": int((irr > 0).sum()),
        "nitrogen_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "overview_is_linked": "IRRIG   :L" in modes and "FERT :L" in modes,
        "interface_pass": "IRRIG   :L" in modes and "FERT :L" in modes and abs(safe_i - summary_i) < 1e-6 and abs(safe_n - summary_n) < 1e-6,
        "action_sequence": action_sequence,
        "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
    }


def run_rule(rule: str, config: dict[str, Any], env_config: dict[str, Any], weather: pd.DataFrame) -> dict[str, Any]:
    env = base032.make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_{rule}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, YEAR)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            amir, anfer = rule_action(rule, dap)
            action_index = grid_action_index(config, amir, anfer)
            obs, reward, terminated, truncated, info = env.step(action_index)
            done = bool(terminated or truncated)
            latest_after = base032.latest_observation_dict(env, obs, info)
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
        return summarize(rule, daily, daily_path, snapshot, metrics)
    finally:
        env.close()


def marginal_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.sort_values("summary_nitrogen_kg_ha").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for prev, cur in zip(work.iloc[:-1].to_dict("records"), work.iloc[1:].to_dict("records")):
        dn = float(cur["summary_nitrogen_kg_ha"]) - float(prev["summary_nitrogen_kg_ha"])
        dy = float(cur["grain_yield_kg_ha"]) - float(prev["grain_yield_kg_ha"])
        dp = float(cur["project_simple_profit"]) - float(prev["project_simple_profit"])
        rows.append(
            {
                "from_rule": prev["rule"],
                "to_rule": cur["rule"],
                "delta_n_kg_ha": dn,
                "delta_yield_kg_ha": dy,
                "marginal_yield_per_kgN": dy / dn if dn else np.nan,
                "delta_project_simple_profit": dp,
                "current_n_cost": PROJECT_N_COST,
                "marginal_exceeds_current_n_cost": (dy / dn) > PROJECT_N_COST if dn else False,
            }
        )
    return pd.DataFrame(rows)


def write_record(summary: pd.DataFrame, deltas: pd.DataFrame, elapsed: float) -> None:
    n240_row = summary[summary["summary_nitrogen_kg_ha"].eq(240.0)]
    best = summary.sort_values("project_simple_profit", ascending=False).iloc[0]
    n200_to_240 = deltas[(deltas["from_rule"].eq("i45_n200")) & (deltas["to_rule"].eq("i45_n240"))]
    if len(n200_to_240):
        last_marginal = float(n200_to_240["marginal_yield_per_kgN"].iloc[0])
        last_msg = f"N200→N240 边际产量收益为 {last_marginal:.3f} kg grain/kg N。"
    else:
        last_msg = "未找到 N200→N240 边际对照。"
    lines = [
        "# 035_05 FQA2014 氮边际收益审计记录",
        "",
        "## 结论先说",
        "",
        f"- 固定 I45 与施氮窗口后，project_simple_profit 最高规则是 `{best['rule']}`。",
        f"- {last_msg}",
        f"- 当前 nitrogen_cost = {PROJECT_N_COST} kg yield-equivalent/kg N；若某档边际收益高于该值，当前 reward 会认为继续加氮仍然划算。",
        "- 本任务不训练模型，只做受控 DSSAT 前向模拟。",
        "",
        "## 固定规则结果",
        "",
        md_table(
            summary[
                [
                    "rule",
                    "interface_pass",
                    "grain_yield_kg_ha",
                    "summary_irrigation_mm",
                    "summary_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "project_simple_profit",
                    "swfac_stress_days_gt_0p05",
                    "nstres_days_gt_0p05",
                    "action_sequence",
                ]
            ].sort_values("summary_nitrogen_kg_ha"),
            30,
        ),
        "",
        "## 相邻施氮档位边际收益",
        "",
        md_table(deltas, 30),
        "",
        "## 解释边界",
        "",
        "- 这是 FQA2014、固定 I45、固定早期分期施氮窗口下的局部审计。",
        "- 它不能直接证明所有站点年份都需要同样的 nitrogen_cost。",
        "- 它可以用于判断 035_04 中 PPO 偏向高氮是否与当前 reward 算术一致。",
        "",
        f"耗时：{elapsed:.1f} 秒",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    weather = direct_ppo.weather_for_daily(config)
    rows = [run_rule(rule, config, env_config, weather) for rule in N_SCHEDULES]
    summary = pd.DataFrame(rows).sort_values("summary_nitrogen_kg_ha").reset_index(drop=True)
    deltas = marginal_deltas(summary)
    summary.to_csv(OUT / "evaluation" / "035_05_n_marginal_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "evaluation" / "035_05_n_marginal_deltas.csv", index=False, encoding="utf-8-sig")
    write_record(summary, deltas, time.time() - start)
    print(
        {
            "task": TASK_ID,
            "rules": int(len(summary)),
            "interface_pass": int(summary["interface_pass"].astype(bool).sum()),
            "record_md": str(DOC.relative_to(ROOT)),
            "summary_csv": str((OUT / "evaluation" / "035_05_n_marginal_summary.csv").relative_to(ROOT)),
            "deltas_csv": str((OUT / "evaluation" / "035_05_n_marginal_deltas.csv").relative_to(ROOT)),
        }
    )
    print(summary[["rule", "grain_yield_kg_ha", "summary_irrigation_mm", "summary_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "project_simple_profit"]].to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()

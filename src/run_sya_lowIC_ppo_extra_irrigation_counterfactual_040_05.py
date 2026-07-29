from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as lc_multi


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_05"
TASK_NAME = "sya_lowIC_ppo_extra_irrigation_counterfactual"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
PPO_DETAIL = (
    ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)

STATION = "SYA"
SEED = 0
PPO_CHECKPOINT = 75_000
YEARS = [2014, 2017]
EXTRA_IRRIGATION_MM = 45.0
RELAXED_SEASON_IRRIGATION_LIMIT = 205.0


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "daily_outputs", "snapshots"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return base032.scalar(value, default=default)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config = base03222.load_config()
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


def action_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    return base032.action_grid(config)


def action_index_for(config: dict[str, Any], irrigation: float, nitrogen: float) -> int:
    grid = action_grid(config)
    for idx, action in enumerate(grid):
        if abs(float(action["amir"]) - float(irrigation)) < 1e-9 and abs(float(action["anfer"]) - float(nitrogen)) < 1e-9:
            return int(idx)
    raise ValueError(f"找不到动作档位 I{irrigation}/N{nitrogen}")


def load_ppo_rows() -> pd.DataFrame:
    df = pd.read_csv(PPO_DETAIL)
    rows = df[
        (df["station_code"].eq(STATION))
        & (df["checkpoint_step"].astype(int).eq(PPO_CHECKPOINT))
        & (df["year"].astype(int).isin(YEARS))
    ].copy()
    if len(rows) != len(YEARS):
        raise RuntimeError(f"缺少PPO行: expected {len(YEARS)}, got {len(rows)}")
    return rows.sort_values("year").reset_index(drop=True)


def original_actions_by_dap(daily_csv_path: str) -> dict[int, tuple[float, float]]:
    daily = pd.read_csv(ROOT / daily_csv_path)
    out: dict[int, tuple[float, float]] = {}
    for _, row in daily.iterrows():
        dap = int(round(float(row["dap"])))
        i = float(row.get("safe_action_amir", 0.0) or 0.0)
        n = float(row.get("safe_action_anfer", 0.0) or 0.0)
        if abs(i) > 1e-9 or abs(n) > 1e-9:
            out[dap] = (i, n)
    return out


def choose_intervention_dap(daily_csv_path: str, config: dict[str, Any]) -> int:
    daily = pd.read_csv(ROOT / daily_csv_path)
    stress = daily[pd.to_numeric(daily["swfac"], errors="coerce") > 0.05].copy()
    if stress.empty:
        raise RuntimeError(f"原PPO轨迹没有swfac>0.05，无法定义加灌DAP: {daily_csv_path}")
    candidate = int(stress["dap"].iloc[0])
    actions = original_actions_by_dap(daily_csv_path)
    previous_irrigations = sorted([dap for dap, (i, _n) in actions.items() if i > 0])
    min_gap = int(config["action_safety"]["min_days_between_irrigation"])
    while True:
        last_before = [dap for dap in previous_irrigations if dap < candidate]
        if not last_before or candidate - max(last_before) >= min_gap:
            return candidate
        candidate += 1


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def replay_fixed_schedule(
    config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    branch: str,
    actions_by_dap: dict[int, tuple[float, float]],
    relaxed_irrigation_limit: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = json.loads(json.dumps(config))
    if relaxed_irrigation_limit is not None:
        cfg["action_safety"]["season_irrigation_soft_limit"] = float(relaxed_irrigation_limit)

    weather = direct_ppo.weather_for_daily(cfg)
    records: list[dict[str, Any]] = []
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_out = direct_ppo.OUTPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    direct_ppo.OUTPUT_ROOT = OUT
    env = None
    try:
        env = base032.make_env(cfg, env_config, STATION, int(year), SEED, f"{STATION}_{year}_{TASK_ID}_{branch}", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(cfg["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            i, n = actions_by_dap.get(dap, (0.0, 0.0))
            action = action_index_for(cfg, i, n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base032.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "year": int(year),
                    "branch": branch,
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
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input
        direct_ppo.OUTPUT_ROOT = old_out

    daily = pd.DataFrame(records)
    daily_path = OUT / "daily_outputs" / f"{STATION}_{year}_{branch}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary = base032.summarize_daily("fixed_schedule_counterfactual", daily, daily_path, ROOT / "not_a_model.zip")
    summary.update(
        {
            "station_code": STATION,
            "year": int(year),
            "branch": branch,
            "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
            "relaxed_irrigation_limit": relaxed_irrigation_limit if relaxed_irrigation_limit is not None else "",
        }
    )
    summary.update(stress_summary(daily))
    return daily, summary


def build_plan() -> pd.DataFrame:
    config, _env_config, _selection = load_config_and_env_config()
    rows = []
    for _, row in load_ppo_rows().iterrows():
        year = int(row["year"])
        daily_path = str(row["daily_csv_path"])
        intervention_dap = choose_intervention_dap(daily_path, config)
        actions = original_actions_by_dap(daily_path)
        rows.append(
            {
                "station_code": STATION,
                "year": year,
                "ppo_checkpoint_step": PPO_CHECKPOINT,
                "original_total_irrigation": float(row["total_irrigation"]),
                "original_total_n": float(row["total_n"]),
                "original_swfac_days_gt_0p05": int(row["swfac_stress_days_gt_0p05"]),
                "original_yield": float(row["final_grnwt"]),
                "intervention_dap": intervention_dap,
                "intervention_irrigation_mm": EXTRA_IRRIGATION_MM,
                "intervention_n_kg_ha": 0.0,
                "relaxed_irrigation_limit": RELAXED_SEASON_IRRIGATION_LIMIT,
                "original_action_sequence": row["action_sequence"],
                "original_daily_csv_path": daily_path,
                "nonzero_action_count": len(actions),
            }
        )
    return pd.DataFrame(rows)


def write_record(plan: pd.DataFrame, summary: pd.DataFrame | None = None) -> None:
    lines = [
        "# 040_05 SYA lowIC PPO 额外灌溉反事实审计记录",
        "",
        "## 结论先说",
        "",
        "- 本任务固定 PPO checkpoint 75000，不重新训练。",
        "- 目标是判断 2014/2017 的水分胁迫失败是否可由一次中后期额外灌溉缓解。",
        "- 因 PPO 原策略已用 I150，而原季节软上限为 I160、单次最小正灌溉为 I15，本任务的加灌分支仅作为机制反事实，临时放宽灌溉上限到 I205。",
        "",
        "## 预注册干预计划",
        "",
        md_table(plan, max_rows=20),
    ]
    if summary is not None and not summary.empty:
        lines.extend(
            [
                "",
                "## 反事实结果汇总",
                "",
                md_table(summary, max_rows=40),
            ]
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 如果加灌改善明显，只能说明当前 PPO 策略存在水量/时机不足的机制风险；不能直接作为正式训练新约束。",
            "- 如果加灌改善不明显，则说明失败不只是中后期水分不足，下一步应检查早期生长轨迹或 reward/策略表达。",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full() -> None:
    ensure_dirs()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)
    config, env_config, selection = load_config_and_env_config()
    # On the Windows-mounted Docker workspace, copying file metadata can fail
    # with PermissionError.  The prompt snapshot only needs content fidelity.
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"{TASK_ID}_resolved_env_config.yaml")
    selection.to_csv(OUT / "configs" / f"{TASK_ID}_selection.csv", index=False, encoding="utf-8-sig")
    plan = build_plan()
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_intervention_plan.csv", index=False, encoding="utf-8-sig")

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for _, prow in plan.iterrows():
        year = int(prow["year"])
        original_actions = original_actions_by_dap(str(prow["original_daily_csv_path"]))
        original_daily, original_summary = replay_fixed_schedule(config, env_config, year, "ppo_original_replay", dict(original_actions))
        daily_frames.append(original_daily)
        summary_rows.append(original_summary)

        plus_actions = dict(original_actions)
        plus_actions[int(prow["intervention_dap"])] = (EXTRA_IRRIGATION_MM, 0.0)
        plus_daily, plus_summary = replay_fixed_schedule(
            config,
            env_config,
            year,
            "ppo_plus_i45_first_water_stress_relaxed_cap",
            plus_actions,
            relaxed_irrigation_limit=RELAXED_SEASON_IRRIGATION_LIMIT,
        )
        daily_frames.append(plus_daily)
        summary_rows.append(plus_summary)

    summary = pd.DataFrame(summary_rows)
    # add deltas versus replayed original by year
    delta_rows = []
    for year, group in summary.groupby("year"):
        orig = group[group["branch"].eq("ppo_original_replay")].iloc[0]
        for _, row in group.iterrows():
            out = row.to_dict()
            out["delta_yield_vs_original_replay"] = float(row["final_grnwt"]) - float(orig["final_grnwt"])
            out["delta_swfac_days_gt_0p05_vs_original_replay"] = int(row["swfac_days_gt_0p05"]) - int(orig["swfac_days_gt_0p05"])
            out["delta_total_irrigation_vs_original_replay"] = float(row["total_irrigation"]) - float(orig["total_irrigation"])
            delta_rows.append(out)
    summary = pd.DataFrame(delta_rows)
    summary.to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    write_record(plan, summary)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "counterfactual_completed",
        "years": YEARS,
        "summary_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv").relative_to(ROOT).as_posix(),
        "daily_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_dry() -> None:
    ensure_dirs()
    plan = build_plan()
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_intervention_plan_dry_run.csv", index=False, encoding="utf-8-sig")
    write_record(plan, None)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "next_step_allowed": bool(len(plan) == len(YEARS)),
        "intervention_plan": plan.to_dict(orient="records"),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_dry_run_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="040_05 PPO extra irrigation counterfactual for SYA lowIC 2014/2017")
    parser.add_argument("--dry-run", action="store_true", help="Only write the intervention plan; do not run DSSAT.")
    args = parser.parse_args()
    if args.dry_run:
        run_dry()
    else:
        run_full()


if __name__ == "__main__":
    main()

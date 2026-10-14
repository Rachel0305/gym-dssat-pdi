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


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_08"
TASK_NAME = "sya_lowIC_ppo_all_year_irrigation_reserve_counterfactual"
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
YEARS = list(range(2014, 2024))
EARLY_CAP_DAP_END = 30
EARLY_IRRIGATION_CAP_MM = 90.0
STRESS_THRESHOLD = 0.05


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "daily_outputs"]:
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
        missing = sorted(set(YEARS) - set(rows["year"].astype(int).tolist()))
        raise RuntimeError(f"缺少 PPO 75000 checkpoint 行：missing={missing}")
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


def cap_early_irrigation(
    actions: dict[int, tuple[float, float]],
    early_cap_mm: float = EARLY_IRRIGATION_CAP_MM,
    early_dap_end: int = EARLY_CAP_DAP_END,
) -> tuple[dict[int, tuple[float, float]], float, list[dict[str, Any]]]:
    capped: dict[int, tuple[float, float]] = {}
    used_early = 0.0
    withheld = 0.0
    levels = [0.0, 15.0, 30.0, 45.0]
    edits: list[dict[str, Any]] = []
    for dap in sorted(actions):
        i, n = actions[dap]
        new_i = i
        if dap <= early_dap_end and i > 0:
            remaining = max(0.0, early_cap_mm - used_early)
            allowed_levels = [x for x in levels if x <= min(i, remaining) + 1e-9]
            new_i = max(allowed_levels) if allowed_levels else 0.0
            used_early += new_i
            withheld += i - new_i
        if new_i > 0 or n > 0:
            capped[dap] = (float(new_i), float(n))
        edits.append(
            {
                "dap": dap,
                "original_i": i,
                "original_n": n,
                "capped_i": new_i,
                "capped_n": n,
                "withheld_i": i - new_i,
            }
        )
    return capped, float(withheld), edits


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def irrigation_gap_ok(dap: int, actions_done: list[dict[str, Any]], min_gap: int) -> bool:
    prior = [int(x["dap"]) for x in actions_done if float(x.get("safe_action_amir", 0.0) or 0.0) > 0]
    return not prior or all(abs(dap - old) >= min_gap for old in prior)


def reserve_release_amount(reserve: float) -> float:
    if reserve >= 45.0 - 1e-9:
        return 45.0
    if reserve >= 30.0 - 1e-9:
        return 30.0
    if reserve >= 15.0 - 1e-9:
        return 15.0
    return 0.0


def replay_schedule(
    config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    branch: str,
    base_actions_by_dap: dict[int, tuple[float, float]],
    reserve_mm: float = 0.0,
    dynamic_transfer: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    transfer_records: list[dict[str, Any]] = []
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_out = direct_ppo.OUTPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    direct_ppo.OUTPUT_ROOT = OUT
    env = None
    remaining_reserve = float(reserve_mm)
    min_gap = int(config["action_safety"]["min_days_between_irrigation"])
    try:
        env = base032.make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_{TASK_ID}_{branch}", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_before = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_before.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            base_i, base_n = base_actions_by_dap.get(dap, (0.0, 0.0))
            i = base_i
            n = base_n

            swfac_before = scalar(latest_before.get("swfac"))
            triggered = (
                dynamic_transfer
                and dap > EARLY_CAP_DAP_END
                and remaining_reserve >= 15.0 - 1e-9
                and swfac_before > STRESS_THRESHOLD
                and irrigation_gap_ok(dap, records, min_gap)
            )
            if triggered:
                amount = reserve_release_amount(remaining_reserve)
                if i + amount > 45.0 + 1e-9:
                    amount = max(0.0, 45.0 - i)
                    amount = reserve_release_amount(amount)
                if amount > 0:
                    i += amount
                    remaining_reserve -= amount
                    transfer_records.append(
                        {
                            "station_code": STATION,
                            "year": int(year),
                            "branch": branch,
                            "dap": dap,
                            "swfac_before_action": swfac_before,
                            "released_irrigation": amount,
                            "reserve_remaining_after": remaining_reserve,
                        }
                    )

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
                    "year": int(year),
                    "branch": branch,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac_before_action": swfac_before,
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    "requested_amir": i,
                    "requested_anfer": n,
                    **dict(env.last_action_info),
                    "reserve_remaining": remaining_reserve,
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
    summary = base032.summarize_daily("irrigation_reserve_counterfactual", daily, daily_path, ROOT / "not_a_model.zip")
    summary.update(
        {
            "station_code": STATION,
            "year": int(year),
            "branch": branch,
            "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
            "initial_reserve_mm": float(reserve_mm),
            "unused_reserve_mm": float(remaining_reserve),
            "transfer_event_count": len(transfer_records),
        }
    )
    summary.update(stress_summary(daily))
    return daily, summary, pd.DataFrame(transfer_records)


def build_plan() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], tuple[dict[int, tuple[float, float]], float, bool]]]:
    plan_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    schedules: dict[tuple[int, str], tuple[dict[int, tuple[float, float]], float, bool]] = {}
    for _, row in load_ppo_rows().iterrows():
        year = int(row["year"])
        original = original_actions_by_dap(str(row["daily_csv_path"]))
        capped, withheld, edits = cap_early_irrigation(original)
        branches = [
            ("original_replay", original, 0.0, False),
            ("early_cap_I90_no_transfer", capped, 0.0, False),
            ("early_cap_I90_stress_trigger_transfer", capped, withheld, True),
        ]
        for branch, actions, reserve, dynamic in branches:
            schedules[(year, branch)] = (actions, reserve, dynamic)
            plan_rows.append(
                {
                    "station_code": STATION,
                    "year": year,
                    "branch": branch,
                    "ppo_checkpoint_step": PPO_CHECKPOINT,
                    "planned_base_irrigation": sum(i for i, _n in actions.values()),
                    "planned_total_n": sum(n for _i, n in actions.values()),
                    "early_I_DAP1_30": sum(i for dap, (i, _n) in actions.items() if dap <= EARLY_CAP_DAP_END),
                    "reserve_available_for_dynamic_transfer": reserve,
                    "dynamic_transfer_enabled": dynamic,
                    "original_total_irrigation": float(row["total_irrigation"]),
                    "original_yield": float(row["final_grnwt"]),
                    "original_swfac_days_gt_0p05": int(row["swfac_stress_days_gt_0p05"]),
                }
            )
        for item in edits:
            item.update({"station_code": STATION, "year": year, "edit_type": "early_cap"})
            edit_rows.append(item)
    return pd.DataFrame(plan_rows), pd.DataFrame(edit_rows), schedules


def write_record(plan: pd.DataFrame, edits: pd.DataFrame, summary: pd.DataFrame | None = None) -> None:
    lines = [
        "# 040_08 SYA lowIC PPO 全验证年灌溉预算保留反事实记录",
        "",
        "## 结论先说",
        "",
        "- 本任务固定 PPO 75k checkpoint，不重新训练。",
        "- 本任务把 040_07 的早期灌溉预算保留机制扩展到 2014-2023 全部验证年份。",
        "- `stress_trigger_transfer` 分支不增加季节总灌溉量，只在 DAP30 后真实出现水分胁迫时释放早期截留的水。",
        "",
        "## 动作计划",
        "",
        md_table(plan, max_rows=35),
        "",
        "## 早期截水明细",
        "",
        md_table(edits, max_rows=80),
    ]
    if summary is not None and not summary.empty:
        lines.extend(["", "## 反事实结果", "", md_table(summary, max_rows=40)])
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry() -> None:
    ensure_dirs()
    plan, edits, _schedules = build_plan()
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_action_plan_dry_run.csv", index=False, encoding="utf-8-sig")
    edits.to_csv(OUT / "tables" / f"{TASK_ID}_action_edits_dry_run.csv", index=False, encoding="utf-8-sig")
    write_record(plan, edits, None)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "next_step_allowed": True,
        "action_plan": (OUT / "tables" / f"{TASK_ID}_action_plan_dry_run.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_dry_run_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_full() -> None:
    ensure_dirs()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)
    config, env_config, selection = load_config_and_env_config()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"{TASK_ID}_resolved_env_config.yaml")
    selection.to_csv(OUT / "configs" / f"{TASK_ID}_selection.csv", index=False, encoding="utf-8-sig")
    plan, edits, schedules = build_plan()
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_action_plan.csv", index=False, encoding="utf-8-sig")
    edits.to_csv(OUT / "tables" / f"{TASK_ID}_action_edits.csv", index=False, encoding="utf-8-sig")

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    transfer_frames: list[pd.DataFrame] = []
    for year in YEARS:
        for branch in ["original_replay", "early_cap_I90_no_transfer", "early_cap_I90_stress_trigger_transfer"]:
            actions, reserve, dynamic = schedules[(year, branch)]
            daily, summary, transfers = replay_schedule(config, env_config, year, branch, actions, reserve_mm=reserve, dynamic_transfer=dynamic)
            daily_frames.append(daily)
            summary_rows.append(summary)
            if not transfers.empty:
                transfer_frames.append(transfers)

    summary = pd.DataFrame(summary_rows)
    delta_rows: list[dict[str, Any]] = []
    for year, group in summary.groupby("year"):
        orig = group[group["branch"].eq("original_replay")].iloc[0]
        for _, row in group.iterrows():
            out = row.to_dict()
            out["delta_yield_vs_original_replay"] = float(row["final_grnwt"]) - float(orig["final_grnwt"])
            out["delta_swfac_days_gt_0p05_vs_original_replay"] = int(row["swfac_days_gt_0p05"]) - int(orig["swfac_days_gt_0p05"])
            out["delta_total_irrigation_vs_original_replay"] = float(row["total_irrigation"]) - float(orig["total_irrigation"])
            delta_rows.append(out)
    summary = pd.DataFrame(delta_rows)
    summary.to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    transfers_all = pd.concat(transfer_frames, ignore_index=True) if transfer_frames else pd.DataFrame()
    transfers_all.to_csv(OUT / "tables" / f"{TASK_ID}_dynamic_transfer_events.csv", index=False, encoding="utf-8-sig")
    write_record(plan, edits, summary)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "completed",
        "years": YEARS,
        "summary_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv").relative_to(ROOT).as_posix(),
        "daily_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv").relative_to(ROOT).as_posix(),
        "transfer_events_csv": (OUT / "tables" / f"{TASK_ID}_dynamic_transfer_events.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="040_08 SYA lowIC all-year PPO irrigation reserve counterfactual")
    parser.add_argument("--dry-run", action="store_true", help="Only write action plan; do not run DSSAT.")
    args = parser.parse_args()
    if args.dry_run:
        run_dry()
    else:
        run_full()


if __name__ == "__main__":
    main()

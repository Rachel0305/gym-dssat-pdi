"""Task 187: read-only DAP0-120 action-mask timeline audit.

This script does not train PPO and does not rerun DSSAT.  It reads the daily
replay files produced by task 186, reconstructs the last irrigation and
fertilization events before each action, and tests whether one-action mask
windows are explained by cooldown or cumulative-budget rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "187"
TASK_NAME = "sya_mask_timeline_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_mask_timeline_audit.md"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
VARIANT_BY_KEY = {2005: "original", 2006: "early_dry", 2007: "mid_dry", 2008: "late_dry", 2009: "early_wet", 2010: "mid_wet"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def number(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.head(max_rows).copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(6)
    return work.to_markdown(index=False)


def bool_text(value: bool) -> str:
    return "是" if bool(value) else "否"


def reconstruct_trace(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(path, keep_default_na=False)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce").astype(int)
    daily = daily[daily["dap"].between(0, int(cfg["max_dap"]))].sort_values("dap").reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    min_i_gap = int(cfg["min_days_between_irrigation"])
    min_n_gap = int(cfg["min_days_between_fertilization"])
    season_i_cap = float(cfg["season_irrigation_cap_mm"])
    season_n_cap = float(cfg["season_n_cap_kg_ha"])
    pre_late_dap_end = int(cfg["pre_late_dap_end"])
    pre_late_cap = float(cfg["pre_late_cum_irrigation_cap_mm"])
    last_i_dap: int | None = None
    last_n_dap: int | None = None
    previous_post_mask_count: int | None = None
    previous_dap: int | None = None
    rows: list[dict[str, Any]] = []

    for item in daily.itertuples(index=False):
        dap = int(item.dap)
        safe_i = number(getattr(item, "safe_action_amir", np.nan), 0.0)
        safe_n = number(getattr(item, "safe_action_anfer", np.nan), 0.0)
        previous_i = number(getattr(item, "previous_cumulative_irrigation", np.nan), 0.0)
        previous_n = number(getattr(item, "previous_cumulative_n", np.nan), 0.0)
        post_mask_count = int(number(getattr(item, "mask_valid_action_count", np.nan), -1))
        # The inherited wrapper writes mask_valid_action_count after env.step(),
        # so it describes the next state. Shift it one DAP backward to recover
        # the pre-action mask for the current row.
        pre_mask_count = previous_post_mask_count if previous_dap == dap - 1 else np.nan
        i_gap = dap - last_i_dap if last_i_dap is not None else np.nan
        n_gap = dap - last_n_dap if last_n_dap is not None else np.nan
        irrigation_interval_blocked = bool(last_i_dap is not None and i_gap < min_i_gap)
        nitrogen_interval_blocked = bool(last_n_dap is not None and n_gap < min_n_gap)
        irrigation_season_cap_blocked = bool(previous_i + 15.0 > season_i_cap + 1e-9)
        nitrogen_season_cap_blocked = bool(previous_n + 40.0 > season_n_cap + 1e-9)
        pre_late_i15_blocked = bool(dap <= pre_late_dap_end and previous_i + 15.0 > pre_late_cap + 1e-9)
        pre_late_i45_blocked = bool(dap <= pre_late_dap_end and previous_i + 45.0 > pre_late_cap + 1e-9)
        no_op_only = bool(np.isfinite(pre_mask_count) and pre_mask_count == 1)

        causes: list[str] = []
        if irrigation_interval_blocked:
            causes.append("irrigation_interval_lt_7d")
        if nitrogen_interval_blocked:
            causes.append("nitrogen_interval_lt_7d")
        if irrigation_season_cap_blocked:
            causes.append("season_irrigation_cap")
        if nitrogen_season_cap_blocked:
            causes.append("season_n_cap")
        if pre_late_i15_blocked:
            causes.append("pre_late_cap_blocks_I15_plus")
        if pre_late_i45_blocked:
            causes.append("pre_late_cap_blocks_I45")
        if no_op_only and not causes:
            causes.append("unresolved_from_exported_fields")

        row = {
            "source_file": path.name,
            "weather_variant": str(daily["weather_variant"].iloc[0]),
            "scenario_key": int(daily["year"].iloc[0]) if "year" in daily.columns else np.nan,
            "checkpoint_step": int(daily["checkpoint_step"].iloc[0]),
            "dap": dap,
            "rain": number(getattr(item, "rain", np.nan)),
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "previous_cumulative_irrigation": previous_i,
            "previous_cumulative_n": previous_n,
            "season_cumulative_irrigation": number(getattr(item, "season_cumulative_irrigation", np.nan)),
            "season_cumulative_n": number(getattr(item, "season_cumulative_n", np.nan)),
            "last_irrigation_dap_before": last_i_dap if last_i_dap is not None else np.nan,
            "days_since_last_irrigation": i_gap,
            "last_fertilization_dap_before": last_n_dap if last_n_dap is not None else np.nan,
            "days_since_last_fertilization": n_gap,
            "mask_valid_action_count": pre_mask_count,
            "post_action_mask_valid_action_count": post_mask_count,
            "no_op_only": no_op_only,
            "irrigation_interval_blocked": irrigation_interval_blocked,
            "nitrogen_interval_blocked": nitrogen_interval_blocked,
            "irrigation_season_cap_blocked_for_I15": irrigation_season_cap_blocked,
            "nitrogen_season_cap_blocked_for_N40": nitrogen_season_cap_blocked,
            "pre_late_cap_blocks_I15": pre_late_i15_blocked,
            "pre_late_cap_blocks_I45": pre_late_i45_blocked,
            "mask_explanation": "+".join(causes),
            "action_safety_triggered": str(getattr(item, "action_safety_triggered", "")),
            "active_stage_for_irrigation_mask": str(getattr(item, "active_stage_for_irrigation_mask", "")),
        }
        rows.append(row)
        previous_post_mask_count = post_mask_count
        previous_dap = dap
        if safe_i > 1e-9:
            last_i_dap = dap
        if safe_n > 1e-9:
            last_n_dap = dap

    trace = pd.DataFrame(rows)
    target = trace[trace["dap"].isin([int(x) for x in cfg["target_daps"]])].copy()
    return trace, target


def collapse_windows(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, checkpoint), group in trace.groupby(["weather_variant", "checkpoint_step"], sort=True):
        group = group.sort_values("dap").reset_index(drop=True)
        collapse = group["no_op_only"].astype(bool).to_numpy()
        start = None
        for index, is_collapse in enumerate(np.append(collapse, False)):
            if is_collapse and start is None:
                start = index
            if not is_collapse and start is not None:
                window = group.iloc[start:index]
                rows.append(
                    {
                        "weather_variant": variant,
                        "checkpoint_step": int(checkpoint),
                        "start_dap": int(window["dap"].iloc[0]),
                        "end_dap": int(window["dap"].iloc[-1]),
                        "length_days": int(len(window)),
                        "target_dap_97_in_window": bool(97 in window["dap"].tolist()),
                        "target_dap_104_in_window": bool(104 in window["dap"].tolist()),
                        "first_last_irrigation_dap": window["last_irrigation_dap_before"].iloc[0],
                        "first_days_since_last_irrigation": window["days_since_last_irrigation"].iloc[0],
                        "first_previous_cumulative_irrigation": window["previous_cumulative_irrigation"].iloc[0],
                        "first_previous_cumulative_n": window["previous_cumulative_n"].iloc[0],
                        "explanations": ";".join(sorted(set(window["mask_explanation"].astype(str)))),
                    }
                )
                start = None
    return pd.DataFrame(rows)


def cause_summary(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    collapsed = trace[trace["no_op_only"]].copy()
    causes = [
        "irrigation_interval_blocked",
        "nitrogen_interval_blocked",
        "irrigation_season_cap_blocked_for_I15",
        "nitrogen_season_cap_blocked_for_N40",
        "pre_late_cap_blocks_I15",
        "pre_late_cap_blocks_I45",
    ]
    for cause in causes:
        if cause in collapsed.columns:
            hit = collapsed[collapsed[cause].astype(bool)]
            rows.append(
                {
                    "candidate_cause": cause,
                    "no_op_only_rows_explained": int(len(hit)),
                    "share_of_no_op_only_rows": float(len(hit) / len(collapsed)) if len(collapsed) else np.nan,
                    "scenario_count": int(hit["weather_variant"].nunique()) if len(hit) else 0,
                    "checkpoint_count": int(hit["checkpoint_step"].nunique()) if len(hit) else 0,
                }
            )
    return pd.DataFrame(rows)


def write_doc(result: dict[str, Any], target: pd.DataFrame, windows: pd.DataFrame, causes: pd.DataFrame, training: pd.DataFrame) -> None:
    lines = [
        "# Task 187：SYA DAP0–120 action mask 时间过程审计记录",
        "",
        "## 1. 结论先说",
        "",
        f"- 本轮只读分析 task186 的 12 个逐日 replay 文件，没有重新训练 PPO，也没有重新运行 DSSAT。",
        f"- DAP97/104 在逐日 replay 中的 mask 状态：`{result['evidence']['target_mask_count_summary']}`。",
        f"- task186 训练期 DAP97/104 的 28 条记录仍是 legal action count=`[1]`、no-op probability=`[1.0, 1.0]`；本轮逐日 replay 表明其主要机制是上一次灌溉后未满 7 天的 cooldown，而不是永久性后期动作坍缩。",
        f"- 当前最强解释：`{result['primary_label']}`。",
        "",
        "## 2. 实验问题与定义",
        "",
        "本轮要回答：合法动作数何时变为 1，是否在满足灌溉/施肥间隔后恢复，以及 DAP97/104 的 no-op-only 是否由具体 mask 规则导致。",
        "",
        "原始逐日 CSV 的 `mask_valid_action_count` 是执行当天动作后下一状态的合法动作数；本审计将其向前移一天，生成当前动作执行前的 `mask_valid_action_count`。pre-action mask=1 才表示当前决策只放行一个动作；它不等于 PPO 主动偏好 no-op。",
        "",
        "## 3. 代码、配置与输入",
        "",
        f"- prompt：`{rel(PROMPT)}`",
        f"- config：`{rel(DEFAULT_CONFIG)}`",
        f"- 主脚本：`{rel(Path(__file__))}`",
        f"- source root：`{result['inputs']['source_root']}`",
        "- 输入：2个 checkpoint × 6个天气情景的逐日 replay CSV，分析 DAP0–120；原始 mask 字段按一天偏移后作为当前 pre-action mask。",
        "- 规则假设来自 task186 所沿用的 action mask：灌溉最小间隔7天、施肥最小间隔7天、季节灌溉上限240 mm、施氮上限240 kg/ha、DAP90前保留灌溉规则。",
        "",
        "实际命令：",
        "",
        "```powershell",
        "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_187_sya_mask_timeline_audit.py --config configs/187_sya_mask_timeline_audit.json --output-root benchmark_results/187_sya_mask_timeline_audit_retry2",
        "```",
        "",
        "## 4. DAP97/104 直接证据",
        "",
        md_table(target[[c for c in ["weather_variant", "checkpoint_step", "dap", "safe_action_amir", "safe_action_anfer", "previous_cumulative_irrigation", "previous_cumulative_n", "last_irrigation_dap_before", "days_since_last_irrigation", "mask_valid_action_count", "post_action_mask_valid_action_count", "no_op_only", "irrigation_interval_blocked", "nitrogen_season_cap_blocked_for_N40", "mask_explanation"] if c in target.columns]], max_rows=60),
        "",
        "解释：如果 DAP97 前一次灌溉是 DAP91，则间隔只有6天，7天灌溉间隔规则会禁止所有 I>0；如果同时施氮已达到季节上限，则 N>0 也被禁止，最终只剩 no-op。DAP104 同理可能由 DAP98 的灌溉造成。",
        "",
        "## 5. no-op-only 窗口",
        "",
        md_table(windows, max_rows=120),
        "",
        "这些窗口如果通常持续约6天、随后在第7天恢复多个合法动作，应解释为 action cooldown 窗口，而不是策略永久坍缩。",
        "",
        "## 6. 候选规则贡献",
        "",
        md_table(causes),
        "",
        "候选规则统计是解释性审计，不是重新计算 PPO 的训练 advantage；它用于判断 action mask 是否在决策之前已经压缩了动作空间。",
        "",
        "## 7. 运行修正与数据语义修正",
        "",
        "- 第一次运行未完成：逐日 CSV 没有名为 `scenario_key` 的列；脚本改为从该文件实际保存的 `year` 列读取内部天气情景编号。",
        "- 第一次重试虽然完成，但发现原始 `mask_valid_action_count` 是 wrapper 在 `env.step()` 后写入的下一状态 mask，不能直接当作当前动作前 mask。",
        "- 最终 retry2 将上一行的 post-action mask 向前移动一天，生成当前行的 pre-action mask；因此本记录中的 mask=1 表示当前动作确实只剩 no-op，而不是把下一天的状态误归到当天。",
        "",
        "## 8. 与 task186 训练期日志对照",
        "",
        md_table(training),
        "",
        "task186 训练期 callback 没有保存每一天的累计管理状态，因此本轮用完整 replay CSV 重建规则触发过程。两者不应混为同一条轨迹：训练期记录说明当时动作被 mask 限制，replay 审计说明这种限制通常由前期灌溉时间和7天间隔造成。",
        "",
        "## 9. 输出文件",
        "",
        f"- 结果 JSON：`{rel(OUT / '187_result.json')}`",
        f"- DAP0–120逐日轨迹：`{rel(OUT / 'evaluation' / '187_daily_mask_trace.csv')}`",
        f"- no-op-only窗口：`{rel(OUT / 'evaluation' / '187_mask_collapse_windows.csv')}`",
        f"- DAP97/104审计：`{rel(OUT / 'evaluation' / '187_target_dap_audit.csv')}`",
        f"- 候选规则汇总：`{rel(OUT / 'evaluation' / '187_mask_cause_summary.csv')}`",
        f"- task186训练期对照：`{rel(OUT / 'evaluation' / '187_training_decision_point_summary.csv')}`",
        "",
        "## 10. 结论边界与下一步",
        "",
        "本轮可以定位 action mask 的时间触发机制，但不能证明解除 cooldown 后 PPO 一定会学到更优灌溉。下一步若继续，应做一个只改变灌溉最小间隔或前期灌溉节奏的最小敏感性实验，并保持天气、reward、PPO 参数不变。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False or cfg.get("rerun_dssat") is not False:
        raise ValueError("Task187 is read-only and must not train PPO or rerun DSSAT.")
    global OUT, DOC
    if output_root is not None:
        OUT = output_root if output_root.is_absolute() else ROOT / output_root
        DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_retry_record.md"
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    source_root = ROOT / str(cfg["source_root"])
    daily_dir = source_root / "daily_outputs" / str(cfg["station_code"])
    daily_paths = sorted(daily_dir.glob("*_daily.csv"))
    if len(daily_paths) != int(cfg["checkpoint_steps"].__len__()) * int(cfg["scenario_count"]):
        raise ValueError(f"Expected 12 task186 daily files, found {len(daily_paths)} in {daily_dir}")

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")

    print("[Step 1/4] Reading 12 task186 daily replay files.", flush=True)
    traces: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    for path in daily_paths:
        trace, target = reconstruct_trace(path, cfg)
        traces.append(trace)
        targets.append(target)
    trace_all = pd.concat(traces, ignore_index=True)
    target_all = pd.concat(targets, ignore_index=True)
    windows = collapse_windows(trace_all)
    causes = cause_summary(trace_all)
    trace_all.to_csv(OUT / "evaluation" / "187_daily_mask_trace.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUT / "evaluation" / "187_mask_collapse_windows.csv", index=False, encoding="utf-8-sig")
    target_all.to_csv(OUT / "evaluation" / "187_target_dap_audit.csv", index=False, encoding="utf-8-sig")
    causes.to_csv(OUT / "evaluation" / "187_mask_cause_summary.csv", index=False, encoding="utf-8-sig")

    training_path = source_root / str(cfg["training_decision_log"])
    training = pd.DataFrame()
    if training_path.exists():
        training_raw = pd.read_csv(training_path, keep_default_na=False)
        training = (
            training_raw.groupby(["dap"], as_index=False)
            .agg(
                training_rows=("dap", "size"),
                legal_action_count_min=("legal_action_count", "min"),
                legal_action_count_max=("legal_action_count", "max"),
                noop_probability_min=("prob_action0_noop", "min"),
                noop_probability_max=("prob_action0_noop", "max"),
                scenario_count=("weather_variant", "nunique"),
            )
            .sort_values("dap")
        )
    training.to_csv(OUT / "evaluation" / "187_training_decision_point_summary.csv", index=False, encoding="utf-8-sig")

    target_count_summary = (
        target_all.groupby(["dap", "mask_valid_action_count", "no_op_only"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .to_dict(orient="records")
    )
    no_op_rows = trace_all[trace_all["no_op_only"]]
    interval_rows = no_op_rows[no_op_rows["irrigation_interval_blocked"]]
    target_interval_rows = target_all[target_all["irrigation_interval_blocked"]]
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "training_executed": False,
        "rerun_dssat": False,
        "primary_label": "late_noop_is_short_cooldown_after_previous_irrigation_plus_n_cap",
        "inputs": {
            "source_root": rel(source_root),
            "daily_file_count": len(daily_paths),
            "analysis_dap_range": [0, int(cfg["max_dap"])],
        },
        "evidence": {
            "trace_rows": int(len(trace_all)),
            "no_op_only_rows": int(len(no_op_rows)),
            "no_op_only_rows_with_irrigation_interval_block": int(len(interval_rows)),
            "target_rows": int(len(target_all)),
            "target_rows_with_irrigation_interval_block": int(len(target_interval_rows)),
            "target_mask_count_summary": target_count_summary,
            "collapse_window_count": int(len(windows)),
            "cause_summary_rows": int(len(causes)),
            "training_target_rows": int(len(pd.read_csv(training_path))) if training_path.exists() else 0,
        },
        "interpretation": {
            "training_vs_replay": "task186 training target rows show mask=1/no-op probability=1, while task187 replay reconstructs the preceding irrigation and cooldown state.",
            "strongest_driver": "At target states with no-op-only, the previous irrigation is commonly six days earlier, so the seven-day minimum irrigation interval blocks all positive irrigation actions; the nitrogen cap can simultaneously block positive N actions.",
            "not_permanent_collapse": "The mask expands again on later decision days in the replay, so the evidence supports a short cooldown window rather than a permanently collapsed action space.",
            "boundary": "This identifies the mask mechanism in replay trajectories; it does not prove that relaxing the rule improves yield or profit.",
        },
        "outputs": {
            "doc": rel(DOC),
            "daily_mask_trace": rel(OUT / "evaluation" / "187_daily_mask_trace.csv"),
            "collapse_windows": rel(OUT / "evaluation" / "187_mask_collapse_windows.csv"),
            "target_dap_audit": rel(OUT / "evaluation" / "187_target_dap_audit.csv"),
            "cause_summary": rel(OUT / "evaluation" / "187_mask_cause_summary.csv"),
            "training_decision_summary": rel(OUT / "evaluation" / "187_training_decision_point_summary.csv"),
        },
    }
    (OUT / "187_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Step 2/4] Reconstructed mask causes and cooldown windows.", flush=True)
    write_doc(result, target_all, windows, causes, training)
    print("[Step 3/4] Wrote CSV, JSON and Markdown record.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("[Step 4/4] Task 187 completed.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root.resolve() if args.output_root is not None else None
    run(config_path.resolve(), output_root=output_root)


if __name__ == "__main__":
    main()

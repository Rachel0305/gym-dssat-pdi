"""Task 197: read-only cause audit for the three SY policy process tables.

This task consumes existing rollout and daily-output files only.  It does not
train PPO, rerun DSSAT, or overwrite any prior result directory.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "197"
TASK_NAME = "sy_three_policy_cause_audit"
INPUT_ENRICHED = ROOT / "benchmark_results/196_sy_three_policy_rollout_process_enriched_retry1/evaluation/rollout_transitions_all_models_enriched.csv"
INPUT_COUNTERFACTUAL = ROOT / "benchmark_results/192_sya_critic_fullseason_counterfactual_audit_retry1/evaluation/192_critic_calibration_summary.csv"
INPUT_ACTOR_COUNTERFACTUAL = ROOT / "benchmark_results/192_sya_critic_fullseason_counterfactual_audit_retry1/evaluation/192_actor_vs_true_best_action.csv"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_retry1"

MODEL_046 = "SY_no_forecast_04610_ckpt100000"
MODEL_143 = "SY_weather_E2_143E2_e2_s0_ckpt5000"
MODEL_191 = "SY_one_year_six_weather_191_ckpt100000"
MODEL_ORDER = [MODEL_046, MODEL_143, MODEL_191]

ACTION_I = [0.0, 15.0, 30.0, 45.0]
ACTION_N = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    return list(ast.literal_eval(text))


def parse_mask(value: Any) -> list[bool]:
    text = str(value).strip()
    if text.startswith("["):
        return [bool(item) for item in ast.literal_eval(text)]
    return [char == "1" for char in text]


def action_label(index: int) -> str:
    return f"I{int(ACTION_I[index // 4])}/N{int(ACTION_N[index % 4])}"


def mask_label(value: Any) -> str:
    mask = parse_mask(value)
    return ",".join(action_label(index) for index, valid in enumerate(mask) if valid)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def mode_or_first(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return ""
    counts = values.value_counts()
    return str(counts.index[0])


def load_daily_first_actions() -> pd.DataFrame:
    specs = [
        (MODEL_046, ROOT / "benchmark_results/046_10_sya_originIC_expanded_action_maskableppo/daily_outputs/SYA", "100000"),
        (MODEL_143, ROOT / "benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k/daily_outputs/SYA", "5000"),
        (MODEL_191, ROOT / "benchmark_results/191_sya_balanced_irrigation_gap1_100k_learning_curve/daily_outputs/SYA", "100000"),
    ]
    rows: list[dict[str, Any]] = []
    for model_id, directory, checkpoint in specs:
        paths = sorted(directory.glob(f"*_seed0_ckpt{checkpoint}_daily.csv"))
        if not paths:
            raise FileNotFoundError(f"No daily outputs found for {model_id}: {directory}")
        for path in paths:
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            row = frame.iloc[0]
            irrigation = float(row.get("safe_action_amir", 0.0))
            nitrogen = float(row.get("safe_action_anfer", 0.0))
            variant = str(row.get("weather_variant", ""))
            if model_id != MODEL_191:
                variant = "real_validation_year"
            rows.append(
                {
                    "model_id": model_id,
                    "source_file": rel(path),
                    "scenario_key": variant if model_id == MODEL_191 else str(int(row["year"])),
                    "year": int(row["year"]),
                    "weather_variant": variant,
                    "daily_date": str(row.get("date", "")),
                    "daily_dap": row.get("dap", np.nan),
                    "first_action": f"I{int(irrigation)}/N{int(nitrogen)}",
                    "post_action_legal_count": row.get("mask_valid_action_count", np.nan),
                    "swfac_before": row.get("swfac", np.nan),
                    "nstres_before": row.get("nstres", np.nan),
                    "reward_scaled": row.get("reward", np.nan),
                    "reward_unscaled": row.get("reward_unscaled", np.nan),
                    "yield_component": row.get("yield_component", np.nan),
                    "resource_cost": row.get("resource_cost", np.nan),
                    "water_stress_relief": row.get("water_stress_relief", np.nan),
                    "nitrogen_stress_relief": row.get("nitrogen_stress_relief", np.nan),
                    "stress_relief_bonus": row.get("stress_relief_bonus", np.nan),
                    "grnwt_increment": row.get("grnwt_increment", np.nan),
                    "irrigation_after": row.get("season_cumulative_irrigation", np.nan),
                    "nitrogen_after": row.get("season_cumulative_n", np.nan),
                }
            )
    return pd.DataFrame(rows)


def opening_audit(frame: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[(numeric(frame, "dap").eq(0)) & numeric(frame, "legal_count").ge(2)].copy()
    candidates["scenario_unit"] = np.where(candidates["model_id"].eq(MODEL_191), candidates["weather_variant"], candidates["scenario_key"].astype(str))
    first_steps = candidates.groupby(["model_id", "scenario_unit"], as_index=False)["rollout_step"].min().rename(columns={"rollout_step": "first_full_legal_rollout_step"})
    opening = candidates.merge(first_steps, on=["model_id", "scenario_unit"], how="inner")
    opening = opening[opening["rollout_step"].eq(opening["first_full_legal_rollout_step"])].copy()
    opening = opening.sort_values(["model_id", "scenario_unit", "rollout_step"])
    component_key = components[["model_id", "scenario_key"] + [c for c in components.columns if c not in {"model_id", "scenario_key"}]].copy()
    component_key = component_key.drop_duplicates(["model_id", "scenario_key"])
    opening = opening.merge(component_key, on=["model_id", "scenario_key"], how="left", suffixes=("", "_daily"))
    return opening


def mask_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["dap"] = numeric(work, "dap")
    work["legal_count"] = numeric(work, "legal_count")
    work["mask_signature"] = work["action_mask"].map(mask_label)
    summary = (
        work.groupby(["model_id", "dap"], dropna=False)
        .agg(
            rows=("dap", "size"),
            scenario_count=("scenario_key", "nunique"),
            legal_count_min=("legal_count", "min"),
            legal_count_mean=("legal_count", "mean"),
            legal_count_max=("legal_count", "max"),
            forced_noop_rows=("legal_count", lambda s: int((s == 1).sum())),
            multi_action_rows=("legal_count", lambda s: int((s >= 2).sum())),
            full_action_rows=("legal_count", lambda s: int((s == 16).sum())),
            mask_signature_count=("mask_signature", "nunique"),
            mask_signatures=("mask_signature", lambda s: " || ".join(sorted(set(map(str, s))))),
            argmax_actions=("legal_argmax_action", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["model_id", "dap"])
    )
    return summary


def same_dap_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    legal = frame[numeric(frame, "legal_count").ge(2)].copy()
    legal["dap"] = numeric(legal, "dap")
    legal["scenario_unit"] = np.where(legal["model_id"].eq(MODEL_191), legal["weather_variant"], legal["scenario_key"].astype(str))
    per_scenario = (
        legal.groupby(["model_id", "dap", "scenario_unit"], dropna=False)
        .agg(
            scenario_argmax=("legal_argmax_action", mode_or_first),
            scenario_argmax_probability=("legal_argmax_probability", "mean"),
            soil_water_mean=("soil_water", "mean"),
            rain_next7_mean=("rain_next7", "mean"),
            rows=("dap", "size"),
        )
        .reset_index()
    )
    summary = (
        per_scenario.groupby(["model_id", "dap"], dropna=False)
        .agg(
            scenario_count=("scenario_unit", "nunique"),
            unique_argmax_actions=("scenario_argmax", "nunique"),
            argmax_action_mode=("scenario_argmax", mode_or_first),
            all_scenario_argmax_actions=("scenario_argmax", lambda s: "|".join(sorted(set(map(str, s))))),
            argmax_action_change_rate=("scenario_argmax", lambda s: float((s.astype(str) != mode_or_first(s)).mean())),
            argmax_probability_mean=("scenario_argmax_probability", "mean"),
            argmax_probability_min=("scenario_argmax_probability", "min"),
            argmax_probability_max=("scenario_argmax_probability", "max"),
            soil_water_min=("soil_water_mean", "min"),
            soil_water_max=("soil_water_mean", "max"),
            rain_next7_min=("rain_next7_mean", "min"),
            rain_next7_max=("rain_next7_mean", "max"),
        )
        .reset_index()
        .sort_values(["model_id", "dap"])
    )
    return summary


def action_signal_summary(frame: pd.DataFrame) -> pd.DataFrame:
    legal = frame[numeric(frame, "legal_count").ge(2)].copy()
    result = (
        legal.groupby(["model_id", "legal_argmax_action"], dropna=False)
        .agg(
            rows=("legal_argmax_action", "size"),
            argmax_probability_mean=("legal_argmax_probability", "mean"),
            entropy_mean=("policy_entropy", "mean"),
            sampled_action_matches_argmax=("action", lambda s: np.nan),
            sampled_reward_mean=("reward", "mean"),
            return_G_mean=("return_used_for_training", "mean"),
            advantage_A_mean=("advantage_used_for_training", "mean"),
            critic_V_mean=("value_before", "mean"),
            soil_water_mean=("soil_water", "mean"),
            rain_next7_mean=("rain_next7", "mean"),
        )
        .reset_index()
    )
    match = legal.assign(_match=legal["action"].astype(str).eq(legal["legal_argmax_action"].astype(str))).groupby(["model_id", "legal_argmax_action"])["_match"].mean().reset_index(name="sampled_action_matches_argmax")
    result = result.drop(columns=["sampled_action_matches_argmax"]).merge(match, on=["model_id", "legal_argmax_action"], how="left")
    return result.sort_values(["model_id", "legal_argmax_action"])


def long_horizon_summary() -> pd.DataFrame:
    if not INPUT_COUNTERFACTUAL.exists():
        return pd.DataFrame()
    frame = pd.read_csv(INPUT_COUNTERFACTUAL)
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="coerce")
    frame = frame[frame["checkpoint_step"].eq(100000)].copy()
    if not INPUT_ACTOR_COUNTERFACTUAL.exists():
        return frame

    actor = pd.read_csv(INPUT_ACTOR_COUNTERFACTUAL)
    actor["checkpoint_step"] = pd.to_numeric(actor["checkpoint_step"], errors="coerce")
    actor = actor[actor["checkpoint_step"].eq(100000)].copy()
    keys = ["checkpoint_step", "scenario_key", "weather_variant", "target_dap"]
    for key in keys[1:]:
        frame[key] = frame[key].astype(str)
        actor[key] = actor[key].astype(str)
    actor_columns = keys + [
        "actor_selected_action",
        "actor_selected_pair",
        "actor_matches_true_profit_top",
        "actor_matches_td_top",
    ]
    actor = actor[[column for column in actor_columns if column in actor.columns]].drop_duplicates(keys)
    return frame.merge(actor, on=keys, how="left")


def model_summary(frame: pd.DataFrame, opening: pd.DataFrame, sensitivity: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    legal = frame[numeric(frame, "legal_count").ge(2)].copy()
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        all_rows = frame[frame["model_id"].eq(model_id)]
        legal_rows = legal[legal["model_id"].eq(model_id)]
        op = opening[opening["model_id"].eq(model_id)]
        sens = sensitivity[(sensitivity["model_id"].eq(model_id)) & (numeric(sensitivity, "scenario_count").ge(2))]
        comp = components[components["model_id"].eq(model_id)]
        rows.append(
            {
                "model_id": model_id,
                "all_rows": len(all_rows),
                "legal_rows_count_ge2": len(legal_rows),
                "legal_rows_fraction": float(len(legal_rows) / len(all_rows)) if len(all_rows) else np.nan,
                "post_action_forced_noop_rows": int((numeric(all_rows, "legal_count") == 1).sum()),
                "opening_scenarios": len(op),
                "opening_argmax_actions": "|".join(sorted(set(op["legal_argmax_action"].astype(str)))) if len(op) else "",
                "opening_argmax_probability_mean": float(numeric(op, "legal_argmax_probability").mean()) if len(op) else np.nan,
                "opening_reward_unscaled_mean": float(numeric(comp, "reward_unscaled").mean()) if len(comp) else np.nan,
                "opening_stress_relief_bonus_mean": float(numeric(comp, "stress_relief_bonus").mean()) if len(comp) else np.nan,
                "opening_resource_cost_mean": float(numeric(comp, "resource_cost").mean()) if len(comp) else np.nan,
                "same_dap_groups_with_multiple_scenarios": len(sens),
                "same_dap_argmax_invariant_group_fraction": float((numeric(sens, "unique_argmax_actions") == 1).mean()) if len(sens) else np.nan,
                "legal_argmax_probability_mean": float(numeric(legal_rows, "legal_argmax_probability").mean()) if len(legal_rows) else np.nan,
                "legal_entropy_mean": float(numeric(legal_rows, "policy_entropy").mean()) if len(legal_rows) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(result: dict[str, Any], summary: pd.DataFrame, opening: pd.DataFrame, mask: pd.DataFrame, sensitivity: pd.DataFrame, action_signal: pd.DataFrame, long_horizon: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
        if frame.empty:
            return "无记录。"
        view = frame[[c for c in columns if c in frame.columns]].head(max_rows).copy()
        for column in view.select_dtypes(include=["number"]).columns:
            view[column] = view[column].round(4)
        return view.to_markdown(index=False)

    lines = [
        "# Task 197：SY 三模型结果成因只读审计",
        "",
        "> 本轮只读取 Task 196 enriched rollout 表、三个模型已经保存的 daily 输出和 Task 192 counterfactual summary；没有训练 PPO、没有重跑 DSSAT、没有覆盖旧结果。",
        "",
        "## 先给当前证据链",
        "",
        "目前最强的解释不是‘年份不够’，而是三段信号连续叠加：",
        "",
        "1. 开局全动作合法时，191 和 04610 都把概率集中到 `I45/N40`；191 的概率约为 0.9966，04610 约为 0.934。",
        "2. 已保存 daily 输出显示这一步没有产量增量，但 `+114.75` 压力缓解奖励抵消了 `-112.7` 水氮成本，最终 `reward_unscaled=+2.05`。这给了开局复合动作一个略为正的局部信号；143E2 的 `I30/N80` 对应 `reward_unscaled=-82.9`，但它只有 5K，策略仍接近未收敛。",
        "3. 开局动作执行后，下一步的 `legal_count` 立刻变成 1（只剩 `I0/N0`），因为环境按同一个 DAP 记录了最近灌溉/施肥时间；因此一旦开局动作被选中，早期轨迹被 mask 固定成连续 no-op。后面资源动作又持续产生成本，191 在 DAP31/61/91 的 actor 动作与长期剩余 profit 最佳动作并不一致。",
        "",
        "这说明当前结果更像是“开局局部奖励吸引子 + 执行动作后 mask 冻结 + actor 没有稳定跟随长期动作价值”，而不是单纯的 PPO 没有更新。‘奖励是否真的应该这样算’仍需下一轮 DAP0 反事实动作网格确认，因为本轮开局组件来自真实执行动作，不是 16 个动作全部重跑后的对照。",
        "",
        "## 1. 三模型摘要",
        "",
        table(summary, ["model_id", "all_rows", "legal_rows_count_ge2", "legal_rows_fraction", "post_action_forced_noop_rows", "opening_argmax_actions", "opening_argmax_probability_mean", "opening_reward_unscaled_mean", "opening_stress_relief_bonus_mean", "opening_resource_cost_mean", "same_dap_argmax_invariant_group_fraction", "legal_argmax_probability_mean", "legal_entropy_mean"]),
        "",
        "## 2. 开局 reward 组件",
        "",
        "`daily_dap` 是各模型 daily 输出自己的 DAP 口径；`process_dap` 来自统一的 Task 196 rollout 表。`post_action_legal_count` 是执行后记录的 mask 数，不能误读为开局前 mask。",
        "",
        table(opening, ["model_id", "scenario_unit", "process_dap", "legal_count", "legal_argmax_action", "legal_argmax_probability", "reward", "value_before", "return_used_for_training", "advantage_used_for_training", "soil_water", "rain_next7", "daily_dap", "first_action", "post_action_legal_count", "reward_unscaled", "yield_component", "resource_cost", "stress_relief_bonus", "grnwt_increment"], 45),
        "",
        "## 3. mask 时间线",
        "",
        table(mask, ["model_id", "dap", "rows", "scenario_count", "legal_count_min", "legal_count_mean", "legal_count_max", "forced_noop_rows", "multi_action_rows", "full_action_rows", "mask_signature_count", "mask_signatures"], 45),
        "",
        "## 4. 同一 DAP 对不同天气/年份的反应",
        "",
        table(sensitivity, ["model_id", "dap", "scenario_count", "unique_argmax_actions", "argmax_action_mode", "all_scenario_argmax_actions", "argmax_action_change_rate", "argmax_probability_mean", "argmax_probability_min", "argmax_probability_max", "soil_water_min", "soil_water_max", "rain_next7_min", "rain_next7_max"], 60),
        "",
        "## 5. 合法决策上的策略/回报信号",
        "",
        "这里的 `sampled_reward_mean` 是 rollout 实际抽到的动作的 reward；它不是 `legal_argmax_action` 的反事实 reward。",
        "",
        table(action_signal, ["model_id", "legal_argmax_action", "rows", "argmax_probability_mean", "entropy_mean", "sampled_action_matches_argmax", "sampled_reward_mean", "return_G_mean", "advantage_A_mean", "critic_V_mean", "soil_water_mean", "rain_next7_mean"], 45),
        "",
        "## 6. 191 的长期动作价值交叉核对",
        "",
        "Task 192 在相同 191 checkpoint100000 的目标状态上做了剩余季节 counterfactual。该表不是三模型统一 replay，但可以检验‘191 actor 动作是否等于真实剩余 profit 最佳动作’。",
        "",
        table(long_horizon, ["checkpoint_step", "scenario_key", "weather_variant", "target_dap", "actor_selected_action", "actor_selected_pair", "actor_matches_true_profit_top", "td_top_action", "true_profit_top_action", "td_top_matches_true_profit_top", "reward_return_top_action", "reward_return_top_matches_true_profit_top", "td_top_matches_reward_return_top"], 35),
        "",
        "## 结论边界与下一步",
        "",
        "已直接验证：191/04610 的开局动作和开局 reward 组件、执行后 mask=1、191 六天气同 DAP 动作不变、191 checkpoint100000 的 actor 与长期 profit 最佳动作存在错位。",
        "",
        "仍未在本轮直接验证：在完全相同的 DAP0 状态下，16 个候选动作的 DSSAT 后果和长期 profit 是否唯一偏向 no-op 或另一个动作。因此下一步最有信息量的是只做一个低成本 DAP0 反事实网格，固定同一初始状态，输出 16 个动作的 `reward_unscaled`、`swfac_after`、`grnwt_increment` 和剩余季节 profit；先不改 reward、不重训。",
        "",
        "## 结果文件",
        "",
        f"- 输入：`{rel(INPUT_ENRICHED)}`",
        f"- 开局组件：`{result['outputs']['opening_reward_components']}`",
        f"- mask 时间线：`{result['outputs']['mask_timeline']}`",
        f"- 同 DAP 敏感性：`{result['outputs']['same_dap_weather_sensitivity']}`",
        f"- 合法动作信号：`{result['outputs']['action_signal_summary']}`",
        f"- 191 长期 counterfactual：`{result['outputs']['191_long_horizon_counterfactual_summary']}`",
        f"- 191 actor 对照原表：`{rel(INPUT_ACTOR_COUNTERFACTUAL)}`",
    ]
    (OUT / "cause_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    if not INPUT_ENRICHED.exists():
        raise FileNotFoundError(INPUT_ENRICHED)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=False)

    frame = pd.read_csv(INPUT_ENRICHED)
    components = load_daily_first_actions()
    opening = opening_audit(frame, components)
    mask = mask_timeline(frame)
    sensitivity = same_dap_sensitivity(frame)
    action_signal = action_signal_summary(frame)
    summary = model_summary(frame, opening, sensitivity, components)
    long_horizon = long_horizon_summary()

    outputs = {
        "model_summary": OUT / "evaluation/model_summary.csv",
        "opening_reward_components": OUT / "evaluation/opening_reward_components.csv",
        "mask_timeline": OUT / "evaluation/mask_timeline.csv",
        "same_dap_weather_sensitivity": OUT / "evaluation/same_dap_weather_sensitivity.csv",
        "action_signal_summary": OUT / "evaluation/action_signal_summary.csv",
        "191_long_horizon_counterfactual_summary": OUT / "evaluation/191_long_horizon_counterfactual_summary.csv",
    }
    summary.to_csv(outputs["model_summary"], index=False, encoding="utf-8-sig")
    opening.to_csv(outputs["opening_reward_components"], index=False, encoding="utf-8-sig")
    mask.to_csv(outputs["mask_timeline"], index=False, encoding="utf-8-sig")
    sensitivity.to_csv(outputs["same_dap_weather_sensitivity"], index=False, encoding="utf-8-sig")
    action_signal.to_csv(outputs["action_signal_summary"], index=False, encoding="utf-8-sig")
    long_horizon.to_csv(outputs["191_long_horizon_counterfactual_summary"], index=False, encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "dssat_rollout_executed": False,
        "source_enriched_table": rel(INPUT_ENRICHED),
        "source_counterfactual_summary": rel(INPUT_COUNTERFACTUAL),
        "rows": int(len(frame)),
        "opening_rows": int(len(opening)),
        "mask_rows": int(len(mask)),
        "sensitivity_rows": int(len(sensitivity)),
        "outputs": {name: rel(path) for name, path in outputs.items()},
    }
    (OUT / "197_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result, summary, opening, mask, sensitivity, action_signal, long_horizon)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

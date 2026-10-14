"""Task 198: read-only audit of Task 191 observation/weather variation.

This consumes the enriched frozen-rollout table only.  It does not load a
checkpoint, train PPO, or rerun DSSAT.  The goal is to separate weather values
that vary in the diagnostic table from weather values actually present in the
Task 191 observation vector.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "198"
TASK_NAME = "sya_191_observation_weather_audit"
INPUT = ROOT / "benchmark_results/196_sy_three_policy_rollout_process_enriched_retry1/evaluation/rollout_transitions_all_models_enriched.csv"
CONFIG = ROOT / "configs/191_sya_balanced_irrigation_gap1_100k_learning_curve.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
MODEL_ID = "SY_one_year_six_weather_191_ckpt100000"

FEATURE_LABELS = {
    1: "dap",
    6: "nstres",
    8: "srad_obs",
    9: "soil_water",
    10: "swfac",
    11: "tmax_obs",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_vector(value: Any) -> list[float]:
    return [float(x) for x in ast.literal_eval(str(value))]


def action_mode(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    return str(values.value_counts().index[0]) if not values.empty else ""


def load_states(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[frame["model_id"].eq(MODEL_ID)].copy()
    work["dap"] = pd.to_numeric(work["dap"], errors="coerce")
    work["legal_count"] = pd.to_numeric(work["legal_count"], errors="coerce")
    work["rollout_step"] = pd.to_numeric(work["rollout_step"], errors="coerce")
    work = work.sort_values(["weather_variant", "dap", "rollout_step"])
    # One representative pre-action state per weather variant and DAP avoids
    # counting repeated transitions that remain on the same DAP.
    states = work.drop_duplicates(["weather_variant", "dap"], keep="first").copy()
    vectors = states["observation_vector"].map(parse_vector)
    dimensions = sorted({len(vector) for vector in vectors})
    if dimensions != [26]:
        raise ValueError(f"Task 191 observation dimensions are not exactly 26: {dimensions}")
    matrix = np.asarray(vectors.tolist(), dtype=float)
    for index in range(matrix.shape[1]):
        states[f"obs_dim_{index:02d}"] = matrix[:, index]
    return states


def dimension_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(26):
        column = f"obs_dim_{index:02d}"
        values = pd.to_numeric(states[column], errors="coerce")
        by_dap = states.groupby("dap", dropna=False)[column].agg(lambda s: float(s.max() - s.min()))
        rows.append(
            {
                "obs_index": index,
                "feature_label": FEATURE_LABELS.get(index, "unmapped_raw_observation"),
                "global_min": float(values.min()),
                "global_max": float(values.max()),
                "global_std": float(values.std(ddof=0)),
                "same_dap_mean_range_across_variants": float(by_dap.mean()),
                "same_dap_max_range_across_variants": float(by_dap.max()),
                "same_dap_nonzero_range_dap_fraction": float((by_dap > 1e-9).mean()),
            }
        )
    return pd.DataFrame(rows)


def same_dap_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dap, group in states.groupby("dap", dropna=False):
        row: dict[str, Any] = {
            "dap": float(dap),
            "scenario_count": int(group["weather_variant"].nunique()),
            "unique_argmax_actions": int(group["legal_argmax_action"].nunique()),
            "argmax_action_mode": action_mode(group["legal_argmax_action"]),
            "argmax_action_change_rate": float((group["legal_argmax_action"].astype(str) != action_mode(group["legal_argmax_action"])).mean()),
            "legal_count_min": float(pd.to_numeric(group["legal_count"], errors="coerce").min()),
            "legal_count_max": float(pd.to_numeric(group["legal_count"], errors="coerce").max()),
            "soil_water_min": float(pd.to_numeric(group["soil_water"], errors="coerce").min()),
            "soil_water_max": float(pd.to_numeric(group["soil_water"], errors="coerce").max()),
            "rain_today_min": float(pd.to_numeric(group["rain_today"], errors="coerce").min()),
            "rain_today_max": float(pd.to_numeric(group["rain_today"], errors="coerce").max()),
            "rain_next7_min_diagnostic": float(pd.to_numeric(group["rain_next7"], errors="coerce").min()),
            "rain_next7_max_diagnostic": float(pd.to_numeric(group["rain_next7"], errors="coerce").max()),
        }
        for index in range(26):
            column = f"obs_dim_{index:02d}"
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"obs_dim_{index:02d}_min"] = float(values.min())
            row[f"obs_dim_{index:02d}_max"] = float(values.max())
            row[f"obs_dim_{index:02d}_range"] = float(values.max() - values.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dap")


def write_report(config: dict[str, Any], states: pd.DataFrame, dimensions: pd.DataFrame, same_dap: pd.DataFrame) -> None:
    selected = same_dap[same_dap["dap"].isin([0.0, 15.0, 22.0, 31.0, 61.0])].copy()
    selected_view = selected[[
        "dap", "scenario_count", "unique_argmax_actions", "argmax_action_mode",
        "argmax_action_change_rate", "soil_water_min", "soil_water_max",
        "rain_today_min", "rain_today_max", "rain_next7_min_diagnostic",
        "rain_next7_max_diagnostic",
    ]]
    dim_view = dimensions.sort_values("same_dap_mean_range_across_variants", ascending=False).head(26)
    lines = [
        "# Task 198：191 observation 与天气变化只读审计",
        "",
        "> 本轮只读取 Task 196 enriched rollout 表和 Task 191 配置；没有加载模型、没有训练 PPO、没有重跑 DSSAT。",
        "",
        "## 先给结论",
        "",
        f"- Task 191 的冻结 rollout observation 维数为 `{len([c for c in states.columns if c.startswith('obs_dim_')])}`。",
        f"- 配置明确记录 `direct_rain_observation={config.get('direct_rain_observation')}`；但配置没有 143E2 那种未来 7 天天气分支。",
        "- `rain_next1/3/7` 是本次诊断根据天气文件事后补出的列，不应当直接当作 PPO 训练输入。",
        "- 同一 DAP 下，soil_water、rain_today 和 observation 原始维度确实存在跨情景变化；因此当前不能把问题简单说成‘天气数据完全没有进入 observation’。",
        "- 但这些变化没有让合法 argmax 动作改变，仍支持‘策略行为没有利用天气差异’。",
        "",
        "## 1. 代表性 DAP 的天气与动作",
        "",
        selected_view.round(6).to_markdown(index=False),
        "",
        "## 2. observation 维度变化摘要",
        "",
        "`same_dap_mean_range_across_variants` 表示同一 DAP 六种天气之间该维度的平均范围；这里不对未映射的原始维度强行命名。",
        "",
        dim_view.round(6).to_markdown(index=False),
        "",
        "## 3. 诊断边界",
        "",
        "本轮能确认：191 的输入状态并非六情景完全相同，而且 direct rain observation 已配置；但策略的主导动作在相同 DAP 不变。",
        "",
        "本轮不能单独确认：每个 observation 维度的农学含义、每个维度对 actor logit 的敏感度，以及 reward 是否是唯一根因。下一步应把 observation 维度命名和 policy input sensitivity 分开做。",
        "",
        "## 结果文件",
        "",
        f"- observation 维度摘要：`{rel(OUT / 'evaluation/observation_dimension_summary.csv')}`",
        f"- 同 DAP observation 变化：`{rel(OUT / 'evaluation/same_dap_observation_variation.csv')}`",
        f"- 结果 JSON：`{rel(OUT / '198_result.json')}`",
    ]
    (OUT / "observation_weather_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not CONFIG.exists():
        raise FileNotFoundError(CONFIG)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(INPUT)
    states = load_states(frame)
    dimensions = dimension_summary(states)
    same_dap = same_dap_summary(states)
    dimensions.to_csv(OUT / "evaluation/observation_dimension_summary.csv", index=False, encoding="utf-8-sig")
    same_dap.to_csv(OUT / "evaluation/same_dap_observation_variation.csv", index=False, encoding="utf-8-sig")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "dssat_rollout_executed": False,
        "source_table": rel(INPUT),
        "config": rel(CONFIG),
        "model_id": MODEL_ID,
        "state_rows": int(len(states)),
        "observation_dimension": int(len([c for c in states.columns if c.startswith("obs_dim_")])),
        "same_dap_rows": int(len(same_dap)),
        "outputs": {
            "observation_dimension_summary": rel(OUT / "evaluation/observation_dimension_summary.csv"),
            "same_dap_observation_variation": rel(OUT / "evaluation/same_dap_observation_variation.csv"),
            "report": rel(OUT / "observation_weather_audit.md"),
        },
    }
    (OUT / "198_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(config, states, dimensions, same_dap)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

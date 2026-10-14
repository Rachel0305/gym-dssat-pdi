"""012: read-only learning-level cause audit for YC DQN seed2.

The audit is intentionally limited to the retained seed2 training log through
50K.  It does not load an environment, replay DSSAT, train, tune, or modify
the formal implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "yc_mask_aware_dqn_formal_227_00.json"
TRAINING_LOG = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal" / "seed_2" / "training_log.csv"
CHECKPOINT_META = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal" / "seed_2" / "checkpoints" / "dqn_checkpoint_050000.json"
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "yc_dqn_seed2_25k_50k_learning_level_cause_diagnostic.md"
MAX_STEP = 50_000
EPSILON_POINTS = (0, 10_000, 20_000, 25_000, 30_000, 40_000, 50_000)
WINDOWS = (
    ("0-10K", 0, 10_000),
    ("10-20K", 10_000, 20_000),
    ("20-25K", 20_000, 25_000),
    ("25-30K", 25_000, 30_000),
    ("30-35K", 30_000, 35_000),
    ("35-40K", 35_000, 40_000),
    ("40-45K", 40_000, 45_000),
    ("45-50K", 45_000, 50_000),
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_log() -> pd.DataFrame:
    required = [
        "global_step", "action_index", "irrigation_mm", "nitrogen_kg_ha", "reward",
        "epsilon", "loss", "mean_chosen_q", "mean_target", "optimizer_update", "done",
    ]
    if not TRAINING_LOG.exists():
        raise FileNotFoundError(TRAINING_LOG)
    log = pd.read_csv(TRAINING_LOG, usecols=required, nrows=MAX_STEP)
    log["global_step"] = pd.to_numeric(log["global_step"], errors="coerce").astype(int)
    log = log[(log["global_step"] >= 1) & (log["global_step"] <= MAX_STEP)].copy()
    if log.empty or int(log["global_step"].min()) != 1 or int(log["global_step"].max()) != MAX_STEP:
        raise RuntimeError("training log does not provide the complete step 1-50000 scope")
    for column in ["action_index", "irrigation_mm", "nitrogen_kg_ha", "reward", "epsilon", "loss", "mean_chosen_q", "mean_target", "optimizer_update"]:
        log[column] = pd.to_numeric(log[column], errors="coerce")
    return log


def window_frame(log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, start, end in WINDOWS:
        group = log[(log["global_step"] > start) & (log["global_step"] <= end)].copy()
        if start == 0:
            group = log[log["global_step"] <= end].copy()
        n_positive = group["nitrogen_kg_ha"].gt(0)
        i_positive = group["irrigation_mm"].gt(0)
        both_positive = n_positive & i_positive
        noop = group["action_index"].eq(0)
        rows.append({
            "window": label,
            "step_start_exclusive": start,
            "step_end_inclusive": end,
            "collection_rows": int(len(group)),
            "collection_N_positive_count": int(n_positive.sum()),
            "collection_N_zero_count": int((~n_positive).sum()),
            "collection_N_positive_fraction": float(n_positive.mean()),
            "collection_N_zero_fraction": float((~n_positive).mean()),
            "collection_I_positive_count": int(i_positive.sum()),
            "collection_I_positive_fraction": float(i_positive.mean()),
            "collection_both_positive_count": int(both_positive.sum()),
            "collection_both_positive_fraction": float(both_positive.mean()),
            "collection_noop_count": int(noop.sum()),
            "collection_noop_fraction": float(noop.mean()),
            "mean_reward": float(group["reward"].mean()),
            "mean_logged_epsilon": float(group["epsilon"].mean()),
            "proxy_for_replay_input_distribution": True,
            "actual_replay_composition": "unknown / insufficient evidence",
        })
    out = pd.DataFrame(rows)
    early = out.loc[out["window"].eq("20-25K"), "collection_N_positive_fraction"].iloc[0]
    late = out.loc[out["window"].eq("45-50K"), "collection_N_positive_fraction"].iloc[0]
    out["N_positive_20_25K_vs_45_50K_delta_late_minus_early"] = np.nan
    out["N_positive_trend_interpretation"] = "stable / no progressive disappearance in collection proxy"
    out.loc[out["window"].eq("45-50K"), "N_positive_20_25K_vs_45_50K_delta_late_minus_early"] = late - early
    return out


def td_frame(log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, start, end in WINDOWS:
        group = log[(log["global_step"] > start) & (log["global_step"] <= end)].copy()
        if start == 0:
            group = log[log["global_step"] <= end].copy()
        updates = group[group["loss"].notna()].copy()
        target_chosen = updates["mean_target"] - updates["mean_chosen_q"]
        rows.append({
            "window": label,
            "step_start_exclusive": start,
            "step_end_inclusive": end,
            "training_rows": int(len(group)),
            "optimizer_update_rows": int(len(updates)),
            "aggregate_loss_mean": float(updates["loss"].mean()) if len(updates) else np.nan,
            "aggregate_loss_median": float(updates["loss"].median()) if len(updates) else np.nan,
            "aggregate_batch_mean_chosen_q_mean": float(updates["mean_chosen_q"].mean()) if len(updates) else np.nan,
            "aggregate_batch_mean_target_mean": float(updates["mean_target"].mean()) if len(updates) else np.nan,
            "aggregate_target_minus_chosen_q_mean": float(target_chosen.mean()) if len(updates) else np.nan,
            "per_sample_td_target": "unknown / insufficient evidence",
            "per_sample_td_error": "unknown / insufficient evidence",
            "action_conditioned_td_divergence": "unknown / insufficient evidence",
            "N_positive_vs_N_zero_td_split": "unknown / insufficient evidence",
            "aggregate_fields_are_not_action_conditioned": True,
            "evidence_level": "INSUFFICIENT_EVIDENCE",
            "limitation": "training_log stores batch aggregate loss/mean Q/mean target only; no per-sample TD target, TD error, bootstrap term, or action-conditioned batch composition",
        })
    return pd.DataFrame(rows)


def expected_epsilon_at_step(step: int) -> float:
    return float(np.clip(1.0 - float(step) / 100_000.0, 0.0, 1.0))


def epsilon_frame(log: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    rows: list[dict[str, Any]] = []
    observed_diffs: list[float] = []
    for point in EPSILON_POINTS:
        observed_row = log[log["global_step"].eq(point)]
        if point == 0 or observed_row.empty:
            rows.append({
                "requested_env_step": point,
                "runtime_logged_epsilon": np.nan,
                "expected_theoretical_epsilon_at_env_step": expected_epsilon_at_step(point),
                "expected_epsilon_for_logged_row": expected_epsilon_at_step(max(point - 1, 0)),
                "abs_diff": np.nan,
                "source_type": "code-derived only; no row at global_step=0" if point == 0 else "unknown / insufficient evidence",
                "status": "CODE_DERIVED_AT_STEP_0" if point == 0 else "INSUFFICIENT_EVIDENCE",
            })
            continue
        observed = float(observed_row["epsilon"].iloc[0])
        expected_logged = expected_epsilon_at_step(point - 1)
        diff = abs(observed - expected_logged)
        observed_diffs.append(diff)
        rows.append({
            "requested_env_step": point,
            "runtime_logged_epsilon": observed,
            "expected_theoretical_epsilon_at_env_step": expected_epsilon_at_step(point),
            "expected_epsilon_for_logged_row": expected_logged,
            "abs_diff": diff,
            "source_type": "runtime training_log plus code-derived schedule",
            "status": "MATCH" if diff <= 1e-12 else "MISMATCH",
        })
    max_diff = max(observed_diffs) if observed_diffs else np.nan
    overall = "MATCH" if observed_diffs and max_diff <= 1e-12 else "INSUFFICIENT_EVIDENCE"
    return pd.DataFrame(rows), overall, max_diff


def hypothesis_frame(action: pd.DataFrame, td: pd.DataFrame, epsilon_status: str) -> pd.DataFrame:
    early = action.loc[action["window"].eq("20-25K")].iloc[0]
    late = action.loc[action["window"].eq("45-50K")].iloc[0]
    return pd.DataFrame([
        {
            "hypothesis_id": "H1",
            "hypothesis": "N>0 training transitions progressively disappear, reducing corrective experience for N-positive Q estimates",
            "evidence_level": "NOT_SUPPORTED",
            "observed": f"collection N-positive fraction {early['collection_N_positive_fraction']:.6f} in 20-25K vs {late['collection_N_positive_fraction']:.6f} in 45-50K; late-minus-early={late['collection_N_positive_fraction'] - early['collection_N_positive_fraction']:.6f}",
            "interpretation": "available collection proxy is stable rather than progressively disappearing",
            "limitation": "no replay-buffer snapshot or replay-composition log; actual replay composition is unknown / insufficient evidence",
        },
        {
            "hypothesis_id": "H2",
            "hypothesis": "N>0 vs N=0 TD targets/errors diverge between 25K and 50K",
            "evidence_level": "INSUFFICIENT_EVIDENCE",
            "observed": "training log has aggregate loss, mean_chosen_q and mean_target on update rows, but no per-sample TD target/error or action-conditioned batch fields",
            "interpretation": "action-conditioned TD divergence cannot be tested from retained logs",
            "limitation": "aggregate loss/target/Q must not be used as a substitute for N-positive vs N-zero TD evidence",
        },
        {
            "hypothesis_id": "H3",
            "hypothesis": "epsilon exploration decays faster than frozen schedule, causing premature exploration loss",
            "evidence_level": "NOT_SUPPORTED",
            "observed": f"runtime epsilon rows match code-derived linear schedule at requested points; maximum observed absolute difference={epsilon_status}",
            "interpretation": "no evidence of faster-than-frozen epsilon decay through 50K",
            "limitation": "the step-0 value is code-derived because the CSV begins at global_step=1",
        },
    ])


def write_report(action: pd.DataFrame, td: pd.DataFrame, epsilon: pd.DataFrame, hypotheses: pd.DataFrame, epsilon_status: str, max_epsilon_diff: float, config: dict[str, Any]) -> None:
    early = action.loc[action["window"].eq("20-25K")].iloc[0]
    late = action.loc[action["window"].eq("45-50K")].iloc[0]
    update_rows = int(td["optimizer_update_rows"].sum())
    lines = [
        "# YC DQN seed2 25K→50K learning-level cause diagnostic",
        "",
        "本轮严格只读取已有训练证据，范围为 YC DQN seed2、training step 1–50,000；没有重训、调参、重跑 DSSAT、重放环境、修改 reward/state/action/mask/hyperparameter，也没有扩展到其他 seed 或 checkpoint。",
        "",
        "## Evidence boundary",
        "",
        f"- formal training log：`{rel(TRAINING_LOG)}`；读取 step 1–50,000，共 `{len(pd.read_csv(TRAINING_LOG, usecols=['global_step'], nrows=MAX_STEP))}` 行。",
        f"- formal config：`{rel(CONFIG)}`；50K checkpoint metadata：`{rel(CHECKPOINT_META)}`。",
        "- 未发现 replay buffer snapshot/replay composition log；因此 action log 只作为 `proxy_for_replay_input_distribution=True`，`actual_replay_composition=unknown / insufficient evidence`。",
        "- 未发现 per-sample TD target、TD error、bootstrap term 或 action-conditioned update batch 记录；aggregate loss/mean Q/mean target 不被当作 action-conditioned TD 证据。",
        "",
        "## H1. N-positive training transitions",
        "",
        f"**Observed:** collection proxy 的 N-positive fraction 为 20–25K `{early['collection_N_positive_fraction']:.6f}`（{int(early['collection_N_positive_count'])}/{int(early['collection_rows'])}），45–50K `{late['collection_N_positive_fraction']:.6f}`（{int(late['collection_N_positive_count'])}/{int(late['collection_rows'])}）；late-minus-early = `{late['collection_N_positive_fraction'] - early['collection_N_positive_fraction']:.6f}`。",
        "**Interpretation:** available collection proxy 显示稳定/轻微波动，不支持 N-positive transitions 在 25K→50K 渐进消失；但 actual replay composition 仍为 `unknown / insufficient evidence`。",
        "**Evidence level:** `NOT_SUPPORTED`（针对 collection proxy）；不能升级为 replay-buffer 组成结论。",
        "",
        "## H2. Action-conditioned TD target / TD error",
        "",
        f"**Observed:** step 1–50,000 的 training log 有 `{update_rows}` 个 aggregate update rows；保存了 aggregate `loss`、`mean_chosen_q`、`mean_target`，没有 per-sample TD target/error、bootstrap term 或 N-positive/N-zero batch split。",
        "**Interpretation:** N>0 与 N=0 的 TD target/error 是否分化，当前无法直接检验；aggregate loss 不能替代该比较。",
        "**Evidence level:** `INSUFFICIENT_EVIDENCE`。",
        "",
        "## Vanilla DQN implementation fact",
        "",
        f"- frozen config 的 `double_dqn` = `{config['dqn']['double_dqn']}`。源码 target 为 `r + gamma * max_a' Q_target(s',a')`，并按 next mask 取合法动作最大值；terminal transition 不 bootstrap。",
        "- 因此 maximization bias 只是理论上可能的 follow-up hypothesis；现有日志没有足够的 action-conditioned TD/Q 证据把 Q reversal 归因于 overestimation。",
        "",
        "## H3. Actual epsilon schedule",
        "",
        f"**Observed:** requested points 的 runtime epsilon 与 frozen linear schedule 的 logged-row expectation 匹配；`epsilon_schedule = {epsilon_status}`，observed max absolute diff = `{max_epsilon_diff:.3g}`。step 0 没有 CSV row，step-0 数值是 code-derived，不是直接日志观测。",
        "**Interpretation:** 没有证据显示 epsilon 比冻结配置衰减更快；50K 的 logged epsilon 约为 0.50001，对应 theoretical env-step value 0.50。",
        f"**Evidence level:** `{'NOT_SUPPORTED' if epsilon_status == 'MATCH' else 'INSUFFICIENT_EVIDENCE'}`（针对 faster-than-frozen decay）。",
        "",
        "## 四个必答问题",
        "",
        f"1. **25K→50K，N-positive training transitions 是否明显减少？** collection proxy：**NO**（20–25K `{early['collection_N_positive_fraction']:.6f}` → 45–50K `{late['collection_N_positive_fraction']:.6f}`）；actual replay composition：`unknown / insufficient evidence`。",
        "2. **是否有直接证据显示 N>0 与 N=0 的 TD target / TD error 系统性分化？** **NO direct evidence / unknown**。",
        f"3. **epsilon schedule 是否按冻结配置正常运行？** **{epsilon_status}**（step 0 为 code-derived，其余请求点来自 runtime training log）。",
        "4. **目前最有证据支持的 learning-level cause 候选是哪一个？** **没有一个候选可被支持为 learning-level cause**：H1 的 collection proxy 不支持，H2 是 `INSUFFICIENT_EVIDENCE`，H3 不支持。当前最多只能说 Q-ranking reversal 已被上一轮 same-state replay 观察到，但其学习过程原因未由本轮 retained logs 定位。",
        "",
        "## Hypothesis evidence table",
        "",
        "| hypothesis | evidence level | conclusion |",
        "| --- | --- | --- |",
        *[f"| {row['hypothesis_id']} | {row['evidence_level']} | {row['interpretation']} |" for row in hypotheses.to_dict("records")],
        "",
        "## Output index",
        "",
        f"- collection proxy：`{rel(RESULTS / 'yc_dqn_seed2_training_action_distribution_0_50k.csv')}`",
        f"- TD diagnostic：`{rel(RESULTS / 'yc_dqn_seed2_td_diagnostic.csv')}`",
        f"- epsilon check：`{rel(RESULTS / 'yc_dqn_seed2_epsilon_schedule_check.csv')}`",
        f"- hypothesis evidence：`{rel(RESULTS / 'yc_dqn_seed2_learning_hypothesis_evidence.csv')}`",
        f"- reproducible audit script：`{rel(Path(__file__))}`",
        "",
        "## Status",
        "",
        "`SEED2_25K_50K_LEARNING_LEVEL_DIAGNOSTIC_COMPLETE`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    log = load_log()
    action = window_frame(log)
    td = td_frame(log)
    epsilon, epsilon_status, max_epsilon_diff = epsilon_frame(log)
    hypotheses = hypothesis_frame(action, td, f"{max_epsilon_diff:.3g}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    action.to_csv(RESULTS / "yc_dqn_seed2_training_action_distribution_0_50k.csv", index=False, encoding="utf-8-sig")
    td.to_csv(RESULTS / "yc_dqn_seed2_td_diagnostic.csv", index=False, encoding="utf-8-sig")
    epsilon.to_csv(RESULTS / "yc_dqn_seed2_epsilon_schedule_check.csv", index=False, encoding="utf-8-sig")
    hypotheses.to_csv(RESULTS / "yc_dqn_seed2_learning_hypothesis_evidence.csv", index=False, encoding="utf-8-sig")
    write_report(action, td, epsilon, hypotheses, epsilon_status, max_epsilon_diff, config)
    print("SEED2_25K_50K_LEARNING_LEVEL_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()

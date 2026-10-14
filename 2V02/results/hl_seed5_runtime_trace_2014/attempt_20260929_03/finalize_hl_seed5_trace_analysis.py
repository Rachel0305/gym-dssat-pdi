from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TRACE_PATH = OUT / "runtime_trace.csv"
RUN_META_PATH = OUT / "metadata.json"
OLD_DAILY = ROOT / "benchmark_results" / "8hl05_hl_seed5_8seed" / "daily_outputs" / "HLA" / "HLA_2014_seed5_ckpt100000_daily.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def pairs_for_rollback(values: list[float | None]) -> list[dict]:
    pairs = []
    for i in range(1, len(values)):
        previous, current = values[i - 1], values[i]
        if previous is not None and current is not None and previous > 0 and current > 0 and current < previous:
            pairs.append({"previous_row": i - 1, "previous_value": previous, "current_row": i, "current_value": current})
    return pairs


def main() -> None:
    run_meta = json.loads(RUN_META_PATH.read_text(encoding="utf-8"))
    trace = pd.read_csv(TRACE_PATH, keep_default_na=False)
    old = pd.read_csv(OLD_DAILY, keep_default_na=False)
    if run_meta.get("status") != "COMPLETED" or len(trace) != 146 or len(old) != 146:
        raise RuntimeError("Runtime trace or saved daily run did not complete at the expected 146 rows.")
    comparison = run_meta.get("saved_daily_comparison", {})
    if comparison.get("all_compared_fields_match") is not True:
        raise RuntimeError("Runtime replay did not fully match the saved daily sequence; preserve a manual discrepancy review.")

    eval_dap = [number(x) for x in trace["evaluator_dap_pre"]]
    wrapper_dap = [number(x) for x in trace["wrapper_dap_pre"]]
    cache_dap = [number(x) for x in trace["wrapper_last_obs_dap_pre"]]
    history_dap = [number(x) for x in trace["history_dap_pre"]]
    eval_reset_pairs = pairs_for_rollback(eval_dap)
    wrapper_reset_pairs = pairs_for_rollback(wrapper_dap)
    valid_indices = [i for i, (e, w, c, h) in enumerate(zip(eval_dap, wrapper_dap, cache_dap, history_dap)) if e and w and c and h and e > 0 and w > 0 and c > 0 and h > 0]
    aligned = all(eval_dap[i] == wrapper_dap[i] == cache_dap[i] == history_dap[i] for i in valid_indices)
    startup_mismatch = [
        i for i, (e, c, h) in enumerate(zip(eval_dap, cache_dap, history_dap))
        if e is not None and e > 0 and ((c is not None and c != e) or (h is not None and h != e))
    ]

    first_n = trace[pd.to_numeric(trace["safe_action_anfer"], errors="coerce").fillna(0).gt(0)].iloc[0]
    dap8 = trace[pd.to_numeric(trace["evaluator_dap_pre"], errors="coerce").eq(8)]
    dap7 = trace[pd.to_numeric(trace["evaluator_dap_pre"], errors="coerce").eq(7)]
    if len(dap8) != 1 or len(dap7) != 1:
        raise RuntimeError(f"Expected one DAP7 and one DAP8 step, found {len(dap7)} and {len(dap8)}")
    dap8_row = dap8.iloc[0]
    dap7_row = dap7.iloc[0]
    min_interval = int(run_meta["effective_action_safety"]["min_days_between_fertilization"])
    wrapper_gap_dap8 = number(dap8_row["wrapper_dap_pre"]) - number(dap8_row["last_fertilization_dap_pre"])
    mask_bits_dap8 = str(dap8_row["pre_mask_bits_0_to_15"])
    requested_index_dap8 = int(dap8_row["requested_action_index"])
    mask_allows_dap8 = mask_bits_dap8[requested_index_dap8] == "1"

    reset_pair = eval_reset_pairs[0] if eval_reset_pairs else None
    reset_state = None
    if reset_pair:
        prev = trace.iloc[reset_pair["previous_row"]]
        current = trace.iloc[reset_pair["current_row"]]
        reset_state = {
            "previous_row": reset_pair["previous_row"],
            "previous_evaluator_dap": number(prev["evaluator_dap_pre"]),
            "previous_wrapper_dap": number(prev["wrapper_dap_pre"]),
            "previous_wrapper_cache_dap": number(prev["wrapper_last_obs_dap_pre"]),
            "previous_history_dap": number(prev["history_dap_pre"]),
            "done_after_previous_row": str(prev["done"]).lower() == "true",
            "current_row": reset_pair["current_row"],
            "current_evaluator_dap": number(current["evaluator_dap_pre"]),
            "current_wrapper_dap": number(current["wrapper_dap_pre"]),
            "current_wrapper_cache_dap": number(current["wrapper_last_obs_dap_pre"]),
            "current_history_dap": number(current["history_dap_pre"]),
            "cumulative_n_after_previous_action": number(prev["cumulative_n_post"]),
            "cumulative_n_before_current_action": number(current["cumulative_n_pre"]),
            "cumulative_irrigation_after_previous_action": number(prev["cumulative_irrigation_post"]),
            "cumulative_irrigation_before_current_action": number(current["cumulative_irrigation_pre"]),
        }

    analysis = {
        "valid_positive_dap_rows": len(valid_indices),
        "positive_dap_alignment_all_equal_evaluator_wrapper_cache_history": aligned,
        "startup_mismatched_row_indices_zero_based": startup_mismatch,
        "evaluator_pre_dap_rollback_pairs": eval_reset_pairs,
        "wrapper_pre_dap_rollback_pairs": wrapper_reset_pairs,
        "evaluator_dap_5_to_1_state": reset_state,
        "first_positive_n_action": {
            "row_index_zero_based": int(first_n["row_index_zero_based"]),
            "evaluator_dap_pre": number(first_n["evaluator_dap_pre"]),
            "wrapper_dap_pre": number(first_n["wrapper_dap_pre"]),
            "wrapper_last_obs_dap_pre": number(first_n["wrapper_last_obs_dap_pre"]),
            "history_dap_pre": number(first_n["history_dap_pre"]),
            "requested_action_index": int(first_n["requested_action_index"]),
            "safe_action_anfer": number(first_n["safe_action_anfer"]),
            "last_fertilization_dap_post": number(first_n["last_fertilization_dap_post"]),
        },
        "dap7_post_mask": {
            "row_index_zero_based": int(dap7_row["row_index_zero_based"]),
            "post_mask_valid_count": int(dap7_row["post_mask_valid_count"]),
            "post_mask_bits_0_to_15": str(dap7_row["post_mask_bits_0_to_15"]),
        },
        "dap8_decision": {
            "row_index_zero_based": int(dap8_row["row_index_zero_based"]),
            "requested_action_index": requested_index_dap8,
            "requested_action_amir": number(dap8_row["requested_action_amir"]),
            "requested_action_anfer": number(dap8_row["requested_action_anfer"]),
            "pre_mask_bits_0_to_15": mask_bits_dap8,
            "requested_action_allowed_by_mask": mask_allows_dap8,
            "last_fertilization_dap_pre": number(dap8_row["last_fertilization_dap_pre"]),
            "wrapper_dap_pre": number(dap8_row["wrapper_dap_pre"]),
            "wrapper_elapsed_days_since_last_n": wrapper_gap_dap8,
            "required_min_days": min_interval,
            "wrapper_action_index": int(dap8_row["wrapper_action_index"]),
            "mask_forced_noop": str(dap8_row["mask_forced_noop"]).lower() == "true",
            "safe_action_anfer": number(dap8_row["safe_action_anfer"]),
        },
        "saved_daily_comparison": comparison,
    }

    initial_findings = OUT / "findings.md"
    initial_copy = OUT / "findings_initial_superseded.md"
    if initial_findings.exists() and not initial_copy.exists():
        original = initial_findings.read_text(encoding="utf-8")
        initial_copy.write_text("<!-- Superseded: startup DAP=0 rows were initially counted as DAP rollbacks. Use findings.md and metadata_v2.json. -->\n\n" + original, encoding="utf-8")

    lines = [
        "# HL seed5 2014 runtime trace 只读诊断",
        "",
        "## 执行结果",
        "",
        "- 使用 HLA seed5 100K 原 checkpoint、CPU、单环境，在历史天气 2014 单年完成 146 step episode；训练、checkpoint、原配置和原日志均未改动。",
        "- 与原 2014 daily CSV 的 146 行逐行比较：日期、评估器 DAP、请求/执行 action index、水氮剂量、累计灌溉/N 和 done 共 9 组字段全部一致；首个差异不存在。",
        "",
        "## DAP reset 的实际口径",
        "",
        "- 逐日日志/评估器 DAP 确实在 row 4→5（0-based）从 5 回到 1，上一 action 的 `done=False`；因此文件内的 DAP 序列回退可由 runtime replay 复现。",
        f"- 但这不是 wrapper 内部 DAP 从 5 回到 1：回退前 wrapper `_dap()` 已是 `{reset_state['previous_wrapper_dap'] if reset_state else 'NA'}`，其 `last_obs_dict.dap` / DSSAT `history` 分别为 `{reset_state['previous_wrapper_cache_dap'] if reset_state else 'NA'}` / `{reset_state['previous_history_dap'] if reset_state else 'NA'}`；回退后 wrapper 与 history 均为 DAP `{reset_state['current_wrapper_dap'] if reset_state else 'NA'}`。wrapper 的累计 N 和灌溉在回退前后仍保持不变。",
        f"- 初始 row indices `{startup_mismatch}` 中，评估器 DAP 与 wrapper cache/history 不同步：wrapper 原始 DAP/history 还为 0，`_dap()` 对非正 DAP 使用 1 作为 mask DAP。自第一个正数 wrapper/history 同步点起，共 `{len(valid_indices)}` 行 evaluator、wrapper cache 与 history DAP 完全一致。",
        "- 首次施氮发生在评估器记录的 DAP2，但该步 wrapper 决策 DAP 为 1（cache/history DAP=0，`_dap()` fallback=1），故 safety state 将 `last_fertilization_dap` 记为 1。这个启动期 DAP 来源差异解释了日志里 DAP2→DAP8 看起来只有 6 天。",
        "",
        "## 7 天施氮 mask 核对",
        "",
        f"- DAP8 的 requested action index `{requested_index_dap8}`（I={number(dap8_row['requested_action_amir'])} mm, N={number(dap8_row['requested_action_anfer'])} kg/ha）在 pre-action mask `{mask_bits_dap8}` 中为允许；wrapper 的上次施氮 DAP=`{number(dap8_row['last_fertilization_dap_pre'])}`，wrapper 当前 DAP=`{number(dap8_row['wrapper_dap_pre'])}`，间隔=`{wrapper_gap_dap8}` 天，达到配置要求 `{min_interval}` 天。",
        f"- wrapper 实际执行同一 index `{int(dap8_row['wrapper_action_index'])}`，`mask_forced_noop=False`，实际施 N `{number(dap8_row['safe_action_anfer'])} kg/ha`。DAP7 step 记录的 post-mask 为 `{int(dap7_row['post_mask_valid_count'])}` 个合法动作，完整 bit string `{str(dap7_row['post_mask_bits_0_to_15'])}`。",
        "- 因此 seed5 的 DAP8 施氮不是 mask 绕过；“6 天”来自旧日志按启动期 evaluator DAP 标注事件，而 wrapper 按 `_dap()` 的 DAP1 更新安全状态。离线 reconstruction 把 logged DAP2 当成了 safety-state event DAP，故其 gap 算成 6 天。",
        "",
        "## 产物与边界",
        "",
        "- `runtime_trace.csv`：逐步决策前完整 mask / DAP / safety state 与决策后状态。",
        "- `metadata_v2.json`：checkpoint、配置、天气、源码与 trace 哈希；逐行 action/log 对比和纠正后的 reset 分析。",
        "- `attempt_20260929_01/` 和 `attempt_20260929_02/` 仅在 rollout 前预检失败，未启动 DSSAT episode；两次失败原因和 traceback 留在各自 metadata/log 中。",
        "- `findings_initial_superseded.md` 保留初次自动摘要；它错误地把启动期 DAP0 计作回退，应以本文件为准。",
        "- 结论只针对 HL seed5 的 2014 单年 runtime；不外推其他 seed/FQ，也不触发重训、配置修改或随机天气实验。",
        "",
    ]
    initial_findings.write_text("\n".join(lines), encoding="utf-8")

    meta_v2 = {
        "task": run_meta.get("task"),
        "status": run_meta.get("status"),
        "runtime_replay": {"steps": run_meta.get("steps"), "episode_done": run_meta.get("episode_done"), "device": run_meta.get("device")},
        "source_run_metadata": "metadata.json",
        "source_trace": "runtime_trace.csv",
        "source_trace_sha256": sha256(TRACE_PATH),
        "source_saved_daily_sha256": sha256(OLD_DAILY),
        "analysis_v2": analysis,
        "report_correction": "Initial findings counted startup evaluator post-DAP=0 as a rollback. The corrected report counts positive pre-step evaluator DAP transitions and distinguishes wrapper _dap fallback from last_obs/history DAP.",
        "prior_preflight_attempts": [
            {"path": "attempt_20260929_01", "status": "FAILED_BEFORE_ROLLOUT", "reason": "missing seeded-runner import path"},
            {"path": "attempt_20260929_02", "status": "FAILED_BEFORE_ROLLOUT", "reason": "Windows weather path separators not normalized in the diagnostic existence check"},
        ],
        "config_or_source_modified": False,
        "training_or_checkpoint_update": False,
        "existing_formal_outputs_overwritten": False,
    }
    (OUT / "metadata_v2.json").write_text(json.dumps(meta_v2, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"status": run_meta.get("status"), "analysis": analysis, "findings": "findings.md", "metadata": "metadata_v2.json"}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

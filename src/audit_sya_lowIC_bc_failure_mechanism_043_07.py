from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05 as base04305  # noqa: E402


TASK_ID = "043_07"
TASK_NAME = "sya_lowIC_bc_failure_mechanism_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SOURCE = ROOT / "benchmark_results" / "043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit"
SNAPSHOT_EPOCHS = [0, 5, 20, 50]


ACTION_RE = re.compile(r"DAP(?P<dap>-?\d+)\s+I(?P<i>[0-9.]+)/N(?P<n>[0-9.]+)")


def ensure_dirs() -> None:
    for rel in ["tables", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    return base04305.md_table(df, max_rows=max_rows)


def load_bc_dataset() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    npz_path = SOURCE / "bc_dataset" / "043_05_bc_dataset.npz"
    meta_path = SOURCE / "bc_dataset" / "043_05_bc_dataset_meta.csv"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing 043_06 BC dataset: {npz_path} / {meta_path}")
    with np.load(npz_path) as data:
        dataset = {k: data[k] for k in data.files}
    meta = pd.read_csv(meta_path, keep_default_na=False)
    return dataset, meta


def predict_dataset(model_path: Path, dataset: dict[str, np.ndarray]) -> np.ndarray:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="auto")
    device = model.device
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(dataset["masks"], dtype=torch.bool, device=device)
    preds: list[np.ndarray] = []
    batch = 512
    with torch.no_grad():
        for start in range(0, int(obs.shape[0]), batch):
            batch_obs = obs[start : start + batch]
            batch_masks = masks[start : start + batch]
            try:
                distribution = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            except TypeError:
                distribution = model.policy.get_distribution(batch_obs)
            probs = distribution.distribution.probs
            pred = torch.argmax(probs.masked_fill(~batch_masks, -1.0), dim=1)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds).astype(int)


def open_loop_audit(dataset: dict[str, np.ndarray], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actions = np.asarray(dataset["actions"], dtype=int)
    rows = []
    by_action_rows = []
    by_year_rows = []
    for epoch in SNAPSHOT_EPOCHS:
        model_path = SOURCE / "models" / "SYA" / f"SYA_binary_forecast_bc_only_seed0_epoch{epoch}.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        pred = predict_dataset(model_path, dataset)
        nonzero = actions != 0
        rows.append(
            {
                "bc_epoch": epoch,
                "sample_count": int(len(actions)),
                "overall_accuracy": float(np.mean(pred == actions)),
                "nonzero_action_accuracy": float(np.mean(pred[nonzero] == actions[nonzero])) if bool(nonzero.any()) else np.nan,
                "nonzero_true_count": int(nonzero.sum()),
                "nonzero_pred_count": int((pred != 0).sum()),
                "nonzero_pred_rate": float(np.mean(pred != 0)),
                "predicted_action_counts": json.dumps({str(k): int(v) for k, v in Counter(pred).items()}, ensure_ascii=False),
            }
        )
        for action in sorted(set(actions.tolist())):
            mask = actions == action
            by_action_rows.append(
                {
                    "bc_epoch": epoch,
                    "teacher_action_index": int(action),
                    "sample_count": int(mask.sum()),
                    "accuracy": float(np.mean(pred[mask] == actions[mask])) if bool(mask.any()) else np.nan,
                    "predicted_counts_within_teacher_action": json.dumps({str(k): int(v) for k, v in Counter(pred[mask]).items()}, ensure_ascii=False),
                }
            )
        tmp = meta.copy()
        tmp["pred_action_index"] = pred
        tmp["correct"] = pred == actions
        tmp["true_nonzero"] = actions != 0
        tmp["pred_nonzero"] = pred != 0
        for year, g in tmp.groupby("year"):
            nz = g[g["true_nonzero"]]
            by_year_rows.append(
                {
                    "bc_epoch": epoch,
                    "year": int(year),
                    "sample_count": int(len(g)),
                    "nonzero_true_count": int(len(nz)),
                    "overall_accuracy": float(g["correct"].mean()),
                    "nonzero_action_accuracy": float(nz["correct"].mean()) if len(nz) else np.nan,
                    "nonzero_pred_count": int(g["pred_nonzero"].sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(by_action_rows), pd.DataFrame(by_year_rows)


def parse_action_sequence(seq: str) -> tuple[tuple[int, float, float], ...]:
    parts = []
    for match in ACTION_RE.finditer(str(seq)):
        parts.append((int(match.group("dap")), float(match.group("i")), float(match.group("n"))))
    return tuple(parts)


def build_teacher_sequences(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in meta.groupby("year"):
        events = []
        for row in g.itertuples(index=False):
            i = float(getattr(row, "binary_teacher_irrigation_mm"))
            n = float(getattr(row, "binary_teacher_nitrogen_kg_ha"))
            if i > 0 or n > 0:
                events.append((int(getattr(row, "dap")), i, n))
        signature = "; ".join([f"DAP{dap} I{i:g}/N{n:g}" for dap, i, n in events])
        rows.append(
            {
                "year": int(year),
                "teacher_event_count": int(len(events)),
                "teacher_irrigation_event_count": int(sum(1 for _dap, i, _n in events if i > 0)),
                "teacher_n_event_count": int(sum(1 for _dap, _i, n in events if n > 0)),
                "teacher_binary_action_sequence": signature,
                "teacher_binary_action_tuple": events,
            }
        )
    return pd.DataFrame(rows)


def closed_loop_audit(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_path = SOURCE / "evaluation" / "043_06_bc_only_validation_summary.csv"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    teacher = build_teacher_sequences(meta)
    rows = []
    for row in eval_df.itertuples(index=False):
        policy_tuple = parse_action_sequence(getattr(row, "action_sequence"))
        rows.append(
            {
                "year": int(getattr(row, "year")),
                "bc_epoch": int(getattr(row, "checkpoint_step")),
                "policy_action_sequence": getattr(row, "action_sequence"),
                "policy_action_tuple": policy_tuple,
                "policy_event_count": int(len(policy_tuple)),
                "policy_irrigation_total": float(getattr(row, "summary_irrigation_total")),
                "policy_n_total": float(getattr(row, "summary_nitrogen_total")),
                "grain_yield_kg_ha": float(getattr(row, "grain_yield_kg_ha")),
                "any_metric_win_vs_four_max": bool(getattr(row, "any_metric_win_vs_four_max")),
                "all3_win_vs_four_max": bool(getattr(row, "all3_win_vs_four_max")),
            }
        )
    policy = pd.DataFrame(rows)
    merged = policy.merge(teacher.drop(columns=["teacher_binary_action_tuple"]), on="year", how="left")
    merged["exact_sequence_match_teacher"] = merged["policy_action_sequence"] == merged["teacher_binary_action_sequence"]
    summary = (
        merged.groupby("bc_epoch", as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            unique_policy_action_signatures=("policy_action_sequence", "nunique"),
            unique_teacher_action_signatures=("teacher_binary_action_sequence", "nunique"),
            exact_teacher_sequence_match_years=("exact_sequence_match_teacher", "sum"),
            mean_yield=("grain_yield_kg_ha", "mean"),
            any_metric_win_years=("any_metric_win_vs_four_max", "sum"),
            all3_win_years=("all3_win_vs_four_max", "sum"),
            mean_policy_irrigation=("policy_irrigation_total", "mean"),
            mean_policy_n=("policy_n_total", "mean"),
        )
    )
    return merged, summary


def decide_branch(open_summary: pd.DataFrame, closed_summary: pd.DataFrame) -> str:
    max_open_nonzero = float(open_summary["nonzero_action_accuracy"].max()) if not open_summary.empty else 0.0
    max_unique = int(closed_summary["unique_policy_action_signatures"].max()) if not closed_summary.empty else 0
    if max_open_nonzero < 0.8 and max_unique < 3:
        return "B_open_loop_imitation_insufficient_and_closed_loop_template"
    if max_open_nonzero >= 0.8 and max_unique < 3:
        return "C_closed_loop_distribution_drift_after_open_loop_learning"
    if max_unique >= 3:
        return "A_bc_policy_has_some_non_template_rollout_signal"
    return "D_inconclusive"


def run_audit() -> None:
    ensure_dirs()
    dataset, meta = load_bc_dataset()
    open_summary, open_by_action, open_by_year = open_loop_audit(dataset, meta)
    closed_detail, closed_summary = closed_loop_audit(meta)
    branch = decide_branch(open_summary, closed_summary)

    outputs = {
        "open_loop_summary": OUT / "tables" / "043_07_open_loop_summary.csv",
        "open_loop_by_action": OUT / "tables" / "043_07_open_loop_by_action.csv",
        "open_loop_by_year": OUT / "tables" / "043_07_open_loop_by_year.csv",
        "closed_loop_detail": OUT / "tables" / "043_07_closed_loop_teacher_vs_policy_detail.csv",
        "closed_loop_summary": OUT / "tables" / "043_07_closed_loop_summary.csv",
        "result_json": OUT / "043_07_result.json",
    }
    open_summary.to_csv(outputs["open_loop_summary"], index=False, encoding="utf-8-sig")
    open_by_action.to_csv(outputs["open_loop_by_action"], index=False, encoding="utf-8-sig")
    open_by_year.to_csv(outputs["open_loop_by_year"], index=False, encoding="utf-8-sig")
    closed_detail.to_csv(outputs["closed_loop_detail"], index=False, encoding="utf-8-sig")
    closed_summary.to_csv(outputs["closed_loop_summary"], index=False, encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "training_run": False,
        "dssat_run": False,
        "source_task": "043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit",
        "max_open_loop_nonzero_action_accuracy": float(open_summary["nonzero_action_accuracy"].max()),
        "max_closed_loop_unique_policy_action_signatures": int(closed_summary["unique_policy_action_signatures"].max()),
        "max_closed_loop_unique_teacher_action_signatures": int(closed_summary["unique_teacher_action_signatures"].max()),
        "outputs": {k: v.relative_to(ROOT).as_posix() for k, v in outputs.items()},
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    outputs["result_json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, open_summary, open_by_action, open_by_year, closed_summary, closed_detail)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_record(
    result: dict[str, Any],
    open_summary: pd.DataFrame,
    open_by_action: pd.DataFrame,
    open_by_year: pd.DataFrame,
    closed_summary: pd.DataFrame,
    closed_detail: pd.DataFrame,
) -> None:
    lines = [
        "# 043_07 SYA lowIC BC failure mechanism audit 记录",
        "",
        "## 一句话结论",
        "",
        f"分支：`{result['branch']}`。本任务没有训练、没有运行 DSSAT，只读取 043_06 的 BC 模型、BC dataset 和 rollout 结果。",
        "",
        "## 开环预测汇总",
        "",
        md_table(open_summary, 20),
        "",
        "## 开环按动作类别",
        "",
        md_table(open_by_action, 40),
        "",
        "## 闭环 rollout 汇总",
        "",
        md_table(closed_summary, 20),
        "",
        "## 闭环 teacher vs policy 示例",
        "",
        md_table(closed_detail[["bc_epoch", "year", "exact_sequence_match_teacher", "policy_action_sequence", "teacher_binary_action_sequence"]], 20),
        "",
        "## 输出文件",
        "",
        "```json",
        json.dumps(result["outputs"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def dry_run() -> None:
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "prompt_exists": PROMPT.exists(),
        "source_exists": SOURCE.exists(),
        "bc_dataset_exists": (SOURCE / "bc_dataset" / "043_05_bc_dataset.npz").exists(),
        "bc_meta_exists": (SOURCE / "bc_dataset" / "043_05_bc_dataset_meta.csv").exists(),
        "eval_summary_exists": (SOURCE / "evaluation" / "043_06_bc_only_validation_summary.csv").exists(),
        "model_epochs": {
            str(epoch): (SOURCE / "models" / "SYA" / f"SYA_binary_forecast_bc_only_seed0_epoch{epoch}.zip").exists()
            for epoch in SNAPSHOT_EPOCHS
        },
        "next_step_allowed": bool(
            PROMPT.exists()
            and SOURCE.exists()
            and (SOURCE / "bc_dataset" / "043_05_bc_dataset.npz").exists()
            and (SOURCE / "evaluation" / "043_06_bc_only_validation_summary.csv").exists()
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run_audit()


if __name__ == "__main__":
    main()

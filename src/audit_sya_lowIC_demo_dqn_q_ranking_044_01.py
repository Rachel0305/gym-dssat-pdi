from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_binary_forecast_demo_dqn_smoke_044_00 as base04400  # noqa: E402
import run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05 as base04305  # noqa: E402
from strict_maskable_dqn_040 import StrictMaskableDQN  # noqa: E402


TASK_ID = "044_01"
TASK_NAME = "sya_lowIC_demo_dqn_q_ranking_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SOURCE = ROOT / "benchmark_results" / "044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k"
CHECKPOINTS = [1000, 2000]


def ensure_dirs() -> None:
    for rel in ["tables", "configs", "demo_buffer"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    return base04305.md_table(df, max_rows=max_rows)


def q_values_for_dataset(model: StrictMaskableDQN, observations: np.ndarray) -> np.ndarray:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    rows: list[np.ndarray] = []
    batch = 512
    with torch.no_grad():
        for start in range(0, int(obs.shape[0]), batch):
            q = model.online(obs[start : start + batch])
            rows.append(q.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def audit_checkpoint(step: int, dataset: dict[str, np.ndarray], meta: pd.DataFrame) -> pd.DataFrame:
    model_path = SOURCE / "models" / "SYA" / f"SYA_binary_forecast_demo_dqn_seed0_ckpt{int(step)}.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model, saved_step = StrictMaskableDQN.load(model_path)
    q = q_values_for_dataset(model, dataset["observations"][: dataset["size"]])
    actions = dataset["actions"][: dataset["size"]].astype(int)
    masks = dataset["masks"][: dataset["size"]].astype(bool)
    rows = []
    for i in range(int(dataset["size"])):
        valid = masks[i]
        action = int(actions[i])
        q_row = q[i]
        q_teacher = float(q_row[action])
        q_noop = float(q_row[0]) if bool(valid[0]) else np.nan
        other = valid.copy()
        other[action] = False
        q_max_other = float(np.max(q_row[other])) if bool(other.any()) else np.nan
        valid_q = q_row[valid]
        teacher_rank = int(1 + np.sum(valid_q > q_teacher))
        argmax_action = int(np.flatnonzero(valid)[np.argmax(valid_q)])
        m = meta.iloc[i].to_dict()
        rows.append(
            {
                "checkpoint_step": int(step),
                "saved_step": int(saved_step),
                "sample_index": int(i),
                "year": int(m.get("year", -1)),
                "dap": int(m.get("dap", -1)),
                "teacher_action_index": action,
                "teacher_is_nonzero": bool(action != 0),
                "argmax_action_index": argmax_action,
                "teacher_is_argmax": bool(argmax_action == action),
                "teacher_rank_among_valid": teacher_rank,
                "q_teacher": q_teacher,
                "q_noop": q_noop,
                "q_max_other_valid": q_max_other,
                "q_teacher_minus_noop": float(q_teacher - q_noop) if np.isfinite(q_noop) else np.nan,
                "q_teacher_minus_max_other": float(q_teacher - q_max_other) if np.isfinite(q_max_other) else np.nan,
                "dqfd_margin_satisfied": bool(np.isfinite(q_max_other) and q_teacher >= q_max_other + base04400.DEMO_MARGIN),
            }
        )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_checkpoint = (
        detail.groupby("checkpoint_step", as_index=False)
        .agg(
            sample_count=("sample_index", "count"),
            nonzero_count=("teacher_is_nonzero", "sum"),
            teacher_argmax_rate=("teacher_is_argmax", "mean"),
            nonzero_teacher_argmax_rate=("teacher_is_argmax", lambda s: float(s[detail.loc[s.index, "teacher_is_nonzero"]].mean())),
            mean_teacher_rank=("teacher_rank_among_valid", "mean"),
            mean_q_teacher_minus_noop=("q_teacher_minus_noop", "mean"),
            mean_q_teacher_minus_max_other=("q_teacher_minus_max_other", "mean"),
            margin_satisfied_rate=("dqfd_margin_satisfied", "mean"),
        )
    )
    by_action = (
        detail.groupby(["checkpoint_step", "teacher_action_index"], as_index=False)
        .agg(
            sample_count=("sample_index", "count"),
            teacher_argmax_rate=("teacher_is_argmax", "mean"),
            mean_teacher_rank=("teacher_rank_among_valid", "mean"),
            mean_q_teacher_minus_noop=("q_teacher_minus_noop", "mean"),
            mean_q_teacher_minus_max_other=("q_teacher_minus_max_other", "mean"),
            margin_satisfied_rate=("dqfd_margin_satisfied", "mean"),
        )
    )
    by_year = (
        detail.groupby(["checkpoint_step", "year"], as_index=False)
        .agg(
            sample_count=("sample_index", "count"),
            nonzero_count=("teacher_is_nonzero", "sum"),
            teacher_argmax_rate=("teacher_is_argmax", "mean"),
            mean_q_teacher_minus_noop=("q_teacher_minus_noop", "mean"),
            mean_q_teacher_minus_max_other=("q_teacher_minus_max_other", "mean"),
        )
    )
    return by_checkpoint, by_action, by_year


def decide_branch(by_checkpoint: pd.DataFrame, by_action: pd.DataFrame) -> str:
    max_nonzero = float(by_checkpoint["nonzero_teacher_argmax_rate"].max()) if not by_checkpoint.empty else 0.0
    action_nonzero = by_action[by_action["teacher_action_index"].ne(0)].copy()
    min_action = float(action_nonzero["teacher_argmax_rate"].min()) if not action_nonzero.empty else 0.0
    if max_nonzero >= 0.8 and min_action >= 0.5:
        return "A_demo_q_ranking_shows_dqfd_signal"
    if max_nonzero >= 0.5 and min_action < 0.5:
        return "C_demo_q_ranking_action_specific_failure"
    return "B_demo_q_ranking_insufficient"


def run_audit() -> None:
    ensure_dirs()
    config, env_config = base04305.build_config_and_env_config(base04305.YEARS)
    (OUT / "configs" / "044_01_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "configs" / "044_01_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")
    demo, meta, skipped = base04400.collect_demo_buffer(OUT, config, env_config)
    dataset = {
        "observations": demo.observations.copy(),
        "actions": demo.actions.copy(),
        "masks": demo.masks.copy(),
        "size": int(demo.size),
    }
    meta = meta.iloc[: int(demo.size)].reset_index(drop=True)
    detail = pd.concat([audit_checkpoint(step, dataset, meta) for step in CHECKPOINTS], ignore_index=True)
    by_checkpoint, by_action, by_year = summarize(detail)
    branch = decide_branch(by_checkpoint, by_action)
    paths = {
        "detail": OUT / "tables" / "044_01_q_ranking_detail.csv",
        "by_checkpoint": OUT / "tables" / "044_01_q_ranking_by_checkpoint.csv",
        "by_action": OUT / "tables" / "044_01_q_ranking_by_action.csv",
        "by_year": OUT / "tables" / "044_01_q_ranking_by_year.csv",
        "demo_meta": OUT / "demo_buffer" / "044_01_demo_transition_meta.csv",
        "skipped": OUT / "demo_buffer" / "044_01_skipped_masked_teacher_actions.csv",
        "result": OUT / "044_01_result.json",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    by_checkpoint.to_csv(paths["by_checkpoint"], index=False, encoding="utf-8-sig")
    by_action.to_csv(paths["by_action"], index=False, encoding="utf-8-sig")
    by_year.to_csv(paths["by_year"], index=False, encoding="utf-8-sig")
    meta.to_csv(paths["demo_meta"], index=False, encoding="utf-8-sig")
    skipped.to_csv(paths["skipped"], index=False, encoding="utf-8-sig")
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "training_run": False,
        "dssat_rollout_run": False,
        "source_task": "044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k",
        "checkpoints": CHECKPOINTS,
        "demo_transition_count": int(demo.size),
        "demo_nonzero_count": int(meta["is_nonzero_action"].sum()) if "is_nonzero_action" in meta.columns else None,
        "max_nonzero_teacher_argmax_rate": float(by_checkpoint["nonzero_teacher_argmax_rate"].max()),
        "outputs": {k: v.relative_to(ROOT).as_posix() for k, v in paths.items()},
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    paths["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, by_checkpoint, by_action, by_year)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_record(result: dict[str, Any], by_checkpoint: pd.DataFrame, by_action: pd.DataFrame, by_year: pd.DataFrame) -> None:
    lines = [
        "# 044_01 SYA lowIC Demo-DQN Q-ranking audit 记录",
        "",
        "## 一句话结论",
        "",
        f"分支：`{result['branch']}`。本任务不训练，只检查 044_00 Demo-DQN checkpoint 在 teacher/demo states 上的 Q 排序。",
        "",
        "## 按 checkpoint 汇总",
        "",
        md_table(by_checkpoint, 20),
        "",
        "## 按 teacher 动作类别汇总",
        "",
        md_table(by_action, 40),
        "",
        "## 按年份汇总",
        "",
        md_table(by_year, 40),
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
        "checkpoint_exists": {
            str(step): (SOURCE / "models" / "SYA" / f"SYA_binary_forecast_demo_dqn_seed0_ckpt{int(step)}.pt").exists()
            for step in CHECKPOINTS
        },
        "input_root": base04305.base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "input_root_exists": base04305.base04302.LOWIC_INPUT_ROOT.exists(),
        "next_step_allowed": bool(PROMPT.exists() and SOURCE.exists()),
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

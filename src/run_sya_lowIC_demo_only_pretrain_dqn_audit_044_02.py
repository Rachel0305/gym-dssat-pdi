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
import audit_sya_lowIC_demo_dqn_q_ranking_044_01 as q_audit  # noqa: E402
import run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05 as base04305  # noqa: E402


TASK_ID = "044_02"
TASK_NAME = "sya_lowIC_demo_only_pretrain_dqn_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SNAPSHOT_EPOCHS = [0, 50, 200, 500]
SEED = 0


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "logs", "tables", "demo_buffer"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    return base04305.md_table(df, max_rows=max_rows)


def save_model(model: Any, epoch: int) -> Path:
    path = OUT / "models" / "SYA" / f"SYA_demo_only_pretrain_dqn_seed{SEED}_epoch{int(epoch)}.pt"
    model.save(path, global_step=int(epoch))
    return path


def demo_only_train_step(model: Any, demo: Any) -> dict[str, float]:
    demo_batch = demo.sample(base04400.DEMO_BATCH_SIZE, model.rng, model.device)
    demo_td, demo_q, demo_target = base04400.td_loss(model, demo_batch)
    margin = base04400.demo_margin_loss(model, demo_batch)
    loss = base04400.DEMO_TD_COEF * demo_td + base04400.DEMO_MARGIN_COEF * margin
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    preclip_norm = torch.nn.utils.clip_grad_norm_(model.online.parameters(), model.max_grad_norm)
    model.optimizer.step()
    model.optimizer_updates += 1
    return {
        "loss_total": float(loss.detach().cpu()),
        "loss_demo_td": float(demo_td.detach().cpu()),
        "loss_demo_margin": float(margin.detach().cpu()),
        "mean_demo_q": float(demo_q.detach().mean().cpu()),
        "mean_demo_target": float(demo_target.detach().mean().cpu()),
        "preclip_grad_norm": float(preclip_norm.detach().cpu()),
    }


def audit_saved_models(dataset: dict[str, np.ndarray], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    old_source = q_audit.SOURCE
    old_checkpoints = q_audit.CHECKPOINTS
    try:
        # Reuse q_audit helpers by directly adapting the checkpoint function logic locally.
        for epoch in SNAPSHOT_EPOCHS:
            model_path = OUT / "models" / "SYA" / f"SYA_demo_only_pretrain_dqn_seed{SEED}_epoch{int(epoch)}.pt"
            model, saved_epoch = q_audit.StrictMaskableDQN.load(model_path)
            q = q_audit.q_values_for_dataset(model, dataset["observations"][: dataset["size"]])
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
                argmax_action = int(np.flatnonzero(valid)[np.argmax(valid_q)])
                teacher_rank = int(1 + np.sum(valid_q > q_teacher))
                m = meta.iloc[i].to_dict()
                rows.append(
                    {
                        "checkpoint_step": int(epoch),
                        "saved_step": int(saved_epoch),
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
            frames.append(pd.DataFrame(rows))
        detail = pd.concat(frames, ignore_index=True)
        return q_audit.summarize(detail)
    finally:
        q_audit.SOURCE = old_source
        q_audit.CHECKPOINTS = old_checkpoints


def run_audit() -> None:
    ensure_dirs()
    config, env_config = base04305.build_config_and_env_config(base04305.YEARS)
    (OUT / "configs" / "044_02_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "configs" / "044_02_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")
    demo, meta, skipped = base04400.collect_demo_buffer(OUT, config, env_config)
    dataset = {
        "observations": demo.observations.copy(),
        "actions": demo.actions.copy(),
        "masks": demo.masks.copy(),
        "size": int(demo.size),
    }
    meta = meta.iloc[: int(demo.size)].reset_index(drop=True)
    probe_obs = demo.observations[0]
    model = base04400.make_model(config, int(probe_obs.shape[0]), int(demo.action_dim), timesteps=max(SNAPSHOT_EPOCHS))
    rows = []
    if 0 in SNAPSHOT_EPOCHS:
        save_model(model, 0)
    for epoch in range(1, max(SNAPSHOT_EPOCHS) + 1):
        row = demo_only_train_step(model, demo)
        row["demo_epoch"] = int(epoch)
        rows.append(row)
        if epoch in SNAPSHOT_EPOCHS:
            save_model(model, epoch)
    train_log = pd.DataFrame(rows)
    train_log.to_csv(OUT / "logs" / "044_02_demo_only_pretrain_log.csv", index=False, encoding="utf-8-sig")
    by_checkpoint, by_action, by_year = audit_saved_models(dataset, meta)
    paths = {
        "by_checkpoint": OUT / "tables" / "044_02_q_ranking_by_checkpoint.csv",
        "by_action": OUT / "tables" / "044_02_q_ranking_by_action.csv",
        "by_year": OUT / "tables" / "044_02_q_ranking_by_year.csv",
        "train_log": OUT / "logs" / "044_02_demo_only_pretrain_log.csv",
        "demo_meta": OUT / "demo_buffer" / "044_02_demo_transition_meta.csv",
        "skipped": OUT / "demo_buffer" / "044_02_skipped_masked_teacher_actions.csv",
        "result": OUT / "044_02_result.json",
    }
    by_checkpoint.to_csv(paths["by_checkpoint"], index=False, encoding="utf-8-sig")
    by_action.to_csv(paths["by_action"], index=False, encoding="utf-8-sig")
    by_year.to_csv(paths["by_year"], index=False, encoding="utf-8-sig")
    meta.to_csv(paths["demo_meta"], index=False, encoding="utf-8-sig")
    skipped.to_csv(paths["skipped"], index=False, encoding="utf-8-sig")
    max_nonzero = float(by_checkpoint["nonzero_teacher_argmax_rate"].max())
    branch = "A_demo_only_pretrain_anchors_nonzero_q" if max_nonzero >= 0.8 else "B_demo_only_pretrain_still_insufficient"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "training_run": "demo_only_pretrain",
        "online_dqn_run": False,
        "snapshot_epochs": SNAPSHOT_EPOCHS,
        "demo_transition_count": int(demo.size),
        "demo_nonzero_count": int(meta["is_nonzero_action"].sum()) if "is_nonzero_action" in meta.columns else None,
        "max_nonzero_teacher_argmax_rate": max_nonzero,
        "outputs": {k: v.relative_to(ROOT).as_posix() for k, v in paths.items()},
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    paths["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, train_log, by_checkpoint, by_action, by_year)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_record(result: dict[str, Any], train_log: pd.DataFrame, by_checkpoint: pd.DataFrame, by_action: pd.DataFrame, by_year: pd.DataFrame) -> None:
    lines = [
        "# 044_02 SYA lowIC demo-only pretrain DQN audit 记录",
        "",
        "## 一句话结论",
        "",
        f"分支：`{result['branch']}`。本任务只用 demo buffer 预训练 Q 网络，不做 online DQN。",
        "",
        "## 训练日志尾部",
        "",
        md_table(train_log.tail(20), 20),
        "",
        "## Q 排序按 epoch 汇总",
        "",
        md_table(by_checkpoint, 20),
        "",
        "## Q 排序按动作类别汇总",
        "",
        md_table(by_action, 40),
        "",
        "## Q 排序按年份汇总",
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
        "input_root": base04305.base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "input_root_exists": base04305.base04302.LOWIC_INPUT_ROOT.exists(),
        "snapshot_epochs": SNAPSHOT_EPOCHS,
        "next_step_allowed": bool(PROMPT.exists() and base04305.base04302.LOWIC_INPUT_ROOT.exists()),
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

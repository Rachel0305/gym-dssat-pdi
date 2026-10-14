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

import ppo_safe_rendering  # noqa: E402
import run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05 as base04305  # noqa: E402


TASK_ID = "043_06"
TASK_NAME = "sya_lowIC_binary_forecast_bc_only_imitation_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SNAPSHOT_EPOCHS = [0, 5, 20, 50]
SEED = 0


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "bc_dataset"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, epoch: int) -> Path:
    path = OUT / "models" / "SYA" / f"SYA_binary_forecast_bc_only_seed{SEED}_epoch{int(epoch)}.zip"
    model.save(str(path.with_suffix("")))
    return path


def train_bc_snapshots(model: Any, dataset: dict[str, np.ndarray]) -> pd.DataFrame:
    device = model.device
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    masks = torch.as_tensor(dataset["masks"], dtype=torch.bool, device=device)
    weights = torch.as_tensor(dataset["weights"], dtype=torch.float32, device=device)
    nonzero_actions = actions != 0
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=base04305.BC_LR)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    n = int(obs.shape[0])
    max_epoch = max(SNAPSHOT_EPOCHS)

    def evaluate_bc_epoch(epoch: int, last_loss: float | None = None) -> None:
        with torch.no_grad():
            try:
                distribution = model.policy.get_distribution(obs, action_masks=masks)
            except TypeError:
                distribution = model.policy.get_distribution(obs)
            probs = distribution.distribution.probs
            pred = torch.argmax(probs.masked_fill(~masks, -1.0), dim=1)
            overall_acc = (pred == actions).float().mean()
            nonzero_acc = (
                (pred[nonzero_actions] == actions[nonzero_actions]).float().mean()
                if bool(nonzero_actions.any())
                else torch.as_tensor(float("nan"), device=device)
            )
            nonzero_pred_rate = (pred != 0).float().mean()
        rows.append(
            {
                "bc_epoch": int(epoch),
                "bc_loss": float("nan") if last_loss is None else float(last_loss),
                "bc_full_action_accuracy": float(overall_acc.detach().cpu()),
                "bc_full_nonzero_action_accuracy": float(nonzero_acc.detach().cpu()),
                "bc_full_nonzero_pred_rate": float(nonzero_pred_rate.detach().cpu()),
            }
        )

    if 0 in SNAPSHOT_EPOCHS:
        save_model(model, 0)
        evaluate_bc_epoch(0)

    for epoch in range(1, max_epoch + 1):
        order = rng.permutation(n)
        losses: list[float] = []
        for start in range(0, n, base04305.BC_BATCH_SIZE):
            idx_np = order[start : start + base04305.BC_BATCH_SIZE]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_masks = masks[idx]
            batch_weights = weights[idx]
            try:
                distribution = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            except TypeError:
                distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_actions)
            loss = -((log_prob * batch_weights).sum() / batch_weights.sum().clamp_min(1e-9))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
        if not np.isfinite(last_loss):
            raise RuntimeError(f"BC loss is not finite at epoch {epoch}")
        if epoch in SNAPSHOT_EPOCHS:
            save_model(model, epoch)
            evaluate_bc_epoch(epoch, last_loss)

    bc_log = pd.DataFrame(rows)
    bc_log.to_csv(OUT / "logs" / "043_06_bc_snapshot_log.csv", index=False, encoding="utf-8-sig")
    return bc_log


def run_audit() -> None:
    from sb3_contrib import MaskablePPO

    ensure_dirs()
    selected = base04305.base04103.load_selected_teachers()
    config, env_config = base04305.build_config_and_env_config(base04305.YEARS)
    (OUT / "configs" / "043_06_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "configs" / "043_06_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset, bc_meta = base04305.collect_bc_dataset(OUT, selected, config, env_config)
    skipped_path = OUT / "bc_dataset" / "043_05_bc_skipped_masked_teacher_actions.csv"
    skipped = pd.read_csv(skipped_path, keep_default_na=False) if skipped_path.exists() and skipped_path.stat().st_size > 0 else pd.DataFrame()

    train_env = base04305.RandomYearEnv(config, env_config, base04305.YEARS, SEED)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        tensorboard_log=str(OUT / "tensorboard" / "SYA"),
        **base04305.base04302.base03222.base.ppo_kwargs(config),
    )
    try:
        bc_log = train_bc_snapshots(model, dataset)
    finally:
        train_env.close()

    rows = []
    for epoch in SNAPSHOT_EPOCHS:
        path = OUT / "models" / "SYA" / f"SYA_binary_forecast_bc_only_seed{SEED}_epoch{int(epoch)}.zip"
        rows.append(
            {
                "station_code": "SYA",
                "site": "SY",
                "seed": SEED,
                "checkpoint_step": int(epoch),
                "stage": "bc_only",
                "model_path": path.relative_to(ROOT).as_posix(),
                "model_sha256": base04305.sha256_file(path),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory_path = OUT / "evaluation" / "043_06_bc_only_model_inventory.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    thresholds = base04305.base04100.load_baseline_thresholds()
    thresholds = thresholds[thresholds["year"].isin(base04305.YEARS)].copy()
    eval_frames = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04305.base04302.LOWIC_INPUT_ROOT
    try:
        for row in inventory.itertuples(index=False):
            eval_df, _daily = base04305.evaluate_checkpoint(
                OUT,
                ROOT / row.model_path,
                int(row.checkpoint_step),
                str(row.stage),
                config,
                env_config,
                thresholds,
            )
            eval_frames.append(eval_df)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    eval_summary = pd.concat(eval_frames, ignore_index=True)
    eval_summary_path = OUT / "evaluation" / "043_06_bc_only_validation_summary.csv"
    eval_summary.to_csv(eval_summary_path, index=False, encoding="utf-8-sig")
    by_epoch = base04305.summarize_by_checkpoint(eval_summary)
    by_epoch_path = OUT / "evaluation" / "043_06_bc_only_summary_by_epoch.csv"
    by_epoch.to_csv(by_epoch_path, index=False, encoding="utf-8-sig")

    max_unique = int(by_epoch["unique_action_signatures"].max()) if not by_epoch.empty else 0
    branch = "A_bc_can_reproduce_non_template_teacher_timing" if max_unique >= 3 else "C_bc_still_template_like"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "training_run": False,
        "ppo_finetune_run": False,
        "algorithm_shell": "MaskablePPO policy network",
        "training_method": "BC imitation only",
        "snapshot_epochs": SNAPSHOT_EPOCHS,
        "bc_sample_count": int(len(bc_meta)),
        "bc_year_count": int(bc_meta["year"].nunique()),
        "skipped_masked_teacher_action_count": int(len(skipped)),
        "max_unique_action_signatures": max_unique,
        "outputs": {
            "record_md": DOC.relative_to(ROOT).as_posix(),
            "bc_log": (OUT / "logs" / "043_06_bc_snapshot_log.csv").relative_to(ROOT).as_posix(),
            "inventory": inventory_path.relative_to(ROOT).as_posix(),
            "validation_summary": eval_summary_path.relative_to(ROOT).as_posix(),
            "by_epoch": by_epoch_path.relative_to(ROOT).as_posix(),
            "bc_dataset_meta": (OUT / "bc_dataset" / "043_05_bc_dataset_meta.csv").relative_to(ROOT).as_posix(),
            "skipped_masked_teacher_actions": skipped_path.relative_to(ROOT).as_posix(),
            "result_json": (OUT / "043_06_result.json").relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "043_06_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, bc_log, bc_meta, skipped, inventory, eval_summary, by_epoch)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_record(
    result: dict[str, Any],
    bc_log: pd.DataFrame,
    bc_meta: pd.DataFrame,
    skipped: pd.DataFrame,
    inventory: pd.DataFrame,
    eval_summary: pd.DataFrame,
    by_epoch: pd.DataFrame,
) -> None:
    lines = [
        "# 043_06 SYA lowIC binary-forecast BC-only imitation audit 记录",
        "",
        "## 一句话结论",
        "",
        f"分支：`{result['branch']}`。本任务只做 teacher imitation / BC，不做 PPO fine-tune。",
        "",
        "## 关键边界",
        "",
        "- 未调用 PPO `model.learn()`；",
        "- 未修改 reward；",
        "- 未修改 DSSAT 输入；",
        "- 使用 043_02 的 30维天气/预报/归一化 observation；",
        "- 使用 042_10 binary action 与既有 safety mask。",
        "",
        "## BC 数据",
        "",
        f"- 样本数：{len(bc_meta)}",
        f"- 年份数：{bc_meta['year'].nunique() if not bc_meta.empty else 0}",
        f"- 被 mask 跳过的 teacher 正动作数：{len(skipped)}",
        "",
        "### BC 训练日志",
        "",
        base04305.md_table(bc_log, 20),
        "",
        "### BC-only rollout 汇总",
        "",
        base04305.md_table(by_epoch, 20),
        "",
        "### 模型清单",
        "",
        base04305.md_table(inventory, 20),
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
    ensure_dirs()
    selected = base04305.base04103.load_selected_teachers()
    config, _env_config = base04305.build_config_and_env_config(base04305.YEARS)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "prompt_exists": PROMPT.exists(),
        "input_root": base04305.base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "input_root_exists": base04305.base04302.LOWIC_INPUT_ROOT.exists(),
        "years": base04305.YEARS,
        "selected_teacher_years": selected["year"].astype(int).tolist(),
        "snapshot_epochs": SNAPSHOT_EPOCHS,
        "observation_dim": len(base04305.base04302.ENHANCED_OBS_NAMES),
        "discrete_actions": config["discrete_actions"],
        "next_step_allowed": bool(PROMPT.exists() and base04305.base04302.LOWIC_INPUT_ROOT.exists() and not selected.empty),
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

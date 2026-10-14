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
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sya_lowIC_teacher_warmstart_maskableppo_041_03 as base04103


TASK_ID = "041_04"
TASK_NAME = "sya_lowIC_teacher_warmstart_balanced_bc_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

BC_EPOCHS = 20
BC_BATCH_SIZE = 256
BC_LR = 1e-4
BALANCED_NONZERO_FRACTION = 0.5


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def _masked_argmax_from_distribution(distribution: Any, masks: torch.Tensor) -> torch.Tensor:
    probs = distribution.distribution.probs
    masked_probs = probs.masked_fill(~masks, -1.0)
    return torch.argmax(masked_probs, dim=1)


def pretrain_policy_bc_balanced(model: Any, dataset: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    device = model.device
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    masks = torch.as_tensor(dataset["masks"], dtype=torch.bool, device=device)
    weights = torch.as_tensor(dataset["weights"], dtype=torch.float32, device=device)

    action_np = np.asarray(dataset["actions"], dtype=int)
    zero_idx = np.where(action_np == 0)[0]
    nonzero_idx = np.where(action_np != 0)[0]
    if len(nonzero_idx) == 0:
        raise RuntimeError("BC dataset has no nonzero teacher actions; balanced warm-start is not meaningful.")

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=BC_LR)
    rng = np.random.default_rng(base04103.SEED)
    rows: list[dict[str, float | int]] = []
    n_batches = max(1, int(np.ceil(len(action_np) / BC_BATCH_SIZE)))
    nonzero_per_batch = max(1, int(round(BC_BATCH_SIZE * BALANCED_NONZERO_FRACTION)))
    zero_per_batch = max(1, BC_BATCH_SIZE - nonzero_per_batch)

    for epoch in range(1, BC_EPOCHS + 1):
        losses = []
        acc_all = []
        acc_zero = []
        acc_nonzero = []
        pred_nonzero_rates = []
        for _ in range(n_batches):
            nz = rng.choice(nonzero_idx, size=nonzero_per_batch, replace=len(nonzero_idx) < nonzero_per_batch)
            zz = rng.choice(zero_idx, size=zero_per_batch, replace=len(zero_idx) < zero_per_batch)
            idx_np = np.concatenate([nz, zz])
            rng.shuffle(idx_np)
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

            with torch.no_grad():
                pred = _masked_argmax_from_distribution(distribution, batch_masks)
                zero_mask = batch_actions == 0
                nonzero_mask = batch_actions != 0
                acc_all.append(float((pred == batch_actions).float().mean().detach().cpu()))
                if zero_mask.any():
                    acc_zero.append(float((pred[zero_mask] == batch_actions[zero_mask]).float().mean().detach().cpu()))
                if nonzero_mask.any():
                    acc_nonzero.append(float((pred[nonzero_mask] == batch_actions[nonzero_mask]).float().mean().detach().cpu()))
                pred_nonzero_rates.append(float((pred != 0).float().mean().detach().cpu()))
            losses.append(float(loss.detach().cpu()))

        row = {
            "bc_epoch": epoch,
            "bc_loss": float(np.mean(losses)),
            "bc_action_accuracy": float(np.mean(acc_all)),
            "bc_zero_action_accuracy": float(np.mean(acc_zero)) if acc_zero else np.nan,
            "bc_nonzero_action_accuracy": float(np.mean(acc_nonzero)) if acc_nonzero else np.nan,
            "bc_pred_nonzero_rate": float(np.mean(pred_nonzero_rates)),
            "balanced_nonzero_fraction": BALANCED_NONZERO_FRACTION,
            "nonzero_teacher_samples": int(len(nonzero_idx)),
            "zero_teacher_samples": int(len(zero_idx)),
        }
        if not np.isfinite(row["bc_loss"]):
            raise RuntimeError(f"BC loss is not finite at epoch {epoch}")
        rows.append(row)

    bc_log = pd.DataFrame(rows)
    bc_log.to_csv(out / "logs" / f"{TASK_ID}_bc_pretrain_log.csv", index=False)
    return bc_log


def configure_base_module() -> None:
    base04103.TASK_ID = TASK_ID
    base04103.TASK_NAME = TASK_NAME
    base04103.BASE_OUT = BASE_OUT
    base04103.BASE_DOC = BASE_DOC
    base04103.PROMPT = PROMPT
    base04103.BC_EPOCHS = BC_EPOCHS
    base04103.BC_BATCH_SIZE = BC_BATCH_SIZE
    base04103.BC_LR = BC_LR
    base04103.out_for_suffix = out_for_suffix
    base04103.doc_for_suffix = doc_for_suffix
    base04103.pretrain_policy_bc = pretrain_policy_bc_balanced


def main() -> None:
    configure_base_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=base04103.DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    checkpoint_steps = base04103.parse_checkpoint_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        base04103.dry_run(args.timesteps, checkpoint_steps, args.suffix)
    else:
        base04103.run_training(args.timesteps, checkpoint_steps, args.suffix)


if __name__ == "__main__":
    main()

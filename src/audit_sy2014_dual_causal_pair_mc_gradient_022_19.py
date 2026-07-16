from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as a12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork


OUT = ROOT / "benchmark_results" / "022_19"
LR = 1e-4
SEEDS = (0, 1, 2)


def losses(
    model: QNetwork,
    train_obs: torch.Tensor,
    train_actions: torch.Tensor,
    train_targets: torch.Tensor,
    obs65: torch.Tensor,
    target65: torch.Tensor,
    obs110: torch.Tensor,
    target110: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_train = model(train_obs).gather(1, train_actions[:, None]).squeeze(1)
    lmc = torch.nn.functional.smooth_l1_loss(q_train, train_targets)
    q65 = model(obs65)
    l65 = torch.nn.functional.smooth_l1_loss(q65[:, 1] - q65[:, 7], target65)
    q110 = model(obs110)
    l110 = torch.nn.functional.smooth_l1_loss(q110[:, 0] - q110[:, 1], target110)
    return lmc, l65, l110


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = np.load(a12.NPZ_PATH)
    obs = data["observations"]
    actions = data["actions"]
    targets = data["targets"]
    scenarios = data["scenarios"].astype(str)
    stage_indices = data["stage_indices"]
    daps = np.asarray([(1, 30, 50, 65, 85, 110)[int(i)] for i in stage_indices])
    manifest = pd.read_csv(a12.MANIFEST_PATH)
    split = pd.read_csv(a12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    train_mask = np.asarray([s in train_scenarios for s in scenarios])
    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)

    table65, obs65, target65 = a12.build_pair_targets(obs, manifest)
    table65.to_csv(OUT / "022_19_dap65_causal_targets.csv", index=False)
    controlled110 = pd.read_csv(ROOT / "benchmark_results" / "022_17" / "022_17_controlled_action_pairs.csv")
    controlled110 = controlled110.loc[~controlled110["execution_alias"]].copy().sort_values("scenario")
    if len(controlled110) != 8 or not (controlled110["delta_g0_a1_minus_a0"] < -1).all():
        raise ValueError("DAP110 controlled target provenance mismatch")
    indices110: list[int] = []
    for scenario in controlled110["scenario"].astype(str):
        idx = np.where((scenarios == scenario) & (daps == 110))[0]
        if len(idx) != 1:
            raise ValueError(f"{scenario}: expected one DAP110 state")
        indices110.append(int(idx[0]))
    obs110 = torch.tensor(obs[indices110], dtype=torch.float32)
    target110 = torch.tensor(
        (-controlled110["delta_g0_a1_minus_a0"].to_numpy(dtype=np.float32)) / 1000.0,
        dtype=torch.float32,
    )
    target110_table = controlled110[["scenario", "g0_action0", "g0_action1"]].copy()
    target110_table["target_q0_minus_q1_scaled"] = target110.numpy()
    target110_table.to_csv(OUT / "022_19_dap110_causal_targets.csv", index=False)

    cache: dict[int, dict[str, object]] = {}
    ratio_rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        payload = torch.load(a12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = QNetwork()
        model.load_state_dict(payload["model_state_dict"])
        named = list(model.named_parameters())
        params = [p for _, p in named]
        shared_mask = [name.startswith("net.0") or name.startswith("net.2") for name, _ in named]
        lmc, l65, l110 = losses(model, train_obs, train_actions, train_targets, obs65, target65, obs110, target110)
        gmc = a12.gradient_list(lmc, params)
        g65 = a12.gradient_list(l65, params)
        g110 = a12.gradient_list(l110, params)
        smc = a12.flatten(gmc, shared_mask)
        s65 = a12.flatten(g65, shared_mask)
        s110 = a12.flatten(g110, shared_mask)
        nmc = float(torch.linalg.vector_norm(smc))
        n65 = float(torch.linalg.vector_norm(s65))
        n110 = float(torch.linalg.vector_norm(s110))
        if min(nmc, n65, n110) <= 0:
            raise ValueError(f"seed{seed}: zero shared gradient norm")
        lambda65_seed = 0.5 * nmc / n65
        lambda110_seed = 0.5 * nmc / n110
        ratio_rows.append(
            {
                "seed": seed,
                "shared_mc_norm": nmc,
                "shared_dap65_norm": n65,
                "shared_dap110_norm": n110,
                "lambda65_seed": lambda65_seed,
                "lambda110_seed": lambda110_seed,
                "cosine_mc_dap65": a12.cosine(smc, s65),
                "cosine_mc_dap110": a12.cosine(smc, s110),
                "cosine_dap65_dap110": a12.cosine(s65, s110),
            }
        )
        cache[seed] = {
            "model": model,
            "grads": (gmc, g65, g110),
            "losses": (float(lmc.detach()), float(l65.detach()), float(l110.detach())),
            "shared": (smc, s65, s110),
        }

    ratio_df = pd.DataFrame(ratio_rows)
    lambda65 = float(ratio_df["lambda65_seed"].median())
    lambda110 = float(ratio_df["lambda110_seed"].median())
    ratio_df.to_csv(OUT / "022_19_seed_gradient_ratios.csv", index=False)
    audit_rows: list[dict[str, float | int | bool]] = []
    for seed in SEEDS:
        item = cache[seed]
        model = item["model"]
        gmc, g65, g110 = item["grads"]
        before_mc, before65, before110 = item["losses"]
        smc, s65, s110 = item["shared"]
        virtual = copy.deepcopy(model)
        with torch.no_grad():
            for p, gm, g6, g1 in zip(virtual.parameters(), gmc, g65, g110):
                p.add_(-LR * (gm + lambda65 * g6 + lambda110 * g1))
        after_mc_t, after65_t, after110_t = losses(
            virtual, train_obs, train_actions, train_targets, obs65, target65, obs110, target110
        )
        after_mc, after65, after110 = map(lambda x: float(x.detach()), (after_mc_t, after65_t, after110_t))
        mc_relative = (after_mc - before_mc) / max(abs(before_mc), 1e-12)
        total_shared = smc + lambda65 * s65 + lambda110 * s110
        finite = all(
            math.isfinite(v)
            for v in (
                before_mc, before65, before110, after_mc, after65, after110,
                mc_relative, a12.cosine(s65, s110), float(torch.linalg.vector_norm(total_shared)),
            )
        )
        compatible = bool(
            after65 < before65
            and after110 < before110
            and mc_relative <= 0.01
            and a12.cosine(s65, s110) >= -0.20
            and finite
        )
        audit_rows.append(
            {
                "seed": seed,
                "lambda65_fixed": lambda65,
                "lambda110_fixed": lambda110,
                "loss_mc_before": before_mc,
                "loss_mc_after": after_mc,
                "mc_relative_change": mc_relative,
                "loss_dap65_before": before65,
                "loss_dap65_after": after65,
                "loss_dap110_before": before110,
                "loss_dap110_after": after110,
                "cosine_dap65_dap110": a12.cosine(s65, s110),
                "combined_shared_grad_norm": float(torch.linalg.vector_norm(total_shared)),
                "finite": finite,
                "compatible": compatible,
            }
        )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT / "022_19_fixed_weight_virtual_step_audit.csv", index=False)
    count = int(audit_df["compatible"].sum())
    branch = "A_compatible" if count >= 2 else ("B_single_seed" if count == 1 else "C_incompatible")
    result = {
        "status": "completed",
        "branch": branch,
        "compatible_seed_count": count,
        "total_seeds": 3,
        "lambda65_fixed": lambda65,
        "lambda110_fixed": lambda110,
        "causal_gradient_budget_rule": "each pair group gets half the MC shared-gradient norm",
        "soft_anchor_included": False,
        "dqn_training_steps": 0,
        "dssat_calls": 0,
        "saved_updated_checkpoint": False,
        "next_step_allowed": branch == "A_compatible",
    }
    (OUT / "022_19_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(3)
    axes[0].bar(x - 0.18, ratio_df["cosine_mc_dap65"], width=0.36, label="MC vs DAP65")
    axes[0].bar(x + 0.18, ratio_df["cosine_mc_dap110"], width=0.36, label="MC vs DAP110")
    axes[0].axhline(-0.20, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Shared-gradient cosine", xlabel="Seed", ylabel="Cosine")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - 0.2, (audit_df["loss_dap65_after"]-audit_df["loss_dap65_before"])/audit_df["loss_dap65_before"]*100, width=0.4, label="DAP65")
    axes[1].bar(x + 0.2, (audit_df["loss_dap110_after"]-audit_df["loss_dap110_before"])/audit_df["loss_dap110_before"]*100, width=0.4, label="DAP110")
    axes[1].set(title="Virtual-step pair loss change", xlabel="Seed", ylabel="Percent")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "022_19_dual_pair_gradient_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_19_dual_pair_gradient_audit.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

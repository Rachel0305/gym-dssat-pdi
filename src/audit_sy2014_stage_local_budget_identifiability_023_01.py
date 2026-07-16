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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as a12
from stage_separated_q_ensemble_023 import STAGE_DAPS, StageSeparatedQEnsemble


OUT = ROOT / "benchmark_results" / "023_01"
LEARNING_RATE = 1e-4
ROUND_TO = 500
PARAMETERS_PER_STAGE = 6409
SHARED_TRAIN_SAMPLES = 216
STAGE_TRAIN_SAMPLES = 36


def gradients(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(params, values)]


def norm(values: list[torch.Tensor]) -> float:
    return float(torch.linalg.vector_norm(torch.cat([v.reshape(-1) for v in values])))


def stage_losses(
    model: nn.Module,
    mc_obs: torch.Tensor,
    mc_actions: torch.Tensor,
    mc_targets: torch.Tensor,
    pair_obs: torch.Tensor,
    pair_targets: torch.Tensor,
    action_control: int,
    action_treatment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = model(mc_obs).gather(1, mc_actions[:, None]).squeeze(1)
    mc_loss = nn.functional.smooth_l1_loss(selected, mc_targets)
    q = model(pair_obs)
    margins = q[:, action_control] - q[:, action_treatment]
    pair_loss = nn.functional.smooth_l1_loss(margins, pair_targets)
    return mc_loss, pair_loss


def local_step_estimate(margin: float, target: float, improvement: float) -> tuple[str, float]:
    if math.isfinite(margin) and margin >= 0:
        residual = target - margin
        region = "quadratic" if abs(residual) < 1.0 else "linear"
        return region, 0.0
    residual = target - margin
    region = "quadratic" if abs(residual) < 1.0 else "linear"
    if not (math.isfinite(residual) and residual > 0 and math.isfinite(improvement) and improvement > 0):
        return region, math.nan
    if region == "quadratic":
        fraction = improvement / residual
        if not 0 < fraction < 1:
            raise ValueError(f"invalid quadratic fractional improvement {fraction}")
        steps = int(math.ceil(math.log(target / residual) / math.log(1.0 - fraction)))
    else:
        steps = int(math.ceil(-margin / improvement))
    return region, float(max(0, steps))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    data = np.load(a12.NPZ_PATH)
    observations = data["observations"]
    actions = data["actions"]
    targets = data["targets"]
    scenarios = data["scenarios"].astype(str)
    stage_indices = data["stage_indices"]
    daps = np.asarray([STAGE_DAPS[int(i)] for i in stage_indices], dtype=int)
    manifest = pd.read_csv(a12.MANIFEST_PATH)
    split = pd.read_csv(a12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    test_scenarios = set(split.loc[split["split"] == "test", "scenario"].astype(str))
    train_mask = np.asarray([s in train_scenarios for s in scenarios])
    test_mask = np.asarray([s in test_scenarios for s in scenarios])

    pair65_table, pair65_obs, pair65_targets = a12.build_pair_targets(observations, manifest)
    pair110_table = pd.read_csv(ROOT / "benchmark_results" / "022_19" / "022_19_dap110_causal_targets.csv")
    idx110: list[int] = []
    for scenario in pair110_table["scenario"].astype(str):
        indices = np.where((scenarios == scenario) & (daps == 110))[0]
        if len(indices) != 1:
            raise ValueError(f"{scenario}: expected exactly one DAP110 state")
        idx110.append(int(indices[0]))
    pair110_obs = torch.tensor(observations[idx110], dtype=torch.float32)
    pair110_targets = torch.tensor(
        pair110_table["target_q0_minus_q1_scaled"].to_numpy(np.float32), dtype=torch.float32
    )

    support = pd.read_csv(a12.SUPPORT_PATH)
    coverage_rows: list[dict[str, object]] = []
    for dap in STAGE_DAPS:
        stage_train = train_mask & (daps == dap)
        stage_test = test_mask & (daps == dap)
        for action in sorted(np.unique(actions[daps == dap]).astype(int).tolist()):
            coverage_rows.append(
                {
                    "dap": dap,
                    "action_index": action,
                    "train_count": int(np.sum(stage_train & (actions == action))),
                    "test_count": int(np.sum(stage_test & (actions == action))),
                    "total_count": int(np.sum((daps == dap) & (actions == action))),
                    "primary_evaluable": bool(
                        len(support.loc[(support["dap"] == dap) & (support["action_index"] == action)])
                        and support.loc[(support["dap"] == dap) & (support["action_index"] == action), "primary_evaluable"]
                        .astype(str)
                        .str.lower()
                        .isin(("true", "1"))
                        .iloc[0]
                    ),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "023_01_stage_action_coverage.csv", index=False)

    stage_summary_rows: list[dict[str, object]] = []
    for dap in STAGE_DAPS:
        sm = train_mask & (daps == dap)
        tm = test_mask & (daps == dap)
        stage_summary_rows.append(
            {
                "dap": dap,
                "train_samples": int(sm.sum()),
                "test_samples": int(tm.sum()),
                "parameters": PARAMETERS_PER_STAGE,
                "parameters_per_train_sample": PARAMETERS_PER_STAGE / max(int(sm.sum()), 1),
                "mc_target_min": float(np.min(targets[sm])),
                "mc_target_max": float(np.max(targets[sm])),
                "mc_target_std": float(np.std(targets[sm])),
                "planned_training_status": "train" if dap in (65, 110) else "freeze",
            }
        )
    stage_summary = pd.DataFrame(stage_summary_rows)
    stage_summary.to_csv(OUT / "023_01_stage_data_summary.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    for seed in range(3):
        payload = torch.load(
            a12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        ensemble = StageSeparatedQEnsemble()
        ensemble.initialize_from_single_state_dict(payload["model_state_dict"])
        for dap, pair_obs, pair_targets, ac, at, labels in (
            (65, pair65_obs, pair65_targets, 1, 7, pair65_table["prefix"].astype(str).tolist()),
            (110, pair110_obs, pair110_targets, 0, 1, pair110_table["scenario"].astype(str).tolist()),
        ):
            mask = train_mask & (daps == dap)
            mc_obs = torch.tensor(observations[mask], dtype=torch.float32)
            mc_actions = torch.tensor(actions[mask], dtype=torch.long)
            mc_targets = torch.tensor(targets[mask], dtype=torch.float32)
            model = ensemble.network_for_stage(dap)
            model.eval()
            params = list(model.parameters())
            mc_loss, pair_loss = stage_losses(
                model, mc_obs, mc_actions, mc_targets, pair_obs, pair_targets, ac, at
            )
            gmc = gradients(mc_loss, params)
            gpair = gradients(pair_loss, params)
            mc_norm, pair_norm = norm(gmc), norm(gpair)
            lam = mc_norm / pair_norm if pair_norm > 0 else math.nan
            virtual = copy.deepcopy(model)
            with torch.no_grad():
                before_q = model(pair_obs).numpy()
                for p, gm, gp in zip(virtual.parameters(), gmc, gpair):
                    p.add_(-LEARNING_RATE * (gm + lam * gp))
                after_q = virtual(pair_obs).numpy()
            mc_after, pair_after = stage_losses(
                virtual, mc_obs, mc_actions, mc_targets, pair_obs, pair_targets, ac, at
            )
            mc_before_v, pair_before_v = float(mc_loss.detach()), float(pair_loss.detach())
            mc_after_v, pair_after_v = float(mc_after.detach()), float(pair_after.detach())
            mc_relative = (mc_after_v - mc_before_v) / max(abs(mc_before_v), 1e-12)
            finite = all(math.isfinite(v) for v in (mc_norm, pair_norm, lam, mc_before_v, pair_before_v, mc_after_v, pair_after_v, mc_relative))
            seed_rows.append(
                {
                    "seed": seed,
                    "dap": dap,
                    "mc_loss_before": mc_before_v,
                    "pair_loss_before": pair_before_v,
                    "mc_gradient_norm": mc_norm,
                    "pair_gradient_norm": pair_norm,
                    "lambda_stage": lam,
                    "mc_loss_after_virtual_step": mc_after_v,
                    "pair_loss_after_virtual_step": pair_after_v,
                    "mc_loss_relative_change": mc_relative,
                    "pair_loss_decreased": bool(pair_after_v < pair_before_v),
                    "finite": finite,
                    "virtual_step_pass": bool(finite and pair_after_v < pair_before_v and mc_relative <= 0.01),
                }
            )
            for i, label in enumerate(labels):
                margin0 = float(before_q[i, ac] - before_q[i, at])
                margin1 = float(after_q[i, ac] - after_q[i, at])
                target = float(pair_targets[i])
                improvement = margin1 - margin0
                region, estimate = local_step_estimate(margin0, target, improvement)
                margin_rows.append(
                    {
                        "seed": seed,
                        "dap": dap,
                        "case": label,
                        "action_control": ac,
                        "action_treatment": at,
                        "target_margin": target,
                        "margin_before": margin0,
                        "margin_after_virtual_step": margin1,
                        "one_step_improvement": improvement,
                        "smooth_l1_region": region,
                        "local_updates_to_sign_flip": estimate,
                    }
                )

    seed_df = pd.DataFrame(seed_rows)
    margin_df = pd.DataFrame(margin_rows)
    seed_df.to_csv(OUT / "023_01_seed_stage_gradient_audit.csv", index=False)
    margin_df.to_csv(OUT / "023_01_causal_margin_budget_estimates.csv", index=False)

    finite_estimates = margin_df.loc[np.isfinite(margin_df["local_updates_to_sign_flip"]), "local_updates_to_sign_flip"]
    maximum_estimate = int(finite_estimates.max()) if len(finite_estimates) else 0
    recommended = int(math.ceil(maximum_estimate / ROUND_TO) * ROUND_TO) if maximum_estimate > 0 else 0
    coverage_pass = bool(
        (stage_summary["train_samples"] == STAGE_TRAIN_SAMPLES).all()
        and (stage_summary["test_samples"] == 12).all()
        and ((coverage["dap"] == 65) & coverage["action_index"].isin((1, 7)) & (coverage["total_count"] > 0)).sum() == 2
        and ((coverage["dap"] == 110) & coverage["action_index"].isin((0, 1)) & (coverage["total_count"] > 0)).sum() == 2
    )
    audit_pass = bool(
        coverage_pass
        and seed_df["virtual_step_pass"].all()
        and np.isfinite(margin_df["one_step_improvement"]).all()
        and (margin_df["one_step_improvement"] > 0).all()
        and recommended > 0
    )
    branch = "A_budget_identified" if audit_pass else "C_budget_not_identified"
    result = {
        "status": "completed",
        "branch": branch,
        "coverage_pass": coverage_pass,
        "seed_stage_virtual_step_pass_count": int(seed_df["virtual_step_pass"].sum()),
        "seed_stage_virtual_step_total": int(len(seed_df)),
        "shared_parameters_per_sample": PARAMETERS_PER_STAGE / SHARED_TRAIN_SAMPLES,
        "separated_parameters_per_sample": PARAMETERS_PER_STAGE / STAGE_TRAIN_SAMPLES,
        "ratio_worsening_factor": SHARED_TRAIN_SAMPLES / STAGE_TRAIN_SAMPLES,
        "maximum_local_sign_flip_estimate": maximum_estimate,
        "recommended_updates_for_023_02": recommended if audit_pass else None,
        "finite_case_diagnostic_candidate_updates": recommended,
        "budget_rounding_rule": "maximum local sign-flip estimate rounded up to nearest 500; no safety multiplier",
        "stages_to_train": [65, 110],
        "stages_to_freeze": [1, 30, 50, 85],
        "formal_dqn_training_steps": 0,
        "virtual_gradient_steps": 6,
        "dssat_calls": 0,
        "saved_updated_checkpoint": False,
        "next_step_allowed": audit_pass,
        "limitation": "Local one-step extrapolation is a preregistration aid, not a convergence guarantee.",
    }
    (OUT / "023_01_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    pivot = stage_summary.set_index("dap")["parameters_per_train_sample"]
    axes[0].bar([str(v) for v in pivot.index], pivot.values, color=["#999999" if d not in (65, 110) else "#2f6f4e" for d in pivot.index])
    axes[0].axhline(PARAMETERS_PER_STAGE / SHARED_TRAIN_SAMPLES, color="#b22222", linestyle="--", label="Shared-network baseline")
    axes[0].set(xlabel="Stage DAP", ylabel="Parameters per training sample", title="Stage-local identifiability pressure")
    axes[0].legend(frameon=False, fontsize=8)
    for (seed, dap), group in margin_df.groupby(["seed", "dap"]):
        axes[1].scatter([f"s{seed}-D{dap}"] * len(group), group["local_updates_to_sign_flip"], alpha=0.75)
    axes[1].axhline(recommended, color="#b22222", linestyle="--", label=f"Registered budget {recommended}")
    axes[1].set(ylabel="Estimated updates to positive margin", title="Local causal-ranking budget")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "023_01_budget_identifiability_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "023_01_budget_identifiability_audit.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

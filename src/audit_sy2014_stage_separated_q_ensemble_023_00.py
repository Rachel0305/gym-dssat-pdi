from __future__ import annotations

import copy
import hashlib
import io
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

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as audit12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork
from stage_based_dqn_core_022 import valid_action_indices
from stage_separated_q_ensemble_023 import STAGE_DAPS, StageSeparatedQEnsemble, stage_parameter_count


OUT = ROOT / "benchmark_results" / "023_00"
SEEDS = (0, 1, 2)
LR = 1e-4


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def stage_hashes(model: StageSeparatedQEnsemble) -> dict[int, str]:
    return {dap: module_hash(model.network_for_stage(dap)) for dap in STAGE_DAPS}


def stage_q_snapshots(
    model: StageSeparatedQEnsemble, observations: torch.Tensor, daps: np.ndarray
) -> dict[int, torch.Tensor]:
    snapshots: dict[int, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for dap in STAGE_DAPS:
            mask = torch.from_numpy(daps == dap)
            snapshots[dap] = model.network_for_stage(dap)(observations[mask]).detach().clone()
    return snapshots


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).item())


def audit_single_stage_update(
    base: StageSeparatedQEnsemble,
    target_dap: int,
    train_observations: torch.Tensor,
    train_actions: torch.Tensor,
    train_targets: torch.Tensor,
    train_daps: np.ndarray,
    all_observations: torch.Tensor,
    all_daps: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model = copy.deepcopy(base)
    model.train()
    before_hashes = stage_hashes(model)
    before_q = stage_q_snapshots(model, all_observations, all_daps)
    mask_np = train_daps == target_dap
    mask = torch.from_numpy(mask_np)
    q_selected = model.network_for_stage(target_dap)(train_observations[mask]).gather(
        1, train_actions[mask, None]
    ).squeeze(1)
    loss = torch.nn.functional.smooth_l1_loss(q_selected, train_targets[mask])
    for parameter in model.parameters():
        parameter.grad = None
    loss.backward()
    gradient_rows: list[dict[str, object]] = []
    for dap in STAGE_DAPS:
        gradients = [parameter.grad for parameter in model.network_for_stage(dap).parameters()]
        nonzero = [gradient for gradient in gradients if gradient is not None and bool(torch.any(gradient != 0))]
        gradient_rows.append(
            {
                "target_update_dap": target_dap,
                "network_dap": dap,
                "parameter_tensors_with_nonzero_gradient": len(nonzero),
                "gradient_isolated_as_expected": bool(len(nonzero) > 0 if dap == target_dap else len(nonzero) == 0),
            }
        )
    optimizer = torch.optim.Adam(model.network_for_stage(target_dap).parameters(), lr=LR)
    optimizer.step()
    after_hashes = stage_hashes(model)
    after_q = stage_q_snapshots(model, all_observations, all_daps)
    stage_rows: list[dict[str, object]] = []
    for dap in STAGE_DAPS:
        stage_rows.append(
            {
                "target_update_dap": target_dap,
                "network_dap": dap,
                "parameter_hash_changed": before_hashes[dap] != after_hashes[dap],
                "q_max_abs_change": max_abs_difference(before_q[dap], after_q[dap]),
                "isolation_pass": bool(
                    (before_hashes[dap] != after_hashes[dap] and max_abs_difference(before_q[dap], after_q[dap]) > 0)
                    if dap == target_dap
                    else (before_hashes[dap] == after_hashes[dap] and max_abs_difference(before_q[dap], after_q[dap]) == 0)
                ),
            }
        )
    gradient_pass = all(bool(row["gradient_isolated_as_expected"]) for row in gradient_rows)
    update_pass = all(bool(row["isolation_pass"]) for row in stage_rows)
    summary = {
        "target_update_dap": target_dap,
        "loss_before_update": float(loss.detach()),
        "gradient_isolation_pass": gradient_pass,
        "parameter_and_q_isolation_pass": update_pass,
    }
    return summary, gradient_rows + stage_rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    data = np.load(audit12.NPZ_PATH)
    observations_np = data["observations"].astype(np.float32)
    actions_np = data["actions"].astype(np.int64)
    targets_np = data["targets"].astype(np.float32)
    scenarios = data["scenarios"].astype(str)
    stage_indices = data["stage_indices"].astype(int)
    daps = np.asarray([STAGE_DAPS[index] for index in stage_indices], dtype=int)
    observations = torch.from_numpy(observations_np)

    split = pd.read_csv(audit12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"].eq("train"), "scenario"].astype(str))
    test_scenarios = set(split.loc[split["split"].eq("test"), "scenario"].astype(str))
    train_mask_np = np.asarray([scenario in train_scenarios for scenario in scenarios])
    test_mask_np = np.asarray([scenario in test_scenarios for scenario in scenarios])
    train_observations = torch.from_numpy(observations_np[train_mask_np])
    train_actions = torch.from_numpy(actions_np[train_mask_np])
    train_targets = torch.from_numpy(targets_np[train_mask_np])
    train_daps = daps[train_mask_np]

    coverage_rows: list[dict[str, object]] = []
    for split_name, split_mask in (("train", train_mask_np), ("test", test_mask_np)):
        for dap in STAGE_DAPS:
            coverage_rows.append(
                {
                    "split": split_name,
                    "dap": dap,
                    "transition_count": int(np.sum(split_mask & (daps == dap))),
                    "valid_action_indices": ",".join(map(str, valid_action_indices(dap))),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "023_00_stage_data_coverage.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    isolation_rows: list[dict[str, object]] = []
    architecture_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        payload = torch.load(
            audit12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        original = QNetwork()
        original.load_state_dict(payload["model_state_dict"])
        original.eval()
        ensemble = StageSeparatedQEnsemble()
        ensemble.initialize_from_single_state_dict(payload["model_state_dict"])
        ensemble.eval()

        parameter_pointers: list[int] = []
        for dap in STAGE_DAPS:
            network = ensemble.network_for_stage(dap)
            pointers = [int(parameter.data_ptr()) for parameter in network.parameters()]
            parameter_pointers.extend(pointers)
            architecture_rows.append(
                {
                    "seed": seed,
                    "dap": dap,
                    "parameter_count": stage_parameter_count(ensemble, dap),
                    "parameter_tensor_count": len(pointers),
                    "valid_action_indices": ",".join(map(str, valid_action_indices(dap))),
                }
            )
        unique_parameter_storage = len(parameter_pointers) == len(set(parameter_pointers))

        initial_diffs: list[float] = []
        direct_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            dispatched = ensemble(observations, torch.from_numpy(daps))
            for dap in STAGE_DAPS:
                mask = torch.from_numpy(daps == dap)
                original_q = original(observations[mask])
                separated_q = ensemble.network_for_stage(dap)(observations[mask])
                initial_diffs.append(max_abs_difference(original_q, separated_q))
                direct_chunks.append(separated_q)
            direct = torch.empty_like(dispatched)
            for dap, chunk in zip(STAGE_DAPS, direct_chunks):
                direct[torch.from_numpy(daps == dap)] = chunk
        initial_q_max_abs_diff = max(initial_diffs)
        dispatch_q_max_abs_diff = max_abs_difference(dispatched, direct)

        update65, rows65 = audit_single_stage_update(
            ensemble, 65, train_observations, train_actions, train_targets, train_daps, observations, daps
        )
        update110, rows110 = audit_single_stage_update(
            ensemble, 110, train_observations, train_actions, train_targets, train_daps, observations, daps
        )
        for row in rows65 + rows110:
            row["seed"] = seed
            isolation_rows.append(row)

        buffer = io.BytesIO()
        torch.save(ensemble.state_dict(), buffer)
        buffer.seek(0)
        reloaded = StageSeparatedQEnsemble()
        reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        hash_roundtrip = stage_hashes(ensemble) == stage_hashes(reloaded)
        with torch.no_grad():
            roundtrip_q_diff = max_abs_difference(
                ensemble(observations, torch.from_numpy(daps)),
                reloaded(observations, torch.from_numpy(daps)),
            )
        finite = bool(
            torch.isfinite(dispatched).all()
            and math.isfinite(float(update65["loss_before_update"]))
            and math.isfinite(float(update110["loss_before_update"]))
        )
        seed_pass = bool(
            unique_parameter_storage
            and initial_q_max_abs_diff == 0
            and dispatch_q_max_abs_diff == 0
            and update65["gradient_isolation_pass"]
            and update65["parameter_and_q_isolation_pass"]
            and update110["gradient_isolation_pass"]
            and update110["parameter_and_q_isolation_pass"]
            and hash_roundtrip
            and roundtrip_q_diff == 0
            and finite
        )
        seed_rows.append(
            {
                "seed": seed,
                "unique_parameter_storage": unique_parameter_storage,
                "initial_q_max_abs_diff": initial_q_max_abs_diff,
                "dispatch_q_max_abs_diff": dispatch_q_max_abs_diff,
                "dap65_gradient_isolation_pass": update65["gradient_isolation_pass"],
                "dap65_parameter_q_isolation_pass": update65["parameter_and_q_isolation_pass"],
                "dap110_gradient_isolation_pass": update110["gradient_isolation_pass"],
                "dap110_parameter_q_isolation_pass": update110["parameter_and_q_isolation_pass"],
                "roundtrip_hash_equal": hash_roundtrip,
                "roundtrip_q_max_abs_diff": roundtrip_q_diff,
                "finite": finite,
                "seed_pass": seed_pass,
            }
        )

    seed_table = pd.DataFrame(seed_rows)
    isolation = pd.DataFrame(isolation_rows)
    architecture = pd.DataFrame(architecture_rows)
    seed_table.to_csv(OUT / "023_00_seed_structure_checks.csv", index=False)
    isolation.to_csv(OUT / "023_00_stage_isolation_checks.csv", index=False)
    architecture.to_csv(OUT / "023_00_architecture_manifest.csv", index=False)

    coverage_pass = bool((coverage["transition_count"] > 0).all())
    pass_count = int(seed_table["seed_pass"].sum())
    branch = "A_structure_isolated" if pass_count == 3 and coverage_pass else "C_implementation_failure"
    result = {
        "status": "completed" if branch == "A_structure_isolated" else "failed",
        "branch": branch,
        "seed_pass_count": pass_count,
        "total_seeds": 3,
        "coverage_pass": coverage_pass,
        "stage_daps": list(STAGE_DAPS),
        "network_architecture_each_stage": "25-64-64-9",
        "parameters_each_stage": int(architecture["parameter_count"].iloc[0]),
        "total_ensemble_parameters": int(architecture.loc[architecture["seed"].eq(0), "parameter_count"].sum()),
        "formal_dqn_training_steps": 0,
        "unit_test_optimizer_steps": 6,
        "dssat_calls": 0,
        "online_interactions": 0,
        "saved_updated_checkpoint": False,
        "next_step_allowed": branch == "A_structure_isolated",
        "scope": "Structural isolation only; no claim of policy improvement or online success.",
    }
    (OUT / "023_00_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    q_rows = isolation.loc[isolation["q_max_abs_change"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, target_dap in zip(axes, (65, 110)):
        subset = q_rows.loc[q_rows["target_update_dap"].eq(target_dap)]
        grouped = subset.groupby("network_dap", as_index=False)["q_max_abs_change"].max()
        colors = ["#c62828" if int(dap) == target_dap else "#1565c0" for dap in grouped["network_dap"]]
        axis.bar(grouped["network_dap"].astype(str), grouped["q_max_abs_change"], color=colors)
        axis.set_title(f"Unit update at DAP {target_dap}")
        axis.set_xlabel("Independent network DAP")
        axis.set_ylabel("Maximum absolute Q change")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("023_00 stage-separated Q-network isolation audit")
    fig.tight_layout()
    fig.savefig(OUT / "023_00_stage_isolation_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "023_00_stage_isolation_audit.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

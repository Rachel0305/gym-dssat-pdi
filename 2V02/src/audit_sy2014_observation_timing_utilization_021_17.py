from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frozen_nstep_dqn_config_020_11 import ACTION_TABLE_9


MANIFEST = ROOT / "benchmark_results" / "021_06" / "021_06_fixed_state_manifest.csv"
OUT = ROOT / "benchmark_results" / "021_17"
MODELS = {
    "unscaled_021_10_10k": ROOT / "benchmark_results" / "021_10" / "021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1" / "checkpoints" / "checkpoint_10000" / "model.zip",
    "scaled_0p1_021_14_10k": ROOT / "benchmark_results" / "021_14" / "021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1" / "checkpoints" / "checkpoint_10000" / "model.zip",
}
DAILY = {
    "unscaled_021_10_10k": ROOT / "benchmark_results" / "021_10" / "021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1" / "checkpoints" / "checkpoint_10000" / "eval_daily.csv",
    "scaled_0p1_021_14_10k": ROOT / "benchmark_results" / "021_14" / "021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1" / "checkpoints" / "checkpoint_10000" / "eval_daily.csv",
}
DAP_GRID = (69, 76, 83, 89, 90, 94, 99, 102, 109)
EARLY_DAPS = (69, 76, 83, 89)
LATE_DAPS = (94, 99, 102, 109)
IRRIGATION_GROUPS = {0.0: (0, 3, 6), 15.0: (1, 4, 7), 30.0: (2, 5, 8)}


def phase(dap: float) -> str:
    if dap <= 89:
        return "pre_or_at_silking"
    if dap < 99:
        return "silking_to_grain_fill"
    return "grain_fill"


def input_mapping(manifest: pd.DataFrame) -> pd.DataFrame:
    x = np.vstack(manifest["observation_vector_json"].map(json.loads))
    rows: list[dict] = []
    for label in ("dap", "used_irrigation", "used_nitrogen"):
        y = pd.to_numeric(manifest[label], errors="coerce").to_numpy(float)
        for index in range(x.shape[1]):
            rows.append(
                {
                    "metadata_variable": label,
                    "observation_index": index,
                    "mean_absolute_error": float(np.nanmean(np.abs(x[:, index] - y))),
                    "max_absolute_error": float(np.nanmax(np.abs(x[:, index] - y))),
                    "correlation": float(np.corrcoef(x[:, index], y)[0, 1])
                    if np.std(x[:, index]) > 0 and np.std(y) > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def real_actions() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_name, path in DAILY.items():
        frame = pd.read_csv(path)
        frame["model"] = model_name
        frame["decision_dap"] = pd.to_numeric(frame["operation_dap"], errors="coerce")
        frame["phenology_phase"] = frame["decision_dap"].map(phase)
        rows.append(
            frame[
                [
                    "model",
                    "decision_dap",
                    "phenology_phase",
                    "action_index",
                    "irrigation_mm",
                    "fertilizer_kg_ha",
                    "grnwt",
                    "topwt",
                    "swfac",
                    "nstres",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def q_sweep(manifest: pd.DataFrame) -> pd.DataFrame:
    # Primary bases are the two real states closest to silking (DAP80), one from
    # each 021_06 reference trajectory. Other variables remain fixed, so this
    # is a network-sensitivity audit rather than a simulated agronomic path.
    bases = manifest[pd.to_numeric(manifest["dap"], errors="coerce").eq(80)].copy()
    if len(bases) != 2:
        raise RuntimeError(f"Expected two DAP80 fixed states, found {len(bases)}")
    models = {name: DQN.load(str(path)) for name, path in MODELS.items()}
    rows: list[dict] = []
    with torch.no_grad():
        for base in bases.itertuples(index=False):
            original = np.asarray(json.loads(base.observation_vector_json), dtype=np.float32)
            if not np.isclose(original[1], 80.0):
                raise RuntimeError("Observation index 1 is not DAP80 in selected base state")
            for dap in DAP_GRID:
                perturbed = original.copy()
                perturbed[1] = float(dap)
                for model_name, model in models.items():
                    tensor = obs_as_tensor(perturbed.reshape(1, -1), model.device)
                    q = model.q_net(tensor).cpu().numpy()[0]
                    ranks = (-q).argsort().argsort() + 1
                    for action_index, action in ACTION_TABLE_9.items():
                        rows.append(
                            {
                                "model": model_name,
                                "base_state_id": base.state_id,
                                "base_source_checkpoint": int(base.source_checkpoint),
                                "base_actual_dap": int(base.dap),
                                "counterfactual_dap": dap,
                                "counterfactual_phase": phase(dap),
                                "action_index": int(action_index),
                                "requested_irrigation": float(action["amir"]),
                                "requested_nitrogen": float(action["anfer"]),
                                "online_q": float(q[action_index]),
                                "online_rank": int(ranks[action_index]),
                                "online_argmax": int(np.argmax(q)),
                            }
                        )
    return pd.DataFrame(rows)


def q_margins(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    keys = ["model", "base_state_id", "base_source_checkpoint", "counterfactual_dap", "counterfactual_phase"]
    for key, group in q.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        indexed = group.set_index("action_index")
        for irrigation, indices in IRRIGATION_GROUPS.items():
            q0 = float(indexed.loc[indices[0], "online_q"])
            q50 = float(indexed.loc[indices[1], "online_q"])
            q100 = float(indexed.loc[indices[2], "online_q"])
            rows.append(
                {
                    **base,
                    "irrigation_group": irrigation,
                    "q_n0": q0,
                    "q_n50": q50,
                    "q_n100": q100,
                    "nitrogen_preference_margin": max(q50, q100) - q0,
                    "preferred_nitrogen": (0.0, 50.0, 100.0)[int(np.argmax([q0, q50, q100]))],
                }
            )
    margins = pd.DataFrame(rows)
    summaries: list[dict] = []
    for key, group in margins.groupby(["model", "base_state_id", "irrigation_group"], sort=False):
        early = group[group["counterfactual_dap"].isin(EARLY_DAPS)]["nitrogen_preference_margin"]
        late = group[group["counterfactual_dap"].isin(LATE_DAPS)]["nitrogen_preference_margin"]
        argmax_count = q[
            q["model"].eq(key[0])
            & q["base_state_id"].eq(key[1])
        ][["counterfactual_dap", "online_argmax"]].drop_duplicates()["online_argmax"].nunique()
        summaries.append(
            {
                "model": key[0],
                "base_state_id": key[1],
                "irrigation_group": key[2],
                "mean_margin_early": float(early.mean()),
                "mean_margin_late": float(late.mean()),
                "early_minus_late_margin": float(early.mean() - late.mean()),
                "direction_agronomically_consistent": bool(early.mean() > late.mean()),
                "unique_global_argmax_across_dap_grid": int(argmax_count),
            }
        )
    return margins, pd.DataFrame(summaries)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    mapping = input_mapping(manifest)
    mapping.to_csv(OUT / "021_17_observation_index_mapping.csv", index=False)
    best = mapping.sort_values(["metadata_variable", "mean_absolute_error"]).groupby("metadata_variable").head(1)

    actions = real_actions()
    actions.to_csv(OUT / "021_17_real_trajectory_actions.csv", index=False)
    action_phase = (
        actions.groupby(["model", "phenology_phase"], as_index=False)
        .agg(
            irrigation_total=("irrigation_mm", "sum"),
            nitrogen_total=("fertilizer_kg_ha", "sum"),
            nonzero_irrigation_days=("irrigation_mm", lambda s: int((s > 0).sum())),
            nonzero_nitrogen_days=("fertilizer_kg_ha", lambda s: int((s > 0).sum())),
        )
    )
    action_phase.to_csv(OUT / "021_17_real_action_phase_summary.csv", index=False)

    q = q_sweep(manifest)
    q.to_csv(OUT / "021_17_dap_only_q_sweep.csv", index=False)
    margins, sensitivity = q_margins(q)
    margins.to_csv(OUT / "021_17_dap_only_nitrogen_margins.csv", index=False)
    sensitivity.to_csv(OUT / "021_17_dap_sensitivity_summary.csv", index=False)

    configured = json.loads(manifest["configured_observations_json"].iloc[0])
    result = {
        "status": "completed_offline",
        "training_calls": 0,
        "dssat_calls": 0,
        "observation_dimension": int(manifest["observation_dimension"].iloc[0]),
        "exact_dap_index": int(best.loc[best["metadata_variable"].eq("dap"), "observation_index"].iloc[0]),
        "exact_cumulative_irrigation_index": int(best.loc[best["metadata_variable"].eq("used_irrigation"), "observation_index"].iloc[0]),
        "exact_cumulative_nitrogen_index": int(best.loc[best["metadata_variable"].eq("used_nitrogen"), "observation_index"].iloc[0]),
        "all_three_exact_mae_zero": bool(best["max_absolute_error"].eq(0.0).all()),
        "istage_configured": "istage" in configured,
        "vstage_configured": "vstage" in configured,
        "totir_configured": "totir" in configured,
        "models_with_dap_direction_consistent_all_groups": [
            name
            for name, group in sensitivity.groupby("model")
            if bool(group["direction_agronomically_consistent"].all())
        ],
        "interpretation_boundary": (
            "The actual DQN input already contains exact DAP, cumulative irrigation, cumulative nitrogen, "
            "istage and vstage information. DAP-only perturbations test network sensitivity but are partly "
            "off-manifold and cannot by themselves establish an agronomic causal policy rule."
        ),
    }
    (OUT / "021_17_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(best.to_string(index=False))
    print(action_phase.to_string(index=False))
    print(sensitivity.to_string(index=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

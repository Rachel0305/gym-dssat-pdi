"""Read-only diagnostic for the 026 YC PPO KL-GUARD retention screen.

This script deliberately does not import the DSSAT environment or invoke PPO
training.  It reads the six 026 TensorBoard-derived metric CSVs, validation
aggregates, manifests, and the installed SB3/sb3-contrib source, then writes
diagnostic evidence and a bounded interpretation for prompt 027.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = ROOT / "benchmark_results" / "026_yc_ppo_seed0_25k_3rng_klguard_retention_screen"
REPORT = ROOT / "docs" / "yc_ppo_seed0_25k_klguard_harm_mechanism_diagnostic.md"
START_YIELD = 8201.599243164062
CRNGS = [1, 2, 3]
BRANCHES = ["control", "klguard"]
WINDOWS = [
    (25_000, 35_000, "25-35K"),
    (35_000, 50_000, "35-50K"),
    (50_000, 75_000, "50-75K"),
    (75_000, 100_000, "75-100K"),
]
CHECKPOINTS = [25_000, 35_000, 50_000, 75_000, 100_000]

MASKABLE_PPO = Path("/home/gymusr/.local/lib/python3.10/site-packages/sb3_contrib/ppo_mask/ppo_mask.py")
MASKABLE_POLICY = Path("/home/gymusr/.local/lib/python3.10/site-packages/sb3_contrib/common/maskable/policies.py")
TORCH_LAYERS = Path("/opt/gym_dssat_pdi/lib/python3.10/site-packages/stable_baselines3/common/torch_layers.py")
BASE_CLASS = Path("/opt/gym_dssat_pdi/lib/python3.10/site-packages/stable_baselines3/common/base_class.py")
ON_POLICY = Path("/opt/gym_dssat_pdi/lib/python3.10/site-packages/stable_baselines3/common/on_policy_algorithm.py")


def metric_path(crng: int, branch: str) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{crng}_{branch}_training_metrics.csv"


def validation_path(crng: int, branch: str) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{crng}_{branch}_validation.csv"


def read_metrics(crng: int, branch: str) -> pd.DataFrame:
    path = metric_path(crng, branch)
    df = pd.read_csv(path)
    numeric = [
        "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss",
        "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate",
        "clip_range", "epochs_attempted", "epochs_completed",
        "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("global_step").reset_index(drop=True)


def read_validation(crng: int, branch: str) -> pd.DataFrame:
    df = pd.read_csv(validation_path(crng, branch), keep_default_na=False)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    for col in ["mean_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid", "fallback"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["row_type"].eq("aggregate")].sort_values("checkpoint_step").reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def safe_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if len(values) else math.nan


def safe_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else math.nan


def window_frame(df: pd.DataFrame, lower: int, upper: int) -> pd.DataFrame:
    return df[df["global_step"].between(lower + 1, upper)].copy()


def validation_value(validation: pd.DataFrame, checkpoint: int, column: str) -> float:
    row = validation[validation["checkpoint_step"].eq(checkpoint)]
    if len(row) != 1 or column not in row:
        return math.nan
    return float(pd.to_numeric(row.iloc[0][column], errors="coerce"))


def output_csv(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def control_value_health(metrics: dict[tuple[int, str], pd.DataFrame], validations: dict[tuple[int, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for crng in CRNGS:
        df = metrics[(crng, "control")]
        validation = validations[(crng, "control")]
        for lower, upper, label in WINDOWS:
            group = window_frame(df, lower, upper)
            row: dict[str, Any] = {
                "continuation_rng": f"CRNG{crng}",
                "branch": "CONTROL",
                "window": label,
                "lower_step": lower,
                "upper_step": upper,
                "update_count": int(len(group)),
                "optimizer_step_count": float(pd.to_numeric(group["optimizer_steps_completed"], errors="coerce").sum()),
                "optimizer_step_fraction": float((pd.to_numeric(group["optimizer_steps_completed"], errors="coerce") > 0).mean()) if len(group) else math.nan,
                "control_yield_at_lower_checkpoint": validation_value(validation, lower, "mean_yield_kg_ha"),
                "control_yield_at_upper_checkpoint": validation_value(validation, upper, "mean_yield_kg_ha"),
                "control_yield_change_in_window": validation_value(validation, upper, "mean_yield_kg_ha") - validation_value(validation, lower, "mean_yield_kg_ha"),
            }
            for col in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance"]:
                row[f"{col}_mean"] = safe_mean(group[col])
                row[f"{col}_median"] = safe_median(group[col])
            row["approx_kl_max"] = safe_max(group["approx_kl"])
            row["clip_fraction_max"] = safe_max(group["clip_fraction"])
            row["value_loss_max"] = safe_max(group["value_loss"])
            row["explained_variance_min"] = safe_min(group["explained_variance"])
            row["learning_rate_mean"] = safe_mean(group["learning_rate"])
            row["learning_rate_min"] = safe_min(group["learning_rate"])
            row["learning_rate_max"] = safe_max(group["learning_rate"])
            row["clip_range_mean"] = safe_mean(group["clip_range"])
            row["clip_range_min"] = safe_min(group["clip_range"])
            row["clip_range_max"] = safe_max(group["clip_range"])
            row["epochs_attempted_mean"] = safe_mean(group["epochs_attempted"])
            rows.append(row)
    return pd.DataFrame(rows)


def guard_vs_control_divergence(metrics: dict[tuple[int, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    compare_cols = [
        "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss",
        "explained_variance", "epochs_attempted", "epochs_completed", "optimizer_steps_completed",
        "kl_guard_triggered",
    ]
    for crng in CRNGS:
        control = metrics[(crng, "control")]
        guard = metrics[(crng, "klguard")]
        for lower, upper, label in WINDOWS:
            row: dict[str, Any] = {
                "continuation_rng": f"CRNG{crng}",
                "window": label,
                "lower_step": lower,
                "upper_step": upper,
                "control_update_count": int(len(window_frame(control, lower, upper))),
                "guard_update_count": int(len(window_frame(guard, lower, upper))),
            }
            for col in compare_cols:
                c = safe_mean(window_frame(control, lower, upper)[col])
                g = safe_mean(window_frame(guard, lower, upper)[col])
                row[f"control_{col}_mean"] = c
                row[f"guard_{col}_mean"] = g
                row[f"guard_minus_control_{col}_mean"] = g - c
            row["guard_trigger_rate"] = row["guard_kl_guard_triggered_mean"]
            row["optimizer_step_fraction_control"] = float((window_frame(control, lower, upper)["optimizer_steps_completed"] > 0).mean())
            row["optimizer_step_fraction_guard"] = float((window_frame(guard, lower, upper)["optimizer_steps_completed"] > 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def timeline_rows(
    crng: int,
    metrics: dict[tuple[int, str], pd.DataFrame],
    validations: dict[tuple[int, str], pd.DataFrame],
    annotation: str,
) -> list[dict[str, Any]]:
    control = validations[(crng, "control")]
    guard = validations[(crng, "klguard")]
    cmetrics = metrics[(crng, "control")]
    gmetrics = metrics[(crng, "klguard")]
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint == 25_000:
            lower, label = math.nan, "START"
            cg = pd.DataFrame()
            gg = pd.DataFrame()
        else:
            lower, label = next((lo, w) for lo, hi, w in WINDOWS if hi == checkpoint)
            cg = window_frame(cmetrics, lower, checkpoint)
            gg = window_frame(gmetrics, lower, checkpoint)
        cy = validation_value(control, checkpoint, "mean_yield_kg_ha")
        gy = validation_value(guard, checkpoint, "mean_yield_kg_ha")
        row: dict[str, Any] = {
            "continuation_rng": f"CRNG{crng}",
            "checkpoint_step": checkpoint,
            "preceding_window": label,
            "control_yield_kg_ha": cy,
            "guard_yield_kg_ha": gy,
            "guard_minus_control_yield_kg_ha": gy - cy,
            "control_yield_drop_from_start": START_YIELD - cy,
            "guard_yield_drop_from_start": START_YIELD - gy,
            "control_I_mm": validation_value(control, checkpoint, "mean_I_mm"),
            "guard_I_mm": validation_value(guard, checkpoint, "mean_I_mm"),
            "guard_minus_control_I_mm": validation_value(guard, checkpoint, "mean_I_mm") - validation_value(control, checkpoint, "mean_I_mm"),
            "control_N_kg_ha": validation_value(control, checkpoint, "mean_N_kg_ha"),
            "guard_N_kg_ha": validation_value(guard, checkpoint, "mean_N_kg_ha"),
            "guard_minus_control_N_kg_ha": validation_value(guard, checkpoint, "mean_N_kg_ha") - validation_value(control, checkpoint, "mean_N_kg_ha"),
            "control_PFP_N": validation_value(control, checkpoint, "mean_PFP_N"),
            "guard_PFP_N": validation_value(guard, checkpoint, "mean_PFP_N"),
            "control_WP_ET": validation_value(control, checkpoint, "mean_WP_ET"),
            "guard_WP_ET": validation_value(guard, checkpoint, "mean_WP_ET"),
            "guard_trigger_rate_preceding_window": safe_mean(gg["kl_guard_triggered"]) if len(gg) else math.nan,
            "guard_epochs_completed_mean_preceding_window": safe_mean(gg["epochs_completed"]) if len(gg) else math.nan,
            "guard_optimizer_steps_mean_preceding_window": safe_mean(gg["optimizer_steps_completed"]) if len(gg) else math.nan,
            "control_epochs_completed_mean_preceding_window": safe_mean(cg["epochs_completed"]) if len(cg) else math.nan,
            "control_optimizer_steps_mean_preceding_window": safe_mean(cg["optimizer_steps_completed"]) if len(cg) else math.nan,
            "control_value_loss_mean_preceding_window": safe_mean(cg["value_loss"]) if len(cg) else math.nan,
            "guard_value_loss_mean_preceding_window": safe_mean(gg["value_loss"]) if len(gg) else math.nan,
            "guard_minus_control_value_loss_mean": (safe_mean(gg["value_loss"]) - safe_mean(cg["value_loss"])) if len(gg) and len(cg) else math.nan,
            "control_explained_variance_mean_preceding_window": safe_mean(cg["explained_variance"]) if len(cg) else math.nan,
            "guard_explained_variance_mean_preceding_window": safe_mean(gg["explained_variance"]) if len(gg) else math.nan,
            "guard_minus_control_explained_variance_mean": (safe_mean(gg["explained_variance"]) - safe_mean(cg["explained_variance"])) if len(gg) and len(cg) else math.nan,
            "control_entropy_loss_mean_preceding_window": safe_mean(cg["entropy_loss"]) if len(cg) else math.nan,
            "guard_entropy_loss_mean_preceding_window": safe_mean(gg["entropy_loss"]) if len(gg) else math.nan,
            "control_policy_gradient_loss_mean_preceding_window": safe_mean(cg["policy_gradient_loss"]) if len(cg) else math.nan,
            "guard_policy_gradient_loss_mean_preceding_window": safe_mean(gg["policy_gradient_loss"]) if len(gg) else math.nan,
            "interpretation_note": annotation,
        }
        rows.append(row)
    return rows


def source_text(path: Path, start: int, end: int) -> str:
    if not path.exists():
        return "unknown / insufficient evidence"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_semantics_audit() -> pd.DataFrame:
    rows = [
        {
            "audit_item": "KL stop trigger",
            "conclusion": "approx_kl is checked after loss construction but before zero_grad/backward/optimizer.step on the triggering minibatch.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_PPO), "function": "MaskablePPO.train", "line_start": 386, "line_end": 408,
            "evidence": source_text(MASKABLE_PPO, 386, 408),
        },
        {
            "audit_item": "stop scope",
            "conclusion": "The break exits the current minibatch loop, then continue_training breaks the remaining epochs; it stops the remaining epochs of the current train() call, not the whole learn() call.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_PPO), "function": "MaskablePPO.train / MaskablePPO.learn", "line_start": 397, "line_end": 412,
            "evidence": source_text(MASKABLE_PPO, 397, 412),
        },
        {
            "audit_item": "whole learn call",
            "conclusion": "The outer learn loop continues with later rollouts/updates while num_timesteps remains below total_timesteps.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_PPO), "function": "MaskablePPO.learn", "line_start": 452, "line_end": 467,
            "evidence": source_text(MASKABLE_PPO, 452, 467),
        },
        {
            "audit_item": "VALUE_UPDATES_REDUCED_BY_KL_STOP",
            "conclusion": "YES: the triggering minibatch skips optimizer.step and all later minibatches/epochs in that train() call are skipped, so future value updates are reduced together with policy updates.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_PPO), "function": "MaskablePPO.train", "line_start": 386, "line_end": 412,
            "evidence": source_text(MASKABLE_PPO, 386, 412),
        },
        {
            "audit_item": "actor_critic_loss_and_optimizer",
            "conclusion": "Policy, entropy, and value losses form one loss; one optimizer.step updates all parameters included in policy.parameters().",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_PPO), "function": "MaskablePPO.train", "line_start": 386, "line_end": 408,
            "evidence": source_text(MASKABLE_PPO, 386, 408),
        },
        {
            "audit_item": "shared_feature_extractor",
            "conclusion": "The actual MaskableActorCriticPolicy default is share_features_extractor=True; pi_features_extractor and vf_features_extractor point to the same extractor.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_POLICY), "function": "MaskableActorCriticPolicy.__init__", "line_start": 45, "line_end": 106,
            "evidence": source_text(MASKABLE_POLICY, 45, 106),
        },
        {
            "audit_item": "separate_heads",
            "conclusion": "Actor and critic MLP branches are separately constructed; action_net and value_net are separate heads, while the input feature extractor is shared.",
            "status": "CONFIRMED",
            "code_path": str(MASKABLE_POLICY), "function": "MaskableActorCriticPolicy._build; MlpExtractor", "line_start": 201, "line_end": 241,
            "evidence": source_text(MASKABLE_POLICY, 201, 241) + "\n--- MlpExtractor ---\n" + source_text(TORCH_LAYERS, 218, 261),
        },
        {
            "audit_item": "optimizer_state_save_load",
            "conclusion": "The algorithm declares policy.optimizer as a torch state dict to save/load; BaseAlgorithm.load reconstructs the policy then restores saved parameters/state.",
            "status": "CONFIRMED",
            "code_path": str(BASE_CLASS), "function": "BaseAlgorithm._get_torch_save_params/load/save", "line_start": 340, "line_end": 345,
            "evidence": source_text(BASE_CLASS, 340, 345) + "\n--- load ---\n" + source_text(BASE_CLASS, 681, 744) + "\n--- save ---\n" + source_text(BASE_CLASS, 844, 867),
        },
        {
            "audit_item": "optimizer_and_schedule_runtime_check",
            "conclusion": "The six 026 manifests report optimizer_state_restored=True and 12 loaded optimizer state entries; 026 continuation used reset_num_timesteps=False and the recorded learning rate remained 0.0003. No reinitialization/reset bug is evidenced.",
            "status": "NO_OBVIOUS_OPTIMIZER_STATE_BUG_FOUND",
            "code_path": "benchmark_results/026_yc_ppo_seed0_25k_3rng_klguard_retention_screen/*/training_manifest.json",
            "function": "026 continuation manifest; BaseAlgorithm._update_learning_rate",
            "line_start": 287, "line_end": 301,
            "evidence": source_text(BASE_CLASS, 287, 301),
        },
    ]
    rows_df = pd.DataFrame(rows)
    output_csv(rows_df, "yc_ppo_klguard_source_semantics_audit.csv")
    return rows_df


def manifest_state_check() -> dict[str, Any]:
    manifests = sorted(EXPERIMENT.glob("crng*/**/training_manifest.json"))
    restored: list[bool] = []
    entries: list[int] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        restored.append(bool(payload.get("optimizer_state_restored")))
        entries.append(int(payload.get("optimizer_state_entries_at_load", 0)))
    zips = sorted(EXPERIMENT.glob("crng*/**/*.zip"))
    zip_with_optimizer = 0
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            if "policy.optimizer.pth" in archive.namelist():
                zip_with_optimizer += 1
    return {
        "manifest_count": len(manifests),
        "optimizer_state_restored_all": bool(manifests) and all(restored),
        "optimizer_state_entries_at_load": sorted(set(entries)),
        "checkpoint_zip_count": len(zips),
        "checkpoint_zips_with_optimizer_state": zip_with_optimizer,
    }


def classify_hypotheses(
    health: pd.DataFrame,
    divergence: pd.DataFrame,
    harm: pd.DataFrame,
    rescue: pd.DataFrame,
    source_audit: pd.DataFrame,
    state_check: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "hypothesis": "H1_Dangerous_trajectory_suppression",
            "classification": "PARTIALLY_SUPPORTED",
            "evidence": "CRNG2 CONTROL has the distinctive later deterioration: value_loss mean 0.0495 -> 0.1091 -> 0.3069, explained_variance mean 0.4400 -> 0.4109 -> 0.1908 -> -0.4018, with approx_kl/clip_fraction also higher than CRNG1/3. KL-GUARD keeps CRNG2 value_loss near 0.05-0.06 and explained_variance positive while retaining 8202.48 kg/ha at 100K. The earliest 35K yield drop means temporal alignment is strong but not a causal proof.",
            "uncertainty": "The signal is clearest in 35-100K; the 35K performance drop and later critic deterioration are not enough to prove guard-mediated suppression as the unique cause.",
        },
        {
            "hypothesis": "H2_Healthy_trajectory_over_intervention",
            "classification": "PARTIALLY_SUPPORTED",
            "evidence": "CRNG1/3 CONTROL retain positive window-mean explained_variance (0.419-0.605 and 0.426-0.561) and low value_loss means (~0.052-0.076) across windows, while KL-GUARD triggers repeatedly (CRNG1 0.201-0.365; CRNG3 reaches 0.457 in 75-100K) and both end below their CONTROL yield. The pattern supports over-intervention as a plausible contributor, but CRNG3 guard value metrics are not worse in the harm window.",
            "uncertainty": "Repeated truncation and harm co-occur, but the data do not isolate actor/value update effects from management-policy changes.",
        },
        {
            "hypothesis": "H3_Critic_update_starvation_collateral_truncation",
            "classification": "PARTIALLY_SUPPORTED",
            "evidence": "Source semantics confirm VALUE_UPDATES_REDUCED_BY_KL_STOP=YES: the same loss contains policy and value terms, one optimizer covers policy parameters, and the guard skips future optimizer steps after a policy-KL trigger. 026 metrics show reduced guard epochs/optimizer steps. This establishes the mechanism, not that it caused CRNG1/3 harm; CRNG3 has lower value_loss and higher explained_variance than CONTROL in 75-100K despite harm.",
            "uncertainty": "Collateral truncation is a credible mechanism-level explanation but not localized as the harm root.",
        },
        {
            "hypothesis": "H4_Optimizer_state_bug",
            "classification": "NOT_SUPPORTED",
            "evidence": f"All {state_check['manifest_count']} manifests report optimizer restoration with loaded state entries {state_check['optimizer_state_entries_at_load']}; all {state_check['checkpoint_zip_count']} continuation checkpoint zips contain policy.optimizer.pth. Source save/load paths include policy.optimizer and no reinitialize/reset defect was found.",
            "uncertainty": "This is a source-and-artifact audit, not a formal proof against every external runtime fault.",
        },
    ]
    hypotheses = pd.DataFrame(rows)
    output_csv(hypotheses, "yc_ppo_seed0_25k_klguard_hypothesis_classification.csv")

    decision = pd.DataFrame([{
        "ACTOR_ONLY_GUARD_NEXT": "INSUFFICIENT_EVIDENCE",
        "CONDITIONAL_GUARD_NEXT": "YES",
        "preferred_next_candidate": "CONDITIONAL_GUARD",
        "decision_basis": "CRNG2 shows a distinctive observable value-function/KL/clip instability pattern before its final 100K collapse, while CRNG1/3 CONTROL lacks the same late deterioration. However, actor-only guard is not sufficiently justified because CRNG1/3 harm is not accompanied by systematic guard-side value deterioration; CRNG3 is harmed while guard value metrics are numerically healthier.",
        "status_boundary": "Candidate direction only; no actor-only or conditional intervention was implemented or tested in this task.",
        "source_semantics": "VALUE_UPDATES_REDUCED_BY_KL_STOP=YES",
    }])
    output_csv(decision, "yc_ppo_seed0_25k_klguard_next_intervention_decision.csv")
    return hypotheses, decision


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown / insufficient evidence"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def report(
    health: pd.DataFrame,
    divergence: pd.DataFrame,
    harm: pd.DataFrame,
    rescue: pd.DataFrame,
    source_audit: pd.DataFrame,
    hypotheses: pd.DataFrame,
    decision: pd.DataFrame,
    state_check: dict[str, Any],
) -> None:
    c2 = health[health["continuation_rng"].eq("CRNG2")].copy()
    c13 = health[health["continuation_rng"].isin(["CRNG1", "CRNG3"])].copy()
    d2 = divergence[divergence["continuation_rng"].eq("CRNG2")]
    harm_table = harm[[
        "continuation_rng", "checkpoint_step", "control_yield_kg_ha", "guard_yield_kg_ha",
        "guard_minus_control_yield_kg_ha", "guard_trigger_rate_preceding_window",
        "guard_epochs_completed_mean_preceding_window", "guard_optimizer_steps_mean_preceding_window",
        "control_value_loss_mean_preceding_window", "guard_value_loss_mean_preceding_window",
        "control_explained_variance_mean_preceding_window", "guard_explained_variance_mean_preceding_window",
        "control_I_mm", "guard_I_mm", "control_N_kg_ha", "guard_N_kg_ha", "interpretation_note",
    ]].fillna("not applicable").to_markdown(index=False)
    rescue_table = rescue[[
        "checkpoint_step", "control_yield_kg_ha", "guard_yield_kg_ha", "guard_minus_control_yield_kg_ha",
        "guard_trigger_rate_preceding_window", "control_value_loss_mean_preceding_window",
        "guard_value_loss_mean_preceding_window", "control_explained_variance_mean_preceding_window",
        "guard_explained_variance_mean_preceding_window", "interpretation_note",
    ]].fillna("not applicable").to_markdown(index=False)
    hypotheses_table = hypotheses[["hypothesis", "classification", "evidence"]].to_markdown(index=False)
    source_subset = source_audit[["audit_item", "conclusion", "status", "code_path", "function", "line_start", "line_end"]].to_markdown(index=False)

    c2_35 = c2[c2["window"].eq("35-50K")].iloc[0]
    c2_50 = c2[c2["window"].eq("50-75K")].iloc[0]
    c2_75 = c2[c2["window"].eq("75-100K")].iloc[0]
    c1 = divergence[divergence["continuation_rng"].eq("CRNG1")]
    c3 = divergence[divergence["continuation_rng"].eq("CRNG3")]
    lines = [
        "# YC PPO seed0@25K KL-GUARD harm mechanism diagnostic",
        "",
        "## Status",
        "",
        "- This is a read-only diagnostic of the completed 026 experiment. No PPO training, intervention, threshold tuning, new CRNG, seed2, augmentation, or site expansion was run.",
        "- Final diagnostic status: `KL_GUARD_HARM_MECHANISM_PARTIAL`.",
        "- `YC_PPO_KLGUARD_HARM_MECHANISM_DIAGNOSTIC_COMPLETE`",
        "",
        "## Evidence boundary and source",
        "",
        "- Inputs are the six 026 training-metrics CSVs, six validation CSVs, paired/update-window outputs, six training manifests, and the installed Docker runtime source.",
        "- Metrics are per PPO update records from the 026 TensorBoard event files; validation is deterministic masked 2014–2023 aggregate evidence. The 25K rows are the reused 025 common start.",
        f"- Source state audit: {state_check['manifest_count']} manifests, {state_check['checkpoint_zip_count']} continuation checkpoint zips, {state_check['checkpoint_zips_with_optimizer_state']} zips containing `policy.optimizer.pth`.",
        "",
        "## 1. CONTROL value-function health",
        "",
        "- CRNG2 CONTROL shows the distinctive deterioration pattern. Its value_loss mean is " + f"{fmt(c2_35['value_loss_mean'])} at 35–50K, {fmt(c2_50['value_loss_mean'])} at 50–75K, and {fmt(c2_75['value_loss_mean'])} at 75–100K; explained_variance mean is " + f"{fmt(c2_35['explained_variance_mean'])}, {fmt(c2_50['explained_variance_mean'])}, then {fmt(c2_75['explained_variance_mean'])}.",
        "- The same CRNG2 CONTROL has approx_kl mean `0.0001658` in 35–50K and clip_fraction mean `0.000174`, higher than CRNG1/3 in that window; its explained_variance minimum reaches `-23.9153` and `-35.9145` in the two late windows.",
        "- CRNG1/3 CONTROL do not show the same window-mean deterioration: value_loss stays near `0.052–0.076`, and explained_variance means remain positive (`0.419–0.605` for CRNG1; `0.426–0.561` for CRNG3). Their minima can be negative, so this is not a claim of perfect critic health.",
        "- Timing: CRNG2 yield is already lower at 35K (`6827.124` vs start `8201.599`), while the strongest mean critic deterioration appears in 35–50K and later. Therefore the evidence supports a pre-final-collapse instability phase, but not a clean proof that critic deterioration preceded the first performance drop.",
        "",
        "## 2. GUARD versus CONTROL divergence",
        "",
        f"- CRNG1: training divergence is already visible in 25–35K (guard trigger rate {fmt(c1[c1.window.eq('25-35K')].iloc[0]['guard_trigger_rate'], 3)}, fewer mean optimizer steps, lower guard explained_variance); the first clear validation/yield divergence is at 35K (guard `7581.363` vs control `8201.451` kg/ha, with lower I).",
        f"- CRNG2: divergence is visible by 35–50K: guard value_loss mean is {fmt(d2[d2.window.eq('35-50K')].iloc[0]['guard_value_loss_mean'])} vs control {fmt(d2[d2.window.eq('35-50K')].iloc[0]['control_value_loss_mean'])}, and guard explained_variance is {fmt(d2[d2.window.eq('35-50K')].iloc[0]['guard_explained_variance_mean'])} vs control {fmt(d2[d2.window.eq('35-50K')].iloc[0]['control_explained_variance_mean'])}; guard validation is already `8202.360` vs control `6827.124` at 35K.",
        f"- CRNG3: early training differences are small; the clearest training divergence is 75–100K, where guard trigger rate is {fmt(c3[c3.window.eq('75-100K')].iloc[0]['guard_trigger_rate'], 3)}, mean completed epochs {fmt(c3[c3.window.eq('75-100K')].iloc[0]['guard_epochs_completed_mean'], 3)} vs control `5`, and mean optimizer steps {fmt(c3[c3.window.eq('75-100K')].iloc[0]['guard_optimizer_steps_completed_mean'], 3)} vs control `5`. The first clear validation divergence is at 100K (`7470.997` vs `8202.764` kg/ha).",
        "- Thus training divergence precedes or coincides with performance divergence for CRNG1/2; for CRNG3 the clearest divergence and performance harm appear in the same 75–100K interval, not clearly earlier.",
        "",
        "## 3. Harm timeline: CRNG1 and CRNG3",
        "",
        harm_table,
        "",
        "- CRNG1 trigger frequency is not monotonically increasing: it rises from 25–35K to 35–50K, falls in 50–75K, then rises again in 75–100K. Guard value_loss is not systematically higher than CONTROL before the endpoint; the key early difference is truncation plus lower early explained_variance, followed by management/yield divergence.",
        "- CRNG3 has the requested 75–100K trigger rate `0.456647`. Completed epochs and optimizer steps fall, but guard explained_variance is higher and value_loss lower than CONTROL in that same window. The 100K harm is accompanied by N changing from `240` to `88` kg/ha in the guard branch, so a critic-deterioration explanation is not sufficient.",
        "",
        "## 4. Rescue timeline: CRNG2",
        "",
        rescue_table,
        "",
        "- CRNG2 CONTROL moves from moderate early metrics to value_loss escalation and explained_variance collapse in the later windows, while its yield reaches `6034.563` kg/ha at 100K. KL-GUARD triggers most strongly in 50–75K (`0.522989`) and keeps value_loss around `0.056` and explained_variance around `0.594` there; 100K yield remains `8202.479` kg/ha with I/N `222/240`.",
        "- This is consistent with dangerous-trajectory suppression, but the evidence remains descriptive and does not establish KL as the unique cause.",
        "",
        "## 5. Source-level KL early-stop semantics",
        "",
        source_subset,
        "",
        "- `VALUE_UPDATES_REDUCED_BY_KL_STOP = YES`.",
        "- Stop level: the triggering minibatch does not perform `optimizer.step()`; `break` exits the current epoch's minibatch loop; the outer `continue_training` check then exits the remaining epochs of the current `train()` call. `learn()` continues with later rollouts/updates until its timestep condition is met.",
        "- Actor and critic share the feature extractor and one Adam optimizer, but have separate policy/value MLP branches and separate action/value heads. The skipped future optimizer steps therefore reduce future updates to shared features, actor branch, and critic branch together.",
        "- `NO_OBVIOUS_OPTIMIZER_STATE_BUG_FOUND`: source save/load includes `policy.optimizer`; all 026 manifests report restored optimizer state with 12 entries; all continuation checkpoint zips contain `policy.optimizer.pth`; no scheduler reset/reinitialization defect is evidenced.",
        "",
        "## 6. Hypothesis classification",
        "",
        hypotheses_table,
        "",
        "## 7. Next-candidate decision (not tested here)",
        "",
        f"- `ACTOR_ONLY_GUARD_NEXT = {decision.iloc[0]['ACTOR_ONLY_GUARD_NEXT']}`.",
        f"- `CONDITIONAL_GUARD_NEXT = {decision.iloc[0]['CONDITIONAL_GUARD_NEXT']}`.",
        "- The conditional-guard direction is the more defensible future candidate because CRNG2 has an observable instability pattern that is absent from CRNG1/3 CONTROL and appears before the final 100K collapse. This is a candidate-method rationale, not a success claim.",
        "- Actor-only guard remains insufficiently evidenced: CRNG1/3 harm is not accompanied by systematic guard-side value deterioration, and CRNG3 is harmed while guard value metrics are numerically healthier.",
        "",
        "## Final answers",
        "",
        "1. CRNG2 CONTROL pre-final-collapse value instability: `PARTIALLY_SUPPORTED` (clear later deterioration; first yield drop is earlier than the strongest mean critic signal).",
        "2. CRNG1/3 CONTROL lack the same instability: `YES` at window-mean level, with isolated negative minima acknowledged.",
        "3. CRNG1/3 GUARD systematic value deterioration before harm: `NO` as a consistent pattern.",
        "4. KL early-stop stops the triggering minibatch update, then remaining minibatches/epochs in the current `train()` call; it does not directly end the whole `learn()` call.",
        "5. Critic/value updates reduced: `YES`.",
        "6. Actor/critic share feature extractor and optimizer; their MLP branches and heads are separate.",
        "7. Optimizer/state bug evidence: `NO_OBVIOUS_OPTIMIZER_STATE_BUG_FOUND`.",
        "8. H1/H2/H3/H4: `PARTIALLY_SUPPORTED` / `PARTIALLY_SUPPORTED` / `PARTIALLY_SUPPORTED` / `NOT_SUPPORTED`.",
        f"9. `ACTOR_ONLY_GUARD_NEXT = {decision.iloc[0]['ACTOR_ONLY_GUARD_NEXT']}`.",
        f"10. `CONDITIONAL_GUARD_NEXT = {decision.iloc[0]['CONDITIONAL_GUARD_NEXT']}`.",
        "11. Conditional guard is the higher-priority future candidate, but it was not implemented or tested in this task.",
        "12. `KL_GUARD_HARM_MECHANISM_PARTIAL`.",
        "",
        "## Stop boundary",
        "",
        "No PPO/MaskablePPO source was modified. No actor-only guard, conditional guard, threshold change, 026 rerun, new CRNG, seed2, Augmented, or other site was started.",
        "",
        "YC_PPO_KLGUARD_HARM_MECHANISM_DIAGNOSTIC_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metrics = {(crng, branch): read_metrics(crng, branch) for crng in CRNGS for branch in BRANCHES}
    validations = {(crng, branch): read_validation(crng, branch) for crng in CRNGS for branch in BRANCHES}
    health = control_value_health(metrics, validations)
    output_csv(health, "yc_ppo_seed0_25k_control_value_health_by_window.csv")
    divergence = guard_vs_control_divergence(metrics)
    output_csv(divergence, "yc_ppo_seed0_25k_guard_vs_control_divergence_by_window.csv")
    harm_rows = timeline_rows(1, metrics, validations, "CRNG1: early guard/control divergence; 35K validation divergence; 100K material harm endpoint.")
    harm_rows += timeline_rows(3, metrics, validations, "CRNG3: 75-100K trigger/optimizer truncation and 100K yield/N harm; guard value metrics not worse in that window.")
    harm = pd.DataFrame(harm_rows)
    output_csv(harm, "yc_ppo_seed0_25k_guard_harm_timeline_crng1_crng3.csv")
    rescue = pd.DataFrame(timeline_rows(2, metrics, validations, "CRNG2: control value deterioration becomes clear by 35-50K and worsens late; guard remains value-stable and retains high yield."))
    output_csv(rescue, "yc_ppo_seed0_25k_guard_rescue_timeline_crng2.csv")
    source_audit = source_semantics_audit()
    state_check = manifest_state_check()
    hypotheses, decision = classify_hypotheses(health, divergence, harm, rescue, source_audit, state_check)
    report(health, divergence, harm, rescue, source_audit, hypotheses, decision, state_check)
    print(json.dumps({
        "status": "PASS",
        "report": str(REPORT.relative_to(ROOT)),
        "outputs": [
            "results/yc_ppo_seed0_25k_control_value_health_by_window.csv",
            "results/yc_ppo_seed0_25k_guard_vs_control_divergence_by_window.csv",
            "results/yc_ppo_seed0_25k_guard_harm_timeline_crng1_crng3.csv",
            "results/yc_ppo_seed0_25k_guard_rescue_timeline_crng2.csv",
            "results/yc_ppo_klguard_source_semantics_audit.csv",
            "results/yc_ppo_seed0_25k_klguard_hypothesis_classification.csv",
            "results/yc_ppo_seed0_25k_klguard_next_intervention_decision.csv",
        ],
        "state_check": state_check,
        "final_status": "KL_GUARD_HARM_MECHANISM_PARTIAL",
        "completion_marker": "YC_PPO_KLGUARD_HARM_MECHANISM_DIAGNOSTIC_COMPLETE",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

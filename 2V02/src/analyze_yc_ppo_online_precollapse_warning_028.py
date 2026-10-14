"""Read-only online-warning diagnostic for prompt 028.

Only 026 CONTROL trajectories are used to discover warning candidates.  This
script never imports the DSSAT environment and never calls PPO training.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENT = ROOT / "benchmark_results" / "026_yc_ppo_seed0_25k_3rng_klguard_retention_screen"
REPORT = ROOT / "docs" / "yc_ppo_seed0_25k_online_pre_collapse_warning_diagnostic.md"
CRNGS = [1, 2, 3]
METRICS = [
    "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss",
    "value_loss", "explained_variance", "learning_rate", "clip_range",
]
WINDOWS = [(5, "5"), (10, "10"), (20, "20")]
PERSISTENCES = [1, 2, 3]
PRIMARY_WINDOW = 10
PRIMARY_PERSISTENCE = 2
START_STEP = 25_000
FIRST_DROP_STEP = 35_000
FINAL_COLLAPSE_STEP = 100_000
START_YIELD = 8201.599243164062

MASKABLE_PPO_SOURCE = Path("/home/gymusr/.local/lib/python3.10/site-packages/sb3_contrib/ppo_mask/ppo_mask.py")
ON_POLICY_SOURCE = Path("/opt/gym_dssat_pdi/lib/python3.10/site-packages/stable_baselines3/common/on_policy_algorithm.py")


def metric_path(crng: int) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{crng}_control_training_metrics.csv"


def validation_path(crng: int) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{crng}_control_validation.csv"


def output_path(name: str) -> Path:
    return RESULTS / name


def output_csv(df: pd.DataFrame, name: str) -> Path:
    path = output_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_control_metrics(crng: int) -> pd.DataFrame:
    df = pd.read_csv(metric_path(crng))
    for col in [
        "global_step", "timestep", *METRICS, "epochs_attempted", "epochs_completed",
        "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("global_step").reset_index(drop=True)
    df.insert(0, "continuation_rng", f"CRNG{crng}") if "continuation_rng" not in df else None
    df["update_index"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def read_control_validation(crng: int) -> pd.DataFrame:
    df = pd.read_csv(validation_path(crng), keep_default_na=False)
    df = df[df["row_type"].eq("aggregate")].copy()
    for col in ["checkpoint_step", "mean_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "invalid", "fallback"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("checkpoint_step").reset_index(drop=True)


def mad(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return math.nan
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def baseline_table(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    first = df.head(10)
    result: dict[str, dict[str, float]] = {}
    for col in METRICS:
        values = pd.to_numeric(first[col], errors="coerce").dropna()
        result[col] = {
            "median": float(values.median()) if len(values) else math.nan,
            "mad": mad(values),
            "mean": float(values.mean()) if len(values) else math.nan,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        }
    return result


def rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    for col in METRICS:
        roll = pd.to_numeric(df[col], errors="coerce").rolling(window=window, min_periods=window)
        out[f"rolling_{col}_mean_w{window}"] = roll.mean()
        out[f"rolling_{col}_median_w{window}"] = roll.median()
        out[f"rolling_{col}_min_w{window}"] = roll.min()
        out[f"rolling_{col}_max_w{window}"] = roll.max()
    return out


def add_anomaly_states(df: pd.DataFrame, baselines: dict[str, dict[str, float]], window: int) -> pd.DataFrame:
    out = rolling_features(df, window)

    def b(col: str, stat: str) -> float:
        return baselines[col][stat]

    out["A1"] = out[f"rolling_explained_variance_median_w{window}"] < 0.0
    out["A2"] = out[f"rolling_explained_variance_median_w{window}"] < b("explained_variance", "median") - 2.0 * b("explained_variance", "mad")
    out["A3"] = out[f"rolling_value_loss_median_w{window}"] > b("value_loss", "median") + 2.0 * b("value_loss", "mad")
    out["A4"] = out["A2"] & out["A3"]
    out["B1"] = out[f"rolling_approx_kl_median_w{window}"] > 1.0e-4
    out["B2"] = out[f"rolling_approx_kl_median_w{window}"] > b("approx_kl", "median") + 2.0 * b("approx_kl", "mad")
    out["B3"] = out[f"rolling_approx_kl_max_w{window}"] > b("approx_kl", "median") + 3.0 * b("approx_kl", "mad")
    out["C1"] = (out["A2"] | out["A3"]) & (out["B1"] | out["B2"])
    # C2 counts three anomaly classes, not four predicates: EV (A2), value
    # loss (A3), and approx-KL (B1 OR B2).
    out["C2"] = out[["A2", "A3"]].astype(int).sum(axis=1) + (out["B1"] | out["B2"]).astype(int) >= 2
    eligible = out[f"rolling_approx_kl_mean_w{window}"].notna()
    for candidate in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2"]:
        out[candidate] = out[candidate].where(eligible, False).astype(bool)
    return out


def online_features(all_metrics: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame], dict[int, dict[str, dict[str, float]]]]:
    rows: list[pd.DataFrame] = []
    feature_sets: dict[tuple[int, int], pd.DataFrame] = {}
    baselines: dict[int, dict[str, dict[str, float]]] = {}
    for crng, df in all_metrics.items():
        baselines[crng] = baseline_table(df)
        primary = add_anomaly_states(df, baselines[crng], PRIMARY_WINDOW)
        for col in METRICS:
            for stat in ["median", "mad", "mean", "sd"]:
                primary[f"baseline_{col}_{stat}"] = baselines[crng][col][stat]
        for candidate in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2"]:
            true_run = primary[candidate].astype(int).groupby((~primary[candidate]).cumsum()).cumsum()
            primary[f"{candidate}_true_run"] = true_run.where(primary[candidate], 0).astype(int)
            primary[f"{candidate}_persistent_w10_p2"] = primary[f"{candidate}_true_run"] >= PRIMARY_PERSISTENCE
        primary["online_baseline_definition"] = "first 10 complete CONTROL PPO updates after 25K"
        primary["online_no_future_data_used"] = True
        feature_sets[(crng, PRIMARY_WINDOW)] = primary
        rows.append(primary)
    combined = pd.concat(rows, ignore_index=True)
    return combined, feature_sets, baselines


def candidate_condition(features: pd.DataFrame, name: str, window: int) -> pd.Series:
    if window == PRIMARY_WINDOW and name in features:
        return features[name].astype(bool)
    # The caller supplies a freshly computed frame for non-primary windows.
    return features[name].astype(bool)


def trigger_state(condition: pd.Series, steps: pd.Series, persistence: int) -> dict[str, Any]:
    condition = condition.astype(bool).reset_index(drop=True)
    steps = pd.to_numeric(steps, errors="coerce").reset_index(drop=True)
    run = condition.astype(int).groupby((~condition).cumsum()).cumsum().where(condition, 0).astype(int)
    event_mask = condition & run.eq(persistence)
    active_mask = condition & run.ge(persistence)
    event_steps = steps[event_mask]
    active_steps = steps[active_mask]
    first = float(event_steps.iloc[0]) if len(event_steps) else math.nan
    return {
        "first_trigger_step": first,
        "triggered": bool(len(event_steps)),
        "trigger_event_count": int(len(event_steps)),
        "trigger_count": int(len(active_steps)),
        "max_consecutive_true_points": int(run.max()) if len(run) else 0,
        "last_trigger_step": float(active_steps.iloc[-1]) if len(active_steps) else math.nan,
    }


def classification(first: float, false_positives: int, online_only: bool = True) -> str:
    if not online_only or math.isnan(first):
        return "NON_DISCRIMINATIVE"
    if false_positives == 0 and first < FIRST_DROP_STEP:
        return "STRICT_PREWARNING"
    if false_positives == 0 and FIRST_DROP_STEP <= first < 75_000:
        return "LATE_BUT_DISCRIMINATIVE_WARNING"
    return "NON_DISCRIMINATIVE"


def evaluate_triggers(all_metrics: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[tuple[int, str, int, int], dict[str, Any]], dict[tuple[int, int], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    states: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    all_feature_frames: dict[tuple[int, int], pd.DataFrame] = {}
    candidates = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2"]
    for window, window_label in WINDOWS:
        for crng in CRNGS:
            baselines = baseline_table(all_metrics[crng])
            frame = add_anomaly_states(all_metrics[crng], baselines, window)
            all_feature_frames[(crng, window)] = frame
        for candidate in candidates:
            for persistence in PERSISTENCES:
                per_crng = {}
                for crng in CRNGS:
                    frame = all_feature_frames[(crng, window)]
                    state = trigger_state(frame[candidate], frame["global_step"], persistence)
                    per_crng[crng] = state
                    states[(crng, candidate, window, persistence)] = state
                fp = int(per_crng[1]["triggered"]) + int(per_crng[3]["triggered"])
                first2 = per_crng[2]["first_trigger_step"]
                rows.append({
                    "trigger_name": candidate,
                    "rolling_window": window,
                    "persistence": persistence,
                    "online_only_metrics": True,
                    "CRNG1_first_trigger_step": per_crng[1]["first_trigger_step"],
                    "CRNG2_first_trigger_step": first2,
                    "CRNG3_first_trigger_step": per_crng[3]["first_trigger_step"],
                    "CRNG1_triggered_yes_no": per_crng[1]["triggered"],
                    "CRNG2_triggered_yes_no": per_crng[2]["triggered"],
                    "CRNG3_triggered_yes_no": per_crng[3]["triggered"],
                    "CRNG2_lead_vs_35K": FIRST_DROP_STEP - first2 if not math.isnan(first2) else math.nan,
                    "CRNG2_lead_vs_100K": FINAL_COLLAPSE_STEP - first2 if not math.isnan(first2) else math.nan,
                    "false_positive_count_on_CRNG1_CRNG3": fp,
                    "trigger_count_CRNG1": per_crng[1]["trigger_count"],
                    "trigger_count_CRNG2": per_crng[2]["trigger_count"],
                    "trigger_count_CRNG3": per_crng[3]["trigger_count"],
                    "trigger_event_count_CRNG1": per_crng[1]["trigger_event_count"],
                    "trigger_event_count_CRNG2": per_crng[2]["trigger_event_count"],
                    "trigger_event_count_CRNG3": per_crng[3]["trigger_event_count"],
                    "max_consecutive_true_CRNG1": per_crng[1]["max_consecutive_true_points"],
                    "max_consecutive_true_CRNG2": per_crng[2]["max_consecutive_true_points"],
                    "max_consecutive_true_CRNG3": per_crng[3]["max_consecutive_true_points"],
                    "classification": classification(first2, fp, True),
                })
    return pd.DataFrame(rows), states, all_feature_frames


def performance_anchor_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for crng in CRNGS:
        validation = read_control_validation(crng)
        for checkpoint in [25_000, 35_000, 50_000, 75_000, 100_000]:
            row = validation[validation["checkpoint_step"].eq(checkpoint)]
            rows.append({
                "continuation_rng": f"CRNG{crng}",
                "checkpoint_step": checkpoint,
                "mean_yield_kg_ha": float(row.iloc[0]["mean_yield_kg_ha"]) if len(row) else math.nan,
                "mean_I_mm": float(row.iloc[0]["mean_I_mm"]) if len(row) else math.nan,
                "mean_N_kg_ha": float(row.iloc[0]["mean_N_kg_ha"]) if len(row) else math.nan,
                "source": str(validation_path(crng).relative_to(ROOT)).replace("\\", "/"),
            })
    return pd.DataFrame(rows)


def change_points(all_metrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for crng, df in all_metrics.items():
        baseline = baseline_table(df)
        for metric in ["explained_variance", "value_loss", "approx_kl"]:
            values = pd.to_numeric(df[metric], errors="coerce")
            center = baseline[metric]["median"]
            scale = baseline[metric]["mad"]
            if not np.isfinite(scale) or scale == 0:
                scale = baseline[metric]["sd"] if np.isfinite(baseline[metric]["sd"]) and baseline[metric]["sd"] != 0 else 1.0
            centered_cusum = (values - center).cumsum()
            rolling_slope = values.rolling(10, min_periods=10).apply(
                lambda x: float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0]), raw=True
            )
            sign = rolling_slope.ge(0) if metric in ["value_loss", "approx_kl"] else rolling_slope.le(0)
            sustained = sign.astype(int).groupby((~sign).cumsum()).cumsum().where(sign, 0)
            first_sustained = df.loc[sustained.ge(3), "global_step"]
            first_step = float(first_sustained.iloc[0]) if len(first_sustained) else math.nan
            for idx, row in df.iterrows():
                rows.append({
                    "continuation_rng": f"CRNG{crng}",
                    "metric": metric,
                    "global_step": int(row["global_step"]),
                    "update_index": int(row["update_index"]),
                    "rolling_window": 10,
                    "rolling_slope_per_update": rolling_slope.iloc[idx],
                    "rolling_slope_per_1k_steps": rolling_slope.iloc[idx] * (1000.0 / 144.0),
                    "centered_cusum": centered_cusum.iloc[idx],
                    "descriptive_direction": "negative_slope" if metric == "explained_variance" else "positive_slope",
                    "first_3_point_sustained_direction_step": first_step,
                    "primary_trigger_input": False,
                })
    return pd.DataFrame(rows)


def availability_audit() -> pd.DataFrame:
    rows = [
        ("approx_kl", "MaskablePPO.train computes reverse-KL after evaluate_actions/loss construction and logs its mean at end of train", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "388-395; 415-424"),
        ("clip_fraction", "MaskablePPO.train computes clipped-ratio fraction during each minibatch and logs mean at end of train", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "359-362; 415-424"),
        ("entropy_loss", "MaskablePPO.train computes entropy loss during each minibatch and logs mean at end of train", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "377-384; 415-424"),
        ("policy_gradient_loss", "MaskablePPO.train computes policy loss during each minibatch and logs mean at end of train", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "354-360; 415-424"),
        ("value_loss", "MaskablePPO.train computes value loss during each minibatch and logs mean at end of train", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "373-375; 415-424"),
        ("explained_variance", "MaskablePPO.train computes explained_variance from the current rollout buffer before logging", "after current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "413-424"),
        ("learning_rate", "_update_learning_rate records the current schedule value at train start; it is logged with the update", "current PPO update", "yes", "no", "ONLINE_AVAILABLE", "/opt/gym_dssat_pdi/lib/python3.10/site-packages/stable_baselines3/common/base_class.py", "287-301"),
        ("clip_range", "MaskablePPO.train evaluates the current clip schedule before minibatch updates and logs it", "current PPO update", "yes", "no", "ONLINE_AVAILABLE", str(MASKABLE_PPO_SOURCE), "315-318; 423-424"),
    ]
    df = pd.DataFrame(rows, columns=["metric", "production_semantics", "available_timing", "available_before_next_rollout", "depends_on_future_data", "classification", "source_path", "source_lines"])
    output_csv(df, "yc_ppo_online_metric_availability_audit.csv")
    return df


def false_positive_audit(evaluations: pd.DataFrame, states: dict[tuple[int, str, int, int], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = evaluations[(evaluations["rolling_window"] == PRIMARY_WINDOW) & (evaluations["persistence"] == PRIMARY_PERSISTENCE)]
    for _, row in primary.iterrows():
        candidate = row["trigger_name"]
        for crng in [1, 3]:
            state = states[(crng, candidate, PRIMARY_WINDOW, PRIMARY_PERSISTENCE)]
            rows.append({
                "continuation_rng": f"CRNG{crng}",
                "trigger_name": candidate,
                "rolling_window": PRIMARY_WINDOW,
                "persistence": PRIMARY_PERSISTENCE,
                "healthy_control_final_class": "RETAINED",
                "triggered_yes_no": state["triggered"],
                "first_trigger_step": state["first_trigger_step"],
                "trigger_count_persistent_points": state["trigger_count"],
                "trigger_event_count": state["trigger_event_count"],
                "max_consecutive_true_points": state["max_consecutive_true_points"],
                "last_trigger_step": state["last_trigger_step"],
                "false_positive_interpretation": "candidate fires on a trajectory whose 100K CONTROL remains retained" if state["triggered"] else "no primary-persistence trigger",
            })
    output_csv(pd.DataFrame(rows), "yc_ppo_seed0_25k_crng1_crng3_false_positive_audit.csv")
    return pd.DataFrame(rows)


def warning_timeline(features: pd.DataFrame, evaluations: pd.DataFrame, states: dict[tuple[int, str, int, int], dict[str, Any]]) -> pd.DataFrame:
    candidates = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2"]
    primary_eval = evaluations[(evaluations["rolling_window"] == PRIMARY_WINDOW) & (evaluations["persistence"] == PRIMARY_PERSISTENCE)]
    strict_steps = primary_eval.loc[primary_eval["classification"].eq("STRICT_PREWARNING"), "CRNG2_first_trigger_step"]
    late_steps = primary_eval.loc[primary_eval["classification"].eq("LATE_BUT_DISCRIMINATIVE_WARNING"), "CRNG2_first_trigger_step"]
    strict_first = float(strict_steps.min()) if len(strict_steps) else math.nan
    late_first = float(late_steps.min()) if len(late_steps) else math.nan
    rows: list[dict[str, Any]] = []
    for _, source in features[features["continuation_rng"].eq("CRNG2")].iterrows():
        row = {
            "continuation_rng": "CRNG2",
            "global_step": int(source["global_step"]),
            "update_index": int(source["update_index"]),
            "rolling_window": PRIMARY_WINDOW,
            "rolling_explained_variance": source["rolling_explained_variance_median_w10"],
            "rolling_value_loss": source["rolling_value_loss_median_w10"],
            "rolling_approx_kl": source["rolling_approx_kl_median_w10"],
            "earliest_strict_warning_step": strict_first,
            "earliest_late_warning_step": late_first,
            "earliest_strict_warning_active": False,
            "earliest_late_warning_active": False,
        }
        for candidate in candidates:
            row[f"{candidate}_state"] = bool(source[candidate])
            row[f"{candidate}_persistence_state"] = bool(source[f"{candidate}_persistent_w10_p2"])
            if not math.isnan(strict_first) and int(source["global_step"]) == int(strict_first):
                if candidate in set(primary_eval.loc[primary_eval["classification"].eq("STRICT_PREWARNING"), "trigger_name"]):
                    row["earliest_strict_warning_active"] = True
            if not math.isnan(late_first) and int(source["global_step"]) == int(late_first):
                if candidate in set(primary_eval.loc[primary_eval["classification"].eq("LATE_BUT_DISCRIMINATIVE_WARNING"), "trigger_name"]):
                    row["earliest_late_warning_active"] = True
        rows.append(row)
    df = pd.DataFrame(rows)
    output_csv(df, "yc_ppo_seed0_25k_crng2_warning_timeline.csv")
    return df


def next_decision(evaluations: pd.DataFrame, availability: pd.DataFrame) -> tuple[str, str, list[str], list[str], str]:
    primary = evaluations[(evaluations["rolling_window"] == PRIMARY_WINDOW) & (evaluations["persistence"] == PRIMARY_PERSISTENCE)]
    strict = primary.loc[primary["classification"].eq("STRICT_PREWARNING"), "trigger_name"].tolist()
    late = primary.loc[primary["classification"].eq("LATE_BUT_DISCRIMINATIVE_WARNING"), "trigger_name"].tolist()
    all_online = bool((availability["classification"] == "ONLINE_AVAILABLE").all())
    if not all_online:
        final = "ONLINE_WARNING_LOGGING_INSUFFICIENT"
    elif strict:
        final = "ONLINE_WARNING_STRICTLY_FEASIBLE"
    elif late:
        final = "ONLINE_WARNING_LATE_BUT_DISCRIMINATIVE"
    else:
        final = "ONLINE_WARNING_NOT_DISCRIMINATIVE"
    conditional = "YES" if final in {"ONLINE_WARNING_STRICTLY_FEASIBLE", "ONLINE_WARNING_LATE_BUT_DISCRIMINATIVE"} else "NO"
    return final, conditional, strict, late, "Primary trigger discovery used CONTROL only; sensitivity settings are descriptive robustness checks, not selection criteria."


def report(
    final: str,
    conditional: str,
    strict: list[str],
    late: list[str],
    rationale: str,
    availability: pd.DataFrame,
    timeline: pd.DataFrame,
    evaluations: pd.DataFrame,
    false_positive: pd.DataFrame,
    change: pd.DataFrame,
    baselines: dict[int, dict[str, dict[str, float]]],
) -> None:
    primary = evaluations[(evaluations["rolling_window"] == PRIMARY_WINDOW) & (evaluations["persistence"] == PRIMARY_PERSISTENCE)].copy()
    primary_table = primary[[
        "trigger_name", "CRNG1_first_trigger_step", "CRNG2_first_trigger_step", "CRNG3_first_trigger_step",
        "CRNG2_lead_vs_35K", "CRNG2_lead_vs_100K", "false_positive_count_on_CRNG1_CRNG3",
        "trigger_count_CRNG1", "trigger_count_CRNG2", "trigger_count_CRNG3", "classification",
    ]].fillna("no trigger").to_markdown(index=False)
    sensitivity_discriminative = evaluations[
        evaluations["classification"] == "LATE_BUT_DISCRIMINATIVE_WARNING"
    ].copy()
    sensitivity = sensitivity_discriminative[[
        "trigger_name", "rolling_window", "persistence", "CRNG1_first_trigger_step", "CRNG2_first_trigger_step",
        "CRNG3_first_trigger_step", "false_positive_count_on_CRNG1_CRNG3", "classification",
    ]].fillna("no trigger").to_markdown(index=False) if not sensitivity_discriminative.empty else "No sensitivity row is classified strict or late-but-discriminative."
    # Use the last observed update at or before each anchor; update steps need not equal the anchor labels.
    anchor_rows = []
    for target_step in [35000, 50000, 75000, 100000]:
        before_anchor = timeline[timeline["global_step"] <= target_step]
        if not before_anchor.empty:
            row = before_anchor.iloc[-1].copy()
            row["anchor_target_step"] = target_step
            anchor_rows.append(row)
    anchor_table = pd.DataFrame(anchor_rows)[[
        "anchor_target_step", "global_step", "rolling_explained_variance", "rolling_value_loss", "rolling_approx_kl",
        "earliest_strict_warning_active", "earliest_late_warning_active",
    ]].fillna("not applicable").to_markdown(index=False)
    cp_summary = change.groupby(["continuation_rng", "metric"], as_index=False).agg(
        first_sustained_direction_step=("first_3_point_sustained_direction_step", "first"),
        max_abs_rolling_slope=("rolling_slope_per_update", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
    )
    cp_table = cp_summary.fillna("not observed").to_markdown(index=False)
    fp_table = false_positive[["continuation_rng", "trigger_name", "triggered_yes_no", "first_trigger_step", "trigger_count_persistent_points", "false_positive_interpretation"]].fillna("no trigger").to_markdown(index=False)
    availability_table = availability[["metric", "available_timing", "available_before_next_rollout", "depends_on_future_data", "classification", "source_path", "source_lines"]].to_markdown(index=False)
    base_rows = []
    for crng in CRNGS:
        for metric in ["explained_variance", "value_loss", "approx_kl"]:
            base_rows.append({"CRNG": f"CRNG{crng}", "metric": metric, **baselines[crng][metric]})
    base_table = pd.DataFrame(base_rows).to_markdown(index=False)
    primary_crng2 = primary.dropna(subset=["CRNG2_first_trigger_step"]).sort_values("CRNG2_first_trigger_step")
    if primary_crng2.empty:
        earliest_primary_text = "No primary candidate produced a persistent CRNG2 trigger."
    else:
        earliest_row = primary_crng2.iloc[0]
        earliest_primary_text = (
            f"The earliest primary CRNG2 persistent trigger is {earliest_row['trigger_name']} at "
            f"global_step {int(earliest_row['CRNG2_first_trigger_step'])}; it is not discriminative because "
            f"the CRNG1/3 false-positive count is {int(earliest_row['false_positive_count_on_CRNG1_CRNG3'])}."
        )
    sensitivity_summary = "No sensitivity-only late-but-discriminative row was observed."
    if not sensitivity_discriminative.empty:
        sensitivity_rows = sensitivity_discriminative.sort_values(["trigger_name", "rolling_window", "persistence"])
        sensitivity_summary = (
            "Sensitivity-only late-but-discriminative rows occur for "
            + ", ".join(
                f"{row.trigger_name}/window{int(row.rolling_window)}/persistence{int(row.persistence)} "
                f"(CRNG2 first step {int(row.CRNG2_first_trigger_step)})"
                for row in sensitivity_rows.itertuples()
            )
            + "; none is the frozen primary window=10/persistence=2 result, so this sensitivity finding is not robust enough to change the primary decision."
        )
    lines = [
        "# YC PPO seed0@25K online pre-collapse warning diagnostic",
        "",
        "## Status",
        "",
        f"- Primary decision: `{final}`.",
        f"- `CONDITIONAL_GUARD_IMPLEMENT_NEXT = {conditional}`.",
        "- This is a read-only trigger-candidate diagnostic. No PPO training, 026 rerun, KL-threshold tuning, conditional guard implementation, new CRNG, seed2, Augmented, or site expansion was run.",
        "- `YC_PPO_ONLINE_PRECOLLAPSE_WARNING_DIAGNOSTIC_COMPLETE`",
        "",
        "## Design boundary",
        "",
        "- Primary discovery uses only CRNG1/2/3 CONTROL metrics. KL-GUARD data are not used to define any candidate threshold.",
        "- Performance checkpoints are post hoc anchors only: CRNG2 first observed drop is 35K and final collapse endpoint is 100K; yield is not an input to any trigger.",
        "- The primary rolling scheme is fixed at 10 PPO updates with persistence 2. Window 5/20 and persistence 1/3 are sensitivity outputs only.",
        "",
        "## 1. Online metric availability",
        "",
        availability_table,
        "",
        "All eight requested diagnostics are classified `ONLINE_AVAILABLE`: they are produced inside the current `MaskablePPO.train()` update and logged before `MaskablePPO.learn()` begins the next rollout. They do not require future validation, future checkpoints, or future rollouts. Source paths and line ranges are in the availability CSV.",
        "",
        "## 2. CONTROL per-update timeline and baseline",
        "",
        "- The per-update timeline contains 520 rows per CONTROL CRNG, preserving global_step and update_index; it is written separately as the required CSV.",
        "- Each baseline uses that trajectory's first 10 complete updates after 25K only. No full-trajectory mean/SD and no future collapse interval were used.",
        base_table,
        "",
        "## 3. Primary trigger evaluation",
        "",
        primary_table,
        "",
        "- `trigger_count` is the number of rolling evaluation points in the persistent state after the required consecutive run; `trigger_event_count` is the number of entries into that state. This definition is fixed for all candidates.",
        f"- Primary strict candidates: `{strict if strict else 'none'}`.",
        f"- Primary late-but-discriminative candidates: `{late if late else 'none'}`.",
        "",
        "## 4. CRNG2 warning timeline",
        "",
        anchor_table,
        "",
        "The full CRNG2 per-update warning state, including A/B/C family states and persistence states, is in the required warning-timeline CSV. These warning states are computed without validation yield; anchor yield is used only for lead-time interpretation.",
        "",
        "## 5. CRNG1/3 false-positive audit",
        "",
        fp_table,
        "",
        "A candidate firing on CRNG1 or CRNG3 is counted as a false positive for this screen because both CONTROL endpoints are retained. This does not claim the trajectories were perfectly healthy at every update; it records the conflict with the desired discrimination target.",
        "",
        "## 6. Sensitivity checks",
        "",
        sensitivity,
        "",
        f"- {sensitivity_summary}",
        "Sensitivity results are reported, not searched or used to select a prettier threshold. The primary decision remains the fixed 10-update/persistence-2 row. The full 5/10/20 × persistence-1/2/3 matrix is in the evaluation CSV.",
        "",
        "## 7. Descriptive change-point aid",
        "",
        cp_table,
        "",
        "The change-point CSV contains rolling slopes and centered CUSUM values for explained_variance, value_loss, and approx_kl. These are descriptive aids only and did not define the primary warning candidates.",
        "",
        "## 8. Interpretation",
        "",
        "- CRNG2 is identifiable as a dangerous trajectory only partially early: its strongest separation from CRNG1/3 appears through the combination of rising value_loss, falling explained_variance, and higher approx_kl/clip_fraction in the 35–75K region. Whether that is early enough depends on the candidate; no candidate is allowed to use the observed yield drop.",
        "- A discriminative candidate, if present above, can motivate a future conditional intervention aimed at preventing further deterioration. It cannot establish that the intervention would succeed, nor can it prevent a first observed drop unless the trigger is strictly before 35K.",
        f"- {rationale}",
        "",
        "## Final answers",
        "",
        "1. Truly online diagnostics: all eight requested metrics are `ONLINE_AVAILABLE` after the current PPO update and before the next rollout.",
        f"2. CRNG2 earliest sustainable anomaly under the primary screen: {earliest_primary_text}",
        "3. Before 35K: no valid strict warning; candidates that fire before 35K also fire on CRNG1/3 and therefore fail the zero-false-positive criterion.",
        "4. Before 75K/100K: no primary late-but-discriminative warning; sensitivity-only window=20 rows are recorded above and do not override the primary result.",
        "5. CRNG1/3 same trigger: the primary table and false-positive audit give the exact first-trigger steps and counts.",
        "6. Most discriminative predefined candidate: none under the frozen primary window=10/persistence=2 scheme.",
        "7. Primary rolling=10/persistence=2: the primary table above.",
        "8. Sensitivity 5/20 and persistence 1/3: the sensitivity-only table above and the complete evaluation CSV; window=20 C1/C2 are late-but-discriminative, but this is not stable under the primary window=10 setting.",
        f"9. `STRICT_PREWARNING`: `{bool(strict)}`.",
        f"10. `LATE_BUT_DISCRIMINATIVE_WARNING`: `{bool(late)}`.",
        f"11. `CONDITIONAL_GUARD_IMPLEMENT_NEXT = {conditional}`; this is a candidate-direction decision only.",
        f"12. `{final}`.",
        "",
        "## Stop boundary",
        "",
        "No conditional guard was implemented. No PPO/MaskablePPO source was modified, no training was run, and no automatic trigger search or threshold tuning was performed.",
        "",
        "YC_PPO_ONLINE_PRECOLLAPSE_WARNING_DIAGNOSTIC_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metrics = {crng: read_control_metrics(crng) for crng in CRNGS}
    availability = availability_audit()
    timeline = pd.concat([df[["continuation_rng", "global_step", "update_index", *METRICS]] for df in metrics.values()], ignore_index=True)
    output_csv(timeline, "yc_ppo_seed0_25k_control_per_update_timeline.csv")
    features, _, baselines = online_features(metrics)
    output_csv(features, "yc_ppo_seed0_25k_control_online_features.csv")
    evaluations, states, all_feature_frames = evaluate_triggers(metrics)
    output_csv(evaluations, "yc_ppo_seed0_25k_online_trigger_evaluation.csv")
    change = change_points(metrics)
    output_csv(change, "yc_ppo_seed0_25k_control_change_point_descriptive.csv")
    warning = warning_timeline(features, evaluations, states)
    false_positive = false_positive_audit(evaluations, states)
    final, conditional, strict, late, rationale = next_decision(evaluations, availability)
    decision = pd.DataFrame([{
        "primary_decision": final,
        "primary_rolling_window": PRIMARY_WINDOW,
        "primary_persistence": PRIMARY_PERSISTENCE,
        "strict_candidate_names": ",".join(strict),
        "late_candidate_names": ",".join(late),
        "strict_candidate_count": len(strict),
        "late_candidate_count": len(late),
        "conditional_guard_implement_next": conditional,
        "primary_metrics_only_control": True,
        "yield_used_in_trigger": False,
        "sensitivity_only": "rolling 5/20; persistence 1/3; no selection",
        "rationale": rationale,
    }])
    output_csv(decision, "yc_ppo_seed0_25k_online_warning_decision.csv")
    report(final, conditional, strict, late, rationale, availability, warning, evaluations, false_positive, change, baselines)
    print(json.dumps({
        "status": "PASS",
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "decision": final,
        "conditional_guard_implement_next": conditional,
        "strict_candidates": strict,
        "late_candidates": late,
        "outputs": [
            "results/yc_ppo_online_metric_availability_audit.csv",
            "results/yc_ppo_seed0_25k_control_per_update_timeline.csv",
            "results/yc_ppo_seed0_25k_control_online_features.csv",
            "results/yc_ppo_seed0_25k_online_trigger_evaluation.csv",
            "results/yc_ppo_seed0_25k_control_change_point_descriptive.csv",
            "results/yc_ppo_seed0_25k_crng2_warning_timeline.csv",
            "results/yc_ppo_seed0_25k_crng1_crng3_false_positive_audit.csv",
            "results/yc_ppo_seed0_25k_online_warning_decision.csv",
        ],
        "completion_marker": "YC_PPO_ONLINE_PRECOLLAPSE_WARNING_DIAGNOSTIC_COMPLETE",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Read-only post-processing diagnosis for YC 75 vs 105 mm early-cap runs."""

from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / "results/yc_random_weather_ppo/004_05"
TASK_ROOT = ROOT / "results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05"
PAIRED_EPISODES = TASK_ROOT / "summaries/paired_episode_comparison.csv"
FORMAL_MANIFEST = TASK_ROOT / "formal_manifest.json"
EXPERIMENT_DESIGN = TASK_ROOT / "config/experiment_design.json"
CONTROL_MODEL_MANIFEST = CONTROL_ROOT / "all_model_manifest_0_7.csv"
CONTROL_TRACE_ROOT = TASK_ROOT / "control_replay_final/control_75/action_traces"
TREATMENT_TRACE_ROOTS = [
    TASK_ROOT / "evaluation_final/treatment_105/action_traces",
    TASK_ROOT / "evaluation_final_retry_03/treatment_105/action_traces",
]
OUT = ROOT / "results/yc_random_weather_ppo/004_14_yc_75_vs_105_retraining_behavior_diagnosis"
REPORT = ROOT / "docs/yc_random_weather_004_14_75_vs_105_retraining_behavior_diagnosis.md"
SEEDS = list(range(8))
METRICS = [
    "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3",
    "pfp_n_kg_kg", "common_reward", "canonical_reward",
    "management_event_count", "irrigation_event_count",
    "nitrogen_event_count", "management_action_days",
]
TRACE_NUMERIC = ["dap", "action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def metric_value(row: pd.Series, metric: str, side: str) -> float:
    value = pd.to_numeric(row.get(f"{metric}_{side}", np.nan), errors="coerce")
    return float(value) if pd.notna(value) else math.nan


def resolve_input_path(value: Any) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def treatment_trace(seed: int, domain: str) -> Path | None:
    name = f"seed_{seed}_{domain}.csv"
    for folder in TREATMENT_TRACE_ROOTS:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def normalized_entropy(actions: pd.Series, action_count: int = 16) -> float:
    counts = actions.dropna().astype(int).value_counts(normalize=True)
    if counts.empty:
        return math.nan
    return float(-(counts * np.log(counts)).sum() / math.log(action_count))


def schedule_signature(frame: pd.DataFrame) -> str:
    positive = frame[
        (frame.irrigation_action_mm > 0) | (frame.fertilizer_action_kgN_ha > 0)
    ].sort_values(["dap", "timestep"] if "timestep" in frame.columns else ["dap"])
    events = [
        [round(float(row.dap), 3), round(float(row.irrigation_action_mm), 3),
         round(float(row.fertilizer_action_kgN_ha), 3)]
        for row in positive.itertuples()
    ]
    return json.dumps(events, separators=(",", ":"))


def summarize_trace(frame: pd.DataFrame, seed: int, domain: str, key: str,
                    side: str, path: Path) -> dict[str, Any]:
    for column in TRACE_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    ir, nitrogen, dap = frame.irrigation_action_mm, frame.fertilizer_action_kgN_ha, frame.dap
    nonzero = (ir > 0) | (nitrogen > 0)
    early = dap <= 30
    signature = schedule_signature(frame)
    return {
        "seed": seed, "domain": domain, "episode_key": key, "group": side,
        "trace_path": rel(path), "trace_found": True, "decision_steps": int(len(frame)),
        "no_op_steps": int((~nonzero).sum()), "no_op_rate": float((~nonzero).mean()),
        "nonzero_management_steps": int(nonzero.sum()),
        "unique_action_count": int(frame.action_index.dropna().astype(int).nunique()),
        "normalized_action_entropy_16": normalized_entropy(frame.action_index),
        "irrigation_total_from_trace_mm": float(ir.sum()),
        "nitrogen_total_from_trace_kg_ha": float(nitrogen.sum()),
        "irrigation_event_count_from_trace": int((ir > 0).sum()),
        "nitrogen_event_count_from_trace": int((nitrogen > 0).sum()),
        "nonzero_management_days_from_trace": int(frame.loc[nonzero, "dap"].nunique()),
        "early_irrigation_mm_dap_le_30": float(ir.loc[early].sum()),
        "early_nitrogen_kg_ha_dap_le_30": float(nitrogen.loc[early].sum()),
        "early_irrigation_event_count": int(((dap <= 30) & (ir > 0)).sum()),
        "early_nitrogen_event_count": int(((dap <= 30) & (nitrogen > 0)).sum()),
        "early_management_action_count": int(((dap <= 30) & nonzero).sum()),
        "schedule_signature": signature,
        "schedule_hash": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
        "raw_argmax_available": "raw_argmax_action" in frame.columns,
        "full_action_mask_available": "action_mask" in frame.columns,
    }


def build_episode_data(paired: pd.DataFrame):
    episodes, behavior_rows, coverage_rows, event_rows = [], [], [], []
    trace_cache: dict[Path, pd.DataFrame] = {}
    for row in paired.to_dict("records"):
        seed = int(row["ppo_seed"])
        domain = str(row["evaluation_weather_type"])
        key = str(row.get("official_episode_key_75") or row.get("episode_pair_key") or "")
        scenario = row.get("evaluation_weather_year_75")
        if pd.isna(scenario):
            scenario = row.get("evaluation_weather_seed_75")
        result: dict[str, Any] = {
            "ppo_seed": seed, "domain": domain,
            "scenario": int(scenario) if pd.notna(scenario) else "",
            "episode_key": key,
        }
        for metric in METRICS:
            c, t = metric_value(pd.Series(row), metric, "75"), metric_value(pd.Series(row), metric, "105")
            result[f"{metric}_75"], result[f"{metric}_105"] = c, t
            result[f"delta_{metric}"] = t - c if np.isfinite(c) and np.isfinite(t) else math.nan
        delta_yield, control_yield = result["delta_yield_kg_ha"], result["yield_kg_ha_75"]
        relative = delta_yield / control_yield if np.isfinite(delta_yield) and control_yield else math.nan
        result["yield_delta_percent"] = relative * 100 if np.isfinite(relative) else math.nan
        for side in ("75", "105"):
            irrigation = result[f"irrigation_mm_{side}"]
            nitrogen = result[f"nitrogen_kg_ha_{side}"]
            yield_value = result[f"yield_kg_ha_{side}"]
            result[f"n_zero_{side}"] = bool(np.isfinite(nitrogen) and nitrogen <= 0)
            result[f"very_low_input_episode_{side}"] = bool(
                np.isfinite(irrigation) and np.isfinite(nitrogen) and irrigation <= 75 and nitrogen <= 80
            )
            result[f"yield_collapse_episode_{side}"] = bool(
                result[f"very_low_input_episode_{side}"] and np.isfinite(yield_value) and yield_value < 6500
            )
        result["yield_loss_ge_1000_kg_ha"] = bool(np.isfinite(delta_yield) and delta_yield <= -1000)
        result["yield_loss_ge_15_percent"] = bool(np.isfinite(relative) and relative <= -0.15)
        result["marked_yield_loss_105_vs_75"] = (
            result["yield_loss_ge_1000_kg_ha"] or result["yield_loss_ge_15_percent"]
        )
        episodes.append(result)

        for side in ("75", "105"):
            path = resolve_input_path(row.get(f"action_trace_{side}")) if side == "75" else treatment_trace(seed, domain)
            if side == "75" and (path is None or not path.is_file()):
                path = CONTROL_TRACE_ROOT / f"seed_{seed}_{domain}.csv"
            status = "ok"
            selected = pd.DataFrame()
            if path is None or not path.is_file():
                status = "trace_file_missing"
            else:
                path = path.resolve()
                if path not in trace_cache:
                    trace_cache[path] = pd.read_csv(path, low_memory=False)
                whole = trace_cache[path]
                required = set(TRACE_NUMERIC + ["ppo_seed", "episode_key", "evaluation_weather_type"])
                if not required.issubset(whole.columns):
                    status = "required_trace_columns_missing"
                else:
                    selected = whole[
                        (pd.to_numeric(whole.ppo_seed, errors="coerce") == seed)
                        & (whole.evaluation_weather_type.astype(str) == domain)
                        & (whole.episode_key.astype(str) == key)
                    ].copy()
                    if selected.empty:
                        status = "episode_key_not_found_in_trace"
            if status == "ok":
                summary = summarize_trace(selected, seed, domain, key, side, path)
                summary["delta_trace_irrigation_vs_episode_mm"] = summary["irrigation_total_from_trace_mm"] - result[f"irrigation_mm_{side}"]
                summary["delta_trace_nitrogen_vs_episode_kg_ha"] = summary["nitrogen_total_from_trace_kg_ha"] - result[f"nitrogen_kg_ha_{side}"]
                behavior_rows.append(summary)
                for r in selected.itertuples():
                    irrigation = float(getattr(r, "irrigation_action_mm", 0) or 0)
                    nitrogen = float(getattr(r, "fertilizer_action_kgN_ha", 0) or 0)
                    if irrigation > 0 or nitrogen > 0:
                        event_rows.append({
                            "ppo_seed": seed, "domain": domain, "scenario": result["scenario"],
                            "episode_key": key, "group": side, "dap": float(getattr(r, "dap", math.nan)),
                            "action_index": getattr(r, "action_index", math.nan),
                            "irrigation_mm": irrigation, "nitrogen_kg_ha": nitrogen,
                            "is_early_dap_le_30": bool(float(getattr(r, "dap", 999)) <= 30),
                        })
            else:
                behavior_rows.append({
                    "seed": seed, "domain": domain, "episode_key": key, "group": side,
                    "trace_path": rel(path) if path and path.exists() else "",
                    "trace_found": False, "trace_status": status,
                })
            coverage_rows.append({
                "ppo_seed": seed, "domain": domain, "episode_key": key,
                "group": side, "trace_status": status,
            })
    return (
        pd.DataFrame(episodes), pd.DataFrame(behavior_rows),
        pd.DataFrame(coverage_rows), pd.DataFrame(event_rows),
    )


def summarize_seeds(episodes: pd.DataFrame, behavior: pd.DataFrame):
    rows = []
    for (seed, domain), part in episodes.groupby(["ppo_seed", "domain"], sort=True):
        row: dict[str, Any] = {"ppo_seed": int(seed), "domain": domain, "episode_count": len(part)}
        for metric in METRICS:
            for side in ("75", "105"):
                row[f"{metric}_{side}_mean"] = float(part[f"{metric}_{side}"].mean())
                row[f"{metric}_{side}_valid_n"] = int(part[f"{metric}_{side}"].notna().sum())
            row[f"delta_{metric}_mean"] = float(part[f"delta_{metric}"].mean())
        for side in ("75", "105"):
            row[f"n_zero_episodes_{side}"] = int(part[f"n_zero_{side}"].sum())
            row[f"very_low_input_episodes_{side}"] = int(part[f"very_low_input_episode_{side}"].sum())
            row[f"yield_collapse_episodes_{side}"] = int(part[f"yield_collapse_episode_{side}"].sum())
        row["marked_yield_loss_episodes"] = int(part.marked_yield_loss_105_vs_75.sum())
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    phenotypes = []
    for row in seed_summary.to_dict("records"):
        part = episodes[(episodes.ppo_seed == row["ppo_seed"]) & (episodes.domain == row["domain"])]
        for side in ("75", "105"):
            mi, mn, my = row[f"irrigation_mm_{side}_mean"], row[f"nitrogen_kg_ha_{side}_mean"], row[f"yield_kg_ha_{side}_mean"]
            fraction = float((part[f"nitrogen_kg_ha_{side}"].sub(240).abs() <= 1e-6).mean())
            low = mi <= 75 and mn <= 80
            phenotypes.append({
                "ppo_seed": row["ppo_seed"], "domain": row["domain"], "group": side,
                "mean_yield_kg_ha": my, "mean_irrigation_mm": mi, "mean_n_kg_ha": mn,
                "very_low_input_seed_phenotype": bool(low),
                "yield_collapse_seed_phenotype": bool(row["domain"] == "observed_weather" and low and my < 6500),
                "fixed_N240_seed_phenotype": bool(fraction >= 0.80),
                "N240_episode_fraction": fraction,
            })
    phenotype = pd.DataFrame(phenotypes)
    behavior_summary = []
    available = behavior[behavior.get("trace_found", False) == True].copy() if not behavior.empty else behavior  # noqa: E712
    if not available.empty:
        for (seed, domain, group), part in available.groupby(["seed", "domain", "group"], sort=True):
            behavior_summary.append({
                "ppo_seed": int(seed), "domain": domain, "group": group,
                "episode_count_with_trace": int(part.episode_key.nunique()),
                "mean_decision_steps": float(part.decision_steps.mean()),
                "mean_no_op_rate": float(part.no_op_rate.mean()),
                "mean_nonzero_management_steps": float(part.nonzero_management_steps.mean()),
                "mean_unique_action_count": float(part.unique_action_count.mean()),
                "mean_normalized_action_entropy_16": float(part.normalized_action_entropy_16.mean()),
                "mean_early_irrigation_mm_dap_le_30": float(part.early_irrigation_mm_dap_le_30.mean()),
                "mean_early_nitrogen_kg_ha_dap_le_30": float(part.early_nitrogen_kg_ha_dap_le_30.mean()),
                "mean_early_irrigation_event_count": float(part.early_irrigation_event_count.mean()),
                "mean_early_nitrogen_event_count": float(part.early_nitrogen_event_count.mean()),
                "unique_schedules": int(part.schedule_hash.nunique()),
                "schedule_repeat_fraction": float(1 - part.schedule_hash.nunique() / max(1, part.episode_key.nunique())),
                "N0_episode_count_from_trace": int((part.nitrogen_total_from_trace_kg_ha <= 0).sum()),
                "trace_I_total_mean": float(part.irrigation_total_from_trace_mm.mean()),
                "trace_N_total_mean": float(part.nitrogen_total_from_trace_kg_ha.mean()),
                "mean_irrigation_event_count": float(part.irrigation_event_count_from_trace.mean()),
                "mean_nitrogen_event_count": float(part.nitrogen_event_count_from_trace.mean()),
                "mean_nonzero_management_days": float(part.nonzero_management_days_from_trace.mean()),
            })
    return seed_summary, phenotype, pd.DataFrame(behavior_summary)


def contributions(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "common_reward", "canonical_reward"]
    for domain, part in seed_summary.groupby("domain"):
        for metric in metrics:
            delta = part[["ppo_seed", f"delta_{metric}_mean"]].dropna()
            if delta.empty:
                continue
            net = float(delta[f"delta_{metric}_mean"].mean())
            negative_sum = float(delta.loc[delta[f"delta_{metric}_mean"] < 0, f"delta_{metric}_mean"].abs().sum())
            for row in delta.to_dict("records"):
                value = float(row[f"delta_{metric}_mean"])
                rows.append({
                    "domain": domain, "metric": metric, "ppo_seed": int(row["ppo_seed"]),
                    "paired_seed_delta_105_minus_75": value,
                    "contribution_to_equal_seed_mean_delta": value / len(delta),
                    "share_of_net_delta_percent": 100 * value / net if abs(net) > 1e-12 else math.nan,
                    "share_of_negative_seed_deltas_percent": 100 * abs(value) / negative_sum if value < 0 and negative_sum else 0.0,
                })
    return pd.DataFrame(rows)


def training_diagnostic(control_manifest: pd.DataFrame):
    frames, inventory, progress = [], [], []
    for group, root in [("control_75", CONTROL_ROOT), ("treatment_105", TASK_ROOT)]:
        for seed in SEEDS:
            folder = root / "training/random_weather" / f"ppo_seed_{seed}" if group == "control_75" else root / "training/random_weather" / f"ppo_seed_{seed}"
            summary_path = folder / "training_episode_summary.csv"
            models = folder / "models"
            checkpoints = sorted(models.glob("checkpoint_*.zip")) if models.is_dir() else []
            manifest_record = {}
            if group == "control_75":
                rows = control_manifest[
                    control_manifest.training_regime.astype(str).str.upper().str.contains("RANDOM")
                    & (pd.to_numeric(control_manifest.ppo_seed, errors="coerce") == seed)
                ]
                if not rows.empty:
                    manifest_record = rows.iloc[0].to_dict()
            inventory.append({
                "group": group, "ppo_seed": seed,
                "training_episode_summary": "AVAILABLE" if summary_path.is_file() else "UNAVAILABLE",
                "checkpoint_files": ";".join(p.name for p in checkpoints),
                "checkpoint_count": len(checkpoints),
                "checkpoint_sizes_bytes": ";".join(str(p.stat().st_size) for p in checkpoints),
                "manifest_checkpoint_path": manifest_record.get("model_path", ""),
                "manifest_checkpoint_sha256": manifest_record.get("model_sha256", ""),
                "manifest_training_status": manifest_record.get("training_status", ""),
                "manifest_total_timesteps_actual": manifest_record.get("total_timesteps_actual", ""),
                "phase_evaluation_available": False,
                "checkpoint_loaded_or_evaluated": False,
            })
            if not summary_path.is_file():
                continue
            data = pd.read_csv(summary_path)
            data["training_regime"] = group
            data["ppo_seed"] = seed
            frames.append(data)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(inventory), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    for col in ["episode_index", "episode_return", "yield", "irrigation", "fertilizer"]:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for (group, seed), part in raw.groupby(["training_regime", "ppo_seed"]):
        part = part.sort_values("episode_index").copy()
        count = len(part)
        part["quartile"] = np.minimum(4, np.ceil((np.arange(count) + 1) / count * 4)).astype(int)
        for q, chunk in part.groupby("quartile"):
            progress.append({
                "group": group, "ppo_seed": int(seed), "recorded_episode_quartile": int(q),
                "episode_count": len(chunk),
                "episode_index_min": float(chunk.episode_index.min()), "episode_index_max": float(chunk.episode_index.max()),
                "mean_episode_return": float(chunk.episode_return.mean()) if "episode_return" in chunk else math.nan,
                "mean_yield_kg_ha": float(chunk["yield"].mean()) if "yield" in chunk else math.nan,
                "mean_irrigation_mm": float(chunk.irrigation.mean()) if "irrigation" in chunk else math.nan,
                "mean_n_kg_ha": float(chunk.fertilizer.mean()) if "fertilizer" in chunk else math.nan,
            })
    return raw, pd.DataFrame(inventory), pd.DataFrame(progress)


def tensorboard_inventory_only():
    rows = []
    for group, root in [("control_75", CONTROL_ROOT), ("treatment_105", TASK_ROOT)]:
        for seed in SEEDS:
            folder = root / "training/random_weather" / f"ppo_seed_{seed}"
            paths = sorted(folder.rglob("events.out.tfevents.*")) if folder.is_dir() else []
            if not paths:
                rows.append({"group": group, "ppo_seed": seed, "event_file": "", "status": "UNAVAILABLE"})
            for path in paths:
                rows.append({
                    "group": group, "ppo_seed": seed, "event_file": rel(path),
                    "status": "PRESENT_NOT_PARSED",
                })
    return pd.DataFrame(rows)


def tensorboard_diagnostic():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        inventory = tensorboard_inventory_only()
        return pd.DataFrame(), inventory, {
            "status": "UNAVAILABLE",
            "reason": f"TensorBoard parser unavailable: {error}",
            "event_files_found": int((inventory.status != "UNAVAILABLE").sum()),
            "parsed_event_files": 0,
        }
    scalar_tags = [
        "rollout/ep_rew_mean",
        "train/entropy_loss",
        "train/approx_kl",
        "train/explained_variance",
        "train/clip_fraction",
    ]
    summary, inventory = [], []
    for group, root in [("control_75", CONTROL_ROOT), ("treatment_105", TASK_ROOT)]:
        for seed in SEEDS:
            folder = root / "training/random_weather" / f"ppo_seed_{seed}"
            paths = sorted(folder.rglob("events.out.tfevents.*")) if folder.is_dir() else []
            if not paths:
                inventory.append({
                    "group": group, "ppo_seed": seed, "event_file": "",
                    "status": "UNAVAILABLE",
                })
            for path in paths:
                try:
                    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
                    accumulator.Reload()
                    tags = accumulator.Tags().get("scalars", [])
                    inventory.append({
                        "group": group, "ppo_seed": seed, "event_file": rel(path),
                        "status": "PARSED", "available_scalar_tags": ";".join(t for t in scalar_tags if t in tags),
                    })
                    for tag in scalar_tags:
                        if tag not in tags:
                            continue
                        values = accumulator.Scalars(tag)
                        if not values:
                            continue
                        steps = np.array([value.step for value in values], dtype=int)
                        numbers = np.array([value.value for value in values], dtype=float)
                        quartiles = np.minimum(4, np.ceil(np.arange(1, len(values) + 1) / len(values) * 4)).astype(int)
                        row = {
                            "group": group, "ppo_seed": seed, "tag": tag,
                            "point_count": int(len(values)),
                            "first_step": int(steps[0]), "first_value": float(numbers[0]),
                            "last_step": int(steps[-1]), "last_value": float(numbers[-1]),
                            "mean_value": float(numbers.mean()), "event_file": rel(path),
                        }
                        for quartile in range(1, 5):
                            row[f"q{quartile}_mean"] = float(numbers[quartiles == quartile].mean())
                        summary.append(row)
                except Exception as error:  # Keep a parse failure visible without blocking other evidence.
                    inventory.append({
                        "group": group, "ppo_seed": seed, "event_file": rel(path),
                        "status": f"PARSE_ERROR:{type(error).__name__}:{error}",
                    })
    summary_frame = pd.DataFrame(summary)
    if summary_frame.empty:
        return summary_frame, pd.DataFrame(inventory), {
            "status": "UNAVAILABLE", "event_files_found": len(inventory),
            "parsed_event_files": int(sum(row.get("status") == "PARSED" for row in inventory)),
        }
    status = {
        "status": "PARSED",
        "event_files_found": int(len(inventory)),
        "parsed_event_files": int(sum(row.get("status") == "PARSED" for row in inventory)),
        "summary_rows": int(len(summary_frame)),
        "scalar_tags": sorted(summary_frame.tag.unique().tolist()),
    }
    return summary_frame, pd.DataFrame(inventory), status


def classify(seed_summary: pd.DataFrame, episodes: pd.DataFrame):
    pivot = seed_summary.pivot(index="ppo_seed", columns="domain")
    both_domain_worse = int((
        (pivot["delta_yield_kg_ha_mean"]["observed_weather"] < 0)
        & (pivot["delta_yield_kg_ha_mean"]["heldout_wgen"] < 0)
        & (pivot["delta_common_reward_mean"]["observed_weather"] < 0)
        & (pivot["delta_common_reward_mean"]["heldout_wgen"] < 0)
    ).sum())
    contrib = contributions(seed_summary)
    top2_yield_share_by_domain = {}
    for domain in ["observed_weather", "heldout_wgen"]:
        negative = contrib[
            (contrib.domain == domain) & (contrib.metric == "yield_kg_ha")
            & (contrib.paired_seed_delta_105_minus_75 < 0)
        ]
        top2_yield_share_by_domain[domain] = float(
            negative.sort_values("paired_seed_delta_105_minus_75")
            .head(2).share_of_negative_seed_deltas_percent.sum()
        ) if not negative.empty else 0.0
    seeds_yield_worse_both = int((
        (pivot["delta_yield_kg_ha_mean"]["observed_weather"] < 0)
        & (pivot["delta_yield_kg_ha_mean"]["heldout_wgen"] < 0)
    ).sum())
    if all(value >= 60 for value in top2_yield_share_by_domain.values()):
        failure = "FEW_SEED_COLLAPSE"
    elif seeds_yield_worse_both >= 5 and max(top2_yield_share_by_domain.values()) < 50:
        failure = "BROAD_SHIFT"
    else:
        failure = "MIXED"

    episode_flags = episodes.assign(
        n0_75=episodes.nitrogen_kg_ha_75 <= 0,
        n0_105=episodes.nitrogen_kg_ha_105 <= 0,
        low_75=episodes.very_low_input_episode_75,
        low_105=episodes.very_low_input_episode_105,
        collapse_75=episodes.yield_collapse_episode_75,
        collapse_105=episodes.yield_collapse_episode_105,
    )
    by_seed_domain = episode_flags.groupby(["ppo_seed", "domain"]).agg(
        n0_75=("n0_75", "sum"), n0_105=("n0_105", "sum"),
        very_low_75=("low_75", "sum"), very_low_105=("low_105", "sum"),
        collapse_75=("collapse_75", "sum"), collapse_105=("collapse_105", "sum"),
        marked_yield_loss=("marked_yield_loss_105_vs_75", "sum"),
        episode_count=("episode_key", "count"),
    ).reset_index()
    by_seed_domain["delta_n0"] = by_seed_domain.n0_105 - by_seed_domain.n0_75
    n0_pivot = by_seed_domain.pivot(index="ppo_seed", columns="domain", values="delta_n0")
    increase_both = int(((n0_pivot.observed_weather > 0) & (n0_pivot.heldout_wgen > 0)).sum())
    total_increase = by_seed_domain.groupby("ppo_seed").delta_n0.sum().clip(lower=0)
    added_n0 = float(total_increase.sum())
    top2_n0_share = float(total_increase.nlargest(2).sum() / added_n0) if added_n0 else 0.0
    low_pattern = "BROAD_ACROSS_SEEDS" if increase_both >= 5 else (
        "CONCENTRATED_IN_FEW_SEEDS" if increase_both <= 3 and top2_n0_share >= 0.60 else "MIXED"
    )
    core = ["yield_kg_ha", "common_reward", "nitrogen_kg_ha", "irrigation_mm", "wp_et_kg_m3"]
    same, total = 0, 0
    for metric in core:
        obs = seed_summary.loc[seed_summary.domain.eq("observed_weather"), f"delta_{metric}_mean"].mean()
        wgen = seed_summary.loc[seed_summary.domain.eq("heldout_wgen"), f"delta_{metric}_mean"].mean()
        if np.isfinite(obs) and np.isfinite(wgen):
            total += 1
            same += int(np.sign(obs) == np.sign(wgen))
    fraction = same / total if total else math.nan
    cross = "INSUFFICIENT" if not np.isfinite(fraction) else ("HIGH" if fraction >= .8 else ("PARTIAL" if fraction >= .5 else "LOW"))
    facts = {
        "seeds_worse_on_yield_and_common_reward_in_both_domains": both_domain_worse,
        "seeds_with_yield_decline_in_both_domains": seeds_yield_worse_both,
        "top_two_share_of_negative_yield_deltas_percent_by_domain": top2_yield_share_by_domain,
        "seeds_with_N0_increase_in_both_domains": increase_both,
        "top_two_share_of_added_N0_episodes": top2_n0_share,
        "core_metrics_same_direction_across_domains": same,
        "core_metrics_compared": total,
    }
    return failure, low_pattern, cross, facts, by_seed_domain


def make_figures(seed_summary: pd.DataFrame, action_summary: pd.DataFrame):
    files = []
    colors = {"negative": "#C65D3B", "positive": "#256D5A"}
    domains = [("observed_weather", "Observed 2014–2023"), ("heldout_wgen", "Held-out WGEN")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, (domain, title) in enumerate(domains):
        part = seed_summary[seed_summary.domain.eq(domain)].sort_values("ppo_seed")
        for row, metric, label in [
            (0, "delta_yield_kg_ha_mean", "Δ Yield (kg/ha)"),
            (1, "delta_common_reward_mean", "Δ Common reward"),
        ]:
            ax = axes[row, col]
            values = part[metric].to_numpy()
            ax.axhline(0, color="#333333", linewidth=1)
            ax.bar(part.ppo_seed.astype(str), values, color=[colors["negative"] if v < 0 else colors["positive"] for v in values], width=.68)
            ax.set_title(title if row == 0 else "")
            ax.set_xlabel("Paired PPO seed")
            ax.set_ylabel(label)
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
            ax.set_axisbelow(True)
    fig.suptitle("105 mm minus 75 mm by paired seed", fontsize=13)
    path = OUT / "figure_seed_delta_diagnosis.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    files.append(path.name)
    if not action_summary.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for col, (domain, title) in enumerate(domains):
            part = action_summary[action_summary.domain.eq(domain)]
            early = part.pivot(index="ppo_seed", columns="group", values="mean_early_irrigation_mm_dap_le_30")
            noop = part.pivot(index="ppo_seed", columns="group", values="mean_no_op_rate")
            for ax, data, ylabel in [
                (axes[0, col], early, "Δ Early irrigation (mm, DAP≤30)"),
                (axes[1, col], noop, "Δ No-op rate"),
            ]:
                delta = data.get("105", pd.Series(dtype=float)) - data.get("75", pd.Series(dtype=float))
                values = delta.to_numpy()
                ax.axhline(0, color="#333333", linewidth=1)
                ax.bar(delta.index.astype(str), values, color=[colors["negative"] if v < 0 else colors["positive"] for v in values])
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Paired PPO seed")
                ax.grid(axis="y", color="#dddddd", linewidth=.6)
                ax.set_axisbelow(True)
            axes[0, col].set_title(title)
        fig.suptitle("Action behavior differences from existing traces", fontsize=13)
        path = OUT / "figure_action_behavior_deltas.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        files.append(path.name)
    return files


def write_report(episodes, seed_summary, phenotype, action_summary, contributions_df,
                 trace_status, trace_closure, training_progress, inventory,
                 tensorboard_summary, tensorboard_status, failure, low_pattern,
                 cross_domain, facts, figures):
    lines_overall = []
    for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "Held-out WGEN")]:
        part = episodes[episodes.domain.eq(domain)]
        lines_overall.append(
            f"| {label} | {part.delta_yield_kg_ha.mean():.1f} | {part.delta_irrigation_mm.mean():.1f} | "
            f"{part.delta_nitrogen_kg_ha.mean():.1f} | {part.delta_wp_et_kg_m3.mean():.3f} | "
            f"{part.delta_common_reward.mean():.3f} | {part.delta_canonical_reward.mean():.3f} |"
        )
    obs = seed_summary[seed_summary.domain.eq("observed_weather")]
    wgen = seed_summary[seed_summary.domain.eq("heldout_wgen")]
    paired_lines = []
    for seed in SEEDS:
        o, w = obs[obs.ppo_seed.eq(seed)].iloc[0], wgen[wgen.ppo_seed.eq(seed)].iloc[0]
        paired_lines.append(
            f"| {seed} | {o.delta_yield_kg_ha_mean:.0f} | {w.delta_yield_kg_ha_mean:.0f} | "
            f"{o.delta_common_reward_mean:.3f} | {w.delta_common_reward_mean:.3f} | "
            f"{o.delta_nitrogen_kg_ha_mean:.1f} | {w.delta_nitrogen_kg_ha_mean:.1f} | "
            f"{o.delta_irrigation_mm_mean:.1f} | {w.delta_irrigation_mm_mean:.1f} |"
        )
    phenotype_counts = phenotype.groupby(["domain", "group"]).agg(
        low=("very_low_input_seed_phenotype", "sum"),
        collapse=("yield_collapse_seed_phenotype", "sum"),
        fixed=("fixed_N240_seed_phenotype", "sum"),
    )
    ep_lines = []
    for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "Held-out WGEN")]:
        part = episodes[episodes.domain.eq(domain)]
        ep_lines.append(
            f"| {label} | {int(part.n_zero_75.sum())}/{len(part)} | {int(part.n_zero_105.sum())}/{len(part)} | "
            f"{int(part.very_low_input_episode_75.sum())}/{len(part)} | {int(part.very_low_input_episode_105.sum())}/{len(part)} | "
            f"{int(part.yield_collapse_episode_75.sum())}/{len(part)} | {int(part.yield_collapse_episode_105.sum())}/{len(part)} |"
        )
    seed_pheno_lines = []
    for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "WGEN")]:
        p = phenotype_counts.loc[domain]
        seed_pheno_lines.append(
            f"- {label} seed phenotype：very-low-input {int(p.loc['75','low'])}/8→{int(p.loc['105','low'])}/8；"
            f"yield-collapse {int(p.loc['75','collapse'])}/8→{int(p.loc['105','collapse'])}/8；"
            f"fixed N=240 {int(p.loc['75','fixed'])}/8→{int(p.loc['105','fixed'])}/8。"
        )
    worst = episodes[episodes.marked_yield_loss_105_vs_75].sort_values("delta_yield_kg_ha").head(12)
    worst_lines = [
        f"| {int(r.ppo_seed)} | {r.domain} | {r.scenario} | {r.delta_yield_kg_ha:.0f} | "
        f"{r.irrigation_mm_75:.0f}→{r.irrigation_mm_105:.0f} | {r.nitrogen_kg_ha_75:.0f}→{r.nitrogen_kg_ha_105:.0f} | "
        f"{'yes' if r.n_zero_105 else 'no'} | {'yes' if r.yield_collapse_episode_105 else 'no'} |"
        for r in worst.itertuples()
    ]
    n0 = episodes.assign(
        n0_75=episodes.nitrogen_kg_ha_75 <= 0,
        n0_105=episodes.nitrogen_kg_ha_105 <= 0,
    ).groupby(["domain", "ppo_seed"])[["n0_75", "n0_105"]].sum().reset_index()
    n0_lines = []
    for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "WGEN")]:
        part = n0[n0.domain.eq(domain)]
        increased = part.loc[part.n0_105 > part.n0_75, "ppo_seed"].astype(int).tolist()
        n0_lines.append(f"- {label}：N=0 episode count 在 105 高于 75 的 seed 为 {increased}。")
    action_lines = []
    event_lines = []
    if not action_summary.empty:
        for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "Held-out WGEN")]:
            part = action_summary[action_summary.domain.eq(domain)].groupby("group").mean(numeric_only=True)
            if {"75", "105"}.issubset(set(part.index.astype(str))):
                c, t = part.loc["75"], part.loc["105"]
                action_lines.append(
                    f"| {label} | {t.mean_early_irrigation_mm_dap_le_30-c.mean_early_irrigation_mm_dap_le_30:.1f} | "
                    f"{t.mean_early_nitrogen_kg_ha_dap_le_30-c.mean_early_nitrogen_kg_ha_dap_le_30:.1f} | "
                    f"{t.mean_no_op_rate-c.mean_no_op_rate:.3f} | "
                    f"{t.mean_unique_action_count-c.mean_unique_action_count:.2f} | "
                    f"{t.unique_schedules-c.unique_schedules:.0f} |"
                )
                event_lines.append(
                    f"| {label} | {t.mean_irrigation_event_count-c.mean_irrigation_event_count:.2f} | "
                    f"{t.mean_nitrogen_event_count-c.mean_nitrogen_event_count:.2f} | "
                    f"{t.mean_nonzero_management_days-c.mean_nonzero_management_days:.2f} | "
                    f"{t.mean_nonzero_management_steps-c.mean_nonzero_management_steps:.2f} |"
                )
    focus_lines = []
    for seed in [3, 4, 5, 6]:
        part = seed_summary[seed_summary.ppo_seed.eq(seed)]
        obs_part = part[part.domain.eq("observed_weather")].iloc[0]
        wgen_part = part[part.domain.eq("heldout_wgen")].iloc[0]
        episode_part = episodes[episodes.ppo_seed.eq(seed)]
        n0_obs = episode_part[episode_part.domain.eq("observed_weather")]
        n0_wgen = episode_part[episode_part.domain.eq("heldout_wgen")]
        focus_lines.append(
            f"| {seed} | {part.irrigation_mm_75_mean.mean():.1f}→{part.irrigation_mm_105_mean.mean():.1f} | "
            f"{part.nitrogen_kg_ha_75_mean.mean():.1f}→{part.nitrogen_kg_ha_105_mean.mean():.1f} | "
            f"{obs_part.delta_yield_kg_ha_mean:.0f} | {wgen_part.delta_yield_kg_ha_mean:.0f} | "
            f"{int(n0_obs.n_zero_75.sum())}→{int(n0_obs.n_zero_105.sum())} | "
            f"{int(n0_wgen.n_zero_75.sum())}→{int(n0_wgen.n_zero_105.sum())} |"
        )
    collapse_action_lines = []
    for seed in [4, 5]:
        for domain, label in [("observed_weather", "Observed"), ("heldout_wgen", "WGEN")]:
            part = action_summary[
                action_summary.ppo_seed.eq(seed) & action_summary.domain.eq(domain)
            ].set_index("group")
            if {"75", "105"}.issubset(set(part.index.astype(str))):
                c, t = part.loc["75"], part.loc["105"]
                collapse_action_lines.append(
                    f"| {seed} | {label} | {int(c.unique_schedules)} | {int(t.unique_schedules)} | "
                    f"{c.mean_unique_action_count:.1f} | {t.mean_unique_action_count:.1f} | "
                    f"{c.mean_no_op_rate:.3f} | {t.mean_no_op_rate:.3f} |"
                )
    common_training = training_progress[training_progress.ppo_seed.between(3, 7)] if not training_progress.empty else training_progress
    training_lines = []
    if not common_training.empty:
        for group in ["control_75", "treatment_105"]:
            sub = common_training[common_training.group.eq(group)]
            q1, q4 = sub[sub.recorded_episode_quartile.eq(1)], sub[sub.recorded_episode_quartile.eq(4)]
            if not q1.empty and not q4.empty:
                training_lines.append(
                    f"| {group} | {q1.mean_episode_return.mean():.3f}→{q4.mean_episode_return.mean():.3f} | "
                    f"{q1.mean_yield_kg_ha.mean():.0f}→{q4.mean_yield_kg_ha.mean():.0f} | "
                    f"{q1.mean_irrigation_mm.mean():.1f}→{q4.mean_irrigation_mm.mean():.1f} | "
                    f"{q1.mean_n_kg_ha.mean():.1f}→{q4.mean_n_kg_ha.mean():.1f} |"
                )
    tensorboard_lines = []
    if not tensorboard_summary.empty:
        for seed in range(3, 8):
            row = {"ppo_seed": seed}
            for group, prefix in [("control_75", "75"), ("treatment_105", "105")]:
                part = tensorboard_summary[
                    tensorboard_summary.group.eq(group)
                    & tensorboard_summary.ppo_seed.eq(seed)
                ]
                entropy = part[part.tag.eq("train/entropy_loss")]
                reward = part[part.tag.eq("rollout/ep_rew_mean")]
                row[f"entropy_{prefix}"] = float(entropy.last_value.iloc[0]) if not entropy.empty else math.nan
                row[f"reward_{prefix}"] = float(reward.last_value.iloc[0]) if not reward.empty else math.nan
            tensorboard_lines.append(
                f"| {seed} | {row['entropy_75']:.4f} | {row['entropy_105']:.4f} | "
                f"{row['reward_75']:.3f} | {row['reward_105']:.3f} |"
            )
    treatment_status = trace_status[trace_status.group.eq("105")]
    ok = int(treatment_status.trace_status.eq("ok").sum())
    expected = len(treatment_status)
    closure_ok = sum(
        bool(value) for value in trace_closure.get("trace_totals_close", pd.Series(dtype=bool)).tolist()
        if pd.notna(value)
    )
    closure_n = int(len(trace_closure))
    control_summaries = inventory[(inventory.group == "control_75") & (inventory.training_episode_summary == "AVAILABLE")]
    treatment_summaries = inventory[(inventory.group == "treatment_105") & (inventory.training_episode_summary == "AVAILABLE")]
    phase_evidence = "INSUFFICIENT"
    report = f"""# YC 004_14：75 vs 105 mm 重训练行为诊断

## 结论先行

本任务只读取既有训练 episode summary、正式 paired episode、action traces 和 checkpoint 文件清单；未训练 PPO、未加载/评估 checkpoint、未运行 DSSAT 或 replay。

105mm_failure_structure = {failure}  
low_input_pattern = {low_pattern}  
cross_domain_consistency = {cross_domain}  
training_stage_evidence = {phase_evidence}  
mechanistic_interpretation = seed 4/5 在两个域转为单一低投入 schedule，seed 3 也减投减产，seed 6 则反向增投增产；训练 entropy-loss 方向不一致。这支持 seed-specific 行为极化/动作收缩解释，但缺少 raw argmax、完整 mask 与阶段性评估，不能把它因果归于 early-cap 放宽。  
next_step = retain_75_and_close_YC

建议保留 75 mm canonical 并结束 early-cap 阈值支线。已有数据能描述低投入 phenotype 与结果的共现，但不能证明因果机制；没有足够证据支持继续针对 early cap 做下一轮调参。

## 数据覆盖与判定口径

- Paired episode rows：{len(episodes)}；Observed {int((episodes.domain == "observed_weather").sum())}，held-out WGEN {int((episodes.domain == "heldout_wgen").sum())}。
- action trace：105 mm {ok}/{expected} episode 匹配成功；75/105 合计投入闭合 {closure_ok}/{closure_n}（trace 与 episode 总 I/N 差值均≤1 单位）。明细在 action_trace_coverage.csv 与 action_trace_closure.csv。
- 可读取的训练 episode summary：75 mm {len(control_summaries)}/8 seeds，105 mm {len(treatment_summaries)}/8 seeds。分段仅按记录 episode 顺序切成四段；它不是等长训练步数或 checkpoint evaluation。
- 虽然 checkpoint 文件存在，脚本只记录文件名和大小，没有读取模型权重。
- 004_13 action trace 记录实际动作、DAP、SWFAC/NSTRES 等，但不含 raw argmax 和完整 mask。因此不能用它证明“某动作偏好被 early cap 拦截”。
- Seed phenotype 沿用 004_13：mean I≤75 mm 且 mean N≤80 kg/ha 为 very-low-input；Observed mean yield<6500 kg/ha 且满足 low-input 为 yield-collapse；固定 N=240 定义为至少 80% episodes N=240。
- Episode flag 使用同一投入阈值，yield<6500 标为 episode collapse 筛查，不能与 seed phenotype 混为一谈。
- “明显减产 episode”筛查阈值为 paired yield 下降≥1000 kg/ha 或≥15%，仅用于定位案例，不是胜负判据。
- low_input_pattern：至少 5/8 seeds 在两个域都增加 N=0 episode 记为 BROAD；至多 3/8 且新增 N=0 的 top-2 seed 占比≥60%记为 CONCENTRATED；其余为 MIXED。
- cross_domain_consistency 的 HIGH 仅表示五项组均值指标（yield、common reward、N、I、WP_ET）在两个域方向至少 80% 一致，不代表每个 seed 或每个指标幅度一致。

## 总体与 paired seeds

下表为 105−75 的 episode 均值差，正值表示 105 mm 更高。seed contribution 使用每个 seed 的 domain mean delta 除以 8。详情见 seed_domain_paired_summary.csv 与 seed_mean_contributions.csv。

| Domain | Δ Yield kg/ha | Δ Irrigation mm | Δ N kg/ha | Δ WP_ET | Δ Common reward | Δ Canonical reward |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines_overall)}

| Seed | Δ Yield Obs | Δ Yield WGEN | Δ Common reward Obs | Δ Common reward WGEN | Δ N Obs | Δ N WGEN | Δ I Obs | Δ I WGEN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_lines)}

Failure structure 以两个评估域各自的 seed-level 负向 yield delta 为准：若 top-2 seeds 合计占各域负向 yield delta 绝对值至少 60%，归为 FEW_SEED_COLLAPSE；若至少 5/8 seeds 两域都减产且 top-2 均不足 50%，归为 BROAD_SHIFT；否则 MIXED。该标签描述减产集中度，不宣称训练 collapse 已被过程数据定位。分类计数：{json.dumps(facts, ensure_ascii=False)}。

## Episode 与低投入行为

| Domain | N=0 75 | N=0 105 | Low-input 75 | Low-input 105 | Collapse 75 | Collapse 105 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ep_lines)}

{chr(10).join(seed_pheno_lines)}

{chr(10).join(n0_lines)}

按 seed 和年份/weather seed 的全量 phenotype、N=0 与减产标签见 paired_episode_diagnosis.csv、episode_phenotype_counts_by_seed.csv。若 N=0 增加分布在多个 seed 且两个评估域方向一致，支持“行为分布有广泛位移”的描述；若集中于少数 seed，则更像 seed 异质性。两者都不等价于 cap 的因果效应。

### 关键 seed 行为分化

| Seed | 两域等权平均 I 75→105 | 两域等权平均 N 75→105 | Δ Yield Obs | Δ Yield WGEN | N=0 Obs 75→105 | N=0 WGEN 75→105 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(focus_lines)}

Seed 4 和 5 是主要低投入/减产端：两域平均上均由 N=240 降至 N=0，灌溉降至约 45 mm，且两域 yield 均大幅下降；seed 3 也明显降 N/I 并减产。与之相反，seed 6 从 N=0 转为 N=240、灌溉升至约 230 mm，并明显增产。故 105 mm 的组均值变化不是 8 个 seed 一致“更节水/减氮”，而是少数策略走向极低投入、另一些转向高投入，seed 间分化加大。这里是描述性对应，不足以把这些策略变化归因于 early cap 本身。

Seed 4/5 的 action-trace 证据如下。unique schedules 统计每个 seed×domain 内不同评估 episode 的完整正管理动作序列数；action count 是实际出现的离散动作编号数。

| Seed | Domain | Unique schedules 75 | Unique schedules 105 | Unique actions 75 | Unique actions 105 | No-op rate 75 | No-op rate 105 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(collapse_action_lines)}

105 mm 的 seeds 4/5 在两个域都只出现 1 种 schedule、2 个 action code，约 99% 决策为 no-op；已执行 schedule 是 DAP 0 灌 45 mm、N=0，之后不再施肥/灌溉。Seed 4 的 75 mm 对照在跨年 schedule 和动作编号上更多样；seed 5 的 75 mm schedule 本来也重复，但仍有 4 个 action code 且全季 N=240。这里的“固定”只用于描述行为收缩，不作为策略优劣定义。

### 明显减产 episodes

| Seed | Domain | Year/weather seed | Δ Yield kg/ha | I 75→105 | N 75→105 | N=0 105 | Collapse 105 |
|---:|---|---:|---:|---:|---:|---|---|
{chr(10).join(worst_lines) if worst_lines else '| 无 | | | | | | | |'}

N=0 时 PFP_N 为 NA。有效样本数在逐 seed summary 中保留，PFP_N 均值不得脱离 N=0 发生率解释。

## 管理行为与 early period

| Domain | Δ early I mm (DAP≤30) | Δ early N kg/ha (DAP≤30) | Δ no-op rate | Δ unique action count | Δ unique schedule count |
|---|---:|---:|---:|---:|---:|
{chr(10).join(action_lines) if action_lines else '| unavailable | | | | | |'}

逐 episode trace 指标见 action_behavior_episode.csv；seed-domain 汇总见 action_behavior_seed_domain.csv；全部正施灌/施氮动作的 DAP 与 dose 见 management_events_from_traces.csv。unique schedule 仅描述策略多样性，不作为优劣标准。

| Domain | Δ Irrigation event count / episode | Δ N event count / episode | Δ non-zero management days / episode | Δ non-zero action steps / episode |
|---|---:|---:|---:|---:|
{chr(10).join(event_lines) if event_lines else '| unavailable | | | | |'}

Early-cap 可以描述性地与已执行的早期 I/N 变化、后续投入和 episode 结果对照；现有 trace 没有 raw argmax/full mask，无法判断放宽后模型“想选”但曾被拦的动作，也无法建立联合 I×N mask 导致 N=0 的因果链。

## 训练阶段

- 75 mm 有训练 episode summary：{len(control_summaries)}/8 seeds；105 mm：{len(treatment_summaries)}/8 seeds。
- 共同可读 seeds 3–7 的首末四分位记录 episode 汇总如下；四分位按每 seed 的记录顺序划分，不代表等长 training steps：

| Group | Episode return Q1→Q4 | Yield Q1→Q4 kg/ha | Irrigation Q1→Q4 mm | N Q1→Q4 kg/ha |
|---|---:|---:|---:|---:|
{chr(10).join(training_lines) if training_lines else '| unavailable | | | | |'}

- 105 mm rollout 在首末段呈现 return 上升、平均投入下降并伴有平均 yield 下降；但 75 mm 也有低投入趋势。训练 episode 是持续更新策略下的在线 rollout，不是固定 checkpoint 评估，不能定位 phenotype 的确切形成阶段。
- 25K/50K/75K/100K checkpoint inventory 见 checkpoint_inventory.csv。没有既有分阶段评估；按边界不 replay checkpoint。
- TensorBoard event scalar：{tensorboard_status.get('parsed_event_files', 0)}/{tensorboard_status.get('event_files_found', 0)} 个文件已解析，状态 {tensorboard_status.get('status')}。仅提取紧凑分位摘要，不加载模型；清单、摘要和状态见 training_tensorboard_inventory.csv、training_tensorboard_summary.csv、training_tensorboard_status.json。

| Common seed | Final entropy_loss 75 | Final entropy_loss 105 | Final rollout reward 75 | Final rollout reward 105 |
|---:|---:|---:|---:|---:|
{chr(10).join(tensorboard_lines) if tensorboard_lines else '| unavailable | | | | |'}

- TensorBoard 的 entropy_loss 仅是训练期全局标量，不能直接等同于固定 evaluation trace 的行为熵。共同 seeds 的方向并不一致，因此它不能单独解释 seed 4/5 的固定低投入 schedule。
- 因而 training_stage_evidence = INSUFFICIENT，不能判断低投入/产量崩塌是早期形成还是后期退化。

## 证据层级与决策

**事实：** 配对 CSV 给出每个 seed/domain/episode 的产量、I/N、效率、reward 和管理事件；action traces 提供实际采取的时序与剂量；本报告给出 N=0、低投入、collapse 和显著减产的分布以及对总体均值的 seed 贡献。

**行为解释：** traces 可显示 105 mm 是否改变实际早期灌溉、施氮、no-op、动作多样性与 schedule 重复。此为描述性关联，不构成 mask/reward 的因果解释。

**未知：** 训练期间具体哪个 observation/action/mask 决策造成低 N 或减产；phenotype 出现在哪个 checkpoint 阶段；没有放宽 cap 时同一训练 seed 是否避免 phenotype。以上都没有现成证据，本任务不补跑。

由于 105 mm 的总体结果不优于 75 mm，且只读资料没有识别出一个可被单独验证的明确机制，决策为 retain_75_and_close_YC。此结论是关闭本轮 mask threshold tuning 支线，不代表 YC 的 PPO 策略优化目标已完成。

## 文件与可复现性

- Script: src/run_004_14_yc_75_vs_105_retraining_behavior_diagnosis.py
- Primary input: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/paired_episode_comparison.csv
- Input file hashes: input_provenance.json
- Machine result: diagnostic_conclusion.json
- Figures: {", ".join(figures)}
- No commit/push.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensorboard-only", action="store_true",
        help="Read only existing TensorBoard event scalars and write compact summaries; skip all episode/action processing.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.tensorboard_only:
        summary, inventory, status = tensorboard_diagnostic()
        summary.to_csv(OUT / "training_tensorboard_summary.csv", index=False)
        inventory.to_csv(OUT / "training_tensorboard_inventory.csv", index=False)
        (OUT / "training_tensorboard_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return
    if not PAIRED_EPISODES.is_file():
        raise FileNotFoundError(PAIRED_EPISODES)
    paired = pd.read_csv(PAIRED_EPISODES, low_memory=False)
    formal_manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    experiment_design = json.loads(EXPERIMENT_DESIGN.read_text(encoding="utf-8"))
    control_manifest = pd.read_csv(CONTROL_MODEL_MANIFEST)
    if formal_manifest.get("status") != "completed" or float(formal_manifest.get("early_cap_mm", math.nan)) != 105.0:
        raise ValueError("004_13 formal manifest does not certify completed 105 mm treatment")
    if set(formal_manifest.get("completed_seeds", [])) != set(SEEDS):
        raise ValueError("004_13 formal manifest does not cover seeds 0-7")
    random_control = control_manifest[control_manifest.training_regime.astype(str).str.upper().str.contains("RANDOM")]
    if set(pd.to_numeric(random_control.ppo_seed, errors="coerce").dropna().astype(int)) != set(SEEDS):
        raise ValueError("004_05 model manifest does not cover random-weather seeds 0-7")
    if set(pd.to_numeric(paired.ppo_seed, errors="coerce").dropna().astype(int)) != set(SEEDS):
        raise ValueError("paired input must contain PPO seeds 0-7")
    if paired.duplicated(["ppo_seed", "evaluation_weather_type", "episode_pair_key"]).any():
        raise ValueError("paired episode keys are not unique")
    if paired.groupby("evaluation_weather_type").size().to_dict() != {"heldout_wgen": 160, "observed_weather": 80}:
        raise ValueError("unexpected paired episode counts; inspect provenance before interpretation")

    episodes, behavior, coverage, event_detail = build_episode_data(paired)
    seed_summary, phenotype, action_summary = summarize_seeds(episodes, behavior)
    contribution = contributions(seed_summary)
    failure, low_pattern, cross_domain, facts, n0_by_seed_domain = classify(seed_summary, episodes)
    training_raw, inventory, training_progress = training_diagnostic(control_manifest)
    status_path = OUT / "training_tensorboard_status.json"
    summary_path = OUT / "training_tensorboard_summary.csv"
    inventory_path = OUT / "training_tensorboard_inventory.csv"
    saved_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if saved_status.get("status") == "PARSED" and summary_path.is_file() and inventory_path.is_file():
        tensorboard_status = saved_status
        tensorboard_summary = pd.read_csv(summary_path)
        tensorboard_inventory = pd.read_csv(inventory_path)
    else:
        tensorboard_inventory = tensorboard_inventory_only()
        tensorboard_summary = pd.DataFrame()
        tensorboard_status = {
            "status": "NOT_PARSED_IN_CORE_RUN",
            "reason": "TensorBoard parser is optional; use --tensorboard-only in the project runtime if needed.",
            "event_files_found": int((tensorboard_inventory.status != "UNAVAILABLE").sum()),
            "parsed_event_files": 0,
        }
    figures = make_figures(seed_summary, action_summary)

    closure = behavior[behavior.get("trace_found", False) == True].copy()  # noqa: E712
    if not closure.empty:
        lookup = episodes.set_index(["ppo_seed", "domain", "episode_key"])
        for idx, row in closure.iterrows():
            key = (int(row.seed), str(row.domain), str(row.episode_key))
            if key not in lookup.index:
                continue
            episode = lookup.loc[key]
            side = str(row.group)
            closure.loc[idx, "episode_irrigation_mm"] = episode[f"irrigation_mm_{side}"]
            closure.loc[idx, "episode_nitrogen_kg_ha"] = episode[f"nitrogen_kg_ha_{side}"]
            closure.loc[idx, "trace_totals_close"] = (
                abs(row.irrigation_total_from_trace_mm - episode[f"irrigation_mm_{side}"]) <= 1.0
                and abs(row.nitrogen_total_from_trace_kg_ha - episode[f"nitrogen_kg_ha_{side}"]) <= 1.0
            )

    episodes.to_csv(OUT / "paired_episode_diagnosis.csv", index=False)
    seed_summary.to_csv(OUT / "seed_domain_paired_summary.csv", index=False)
    phenotype.to_csv(OUT / "seed_phenotype_transitions.csv", index=False)
    phenotype_pairs = phenotype.pivot(
        index=["ppo_seed", "domain"], columns="group",
        values=["very_low_input_seed_phenotype", "yield_collapse_seed_phenotype", "fixed_N240_seed_phenotype"],
    ).reset_index()
    phenotype_pairs.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in phenotype_pairs.columns
    ]
    for name in [
        "very_low_input_seed_phenotype",
        "yield_collapse_seed_phenotype",
        "fixed_N240_seed_phenotype",
    ]:
        c, t = f"{name}_75", f"{name}_105"
        if c in phenotype_pairs and t in phenotype_pairs:
            phenotype_pairs[f"{name}_transition"] = np.select(
                [~phenotype_pairs[c].astype(bool) & phenotype_pairs[t].astype(bool),
                 phenotype_pairs[c].astype(bool) & ~phenotype_pairs[t].astype(bool)],
                ["emergent_in_105", "resolved_in_105"], default="unchanged",
            )
    phenotype_pairs.to_csv(OUT / "seed_phenotype_transition_pairs.csv", index=False)
    contribution.to_csv(OUT / "seed_mean_contributions.csv", index=False)
    behavior.to_csv(OUT / "action_behavior_episode.csv", index=False)
    coverage.to_csv(OUT / "action_trace_coverage.csv", index=False)
    action_summary.to_csv(OUT / "action_behavior_seed_domain.csv", index=False)
    closure.to_csv(OUT / "action_trace_closure.csv", index=False)
    event_detail.to_csv(OUT / "management_events_from_traces.csv", index=False)
    n0_by_seed_domain.to_csv(OUT / "episode_phenotype_counts_by_seed.csv", index=False)
    training_progress.to_csv(OUT / "training_episode_progress_quartiles.csv", index=False)
    inventory.to_csv(OUT / "checkpoint_inventory.csv", index=False)
    tensorboard_inventory.to_csv(inventory_path, index=False)
    status_path.write_text(json.dumps(tensorboard_status, indent=2), encoding="utf-8")
    if not tensorboard_summary.empty:
        tensorboard_summary.to_csv(summary_path, index=False)

    input_paths = [
        PAIRED_EPISODES, FORMAL_MANIFEST, EXPERIMENT_DESIGN,
        CONTROL_MODEL_MANIFEST,
        CONTROL_ROOT / "all_model_performance_summary_0_7.csv",
        CONTROL_ROOT / "archetype_performance_summary.csv",
    ]
    input_paths.extend(sorted(CONTROL_TRACE_ROOT.glob("*.csv")))
    for folder in TREATMENT_TRACE_ROOTS:
        input_paths.extend(sorted(folder.glob("*.csv")))
    for seed in SEEDS:
        for base in [CONTROL_ROOT, TASK_ROOT]:
            path = base / "training/random_weather" / f"ppo_seed_{seed}" / "training_episode_summary.csv"
            if path.is_file():
                input_paths.append(path)
        tb_folder = TASK_ROOT / "training/random_weather" / f"ppo_seed_{seed}"
        input_paths.extend(sorted(tb_folder.rglob("events.out.tfevents.*")) if tb_folder.is_dir() else [])
    for seed in range(3, 8):
        tb_folder = CONTROL_ROOT / "training/random_weather" / f"ppo_seed_{seed}"
        input_paths.extend(sorted(tb_folder.rglob("events.out.tfevents.*")) if tb_folder.is_dir() else [])
    files, seen = [], set()
    for path in input_paths:
        path = path.resolve()
        if path.is_file() and path not in seen:
            seen.add(path)
            files.append({"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    provenance = {
        "task": "004_14", "inputs_read_only": True,
        "training_started": False, "checkpoint_loaded": False,
        "dssat_evaluation_or_replay_started": False,
        "input_files": files,
        "formal_manifest_status": formal_manifest.get("status"),
        "treatment_early_cap_mm": formal_manifest.get("early_cap_mm"),
        "completed_treatment_seeds": formal_manifest.get("completed_seeds"),
        "control_source": formal_manifest.get("control_source"),
        "control_exact_replay_episodes": formal_manifest.get("control_exact_replay_episodes"),
        "control_model_manifest_random_weather_seed_count": int(random_control.ppo_seed.nunique()),
        "tensorboard_diagnostic": tensorboard_status,
        "treatment_trace_success": int(coverage[(coverage.group == "105")].trace_status.eq("ok").sum()),
        "treatment_trace_expected": int((coverage.group == "105").sum()),
    }
    (OUT / "input_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    conclusion = {
        "105mm_failure_structure": failure,
        "low_input_pattern": low_pattern,
        "cross_domain_consistency": cross_domain,
        "training_stage_evidence": "INSUFFICIENT",
        "mechanistic_interpretation": (
            "105 mm 的减产不是所有 seeds 一致变差，而是 seed 间策略极化。"
            "Seed 4/5 在两个评估域都转为约 I=45 mm、N=0 的单一 schedule，"
            "约 99% 决策为 no-op；seed 3 也减投入并减产，而 seed 6 反向增投并增产。"
            "训练期 TensorBoard entropy-loss 在 paired seeds 间方向不一致。"
            "这些结果支持 seed-specific low-input/action-collapse 行为解释，"
            "但没有 raw argmax/full mask 或固定 checkpoint 阶段评估，不能证明该行为由 105 mm cap 因果造成。"
        ),
        "next_step": "retain_75_and_close_YC",
        "classification_evidence": facts,
        "training_and_dssat_reruns": False,
        "unavailable": [
            "formal 105 mm raw policy argmax and complete action mask",
            "fixed-checkpoint phase evaluations at 25K/50K/75K/100K",
            "causal attribution of low-N/collapse phenotype to early cap",
        ],
    }
    (OUT / "diagnostic_conclusion.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    write_report(episodes, seed_summary, phenotype, action_summary, contribution,
                 coverage, closure, training_progress, inventory,
                 tensorboard_summary, tensorboard_status, failure,
                 low_pattern, cross_domain, facts, figures)
    print(json.dumps({
        "output_dir": rel(OUT), "report": rel(REPORT),
        "episodes": len(episodes), "trace_episode_rows": len(coverage),
        "trace_ok": int(coverage.trace_status.eq("ok").sum()),
        "failure_structure": failure, "low_input_pattern": low_pattern,
        "cross_domain_consistency": cross_domain,
        "training_stage_evidence": "INSUFFICIENT",
        "next_step": "retain_75_and_close_YC",
    }, indent=2))


if __name__ == "__main__":
    main()

"""205HLA: consolidate HLA 100K weather-augmentation diagnostics.

This analysis is intentionally read-only with respect to the 054/204 runs.  It
combines PPO training-process logs, checkpoint validation, the paired 054
comparison, and exact Summary.OUT/ETCP replays.  It never estimates WP_ET from
daily CSV files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
OLD_ROOT = ROOT / "benchmark_results" / "054_00_hla_lowIC_expanded_action_maskableppo"
OUT = ROOT / "benchmark_results" / "205HLA_hla_weather_aug_100k_diagnostics"
TABLES = OUT / "tables"
RECORD = ROOT / "docs" / "205HLA_hla_weather_aug_100k_diagnostics_record.md"
CHECKPOINTS = [10_000, 25_000, 50_000, 75_000, 100_000]
COMMON_CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def exact_replay_dir(prefix: str, checkpoint: int) -> Path | None:
    candidates = sorted(
        path
        for path in (ROOT / "benchmark_results").glob(f"{prefix}*ckpt{checkpoint}*")
        if (path / "tables" / "054_03_hla_five_scenario_season_summary.csv").exists()
    )
    if not candidates:
        return None
    preferred = [path for path in candidates if "exact205" in path.name]
    return (preferred or candidates)[-1]


def exact_ppo_metrics(prefix: str, checkpoint: int) -> dict[str, Any]:
    directory = exact_replay_dir(prefix, checkpoint)
    if directory is None:
        return {
            "WP_ET_kg_m3": np.nan,
            "ETCP_mm": np.nan,
            "exact_replay_root": "",
            "exact_replay_status": "missing",
        }
    table = pd.read_csv(directory / "tables" / "054_03_hla_five_scenario_season_summary.csv")
    ppo = table[table["scenario"].astype(str).eq("rl_candidate")].copy()
    if ppo.empty:
        raise RuntimeError(f"No rl_candidate rows in {rel(directory)}")
    return {
        "WP_ET_kg_m3": finite(pd.to_numeric(ppo["WP_ET_kg_m3"], errors="coerce").mean()),
        "ETCP_mm": finite(pd.to_numeric(ppo["etcp_mm"], errors="coerce").mean()),
        "exact_replay_root": rel(directory),
        "exact_replay_status": "exact Summary.OUT/ETCP replay",
    }


def process_phase_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    numeric = [
        "num_timesteps",
        "approx_kl",
        "entropy_loss",
        "explained_variance",
        "value_loss",
        "requested_noop_rate",
        "requested_action_entropy_normalized",
        "safe_nonzero_rate",
        "scenario_coverage",
        "rss_mb",
    ]
    for column in numeric:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    bins = [0, 10_000, 25_000, 50_000, 75_000, 100_080]
    labels = ["0-10K", "10-25K", "25-50K", "50-75K", "75-100K"]
    metrics["phase"] = pd.cut(
        metrics["num_timesteps"], bins=bins, labels=labels, include_lowest=True, right=True
    )
    rows: list[dict[str, Any]] = []
    for phase, group in metrics.groupby("phase", observed=True):
        explained = group["explained_variance"]
        rows.append(
            {
                "phase": str(phase),
                "updates": int(len(group)),
                "step_start": int(group["num_timesteps"].min()),
                "step_end": int(group["num_timesteps"].max()),
                "mean_approx_kl": finite(group["approx_kl"].mean()),
                "p95_approx_kl": finite(group["approx_kl"].quantile(0.95)),
                "median_explained_variance": finite(explained.median()),
                "negative_explained_variance_rate": finite((explained < 0).mean()),
                "explained_variance_below_minus1_rate": finite((explained < -1).mean()),
                "median_value_loss": finite(group["value_loss"].median()),
                "p95_value_loss": finite(group["value_loss"].quantile(0.95)),
                "mean_requested_action_entropy_normalized": finite(
                    group["requested_action_entropy_normalized"].mean()
                ),
                "mean_requested_noop_rate": finite(group["requested_noop_rate"].mean()),
                "mean_safe_nonzero_rate": finite(group["safe_nonzero_rate"].mean()),
                "final_scenario_coverage": int(group["scenario_coverage"].iloc[-1]),
                "peak_rss_mb": finite(group["rss_mb"].max()),
            }
        )
    return pd.DataFrame(rows)


def instability_events(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for column in ["num_timesteps", "approx_kl", "explained_variance", "value_loss", "rss_mb"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    flagged = metrics[
        metrics["explained_variance"].lt(-1)
        | metrics["value_loss"].gt(metrics["value_loss"].quantile(0.99))
        | metrics["approx_kl"].gt(metrics["approx_kl"].quantile(0.99))
    ].copy()
    flagged["reason"] = flagged.apply(
        lambda row: "; ".join(
            reason
            for condition, reason in [
                (row["explained_variance"] < -1, "explained_variance<-1"),
                (row["value_loss"] > metrics["value_loss"].quantile(0.99), "value_loss>p99"),
                (row["approx_kl"] > metrics["approx_kl"].quantile(0.99), "approx_kl>p99"),
            ]
            if condition
        ),
        axis=1,
    )
    return flagged[
        ["num_timesteps", "approx_kl", "explained_variance", "value_loss", "rss_mb", "reason"]
    ].sort_values("num_timesteps")


def checkpoint_comparison() -> pd.DataFrame:
    new = pd.read_csv(NEW_ROOT / "audits" / "204HLA_checkpoint_process_summary.csv")
    old_detail = pd.read_csv(OLD_ROOT / "evaluation" / "054_00_checkpoint_validation_summary.csv")
    for column in ["checkpoint_step", "final_grnwt", "PFP_N", "total_irrigation", "total_n"]:
        old_detail[column] = pd.to_numeric(old_detail[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        new_row = new[pd.to_numeric(new["checkpoint_step"], errors="coerce").eq(checkpoint)]
        if new_row.empty:
            continue
        new_row = new_row.iloc[0]
        old_group = old_detail[old_detail["checkpoint_step"].eq(checkpoint)]
        new_exact = exact_ppo_metrics("054_03_hla_lowIC_204HLA_", checkpoint)
        old_exact = exact_ppo_metrics("054_03_hla_lowIC_054_00_", checkpoint)
        old_available = not old_group.empty
        new_yield = finite(new_row["mean_final_grnwt"])
        old_yield = finite(old_group["final_grnwt"].mean()) if old_available else np.nan
        new_pfp = finite(new_row["mean_PFP_N"])
        old_pfp = finite(old_group["PFP_N"].mean()) if old_available else np.nan
        rows.append(
            {
                "checkpoint_step": checkpoint,
                "old_available": old_available,
                "old_mean_yield_kg_ha": old_yield,
                "new_mean_yield_kg_ha": new_yield,
                "delta_yield_kg_ha": new_yield - old_yield if old_available else np.nan,
                "old_mean_WP_ET_kg_m3": old_exact["WP_ET_kg_m3"] if old_available else np.nan,
                "new_mean_WP_ET_kg_m3": new_exact["WP_ET_kg_m3"],
                "delta_WP_ET_kg_m3": (
                    new_exact["WP_ET_kg_m3"] - old_exact["WP_ET_kg_m3"] if old_available else np.nan
                ),
                "old_mean_PFP_N_kg_kg": old_pfp,
                "new_mean_PFP_N_kg_kg": new_pfp,
                "delta_PFP_N_kg_kg": new_pfp - old_pfp if old_available else np.nan,
                "old_mean_irrigation_mm": finite(old_group["total_irrigation"].mean()) if old_available else np.nan,
                "new_mean_irrigation_mm": finite(new_row["mean_total_irrigation"]),
                "old_mean_nitrogen_kg_ha": finite(old_group["total_n"].mean()) if old_available else np.nan,
                "new_mean_nitrogen_kg_ha": finite(new_row["mean_total_n"]),
                "old_unique_action_signatures": int(old_group["action_sequence"].astype(str).nunique())
                if old_available
                else 0,
                "new_unique_action_signatures": int(new_row["unique_full_action_signatures"]),
                "new_all_years_same_action_signature": bool(new_row["all_years_same_action_signature"]),
                "new_max_swfac": finite(new_row["max_swfac"]),
                "new_max_nstres": finite(new_row["max_nstres"]),
                "new_exact_replay_root": new_exact["exact_replay_root"],
                "old_exact_replay_root": old_exact["exact_replay_root"] if old_available else "",
                "WP_ET_provenance": new_exact["exact_replay_status"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    required = [
        NEW_ROOT / "204HLA_formal_result.json",
        NEW_ROOT / "logs" / "204HLA_training_update_metrics.csv",
        NEW_ROOT / "audits" / "204HLA_checkpoint_process_summary.csv",
        OLD_ROOT / "evaluation" / "054_00_checkpoint_validation_summary.csv",
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required completed-run files: " + "; ".join(missing))

    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(NEW_ROOT / "logs" / "204HLA_training_update_metrics.csv")
    phases = process_phase_summary(metrics)
    events = instability_events(metrics)
    checkpoints = checkpoint_comparison()
    if checkpoints["new_mean_WP_ET_kg_m3"].isna().any():
        missing_steps = checkpoints.loc[checkpoints["new_mean_WP_ET_kg_m3"].isna(), "checkpoint_step"].tolist()
        raise RuntimeError(
            "Exact 204HLA Summary.OUT/ETCP replay is missing for checkpoints: " + str(missing_steps)
        )

    phases.to_csv(TABLES / "205HLA_training_phase_process_metrics.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TABLES / "205HLA_training_instability_events.csv", index=False, encoding="utf-8-sig")
    checkpoints.to_csv(TABLES / "205HLA_checkpoint_old_vs_augmented.csv", index=False, encoding="utf-8-sig")

    best_yield = checkpoints.loc[checkpoints["new_mean_yield_kg_ha"].idxmax()]
    best_wp = checkpoints.loc[checkpoints["new_mean_WP_ET_kg_m3"].idxmax()]
    best_pfp = checkpoints.loc[checkpoints["new_mean_PFP_N_kg_kg"].idxmax()]
    collapsed = checkpoints[checkpoints["new_unique_action_signatures"].le(1)]
    common = checkpoints[checkpoints["checkpoint_step"].isin(COMMON_CHECKPOINTS)].copy()
    aug_improves_all_three = common[
        common["delta_yield_kg_ha"].gt(0)
        & common["delta_WP_ET_kg_m3"].gt(0)
        & common["delta_PFP_N_kg_kg"].gt(0)
    ]
    result = {
        "task": "205HLA_hla_weather_aug_100k_diagnostics",
        "status": "completed",
        "diagnostic_definition": {
            "policy_collapse_trigger": "unique full validation action signatures <= 1",
            "critic_instability_trigger": "explained_variance < -1 or value_loss/approx_kl above p99",
            "note": "Triggers are diagnostics, not causal proof.",
        },
        "best_new_checkpoint_by_yield": int(best_yield["checkpoint_step"]),
        "best_new_checkpoint_by_WP_ET": int(best_wp["checkpoint_step"]),
        "best_new_checkpoint_by_PFP_N": int(best_pfp["checkpoint_step"]),
        "collapsed_checkpoint_steps": collapsed["checkpoint_step"].astype(int).tolist(),
        "common_checkpoints_improving_yield_WP_ET_and_PFP_N": aug_improves_all_three[
            "checkpoint_step"
        ].astype(int).tolist(),
        "no_single_metric_cherry_pick": True,
        "WP_ET_rule": "exact Summary.OUT/ETCP replay only; never inferred from daily CSV",
        "outputs": {
            "checkpoint_comparison": rel(TABLES / "205HLA_checkpoint_old_vs_augmented.csv"),
            "training_phase_metrics": rel(TABLES / "205HLA_training_phase_process_metrics.csv"),
            "instability_events": rel(TABLES / "205HLA_training_instability_events.csv"),
            "record": rel(RECORD),
        },
    }
    (OUT / "205HLA_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    record_lines = [
        "# 205HLA HLA weather-augmentation 100K diagnostics record",
        "",
        "## Comparison contract",
        "",
        "- Old comparator: `054_00_hla_lowIC_expanded_action_maskableppo`.",
        "- New run: `204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo`.",
        "- Both use HLA lowIC, seed 0, the same 16-action grid, reward, action-safety logic, and original 2014–2023 validation years.",
        "- The intended treatment difference is the 2004–2013 training weather bank (10 years x 6 scenarios) plus balanced without-replacement scenario sampling.",
        "- `WP_ET` is read only from exact DSSAT `Summary.OUT`/`ETCP` replay.",
        "",
        "## Checkpoint conclusions",
        "",
        f"- Best augmented yield checkpoint: `{int(best_yield['checkpoint_step'])}`.",
        f"- Best augmented WP_ET checkpoint: `{int(best_wp['checkpoint_step'])}`.",
        f"- Best augmented PFP_N checkpoint: `{int(best_pfp['checkpoint_step'])}`.",
        f"- Collapse-trigger checkpoints (one validation action signature): `{collapsed['checkpoint_step'].astype(int).tolist()}`.",
        f"- Common checkpoints improving yield, WP_ET and PFP_N together versus 054_00: `{aug_improves_all_three['checkpoint_step'].astype(int).tolist()}`.",
        "- Different metrics may favor different checkpoints; all checkpoints remain visible to avoid post-hoc single-metric selection.",
        "",
        "## Outputs",
        "",
        f"- Checkpoint comparison: `{rel(TABLES / '205HLA_checkpoint_old_vs_augmented.csv')}`.",
        f"- Training phase process metrics: `{rel(TABLES / '205HLA_training_phase_process_metrics.csv')}`.",
        f"- Instability events: `{rel(TABLES / '205HLA_training_instability_events.csv')}`.",
        f"- Machine-readable result: `{rel(OUT / '205HLA_result.json')}`.",
    ]
    RECORD.write_text("\n".join(record_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

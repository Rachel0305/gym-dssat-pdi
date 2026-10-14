"""021_15: offline management timing versus DSSAT phenology audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/021_15"
RUNS = {
    "unscaled_021_10": ROOT
    / "benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1",
    "scaled_0p1_021_14": ROOT
    / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1",
}
CHECKPOINTS = (5000, 10000, 15000, 20000, 25000)
STAGE_PATTERNS = {
    "emergence": r"Emergence",
    "end_juvenile": r"End Juveni",
    "floral_initiation": r"Floral Ini",
    "silking_75pct": r"75% Silkin",
    "begin_grain_fill": r"Beg Gr Fil",
    "end_grain_fill": r"End Gr Fil",
    "maturity": r"Maturity",
}


def parse_phenology(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="latin-1", errors="ignore")
    result: dict[str, int] = {}
    for key, pattern in STAGE_PATTERNS.items():
        matches = re.findall(
            rf"^\s*\d+\s+[A-Z]{{3}}\s+(\d+)\s+{pattern}\b",
            text,
            flags=re.MULTILINE,
        )
        if not matches:
            raise RuntimeError(f"Missing phenology stage {key} in {path}")
        result[key] = int(matches[-1])
    return result


def phase_for_dap(dap: float, stages: dict[str, int]) -> str:
    if dap <= stages["emergence"]:
        return "pre_emergence"
    if dap <= stages["end_juvenile"]:
        return "juvenile"
    if dap <= stages["floral_initiation"]:
        return "juvenile_to_floral_init"
    if dap <= stages["silking_75pct"]:
        return "floral_init_to_silking"
    if dap <= stages["begin_grain_fill"]:
        return "silking_to_grain_fill"
    if dap <= stages["end_grain_fill"]:
        return "grain_fill"
    return "post_grain_fill"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights)) if float(weights.sum()) > 0 else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_frames: list[pd.DataFrame] = []
    phenology_rows: list[dict] = []
    summary_rows: list[dict] = []
    phase_rows: list[dict] = []

    for protocol, run in RUNS.items():
        seasons = pd.read_csv(run / "evaluations/season_summary_all_checkpoints.csv").set_index("checkpoint")
        for step in CHECKPOINTS:
            checkpoint = run / "checkpoints" / f"checkpoint_{step}"
            stages = parse_phenology(checkpoint / "pdi_tmp_snapshot_eval/OVERVIEW.OUT")
            phenology_rows.append({"protocol": protocol, "checkpoint": step, **stages})
            daily = pd.read_csv(checkpoint / "eval_daily.csv")
            operation_dap = pd.to_numeric(daily["operation_dap"], errors="coerce")
            fallback_dap = pd.to_numeric(daily["dap"], errors="coerce")
            daily["event_dap"] = operation_dap.fillna(fallback_dap)
            for operation, column in (
                ("irrigation", "irrigation_mm"),
                ("nitrogen", "fertilizer_kg_ha"),
            ):
                amount = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
                events = daily.loc[amount > 0, ["step", "event_dap", "action_index", "grnwt", "topwt", "swfac", "nstres"]].copy()
                events["amount"] = amount.loc[amount > 0].to_numpy()
                events.insert(0, "operation", operation)
                events.insert(0, "checkpoint", step)
                events.insert(0, "protocol", protocol)
                events["phenology_phase"] = events.event_dap.map(lambda dap: phase_for_dap(float(dap), stages))
                event_frames.append(events)

            row = seasons.loc[step]
            all_events = pd.concat(event_frames[-2:], ignore_index=True)
            irr = all_events[all_events.operation == "irrigation"]
            nitrogen = all_events[all_events.operation == "nitrogen"]
            summary_rows.append(
                {
                    "protocol": protocol,
                    "checkpoint": step,
                    "yield_kg_ha": float(row.yield_kg_ha),
                    "irrigation_mm": float(row.irrigation_mm),
                    "nitrogen_kg_ha": float(row.nitrogen_kg_ha),
                    "irrigation_events": int(len(irr)),
                    "irrigation_first_dap": float(irr.event_dap.min()) if len(irr) else np.nan,
                    "irrigation_last_dap": float(irr.event_dap.max()) if len(irr) else np.nan,
                    "irrigation_weighted_mean_dap": weighted_mean(irr.event_dap, irr.amount),
                    "nitrogen_events": int(len(nitrogen)),
                    "nitrogen_first_dap": float(nitrogen.event_dap.min()) if len(nitrogen) else np.nan,
                    "nitrogen_last_dap": float(nitrogen.event_dap.max()) if len(nitrogen) else np.nan,
                    "nitrogen_weighted_mean_dap": weighted_mean(nitrogen.event_dap, nitrogen.amount),
                    "nitrogen_after_silking_kg_ha": float(
                        nitrogen.loc[nitrogen.event_dap > stages["silking_75pct"], "amount"].sum()
                    ),
                    "nitrogen_after_grain_fill_start_kg_ha": float(
                        nitrogen.loc[nitrogen.event_dap > stages["begin_grain_fill"], "amount"].sum()
                    ),
                }
            )
            for (operation, phase), group in all_events.groupby(["operation", "phenology_phase"]):
                phase_rows.append(
                    {
                        "protocol": protocol,
                        "checkpoint": step,
                        "operation": operation,
                        "phenology_phase": phase,
                        "event_count": int(len(group)),
                        "amount_total": float(group.amount.sum()),
                    }
                )

    events = pd.concat(event_frames, ignore_index=True)
    phenology = pd.DataFrame(phenology_rows)
    summary = pd.DataFrame(summary_rows)
    phases = pd.DataFrame(phase_rows)
    events.to_csv(OUT / "021_15_management_events_with_phenology.csv", index=False, encoding="utf-8-sig")
    phenology.to_csv(OUT / "021_15_dssat_phenology_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "021_15_checkpoint_timing_summary.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(OUT / "021_15_phase_allocation_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    y_map = {"unscaled_021_10": 1.0, "scaled_0p1_021_14": 0.0}
    colors = {"irrigation": "#0072B2", "nitrogen": "#D55E00"}
    markers = {"irrigation": "o", "nitrogen": "^"}
    for ax, step in zip(axes, CHECKPOINTS):
        subset = events[events.checkpoint == step]
        stage = phenology[(phenology.protocol == "scaled_0p1_021_14") & (phenology.checkpoint == step)].iloc[0]
        for protocol in RUNS:
            for operation in ("irrigation", "nitrogen"):
                group = subset[(subset.protocol == protocol) & (subset.operation == operation)]
                if group.empty:
                    continue
                size = 18 + 0.55 * group.amount.to_numpy()
                ax.scatter(
                    group.event_dap,
                    np.full(len(group), y_map[protocol]),
                    s=size,
                    marker=markers[operation],
                    color=colors[operation],
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                )
        for key, style in (
            ("emergence", ":"),
            ("floral_initiation", "--"),
            ("silking_75pct", "-"),
            ("begin_grain_fill", "-."),
        ):
            ax.axvline(float(stage[key]), color="#666666", linestyle=style, linewidth=0.9, alpha=0.7)
        yields = summary[summary.checkpoint == step].set_index("protocol").yield_kg_ha
        ax.set_yticks([0, 1], [f"scaled ({int(yields['scaled_0p1_021_14'])})", f"unscaled ({int(yields['unscaled_021_10'])})"])
        ax.set_title(f"Checkpoint {step // 1000}K: circles=irrigation, triangles=nitrogen")
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("DAP (phenology lines: emergence, floral initiation, silking, grain-fill start)")
    fig.tight_layout()
    fig.savefig(OUT / "021_15_management_timing_vs_phenology.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    indexed = summary.set_index(["protocol", "checkpoint"])
    key = {
        "unscaled_10k": indexed.loc[("unscaled_021_10", 10000)].to_dict(),
        "scaled_10k": indexed.loc[("scaled_0p1_021_14", 10000)].to_dict(),
    }
    conclusion = {
        "status": "completed_offline",
        "training_or_dssat_calls": 0,
        "key_same_total_comparison": key,
        "scaled_10k_nitrogen_all_after_silking": bool(
            key["scaled_10k"]["nitrogen_after_silking_kg_ha"]
            == key["scaled_10k"]["nitrogen_kg_ha"]
        ),
        "unscaled_10k_nitrogen_after_silking_kg_ha": key["unscaled_10k"]["nitrogen_after_silking_kg_ha"],
        "interpretation": (
            "The same seasonal I120/N300 masks a major timing difference.  The scaled 10K "
            "policy applies all N at/after silking and much of it during grain fill, whereas "
            "the unscaled 10K policy applies most N before silking.  This supports real timing "
            "sensitivity and unstable temporal policy quality, but does not by itself prove a "
            "Markov-observation defect or isolate causality."
        ),
    }
    (OUT / "021_15_summary.json").write_text(
        json.dumps(conclusion, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(conclusion, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()


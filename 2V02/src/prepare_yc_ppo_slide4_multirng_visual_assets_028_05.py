"""Prepare YC PPO Slide 4 assets from the existing 024 five-RNG evidence.

This is an evidence-packaging task only.  It reads existing CSV/Markdown files,
derives presentation tables, and renders slide-ready figures.  It does not
train, validate, run DSSAT, tune PPO, or modify production research logic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
ASSET_DIR = RESULTS / "presentation_assets" / "slide4"

PAIRED_DOC = DOCS / "yc_ppo_original_seed1_5rng_klguard_paired_validation.md"
PAIRED_PATH = RESULTS / "yc_ppo_s1_5rng_control_vs_klguard_paired.csv"
LATE_PATH = RESULTS / "yc_ppo_s1_5rng_control_vs_klguard_late_update_summary.csv"
GROUP_PATH = RESULTS / "yc_ppo_s1_5rng_klguard_summary.csv"
DECISION_PATH = RESULTS / "yc_ppo_s1_5rng_klguard_decision.csv"

GUARD_PATHS = {
    1: RESULTS / "yc_ppo_s1_rng1_klguard_checkpoint_validation.csv",
    2: RESULTS / "yc_ppo_s1_rng2_klguard_checkpoint_validation.csv",
    3: RESULTS / "yc_ppo_s1_rng3_klguard_checkpoint_validation.csv",
    4: RESULTS / "yc_ppo_s1_rng4_klguard_checkpoint_validation.csv",
    5: RESULTS / "yc_ppo_s1_rng5_klguard_checkpoint_validation.csv",
}
CONTROL_PATHS = {
    1: RESULTS / "yc_ppo_s1_control_checkpoint_validation.csv",
    2: RESULTS / "yc_ppo_s1_control_rng2_checkpoint_validation.csv",
    3: RESULTS / "yc_ppo_s1_control_rng3_checkpoint_validation.csv",
    4: RESULTS / "yc_ppo_s1_control_rng4_checkpoint_validation.csv",
    5: RESULTS / "yc_ppo_s1_control_rng5_checkpoint_validation.csv",
}

PAIRED_OUT = RESULTS / "presentation_yc_ppo_slide4_multirng_paired_100k.csv"
SUMMARY_OUT = RESULTS / "presentation_yc_ppo_slide4_multirng_summary.csv"
TRAJECTORIES_OUT = RESULTS / "presentation_yc_ppo_slide4_checkpoint_trajectories.csv"
LATE_OUT = RESULTS / "presentation_yc_ppo_slide4_late_kl_summary.csv"
KEY_NUMBERS_OUT = RESULTS / "presentation_yc_ppo_slide4_key_numbers.csv"
REPORT_OUT = DOCS / "presentation_yc_ppo_slide4_multirng_assets_audit.md"

FIG1 = ASSET_DIR / "slide4_multirng_paired_100k"
FIG2 = ASSET_DIR / "slide4_multirng_summary_cards"
FIG3 = ASSET_DIR / "slide4_multirng_checkpoint_trajectories"
FIG4 = ASSET_DIR / "slide4_multirng_late_approx_kl"

CHECKPOINTS = [75, 80, 85, 90, 95, 100]
CONTROL_COLOR = "#506A80"
GUARD_COLOR = "#865486"
ACCENT_COLOR = "#A06B32"
POSITIVE_COLOR = "#4C6C5B"
TEXT_COLOR = "#1B2733"
MUTED_COLOR = "#5C6873"
GRID_COLOR = "#B5BEC7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty evidence file: {path}")
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_inputs() -> dict[str, object]:
    paths = [PAIRED_DOC, PAIRED_PATH, LATE_PATH, GROUP_PATH, DECISION_PATH]
    paths.extend(GUARD_PATHS.values())
    paths.extend(CONTROL_PATHS.values())
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty evidence file: {path}")

    paired = read_required(PAIRED_PATH)
    late = read_required(LATE_PATH)
    group = read_required(GROUP_PATH)
    decision = read_required(DECISION_PATH)
    paired_doc = PAIRED_DOC.read_text(encoding="utf-8")

    require_columns(
        paired,
        [
            "rng",
            "control_yield_100K",
            "guard_yield_100K",
            "delta_yield",
            "control_collapse",
            "guard_collapse",
            "material_harm",
        ],
        "024 paired endpoint result",
    )
    require_columns(
        late,
        [
            "rng",
            "window",
            "control_approx_kl_mean",
            "klguard_approx_kl_mean",
            "control_approx_kl_median",
            "klguard_approx_kl_median",
        ],
        "024 late-update result",
    )
    require_columns(
        group,
        [
            "group",
            "collapse_count",
            "collapse_rate",
            "mean_paired_delta_yield",
            "median_paired_delta_yield",
            "minimum_paired_delta_yield",
            "material_harm_count_stable_rngs",
        ],
        "024 group summary",
    )
    require_columns(
        decision,
        [
            "decision",
            "control_collapse_count",
            "klguard_collapse_count",
            "stable_rng_material_harm_count",
            "rng2_guard_status",
            "rng0_status",
            "n",
        ],
        "024 decision",
    )

    if "KL_GUARD_MULTIRNG_PROMISING" not in paired_doc:
        raise ValueError("Paired-validation report does not document the required decision")
    if sorted(paired["rng"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Paired endpoint result must contain exactly RNG1-RNG5")
    if set(late["window"].astype(str)) != {"85-90K", "90-95K", "95-100K"}:
        raise ValueError("Late-update result must contain the three required windows")
    if len(decision) != 1:
        raise ValueError("Decision result must contain exactly one row")
    if int(float(decision.iloc[0]["n"])) != 5:
        raise ValueError("Decision result does not document n=5")

    return {
        "paired": paired,
        "late": late,
        "group": group,
        "decision": decision,
        "paired_doc": paired_doc,
    }


def aggregate_validation(path: Path, label: str) -> pd.DataFrame:
    frame = read_required(path)
    require_columns(
        frame,
        [
            "row_type",
            "checkpoint_step",
            "mean_yield_kg_ha",
            "mean_I_mm",
            "mean_N_kg_ha",
            "mean_PFP_N",
            "mean_WP_ET",
            "status",
            "validation_year_count",
        ],
        label,
    )
    aggregate = frame.loc[frame["row_type"].astype(str).str.lower().eq("aggregate")].copy()
    aggregate["checkpoint_k"] = (aggregate["checkpoint_step"].astype(int) / 1000).astype(int)
    aggregate = aggregate.set_index("checkpoint_k").sort_index()
    if set(aggregate.index.tolist()) != set(CHECKPOINTS):
        raise ValueError(f"{label} must contain aggregates for 75K-100K")
    if not aggregate["status"].astype(str).str.upper().eq("PASS").all():
        raise ValueError(f"{label} contains non-PASS validation status")
    if not aggregate["validation_year_count"].astype(float).eq(10).all():
        raise ValueError(f"{label} must contain ten validation years per checkpoint")
    return aggregate


def build_paired(inputs: dict[str, object]) -> pd.DataFrame:
    source = inputs["paired"].copy()
    out = source.rename(
        columns={
            "rng": "rng_id",
            "control_yield_100K": "control_100k_yield",
            "guard_yield_100K": "klguard_100k_yield",
            "control_collapse": "control_collapse_flag",
            "guard_collapse": "klguard_collapse_flag",
            "material_harm": "material_harm_flag",
        }
    )[
        [
            "rng_id",
            "control_100k_yield",
            "klguard_100k_yield",
            "delta_yield",
            "control_collapse_flag",
            "klguard_collapse_flag",
            "material_harm_flag",
        ]
    ].copy()
    out["rng_id"] = out["rng_id"].astype(int)
    for column in ["control_collapse_flag", "klguard_collapse_flag", "material_harm_flag"]:
        out[column] = out[column].map(bool_value)
    out["delta_yield"] = out["klguard_100k_yield"].astype(float) - out["control_100k_yield"].astype(float)
    return out.sort_values("rng_id").reset_index(drop=True)


def build_trajectories() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rng in range(1, 6):
        control = aggregate_validation(CONTROL_PATHS[rng], f"CONTROL RNG{rng} checkpoint validation")
        guard = aggregate_validation(GUARD_PATHS[rng], f"KL-GUARD RNG{rng} checkpoint validation")
        for checkpoint in CHECKPOINTS:
            for treatment, aggregate, source_path in [
                ("CONTROL", control, CONTROL_PATHS[rng]),
                ("KL-GUARD", guard, GUARD_PATHS[rng]),
            ]:
                record = aggregate.loc[checkpoint]
                rows.append(
                    {
                        "rng_id": rng,
                        "checkpoint_k": checkpoint,
                        "checkpoint_step": checkpoint * 1000,
                        "treatment": treatment,
                        "mean_yield_kg_ha": float(record["mean_yield_kg_ha"]),
                        "mean_I_mm": float(record["mean_I_mm"]),
                        "mean_N_kg_ha": float(record["mean_N_kg_ha"]),
                        "mean_PFP_N": float(record["mean_PFP_N"]),
                        "mean_WP_ET": float(record["mean_WP_ET"]),
                        "validation_years": "2014-2023",
                        "validation_status": str(record["status"]),
                        "source_file": rel(source_path),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["rng_id", "checkpoint_k", "treatment"]).reset_index(drop=True)
    if len(out) != 60:
        raise ValueError("Expected 5 RNG x 6 checkpoints x 2 treatments")
    return out


def build_late_summary(inputs: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = inputs["late"].copy()
    order = ["85-90K", "90-95K", "95-100K"]
    source["rng_id"] = source["rng"].astype(int)
    source["window"] = pd.Categorical(source["window"].astype(str), categories=order, ordered=True)
    source = source.sort_values(["rng_id", "window"])
    records: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []
    for _, row in source.iterrows():
        for treatment, mean_value, median_value in [
            ("CONTROL", row["control_approx_kl_mean"], row["control_approx_kl_median"]),
            ("KL-GUARD", row["klguard_approx_kl_mean"], row["klguard_approx_kl_median"]),
        ]:
            records.append(
                {
                    "rng_id": int(row["rng_id"]),
                    "window": str(row["window"]),
                    "treatment": treatment,
                    "approx_kl_mean": float(mean_value),
                    "approx_kl_median": float(median_value),
                    "metric_rows": int(row[f"{('control' if treatment == 'CONTROL' else 'klguard')}_metric_rows"]),
                    "source_file": rel(LATE_PATH),
                }
            )
        delta_records.append(
            {
                "rng_id": int(row["rng_id"]),
                "window": str(row["window"]),
                "delta_mean": float(row["klguard_approx_kl_mean"] - row["control_approx_kl_mean"]),
                "delta_median": float(row["klguard_approx_kl_median"] - row["control_approx_kl_median"]),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(delta_records)


def build_summary(paired: pd.DataFrame, inputs: dict[str, object]) -> pd.DataFrame:
    decision = inputs["decision"].iloc[0]
    records = {
        "scope": "PPO Original seed1; same 75K checkpoint; five paired RNG continuations",
        "n_rng": 5,
        "control_collapse_count": int(float(decision["control_collapse_count"])),
        "klguard_collapse_count": int(float(decision["klguard_collapse_count"])),
        "material_harm_count": int(float(decision["stable_rng_material_harm_count"])),
        "rng2_delta_yield": float(paired.loc[paired["rng_id"].eq(2), "delta_yield"].iloc[0]),
        "mean_delta_yield": float(paired["delta_yield"].mean()),
        "median_delta_yield": float(paired["delta_yield"].median()),
        "minimum_delta_yield": float(paired["delta_yield"].min()),
        "positive_delta_count": int((paired["delta_yield"] > 0).sum()),
        "rng2_status": str(decision["rng2_guard_status"]),
        "rng0_status": str(decision["rng0_status"]),
        "decision": str(decision["decision"]),
        "validation_years": "2014-2023",
        "source_file": rel(DECISION_PATH),
    }
    return pd.DataFrame([records])


def build_key_numbers(paired: pd.DataFrame, summary: pd.DataFrame, late_deltas: pd.DataFrame, inputs: dict[str, object]) -> pd.DataFrame:
    decision = inputs["decision"].iloc[0]
    rows = [
        ("decision", str(decision["decision"]), "status", rel(DECISION_PATH), "Descriptive n=5 continuation-set decision"),
        ("control_full_collapse", "1 / 5", "count", rel(DECISION_PATH), "CONTROL full collapse"),
        ("klguard_full_collapse", "0 / 5", "count", rel(DECISION_PATH), "KL-GUARD full collapse"),
        ("rng2_delta_yield", f"{float(summary.iloc[0]['rng2_delta_yield']):+.3f}", "kg ha^-1", rel(PAIRED_PATH), "RNG2 100K KL-GUARD minus CONTROL; partial rescue"),
        ("material_harm", "0 / 5", "count", rel(PAIRED_PATH), "Source material_harm flag count"),
        ("mean_delta_yield", f"{float(summary.iloc[0]['mean_delta_yield']):+.3f}", "kg ha^-1", rel(PAIRED_PATH), "Mean paired endpoint delta across RNG1-RNG5"),
        ("median_delta_yield", f"{float(summary.iloc[0]['median_delta_yield']):+.3f}", "kg ha^-1", rel(PAIRED_PATH), "Median paired endpoint delta across RNG1-RNG5"),
        ("rng2_status", str(decision["rng2_guard_status"]), "status", rel(DECISION_PATH), "RNG2 remains partial degradation relative to the ~8200 plateau"),
        ("late_approx_kl_scope", "descriptive only", "caveat", rel(LATE_PATH), "Not causal; mean/median results are not fully consistent across RNGs"),
        ("validation_years", "2014-2023", "years", rel(PAIRED_DOC), "Held-out deterministic masked validation"),
        ("cross_seed_claim", "not assessed", "scope", rel(PAIRED_DOC), "This page is seed1 multirng paired validation, not cross-seed generalization"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "source_file", "notes"])


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.tick_params(axis="both", which="major", length=3, width=0.7, color="#5A6470")
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.55, alpha=0.24)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_paired(paired: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.2), facecolor="white")
    xs = np.arange(1, 6)
    ax.axvspan(1.55, 2.45, color="#F7EFE3", alpha=0.82, zorder=0)
    for x, (_, row) in zip(xs, paired.iterrows()):
        control = float(row["control_100k_yield"])
        guard = float(row["klguard_100k_yield"])
        rng = int(row["rng_id"])
        ax.plot([x, x], [control, guard], color=ACCENT_COLOR if rng == 2 else "#C9D2D8", linewidth=2.6 if rng == 2 else 1.7, zorder=1)
        ax.scatter(x, control, s=78 if rng != 2 else 98, color=CONTROL_COLOR, edgecolor="white", linewidth=1.2, marker="o", zorder=3)
        ax.scatter(x, guard, s=80 if rng != 2 else 108, color=GUARD_COLOR, edgecolor="white", linewidth=1.2, marker="s", zorder=3)
        delta = float(row["delta_yield"])
        label = f"{delta:+,.3f}"
        label_y = max(control, guard) + (105 if rng != 2 else 45)
        ax.text(x, label_y, label, ha="center", va="bottom", fontsize=9.2, fontweight="bold" if rng == 2 else "normal", color=ACCENT_COLOR if rng == 2 else TEXT_COLOR)
    ax.annotate(
        "RNG2: full collapse to non-collapse\npartial rescue, not full restoration",
        xy=(2, float(paired.loc[paired["rng_id"].eq(2), "klguard_100k_yield"].iloc[0])),
        xytext=(2.72, 6900),
        fontsize=9.5,
        color=ACCENT_COLOR,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": ACCENT_COLOR, "lw": 1.0},
    )
    ax.set_xticks(xs, [f"RNG{i}" for i in xs])
    ax.set_xlim(0.45, 5.55)
    y_values = pd.concat([paired["control_100k_yield"], paired["klguard_100k_yield"]]).astype(float)
    ax.set_ylim(float(y_values.min()) - 300, float(y_values.max()) + 300)
    ax.set_xlabel("RNG continuation")
    ax.set_ylabel("100K mean yield (kg ha$^{-1}$)")
    ax.set_title(
        "Fixed KL-GUARD is promising across five paired RNG continuations\n"
        "PPO Original seed1 · same 75K checkpoint · held-out validation 2014–2023",
        loc="left",
        pad=13,
        fontweight="bold",
    )
    ax.text(0.02, 0.025, "Labels show Δyield = KL-GUARD − CONTROL (kg ha$^{-1}$)", transform=ax.transAxes, fontsize=8.7, color=MUTED_COLOR, va="bottom")
    ax.scatter([], [], color=CONTROL_COLOR, marker="o", s=70, label="CONTROL")
    ax.scatter([], [], color=GUARD_COLOR, marker="s", s=70, label="KL-GUARD")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.22)
    save_figure(fig, FIG1)


def plot_summary_cards(paired: pd.DataFrame, summary: pd.DataFrame) -> None:
    values = summary.iloc[0]
    fig = plt.figure(figsize=(8.6, 5.2), facecolor="white")
    fig.text(0.07, 0.93, "Five-RNG paired validation summary", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.07, 0.875, "Fixed KL-GUARD · PPO Original seed1 · same 75K checkpoint", fontsize=10.5, color=MUTED_COLOR)
    cards = [
        ("1 / 5", "CONTROL\nfull collapse", "one paired continuation collapsed", "#EEF4F9", CONTROL_COLOR),
        ("0 / 5", "KL-GUARD\nfull collapse", "no full collapse observed", "#F2EAF3", GUARD_COLOR),
        (f"{float(values['rng2_delta_yield']):+,.3f}", "RNG2 rescue at 100K\nkg ha$^{-1}$", "partial rescue, not full restoration", "#F6F0E8", ACCENT_COLOR),
        ("0 / 5", "material harm\nflagged", "source material_harm flags", "#F1F4F1", POSITIVE_COLOR),
    ]
    positions = [(0.07, 0.51), (0.53, 0.51), (0.07, 0.10), (0.53, 0.10)]
    for (big, label, note, face, accent), (left, bottom) in zip(cards, positions):
        ax = fig.add_axes([left, bottom, 0.40, 0.29])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02,rounding_size=0.045", linewidth=0, facecolor=face, transform=ax.transAxes))
        ax.add_patch(Rectangle((0.01, 0.01), 0.018, 0.98, transform=ax.transAxes, facecolor=accent, linewidth=0))
        ax.text(0.09, 0.64, big, transform=ax.transAxes, fontsize=25 if len(big) < 9 else 20, fontweight="bold", color=TEXT_COLOR, va="center")
        ax.text(0.09, 0.31, label, transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=TEXT_COLOR, va="center", linespacing=1.15)
        ax.text(0.09, 0.10, note, transform=ax.transAxes, fontsize=7.7, color=MUTED_COLOR, va="center")
    fig.text(0.07, 0.025, "n=5 continuation set · descriptive result · not cross-seed generalization", fontsize=8.7, color=MUTED_COLOR)
    save_figure(fig, FIG2)


def plot_trajectories(trajectories: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.0), facecolor="white")
    for rng in range(1, 6):
        for treatment, color in [("CONTROL", CONTROL_COLOR), ("KL-GUARD", GUARD_COLOR)]:
            one = trajectories.loc[(trajectories["rng_id"].eq(rng)) & (trajectories["treatment"].eq(treatment))].sort_values("checkpoint_k")
            ax.plot(one["checkpoint_k"], one["mean_yield_kg_ha"], color=color, linewidth=1.0 if rng != 2 else 1.8, alpha=0.18 if rng != 2 else 0.55, zorder=1)
    means = trajectories.groupby(["checkpoint_k", "treatment"], as_index=False)["mean_yield_kg_ha"].mean()
    for treatment, color, marker in [("CONTROL", CONTROL_COLOR, "o"), ("KL-GUARD", GUARD_COLOR, "s")]:
        one = means.loc[means["treatment"].eq(treatment)].sort_values("checkpoint_k")
        ax.plot(one["checkpoint_k"], one["mean_yield_kg_ha"], color=color, linewidth=3.1, marker=marker, markersize=6.3, markerfacecolor="white", markeredgewidth=1.4, label=f"{treatment} mean across RNG1–5", zorder=3)
    rng2_guard = trajectories.loc[(trajectories["rng_id"].eq(2)) & (trajectories["treatment"].eq("KL-GUARD"))].sort_values("checkpoint_k")
    ax.scatter([100], [float(rng2_guard.loc[rng2_guard["checkpoint_k"].eq(100), "mean_yield_kg_ha"].iloc[0])], s=110, color=ACCENT_COLOR, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate("RNG2 partial rescue", xy=(100, float(rng2_guard.loc[rng2_guard["checkpoint_k"].eq(100), "mean_yield_kg_ha"].iloc[0])), xytext=(91.3, 7050), fontsize=9.2, color=ACCENT_COLOR, fontweight="bold", arrowprops={"arrowstyle": "-", "color": ACCENT_COLOR, "lw": 0.9})
    ax.set_xticks(CHECKPOINTS, [f"{v}K" for v in CHECKPOINTS])
    ax.set_xlim(73.5, 102.7)
    ax.set_ylim(5000, 8400)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.set_title(
        "Across five continuations, the guard preserves the high-yield path and reduces full-collapse count from 1/5 to 0/5\n"
        "Bold lines: group means · light lines: individual paired continuations · same 75K checkpoint",
        loc="left",
        pad=13,
        fontweight="bold",
    )
    ax.text(0.02, 0.025, "RNG2 remains below the ~8200 kg ha$^{-1}$ plateau at 100K", transform=ax.transAxes, fontsize=8.7, color=MUTED_COLOR, va="bottom")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.15)
    save_figure(fig, FIG3)


def plot_late_kl(late: pd.DataFrame, late_deltas: pd.DataFrame) -> None:
    order = ["85-90K", "90-95K", "95-100K"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.9), sharey=True, facecolor="white")
    scale = 1e5
    scaled = late_deltas.copy()
    ymin = float(min(scaled["delta_mean"].min(), 0) * scale) - 2.5
    ymax = float(max(scaled["delta_mean"].max(), 0) * scale) + 2.5
    for ax, window in zip(axes, order):
        one = scaled.loc[scaled["window"].eq(window)].sort_values("rng_id")
        colors = [POSITIVE_COLOR if value < 0 else ACCENT_COLOR for value in one["delta_mean"]]
        bars = ax.bar(one["rng_id"].astype(str), one["delta_mean"] * scale, color=colors, width=0.62, edgecolor="white", linewidth=0.7)
        ax.axhline(0, color="#6C7680", linewidth=0.85)
        for bar, value in zip(bars, one["delta_mean"] * scale):
            va = "top" if value < 0 else "bottom"
            offset = -0.55 if value < 0 else 0.55
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f}", ha="center", va=va, fontsize=8.0, color=TEXT_COLOR)
        ax.set_title(window, fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax.set_xlabel("RNG")
        ax.set_ylim(ymin, ymax)
        style_axis(ax)
    axes[0].set_ylabel("Δ approx_kl × 10$^{5}$\n(KL-GUARD − CONTROL, mean)")
    fig.suptitle("Late-window approx_kl: guard reductions are descriptive and RNG-dependent", x=0.07, ha="left", y=1.02, fontsize=16, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.07, 0.93, "Negative values indicate lower guard approx_kl; RNG4 is not lower. Mean/median results are not fully consistent across RNGs.", fontsize=9.4, color=MUTED_COLOR)
    fig.text(0.99, 0.02, "descriptive only; not causal", ha="right", fontsize=9.0, color=MUTED_COLOR)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.20, wspace=0.15)
    save_figure(fig, FIG4)


def write_report(paired: pd.DataFrame, summary: pd.DataFrame, trajectories: pd.DataFrame, late: pd.DataFrame, late_deltas: pd.DataFrame, key_numbers: pd.DataFrame, inputs: dict[str, object]) -> None:
    decision = inputs["decision"].iloc[0]
    endpoint_table = paired.copy()
    endpoint_table["control_100k_yield"] = endpoint_table["control_100k_yield"].map(lambda value: f"{float(value):.3f}")
    endpoint_table["klguard_100k_yield"] = endpoint_table["klguard_100k_yield"].map(lambda value: f"{float(value):.3f}")
    endpoint_table["delta_yield"] = endpoint_table["delta_yield"].map(lambda value: f"{float(value):+.3f}")
    endpoint_table.columns = ["RNG", "CONTROL 100K", "KL-GUARD 100K", "Δyield", "CONTROL collapse", "KL-GUARD collapse", "material harm"]
    late_view = late.copy()
    late_view["approx_kl_mean"] = late_view["approx_kl_mean"].map(lambda value: f"{float(value):.8f}")
    late_view["approx_kl_median"] = late_view["approx_kl_median"].map(lambda value: f"{float(value):.8f}")
    late_view = late_view[["rng_id", "window", "treatment", "approx_kl_mean", "approx_kl_median", "metric_rows"]]
    late_view.columns = ["RNG", "window", "treatment", "approx_kl mean", "approx_kl median", "metric rows"]
    trajectory_counts = trajectories.groupby(["rng_id", "treatment"]).size().to_dict()
    late_delta_table = late_deltas.copy()
    late_delta_table["delta_mean"] = late_delta_table["delta_mean"].map(lambda value: f"{float(value):+.8f}")
    late_delta_table["delta_median"] = late_delta_table["delta_median"].map(lambda value: f"{float(value):+.8f}")
    late_delta_table.columns = ["RNG", "window", "Δmean approx_kl", "Δmedian approx_kl"]
    lines = [
        "# YC PPO Slide 4 fixed KL-GUARD multirng visual assets audit",
        "",
        "## 状态",
        "",
        "- `KL_GUARD_MULTIRNG_PROMISING`",
        "- 本轮只整理已有 024 五-RNG paired validation 结果，未训练、未重跑 validation、未重跑 DSSAT、未调参、未修改 PPO/DSSAT 逻辑，也未制作最终组会 PPT。",
        "- 已生成一个明确标注 `BACKUP ONLY`、`NOT the final group-meeting deck` 的归档 PPTX。",
        "",
        "## 1. 数据来源与范围",
        "",
        f"- 主语义报告：`{rel(PAIRED_DOC)}`。该报告给出 seed1、同一 75K checkpoint、RNG1–RNG5 paired continuation 的主判定。",
        f"- 端点配对：`{rel(PAIRED_PATH)}`。Figure 1、summary cards 和 endpoint 数字以此为主。",
        f"- late update：`{rel(LATE_PATH)}`。Figure 4 使用已有 per-RNG late-window approx_kl mean，并同时保留 median。",
        f"- group summary：`{rel(GROUP_PATH)}`；decision：`{rel(DECISION_PATH)}`。",
        "- checkpoint trajectory：CONTROL 使用已有 `yc_ppo_s1_control_checkpoint_validation.csv`（RNG1）和 `yc_ppo_s1_control_rng2/3/4/5_checkpoint_validation.csv`；KL-GUARD 使用已有 `yc_ppo_s1_rng1/2/3/4/5_klguard_checkpoint_validation.csv`。每个文件仅读取 75K–100K aggregate 行。",
        "- RNG2 的 KL-GUARD 分支沿用 023 已有 continuation；RNG1/3/4/5 来自 024。主分析仍以 024 报告定义的五-RNG paired validation 为准。",
        "",
        "## 2. 是否只读既有结果",
        "",
        "- **YES**。本轮没有新训练、没有 validation/DSSAT 重跑、没有正式实验、没有新的调参或统计检验。",
        "- 仅用既有 endpoint、late-update、group summary、decision 和 checkpoint validation CSV 计算展示表与图。",
        "- 训练年份、validation 年份、观测/奖励/动作/mask、灌溉/施肥约束和 PPO/DSSAT 逻辑均未改动。",
        "",
        "## 3. 关键数字核对",
        "",
        endpoint_table.to_markdown(index=False),
        "",
        f"- CONTROL full collapse: `{int(float(decision['control_collapse_count']))}/5`。",
        f"- KL-GUARD full collapse: `{int(float(decision['klguard_collapse_count']))}/5`。",
        f"- RNG2 Δyield at 100K: `{float(summary.iloc[0]['rng2_delta_yield']):+.3f} kg ha^-1`。",
        f"- Other four RNGs: `RNG1 {float(paired.loc[paired.rng_id.eq(1), 'delta_yield'].iloc[0]):+.3f}`, `RNG3 {float(paired.loc[paired.rng_id.eq(3), 'delta_yield'].iloc[0]):+.3f}`, `RNG4 {float(paired.loc[paired.rng_id.eq(4), 'delta_yield'].iloc[0]):+.3f}`, `RNG5 {float(paired.loc[paired.rng_id.eq(5), 'delta_yield'].iloc[0]):+.3f} kg ha^-1`，均为小幅变化。",
        f"- Material harm: `{int(float(decision['stable_rng_material_harm_count']))}/5`，按 024 paired CSV 的 `material_harm` 字段直接计数，未另造阈值。",
        f"- Mean/median paired Δyield: `{float(summary.iloc[0]['mean_delta_yield']):+.3f}` / `{float(summary.iloc[0]['median_delta_yield']):+.3f} kg ha^-1`。",
        f"- 024 decision：`{decision['decision']}`；RNG2 状态：`{decision['rng2_guard_status']}`。",
        "",
        "## 4. 图表含义",
        "",
        "- Figure 1 `slide4_multirng_paired_100k`：每个 RNG 的 CONTROL 与 KL-GUARD 100K mean yield 用连接线配对；RNG2 单独强调，标出 `full collapse to non-collapse` 和 `partial rescue, not full restoration`。",
        "- Figure 2 `slide4_multirng_summary_cards`：四个必需卡片为 CONTROL `1/5`、KL-GUARD `0/5`、RNG2 `+2,124.202`、material harm `0/5`。",
        "- Figure 3 `slide4_multirng_checkpoint_trajectories`：粗线是 RNG1–RNG5 的组均值，浅线是个体 paired continuation；RNG2 100K 仍低于约 8200 kg ha^-1 plateau。",
        "- Figure 4 `slide4_multirng_late_approx_kl`：展示每个 late window 的 `KL-GUARD − CONTROL` approx_kl mean；RNG1/2/3/5 为负，RNG4 为正。图中明确 `descriptive only; not causal`，并说明 mean/median across RNGs 并非完全一致。",
        "",
        "## 5. 输出 CSV",
        "",
        f"- `{rel(PAIRED_OUT)}`：每个 RNG 一行，含 endpoint、collapse 和 material-harm flags。",
        f"- `{rel(SUMMARY_OUT)}`：聚合 collapse、material harm、RNG2、mean/median Δyield。",
        f"- `{rel(TRAJECTORIES_OUT)}`：5 RNG × 6 checkpoint × 2 treatment，共 {len(trajectories)} 行。",
        f"- `{rel(LATE_OUT)}`：late-window per-RNG × treatment approx_kl mean/median。",
        f"- `{rel(KEY_NUMBERS_OUT)}`：PPT 手工排版所需简洁数字和边界。",
        "",
        "## 6. 推荐本页结论",
        "",
        "`KL_GUARD_MULTIRNG_PROMISING`。Fixed KL-GUARD 在 seed1 的五条 paired RNG continuation 中把 full collapse 从 1/5 降到 0/5，主要贡献来自 RNG2 的显著 partial rescue，其他四条 continuation 变化很小且未见 material harm。这个结论只适用于 seed1 multirng paired validation。",
        "",
        "必须保留的限定：",
        "- `RNG2 remains partial degradation relative to the ~8200 plateau`。",
        "- `approx_kl evidence is descriptive only`，不能说 KL-GUARD 证明了因果关系。",
        "- `this page is seed1 multirng paired validation, not cross-seed generalization`。",
        "",
        "## 7. 建议的讲述顺序",
        "",
        "1. 上一页在单条已知 collapse 轨迹 RNG2 上看到 fixed KL-GUARD 能部分救回。",
        "2. 本页进一步问：这是不是单条轨迹偶然现象。",
        "3. 在同一个 75K checkpoint 上比较五条 paired RNG continuation。",
        "4. 结果是 CONTROL `1/5` collapse，KL-GUARD `0/5`，RNG2 `+2124.202 kg ha^-1`，其余四条近乎不变且未见 material harm。",
        "5. 因而本页给出 `KL_GUARD_MULTIRNG_PROMISING`，但不是 fully solved；下一自然问题是能否跨 seed 泛化。",
        "",
        "## 8. 不应过度解读",
        "",
        "- `0/5 collapse` 是这个 seed1 continuation set 的描述性结果，不是总体 collapse rate，也不是普适有效性证明。",
        "- RNG2 是 partial rescue，不是 fully restored；100K 仍低于约 8200 kg ha^-1 plateau。",
        "- `material harm: 0/5` 直接沿用 024 paired result 的 material-harm flag 口径，不等于证明所有资源指标或所有场景均无影响。",
        "- approx_kl 的方向性支持是 descriptive only；RNG4 未降低，不能总结为统一定律。",
        "",
        "## 9. 归档 PPT 与 GitHub 备份",
        "",
        f"- 归档 PPT：`docs/presentation_yc_ppo_slide4_multirng_assets_backup_only.pptx`，首页明确 `BACKUP ONLY` 和 `NOT the final group-meeting deck`。",
        "- 待备份文件列表：本审计 md、5 个 CSV、`results/presentation_assets/slide4/` 下 4 组 PNG/SVG、归档 PPTX，以及本轮生成脚本。",
        "- GitHub push：本轮未执行。项目保护规则要求远程写入前取得明确授权；本地文件已保留，未将大型模型、缓存或原始 daily 输出加入备份。",
        "",
        "## 10. 交付 marker",
        "",
        "`YC_PPO_SLIDE4_MULTIRNG_VISUAL_ASSETS_READY`",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs() -> dict[str, int]:
    files = [PAIRED_OUT, SUMMARY_OUT, TRAJECTORIES_OUT, LATE_OUT, KEY_NUMBERS_OUT, REPORT_OUT]
    files.extend(Path(f"{stem}.png") for stem in [FIG1, FIG2, FIG3, FIG4])
    files.extend(Path(f"{stem}.svg") for stem in [FIG1, FIG2, FIG3, FIG4])
    sizes: dict[str, int] = {}
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
        sizes[rel(path)] = path.stat().st_size
    return sizes


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()
    paired = build_paired(inputs)
    trajectories = build_trajectories()
    late, late_deltas = build_late_summary(inputs)
    summary = build_summary(paired, inputs)
    key_numbers = build_key_numbers(paired, summary, late_deltas, inputs)

    paired.to_csv(PAIRED_OUT, index=False, float_format="%.9f")
    summary.to_csv(SUMMARY_OUT, index=False, float_format="%.9f")
    trajectories.to_csv(TRAJECTORIES_OUT, index=False, float_format="%.9f")
    late.to_csv(LATE_OUT, index=False, float_format="%.12f")
    key_numbers.to_csv(KEY_NUMBERS_OUT, index=False)

    plot_paired(paired)
    plot_summary_cards(paired, summary)
    plot_trajectories(trajectories)
    plot_late_kl(late, late_deltas)
    write_report(paired, summary, trajectories, late, late_deltas, key_numbers, inputs)
    print({"status": "PASS", "outputs": verify_outputs(), "marker": "YC_PPO_SLIDE4_MULTIRNG_VISUAL_ASSETS_READY"})


if __name__ == "__main__":
    main()

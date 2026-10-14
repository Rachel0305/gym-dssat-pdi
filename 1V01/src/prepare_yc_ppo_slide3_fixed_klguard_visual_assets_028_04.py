"""Prepare Slide 3 visual assets from existing YC PPO rescue evidence.

This script only reads existing result files, derives presentation tables, and
renders slide-ready PNGs. It does not train, run validation, run DSSAT, or
modify any source result file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

RESCUE_DOC = DOCS / "yc_ppo_original_seed1_rng2_kl_guard_rescue.md"
TRAINING_PATH = RESULTS / "yc_ppo_s1_rng2_klguard_training_metrics.csv"
GUARD_VALIDATION_PATH = RESULTS / "yc_ppo_s1_rng2_klguard_checkpoint_validation.csv"
CONTROL_VALIDATION_PATH = RESULTS / "yc_ppo_s1_control_rng2_checkpoint_validation.csv"
COMPARISON_PATH = RESULTS / "yc_ppo_s1_rng2_control_vs_klguard.csv"
LATE_PATH = RESULTS / "yc_ppo_s1_rng2_control_vs_klguard_late_update_summary.csv"
DECISION_PATH = RESULTS / "yc_ppo_s1_rng2_klguard_rescue_decision.csv"
PAIRED_DOC = DOCS / "yc_ppo_original_seed1_5rng_klguard_paired_validation.md"

TRAJECTORY_OUT = RESULTS / "presentation_yc_ppo_slide3_rng2_control_vs_klguard_yield_trajectory.csv"
TRAJECTORY_PNG = RESULTS / "presentation_yc_ppo_slide3_rng2_control_vs_klguard_yield_trajectory.png"
LATE_OUT = RESULTS / "presentation_yc_ppo_slide3_rng2_late_update_summary.csv"
LATE_PNG = RESULTS / "presentation_yc_ppo_slide3_rng2_late_update_summary.png"
ENDPOINT_OUT = RESULTS / "presentation_yc_ppo_slide3_rng2_endpoint_table.csv"
KEY_NUMBERS_OUT = RESULTS / "presentation_yc_ppo_slide3_key_numbers.csv"
REPORT_OUT = DOCS / "presentation_yc_ppo_slide3_fixed_klguard_assets.md"

CHECKPOINTS = [75, 80, 85, 90, 95, 100]
CONTROL_COLOR = "#506A80"
GUARD_COLOR = "#865486"
ACCENT_COLOR = "#A06B32"
TEXT_COLOR = "#1B2733"
MUTED_COLOR = "#5C6873"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty evidence file: {path}")
    return pd.read_csv(path)


def read_inputs() -> dict[str, pd.DataFrame | str]:
    for path in [RESCUE_DOC, TRAINING_PATH, GUARD_VALIDATION_PATH, CONTROL_VALIDATION_PATH, COMPARISON_PATH, LATE_PATH, DECISION_PATH, PAIRED_DOC]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty evidence file: {path}")

    training = read_required(TRAINING_PATH)
    guard_validation = read_required(GUARD_VALIDATION_PATH)
    control_validation = read_required(CONTROL_VALIDATION_PATH)
    comparison = read_required(COMPARISON_PATH)
    late = read_required(LATE_PATH)
    decision = read_required(DECISION_PATH)
    rescue_doc = RESCUE_DOC.read_text(encoding="utf-8")
    paired_doc = PAIRED_DOC.read_text(encoding="utf-8")

    require_columns(
        training,
        ["global_step", "approx_kl", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered"],
        "KL-GUARD training metrics",
    )
    aggregate_columns = [
        "row_type",
        "checkpoint_step",
        "mean_yield_kg_ha",
        "mean_I_mm",
        "mean_N_kg_ha",
        "mean_PFP_N",
        "mean_WP_ET",
        "status",
        "validation_year_count",
    ]
    require_columns(guard_validation, aggregate_columns, "KL-GUARD validation")
    require_columns(control_validation, aggregate_columns, "CONTROL validation")
    require_columns(comparison, ["checkpoint_step", "control_mean_yield", "klguard_mean_yield", "control_mean_I", "klguard_mean_I", "control_mean_N", "klguard_mean_N"], "CONTROL vs KL-GUARD comparison")
    require_columns(late, ["window", "control_approx_kl_median", "klguard_approx_kl_median", "control_approx_kl_mean", "klguard_approx_kl_mean"], "late update summary")
    require_columns(
        decision,
        [
            "decision",
            "control_rng2_100k_mean_yield",
            "control_rng2_collapse_unchanged",
            "klguard_100k_mean_yield",
            "improvement_vs_control_100k",
            "effective_kl_threshold",
            "klguard_trigger_rate",
            "klguard_updates_with_optimizer_step_fraction",
            "klguard_not_frozen",
        ],
        "rescue decision",
    )

    if "2014–2023" not in rescue_doc and "2014-2023" not in rescue_doc:
        raise ValueError("Rescue report does not document 2014-2023 validation")
    if "KL_GUARD_MULTIRNG_PROMISING" not in paired_doc:
        raise ValueError("Paired validation report could not be used as semantic background")

    for frame, label in [(guard_validation, "KL-GUARD validation"), (control_validation, "CONTROL validation")]:
        aggregate = frame.loc[frame["row_type"].astype(str).str.lower().eq("aggregate")].copy()
        if set(aggregate["checkpoint_step"].astype(int)) != {value * 1000 for value in CHECKPOINTS}:
            raise ValueError(f"{label} does not contain exactly 75K-100K aggregate checkpoints")
        if not aggregate["status"].astype(str).str.upper().eq("PASS").all():
            raise ValueError(f"{label} has a non-PASS aggregate status")
        if not aggregate["validation_year_count"].astype(float).eq(10).all():
            raise ValueError(f"{label} does not contain ten validation years per checkpoint")

    if set(late["window"].astype(str)) != {"85-90K", "90-95K", "95-100K"}:
        raise ValueError("Late update summary does not contain the three required windows")
    if len(decision) != 1:
        raise ValueError("Rescue decision CSV must contain exactly one decision row")
    if not bool_from_value(decision.iloc[0]["klguard_not_frozen"]):
        raise ValueError("KL-GUARD decision does not document a non-frozen result")

    return {
        "training": training,
        "guard_validation": guard_validation,
        "control_validation": control_validation,
        "comparison": comparison,
        "late": late,
        "decision": decision,
        "rescue_doc": rescue_doc,
        "paired_doc": paired_doc,
    }


def bool_from_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["row_type"].astype(str).str.lower().eq("aggregate")].copy()
    out["checkpoint_k"] = (out["checkpoint_step"].astype(int) / 1000).astype(int)
    return out.set_index("checkpoint_k").sort_index()


def build_trajectory(inputs: dict[str, pd.DataFrame | str]) -> tuple[pd.DataFrame, bool]:
    guard = aggregate_frame(inputs["guard_validation"])
    control = aggregate_frame(inputs["control_validation"])
    comparison = inputs["comparison"].copy()
    comparison["checkpoint_k"] = (comparison["checkpoint_step"].astype(int) / 1000).astype(int)
    comparison = comparison.set_index("checkpoint_k")
    rows = []
    conflict = False
    for checkpoint in CHECKPOINTS:
        c = control.loc[checkpoint]
        g = guard.loc[checkpoint]
        row = {
            "checkpoint_k": checkpoint,
            "checkpoint_step": checkpoint * 1000,
            "control_mean_yield_kg_ha": float(c["mean_yield_kg_ha"]),
            "klguard_mean_yield_kg_ha": float(g["mean_yield_kg_ha"]),
            "delta_yield_kg_ha": float(g["mean_yield_kg_ha"]) - float(c["mean_yield_kg_ha"]),
            "control_mean_I_mm": float(c["mean_I_mm"]),
            "klguard_mean_I_mm": float(g["mean_I_mm"]),
            "control_mean_N_kg_ha": float(c["mean_N_kg_ha"]),
            "klguard_mean_N_kg_ha": float(g["mean_N_kg_ha"]),
            "control_mean_PFP_N": float(c["mean_PFP_N"]),
            "klguard_mean_PFP_N": float(g["mean_PFP_N"]),
            "control_mean_WP_ET": float(c["mean_WP_ET"]),
            "klguard_mean_WP_ET": float(g["mean_WP_ET"]),
            "validation_years": "2014-2023",
            "validation_status": "PASS",
            "control_source_file": str(CONTROL_VALIDATION_PATH.relative_to(ROOT)),
            "klguard_source_file": str(GUARD_VALIDATION_PATH.relative_to(ROOT)),
        }
        if checkpoint in comparison.index:
            cmp = comparison.loc[checkpoint]
            checks = {
                "control_mean_yield_kg_ha": float(cmp["control_mean_yield"]),
                "klguard_mean_yield_kg_ha": float(cmp["klguard_mean_yield"]),
                "control_mean_I_mm": float(cmp["control_mean_I"]),
                "klguard_mean_I_mm": float(cmp["klguard_mean_I"]),
                "control_mean_N_kg_ha": float(cmp["control_mean_N"]),
                "klguard_mean_N_kg_ha": float(cmp["klguard_mean_N"]),
            }
            for key, expected in checks.items():
                if abs(float(row[key]) - expected) > 1e-6:
                    conflict = True
        rows.append(row)
    if conflict:
        raise ValueError("Checkpoint validation and control-vs-KL-GUARD comparison contain a numeric conflict")
    return pd.DataFrame(rows), conflict


def build_late_summary(inputs: dict[str, pd.DataFrame | str]) -> pd.DataFrame:
    source = inputs["late"].copy()
    order = ["85-90K", "90-95K", "95-100K"]
    source["window"] = pd.Categorical(source["window"].astype(str), categories=order, ordered=True)
    source = source.sort_values("window")
    out = source[["window", "control_approx_kl_median", "klguard_approx_kl_median", "control_approx_kl_mean", "klguard_approx_kl_mean", "control_metric_rows", "klguard_metric_rows"]].copy()
    out["window"] = out["window"].astype(str)
    out["delta_median_klguard_minus_control"] = out["klguard_approx_kl_median"].astype(float) - out["control_approx_kl_median"].astype(float)
    out["metric_definition"] = "approx_kl median; descriptive only"
    out["source_file"] = str(LATE_PATH.relative_to(ROOT))
    return out


def build_endpoint_table(trajectory: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    endpoint = trajectory.loc[trajectory["checkpoint_k"].eq(100)].iloc[0]
    decision_row = decision.iloc[0]
    control_yield = float(decision_row["control_rng2_100k_mean_yield"])
    guard_yield = float(decision_row["klguard_100k_mean_yield"])
    if abs(float(endpoint["control_mean_yield_kg_ha"]) - control_yield) > 1e-6 or abs(float(endpoint["klguard_mean_yield_kg_ha"]) - guard_yield) > 1e-6:
        raise ValueError("Decision CSV and trajectory endpoint conflict")
    rows = [
        {
            "branch": "CONTROL",
            "mean_yield_100k": control_yield,
            "mean_irrigation_100k": float(endpoint["control_mean_I_mm"]),
            "mean_n_100k": float(endpoint["control_mean_N_kg_ha"]),
            "pfp_n_100k": float(endpoint["control_mean_PFP_N"]),
            "wp_et_100k": float(endpoint["control_mean_WP_ET"]),
            "collapse_status": "full-collapse",
            "delta_yield_vs_control": 0.0,
        },
        {
            "branch": "KL-GUARD",
            "mean_yield_100k": guard_yield,
            "mean_irrigation_100k": float(endpoint["klguard_mean_I_mm"]),
            "mean_n_100k": float(endpoint["klguard_mean_N_kg_ha"]),
            "pfp_n_100k": float(endpoint["klguard_mean_PFP_N"]),
            "wp_et_100k": float(endpoint["klguard_mean_WP_ET"]),
            "collapse_status": "non-collapse / partial rescue",
            "delta_yield_vs_control": float(decision_row["improvement_vs_control_100k"]),
        },
    ]
    return pd.DataFrame(rows)


def save_axis_style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.tick_params(axis="both", which="major", length=3, width=0.7, color="#5A6470")
    ax.grid(axis=grid_axis, color="#B5BEC7", linewidth=0.55, alpha=0.24)
    ax.set_axisbelow(True)


def plot_main(trajectory: pd.DataFrame, decision: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.8), facecolor="white")
    x = trajectory["checkpoint_k"].astype(int)
    control = trajectory["control_mean_yield_kg_ha"].astype(float)
    guard = trajectory["klguard_mean_yield_kg_ha"].astype(float)
    ax.plot(x, control, color=CONTROL_COLOR, linewidth=2.7, marker="o", markersize=5.8, markerfacecolor="white", markeredgewidth=1.4, label="CONTROL")
    ax.plot(x, guard, color=GUARD_COLOR, linewidth=2.7, marker="s", markersize=5.8, markerfacecolor="white", markeredgewidth=1.4, label="KL-GUARD")
    ax.scatter([100], [float(control.iloc[-1])], s=95, color=CONTROL_COLOR, edgecolor="white", linewidth=1.1, zorder=5)
    ax.scatter([100], [float(guard.iloc[-1])], s=105, color=GUARD_COLOR, edgecolor="white", linewidth=1.1, zorder=5)
    delta = float(decision.iloc[0]["improvement_vs_control_100k"])
    ax.annotate("CONTROL: full collapse", xy=(100, float(control.iloc[-1])), xytext=(100.9, float(control.iloc[-1]) + 120), color=CONTROL_COLOR, fontsize=9, fontweight="bold", va="center")
    ax.annotate(f"KL-GUARD: +{delta:,.3f} kg ha$^{{-1}}$", xy=(100, float(guard.iloc[-1])), xytext=(100.9, float(guard.iloc[-1]) + 65), color=GUARD_COLOR, fontsize=9, fontweight="bold", va="center")
    ax.annotate("same known RNG2 continuation", xy=(75, float(control.iloc[0])), xytext=(76.0, float(control.iloc[0]) + 220), fontsize=8.5, color="#3C4650", arrowprops={"arrowstyle": "-", "color": "#66727D", "lw": 0.8})
    ax.set_xticks(CHECKPOINTS, [f"{v}K" for v in CHECKPOINTS])
    ax.set_xlim(73.5, 109.0)
    combined = pd.concat([control, guard], ignore_index=True)
    low, high = float(combined.min()), float(combined.max())
    margin = max((high - low) * 0.08, 150.0)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.set_title(
        "PPO known-collapse trajectory: CONTROL vs fixed KL-GUARD\n"
        "Known seed1 / RNG2 continuation · held-out validation, 2014–2023",
        loc="left",
        pad=12,
        fontweight="bold",
    )
    ax.text(0.01, 0.03, "Same 75K checkpoint; only the fixed KL-based update early stop differs", transform=ax.transAxes, fontsize=8.2, color=MUTED_COLOR, va="bottom")
    save_axis_style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.22)
    fig.savefig(TRAJECTORY_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_late(late: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor="white")
    x = list(range(len(late)))
    ax.plot(x, late["control_approx_kl_median"].astype(float), color=CONTROL_COLOR, linewidth=2.4, marker="o", markersize=6, label="CONTROL")
    ax.plot(x, late["klguard_approx_kl_median"].astype(float), color=GUARD_COLOR, linewidth=2.4, marker="s", markersize=6, label="KL-GUARD")
    ax.set_xticks(x, late["window"])
    ax.set_ylabel("approx_kl median")
    ax.set_xlabel("Late training window")
    ax.set_title(
        "Late-window approx_kl: CONTROL vs fixed KL-GUARD\n"
        "Descriptive only · lower guard values align with reduced late-window instability",
        loc="left",
        pad=12,
        fontweight="bold",
    )
    ax.text(0.99, 0.03, "descriptive only; not causal", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED_COLOR)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-4, -4))
    ax.yaxis.set_major_formatter(formatter)
    save_axis_style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(LATE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_setup_card(ax: plt.Axes, compact: bool = False) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02,rounding_size=0.035", linewidth=0.8, edgecolor="#D4DCE2", facecolor="#FBFCFD", transform=ax.transAxes))
    ax.text(0.055, 0.93, "为什么这算一次固定干预", transform=ax.transAxes, fontsize=13 if not compact else 9.5, fontweight="bold", color=TEXT_COLOR, va="top")
    rows = [
        ("Known failure", "seed1 / RNG2 / CONTROL"),
        ("Starting point", "same 75K checkpoint"),
        ("Intervention", "fixed KL guard only"),
        ("Effective threshold", "1.0e-04"),
        ("Unchanged", "PPO contract, inputs, constraints"),
        ("Guard status", "triggered and not frozen"),
    ]
    top = 0.80
    row_height = 0.115 if not compact else 0.13
    for idx, (key, value) in enumerate(rows):
        y = top - idx * row_height
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.045, y - row_height + 0.01), 0.91, row_height, transform=ax.transAxes, facecolor="#F3F6F8", linewidth=0))
        ax.text(0.065, y - 0.045, key, transform=ax.transAxes, fontsize=8.2 if not compact else 5.9, color=MUTED_COLOR, va="center")
        ax.text(0.44, y - 0.045, value, transform=ax.transAxes, fontsize=8.2 if not compact else 5.9, color=TEXT_COLOR, va="center")


def plot_key_numbers(repro: pd.DataFrame, decision: pd.DataFrame) -> None:
    row = decision.iloc[0]
    control = float(row["control_rng2_100k_mean_yield"])
    guard = float(row["klguard_100k_mean_yield"])
    delta = float(row["improvement_vs_control_100k"])
    threshold = float(row["effective_kl_threshold"])
    trigger = float(row["klguard_trigger_rate"])
    fig = plt.figure(figsize=(5.8, 4.8), facecolor="white")
    fig.text(0.08, 0.93, "固定 KL guard 的 rescue 结果", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.08, 0.875, "Known collapse · RNG2 · 100K endpoint", fontsize=9.5, color=MUTED_COLOR)
    cards = [
        (f"{control:,.3f}", "CONTROL 100K yield\nkg ha$^{-1}$", "#EEF4F9", CONTROL_COLOR, "full collapse"),
        (f"{guard:,.3f}", "KL-GUARD 100K yield\nkg ha$^{-1}$", "#F2EAF3", GUARD_COLOR, "partial rescue"),
        (f"+{delta:,.3f}", "yield recovered vs CONTROL\nkg ha$^{-1}$", "#F6F0E8", ACCENT_COLOR, "same known trajectory"),
        (f"{threshold:.1e}", "effective KL threshold\ntrigger rate = " + f"{trigger:.4f}", "#F1F4F1", "#4C6C5B", "guard not frozen"),
    ]
    positions = [(0.08, 0.52), (0.54, 0.52), (0.08, 0.08), (0.54, 0.08)]
    for (big, label, face, accent, note), (left, bottom) in zip(cards, positions):
        ax = fig.add_axes([left, bottom, 0.38, 0.31])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02,rounding_size=0.045", linewidth=0, facecolor=face, transform=ax.transAxes))
        ax.add_patch(Rectangle((0.01, 0.01), 0.014, 0.98, transform=ax.transAxes, facecolor=accent, linewidth=0))
        ax.text(0.09, 0.62, big, transform=ax.transAxes, fontsize=24, fontweight="bold", color=TEXT_COLOR, va="center")
        ax.text(0.09, 0.25, label, transform=ax.transAxes, fontsize=8.2, color="#4B5966", va="center", linespacing=1.25)
        ax.text(0.09, 0.10, note, transform=ax.transAxes, fontsize=7.0, color="#63717C", va="center")
    fig.savefig(RESULTS / "presentation_yc_ppo_slide3_key_numbers_visual.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_key_numbers(repro: pd.DataFrame, decision: pd.DataFrame) -> None:
    row = decision.iloc[0]
    records = [
        {"metric": "control_100k_yield", "value": float(row["control_rng2_100k_mean_yield"]), "unit": "kg/ha", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": "CONTROL RNG2 100K mean yield"},
        {"metric": "klguard_100k_yield", "value": float(row["klguard_100k_mean_yield"]), "unit": "kg/ha", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": "KL-GUARD RNG2 100K mean yield"},
        {"metric": "delta_yield_100k", "value": float(row["improvement_vs_control_100k"]), "unit": "kg/ha", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": "KL-GUARD minus CONTROL at 100K"},
        {"metric": "control_collapse_status", "value": "full-collapse", "unit": "status", "source_file": str(RESCUE_DOC.relative_to(ROOT)), "notes": "Known CONTROL collapse retained"},
        {"metric": "klguard_collapse_status", "value": "non-collapse / partial rescue", "unit": "status", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": "Decision is RNG2_KL_GUARD_PARTIAL_RESCUE"},
        {"metric": "effective_kl_threshold", "value": float(row["effective_kl_threshold"]), "unit": "approx_kl", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": "Verified effective threshold"},
        {"metric": "guard_triggered", "value": "TRUE", "unit": "status", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": f"trigger_rate={float(row['klguard_trigger_rate']):.6f}"},
        {"metric": "guard_not_frozen", "value": "TRUE", "unit": "status", "source_file": str(DECISION_PATH.relative_to(ROOT)), "notes": f"optimizer_step_fraction={float(row['klguard_updates_with_optimizer_step_fraction']):.6f}"},
        {"metric": "primary_conclusion", "value": "Fixed KL guard partially rescued a known collapsing trajectory.", "unit": "text", "source_file": str(RESCUE_DOC.relative_to(ROOT)), "notes": "Slide-ready conclusion; descriptive rescue result"},
    ]
    pd.DataFrame(records).to_csv(KEY_NUMBERS_OUT, index=False, float_format="%.9f")


def write_report(inputs: dict[str, pd.DataFrame | str], trajectory: pd.DataFrame, late: pd.DataFrame, endpoint: pd.DataFrame, conflict: bool) -> None:
    row = inputs["decision"].iloc[0]
    trajectory_table = trajectory[["checkpoint_k", "control_mean_yield_kg_ha", "klguard_mean_yield_kg_ha", "delta_yield_kg_ha"]].copy()
    trajectory_table.columns = ["checkpoint", "CONTROL mean yield (kg/ha)", "KL-GUARD mean yield (kg/ha)", "delta (kg/ha)"]
    late_table = late[["window", "control_approx_kl_median", "klguard_approx_kl_median", "delta_median_klguard_minus_control"]].copy()
    late_table.columns = ["window", "CONTROL approx_kl median", "KL-GUARD approx_kl median", "delta (guard - control)"]
    lines = [
        "# YC PPO Slide 3 fixed KL-GUARD rescue visual assets",
        "",
        "## 状态",
        "",
        "- `YC_PPO_SLIDE3_VISUAL_ASSETS_READY`",
        "- 本轮只整理既有结果，未训练、未重跑 validation/DSSAT、未调参、未修改 PPO/DSSAT/reward/action/mask 逻辑，也未制作正式组会 PPT。",
        "- 所有数值来自既有 CSV；75K 使用两份 checkpoint validation aggregate，80K–100K 另外与 CONTROL vs KL-GUARD comparison CSV 做了一致性核验。",
        "",
        "## 1. 数据来源",
        "",
        f"- Rescue semantics and contract: `{RESCUE_DOC.relative_to(ROOT)}`",
        f"- KL-GUARD training metrics: `{TRAINING_PATH.relative_to(ROOT)}`",
        f"- KL-GUARD validation: `{GUARD_VALIDATION_PATH.relative_to(ROOT)}`",
        f"- CONTROL reference validation: `{CONTROL_VALIDATION_PATH.relative_to(ROOT)}`",
        f"- CONTROL vs KL-GUARD comparison: `{COMPARISON_PATH.relative_to(ROOT)}`",
        f"- Late update summary: `{LATE_PATH.relative_to(ROOT)}`",
        f"- Rescue decision: `{DECISION_PATH.relative_to(ROOT)}`",
        f"- Paired-validation semantic background: `{PAIRED_DOC.relative_to(ROOT)}`",
        "",
        "## 2. 输出文件",
        "",
        f"- 主图 CSV：`{TRAJECTORY_OUT.relative_to(ROOT)}`",
        f"- 主图 PNG：`{TRAJECTORY_PNG.relative_to(ROOT)}`",
        f"- 辅助图 CSV：`{LATE_OUT.relative_to(ROOT)}`",
        f"- 辅助图 PNG：`{LATE_PNG.relative_to(ROOT)}`",
        f"- Endpoint table：`{ENDPOINT_OUT.relative_to(ROOT)}`",
        f"- Key numbers：`{KEY_NUMBERS_OUT.relative_to(ROOT)}`",
        f"- 最小任务记录 PPTX：`docs/presentation_yc_ppo_slide3_fixed_klguard_assets.pptx`（已生成并通过 finalizer 结构/版式校验）",
        "",
        "## 3. 主图含义",
        "",
        "主图把同一条已知会 collapse 的 seed1 / RNG2 continuation 放在同一坐标系中比较。两条线共享 75K 起点，CONTROL 在 100K 降到 full-collapse 水平，固定 KL-GUARD 的 100K yield 明显更高，但仍低于约 8200 kg/ha 的正常高产平台。",
        "",
        trajectory_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 4. 辅助图含义",
        "",
        "辅助图采用 late-window `approx_kl` median，展示 CONTROL 与 KL-GUARD 的描述性 update-instability 差异。median 是本页作图口径；图中明确标注 descriptive only，不把该差异表述为因果证明。",
        "",
        late_table.to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 5. Endpoint comparison at 100K",
        "",
        endpoint.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 6. Slide-ready key numbers",
        "",
        f"- CONTROL 100K yield: `{float(row['control_rng2_100k_mean_yield']):.3f} kg/ha`",
        f"- KL-GUARD 100K yield: `{float(row['klguard_100k_mean_yield']):.3f} kg/ha`",
        f"- Yield recovery vs CONTROL: `+{float(row['improvement_vs_control_100k']):.3f} kg/ha`",
        f"- Effective KL threshold: `{float(row['effective_kl_threshold']):.1e}`",
        f"- Guard triggered: **TRUE**, trigger rate `{float(row['klguard_trigger_rate']):.6f}`",
        f"- Guard not frozen: **TRUE**, optimizer-step fraction `{float(row['klguard_updates_with_optimizer_step_fraction']):.6f}`",
        "",
        "## 7. 推荐页标题",
        "",
        "1. 第一次干预：固定 KL guard 能部分救回已知坍缩轨迹",
        "2. Fixed KL guard can partially rescue a known collapsing trajectory",
        "",
        "## 8. 推荐一句话 takeaway",
        "",
        "- 固定 KL guard 把已知 RNG2 坍缩轨迹从 100K 的低产状态部分拉回，但没有完全恢复高产平台。",
        "- 这次干预有效降低了 late-window instability，并显著抬高了 100K yield。",
        "- Fixed KL guard partially rescued a known collapse, recovering `+" + f"{float(row['improvement_vs_control_100k']):.3f}" + " kg ha$^{-1}$ at 100K.",
        "",
        "## 9. 推荐口头汇报说法",
        "",
        "这一页看的是一条已经知道会 collapse 的 RNG2 轨迹。CONTROL 从相同的 75K checkpoint 继续到 100K 后，yield 降到 5348.288 kg/ha。我们只加入固定 KL guard，保持其他 PPO/DSSAT 契约不变，100K yield 回到 7472.490 kg/ha，较 CONTROL 高 2124.202 kg/ha。与此同时，late-window 的 approx_kl median 整体下降，说明 guard 的作用方向与抑制后期 update instability 一致，但这仍是一次 partial rescue，不等于完全恢复，也不单独证明因果机制。",
        "",
        "## 10. 缺失字段、口径限制与冲突",
        "",
        "- 本页所需的 yield、I、N、PFP_N、WP_ET、approx_kl、effective threshold、trigger rate 和 non-freezing evidence 均有既有来源，未伪造缺失字段。",
        "- `control_vs_klguard.csv` 没有 75K 行，因此 75K 轨迹点使用两份 checkpoint validation aggregate；80K–100K 已逐点与 comparison CSV 核对。",
        "- late-KL 图使用 median，属于 descriptive only；不扩展到机制因果结论。",
        f"- 数据冲突：**{'YES' if conflict else 'NO'}**。",
        "- GitHub backup：本轮未执行 push。当前任务未提供单独的远程写入确认，项目保护规则要求 push 前先获得明确授权；生成文件已保留在本地仓库路径。",
        "",
        "## 11. 推荐 Slide 3 组合",
        "",
        "主图放左侧作为 hero panel，右侧放 key numbers，右下或页底放 endpoint table / 一句 takeaway。late-KL 辅助图建议作为小型 inset 或备用图，仅在需要解释‘为什么方向上像 rescue’时使用，并保留 descriptive-only 限定。",
        "",
        "YC_PPO_SLIDE3_VISUAL_ASSETS_READY",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs() -> dict[str, int]:
    outputs = [TRAJECTORY_OUT, TRAJECTORY_PNG, LATE_OUT, LATE_PNG, ENDPOINT_OUT, KEY_NUMBERS_OUT, REPORT_OUT]
    sizes: dict[str, int] = {}
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
        sizes[str(path.relative_to(ROOT))] = path.stat().st_size
    return sizes


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()
    trajectory, conflict = build_trajectory(inputs)
    late = build_late_summary(inputs)
    endpoint = build_endpoint_table(trajectory, inputs["decision"])
    trajectory.to_csv(TRAJECTORY_OUT, index=False, float_format="%.9f")
    late.to_csv(LATE_OUT, index=False, float_format="%.12f")
    endpoint.to_csv(ENDPOINT_OUT, index=False, float_format="%.9f")
    write_key_numbers(trajectory, inputs["decision"])
    plot_main(trajectory, inputs["decision"])
    plot_late(late)
    write_report(inputs, trajectory, late, endpoint, conflict)
    print({"status": "PASS", "outputs": verify_outputs(), "marker": "YC_PPO_SLIDE3_VISUAL_ASSETS_READY"})


if __name__ == "__main__":
    main()

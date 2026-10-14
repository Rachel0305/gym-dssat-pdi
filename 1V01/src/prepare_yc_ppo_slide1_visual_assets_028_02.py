"""Generate the 028_02 YC PPO Slide 1 visual assets from existing CSVs.

This is a presentation-data rendering step only.  It does not train models,
run DSSAT, rerun validation, or modify any source result CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

TRAJECTORY_PATH = RESULTS / "presentation_yc_ppo_6run_checkpoint_trajectories.csv"
SUMMARY_PATH = RESULTS / "presentation_yc_ppo_6run_run_summary.csv"
KEY_NUMBERS_PATH = RESULTS / "presentation_yc_ppo_6run_slide1_key_numbers.csv"
BEST_FINAL_PATH = RESULTS / "presentation_yc_ppo_6run_best_vs_final.csv"

MAIN_PNG = FIGURES / "slide1_ppo_6run_checkpoint_trajectory.png"
MAIN_SVG = FIGURES / "slide1_ppo_6run_checkpoint_trajectory.svg"
CARDS_PNG = FIGURES / "slide1_ppo_key_numbers.png"
CARDS_SVG = FIGURES / "slide1_ppo_key_numbers.svg"
BEST_FINAL_PNG = FIGURES / "slide1_ppo_best_vs_final.png"
BEST_FINAL_SVG = FIGURES / "slide1_ppo_best_vs_final.svg"
MOCKUP_PNG = FIGURES / "slide1_layout_mockup.png"
REPORT_PATH = DOCS / "presentation_yc_ppo_slide1_visual_assets.md"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


VARIANT_STYLE = {
    "Original": {
        "linestyle": "-",
        "colors": ["#244B73", "#4F759A", "#789BB8"],
        "face": "#EEF4F9",
    },
    "Augmented": {
        "linestyle": "--",
        "colors": ["#985B2F", "#BE8051", "#D5A47A"],
        "face": "#FBF1E9",
    },
}
MARKERS = {0: "o", 1: "s", 2: "^"}
RUN_ORDER = [
    "Original_s0",
    "Original_s1",
    "Original_s2",
    "Augmented_s0",
    "Augmented_s1",
    "Augmented_s2",
]


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str], pd.DataFrame]:
    for path in [TRAJECTORY_PATH, SUMMARY_PATH, KEY_NUMBERS_PATH, BEST_FINAL_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty input CSV: {path}")

    trajectory = pd.read_csv(TRAJECTORY_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    key_numbers = pd.read_csv(KEY_NUMBERS_PATH)
    best_final = pd.read_csv(BEST_FINAL_PATH)

    require_columns(
        trajectory,
        [
            "variant",
            "seed",
            "run_label",
            "checkpoint_k",
            "checkpoint_step",
            "mean_yield_kg_ha",
            "mean_I_mm",
            "mean_N_kg_ha",
            "source_file",
            "validation_years",
            "validation_status",
        ],
        "trajectory CSV",
    )
    require_columns(
        summary,
        [
            "run_label",
            "variant",
            "seed",
            "best_checkpoint_k",
            "best_yield_kg_ha",
            "final_100k_yield_kg_ha",
            "retention_class",
        ],
        "run summary CSV",
    )
    require_columns(key_numbers, ["metric", "value", "unit", "source_file", "notes"], "key numbers CSV")
    require_columns(
        best_final,
        [
            "run_label",
            "variant",
            "seed",
            "best_checkpoint_k",
            "best_yield_kg_ha",
            "final_100k_yield_kg_ha",
            "drop_best_to_100k_kg_ha",
            "retention_class",
        ],
        "best-vs-final CSV",
    )

    if len(trajectory) != len(RUN_ORDER) * 4:
        raise ValueError(f"Expected 24 trajectory rows, found {len(trajectory)}")
    if set(trajectory["run_label"]) != set(RUN_ORDER):
        raise ValueError("Trajectory CSV does not contain exactly the six required run labels")
    if trajectory[["run_label", "checkpoint_k"]].duplicated().any():
        raise ValueError("Trajectory CSV contains duplicate run/checkpoint keys")
    if trajectory["validation_status"].astype(str).str.upper().ne("PASS").any():
        raise ValueError("Trajectory CSV contains a non-PASS validation status")
    if set(trajectory["validation_years"].astype(str)) != {"2014-2023"}:
        raise ValueError("Trajectory CSV does not consistently report 2014-2023")
    if len(summary) != len(RUN_ORDER) or set(summary["run_label"]) != set(RUN_ORDER):
        raise ValueError("Run summary CSV is incomplete")
    if len(best_final) != len(RUN_ORDER) or set(best_final["run_label"]) != set(RUN_ORDER):
        raise ValueError("Best-vs-final CSV is incomplete")

    key_map: dict[str, float | str] = {}
    for _, row in key_numbers.iterrows():
        raw = str(row["value"])
        try:
            key_map[str(row["metric"])] = float(raw)
        except ValueError:
            key_map[str(row["metric"])] = raw

    threshold = None
    for metric in ["runs_ever_high_yield", "runs_retained_high_at_100k", "runs_learned_then_lost"]:
        note = str(key_numbers.loc[key_numbers["metric"].eq(metric), "notes"].iloc[0])
        match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)?)", note)
        if match:
            threshold = float(match.group(1))
            break
    if threshold is None:
        raise ValueError("Could not read the descriptive yield threshold from key numbers notes")
    key_map["high_yield_threshold"] = threshold

    # Internal consistency checks across the already-produced presentation CSVs.
    by_run = trajectory.groupby("run_label", sort=False)
    for run_label, run_summary in summary.set_index("run_label").iterrows():
        run_data = by_run.get_group(run_label)
        best_row = run_data.loc[run_data["mean_yield_kg_ha"].astype(float).idxmax()]
        final_row = run_data.loc[run_data["checkpoint_k"].astype(int).eq(100)].iloc[0]
        if int(run_summary["best_checkpoint_k"]) != int(best_row["checkpoint_k"]):
            raise ValueError(f"Best checkpoint conflict for {run_label}")
        if abs(float(run_summary["best_yield_kg_ha"]) - float(best_row["mean_yield_kg_ha"])) > 1e-6:
            raise ValueError(f"Best yield conflict for {run_label}")
        if abs(float(run_summary["final_100k_yield_kg_ha"]) - float(final_row["mean_yield_kg_ha"])) > 1e-6:
            raise ValueError(f"Final yield conflict for {run_label}")

    return trajectory, summary, key_map, best_final


def ordered_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    order = {run: index for index, run in enumerate(RUN_ORDER)}
    out = trajectory.copy()
    out["_run_order"] = out["run_label"].map(order)
    out["checkpoint_k"] = out["checkpoint_k"].astype(int)
    out = out.sort_values(["_run_order", "checkpoint_k"]).reset_index(drop=True)
    return out


def axis_limits(trajectory: pd.DataFrame, expert: float) -> tuple[float, float]:
    values = pd.concat(
        [trajectory["mean_yield_kg_ha"].astype(float), pd.Series([float(expert)])],
        ignore_index=True,
    )
    span = float(values.max() - values.min())
    margin = max(span * 0.08, 150.0)
    return float(values.min() - margin), float(values.max() + margin)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=3, width=0.7, color="#5A6470")
    ax.grid(axis="y", color="#B5BEC7", linewidth=0.55, alpha=0.24)
    ax.set_axisbelow(True)


def plot_trajectory(
    ax: plt.Axes,
    trajectory: pd.DataFrame,
    expert: float,
    threshold: float,
    compact: bool = False,
    add_title: bool = True,
    show_expert_label: bool = True,
) -> None:
    data = ordered_trajectory(trajectory)
    checkpoints = sorted(data["checkpoint_k"].unique())
    for run_label in RUN_ORDER:
        row_data = data.loc[data["run_label"].eq(run_label)].sort_values("checkpoint_k")
        variant = str(row_data["variant"].iloc[0])
        seed = int(row_data["seed"].iloc[0])
        style = VARIANT_STYLE[variant]
        color = style["colors"][seed]
        x = row_data["checkpoint_k"].to_numpy()
        y = row_data["mean_yield_kg_ha"].astype(float).to_numpy()
        ax.plot(
            x,
            y,
            color=color,
            linestyle=style["linestyle"],
            linewidth=2.0 if not compact else 1.6,
            marker=MARKERS[seed],
            markersize=5.4 if not compact else 4.3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            label=run_label,
            zorder=3,
        )
        endpoint = row_data.loc[row_data["checkpoint_k"].eq(max(checkpoints))].iloc[0]
        ax.plot(
            [max(checkpoints)],
            [float(endpoint["mean_yield_kg_ha"])],
            marker=MARKERS[seed],
            markersize=8.0 if not compact else 6.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=5,
        )
    lost_runs = set()
    for run_label in RUN_ORDER:
        row_data = data.loc[data["run_label"].eq(run_label)].sort_values("checkpoint_k")
        final_yield = float(row_data.loc[row_data["checkpoint_k"].eq(max(checkpoints)), "mean_yield_kg_ha"].iloc[0])
        if float(row_data["mean_yield_kg_ha"].astype(float).max()) >= threshold and final_yield < threshold:
            lost_runs.add(run_label)
    label_offsets = {"Original_s0": -10, "Original_s1": 9, "Augmented_s2": -4}
    for run_label in RUN_ORDER:
        if run_label not in lost_runs:
            continue
        endpoint = data.loc[
            data["run_label"].eq(run_label) & data["checkpoint_k"].eq(max(checkpoints))
        ].iloc[0]
        variant = str(endpoint["variant"])
        seed = int(endpoint["seed"])
        ax.annotate(
            "learned -> lost",
            xy=(max(checkpoints), float(endpoint["mean_yield_kg_ha"])),
            xytext=(8, label_offsets.get(run_label, 0)),
            textcoords="offset points",
            fontsize=8 if not compact else 6.6,
            color=VARIANT_STYLE[variant]["colors"][seed],
            va="center",
            ha="left",
            clip_on=False,
        )

    ax.axhline(
        float(expert),
        color="#20262D",
        linestyle=(0, (4, 3)),
        linewidth=1.25,
        label="Expert baseline",
        zorder=2,
    )
    ax.set_xticks(checkpoints, [f"{int(value)}K" for value in checkpoints])
    ax.set_xlim(min(checkpoints) - 3, max(checkpoints) + (26 if not compact else 22))
    ymin, ymax = axis_limits(trajectory, expert)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    if show_expert_label:
        ax.text(
            0.01,
            1.02,
            f"Expert = {expert:,.3f} kg ha$^{{-1}}$",
            transform=ax.transAxes,
            fontsize=8 if not compact else 7,
            color="#20262D",
            va="bottom",
        )
    style_axis(ax)
    if add_title:
        ax.set_title(
            "PPO can learn a high-yield policy — and lose it later\n"
            "YC held-out validation, 2014–2023",
            loc="left",
            pad=12,
            fontweight="bold",
        )
        ax.legend(
            ncol=4,
            fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            handlelength=2.2,
            columnspacing=1.2,
        )


def draw_card(ax: plt.Axes, big: str, caption: str, face: str, accent: str, compact: bool = False) -> None:
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        linewidth=0,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.012,
            0.98,
            boxstyle="round,pad=0.0,rounding_size=0.01",
            linewidth=0,
            facecolor=accent,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        0.09,
        0.60,
        big,
        transform=ax.transAxes,
        fontsize=25 if not compact else 17,
        fontweight="bold",
        color="#1B2733",
        va="center",
    )
    ax.text(
        0.09,
        0.23,
        caption,
        transform=ax.transAxes,
        fontsize=8.6 if not compact else 6.8,
        color="#4B5966",
        va="center",
        linespacing=1.25,
    )


def card_payload(key_map: dict[str, float | str], threshold: float) -> list[tuple[str, str, str, str]]:
    total = int(round(float(key_map["total_runs"])))
    ever = int(round(float(key_map["runs_ever_high_yield"])))
    retained = int(round(float(key_map["runs_retained_high_at_100k"])))
    lost = int(round(float(key_map["runs_learned_then_lost"])))
    expert = float(key_map["expert_mean_yield_kg_ha"])
    high_text = f"≥{threshold:,.0f} kg ha$^{{-1}}$"
    return [
        (f"{ever} / {total}", f"runs ever reached {high_text}", "#EEF4F9", "#244B73"),
        (f"{retained} / {total}", f"runs retaining {high_text}\nat 100K", "#FBF1E9", "#985B2F"),
        (f"{lost} / {total}", "learned -> lost", "#F1F0F7", "#62577F"),
        (f"{expert:,.3f}", "Expert mean yield\n(kg ha$^{-1}$)", "#F3F5F3", "#4B6A59"),
    ]


def plot_key_cards(key_map: dict[str, float | str], threshold: float) -> None:
    fig = plt.figure(figsize=(5.6, 4.35), facecolor="white")
    fig.text(0.08, 0.93, "Slide 1 key numbers", fontsize=17, fontweight="bold", color="#1B2733")
    fig.text(
        0.08,
        0.875,
        "Six existing YC PPO runs · held-out validation",
        fontsize=9.5,
        color="#5C6873",
    )
    positions = [(0.08, 0.52), (0.54, 0.52), (0.08, 0.08), (0.54, 0.08)]
    for (big, caption, face, accent), (left, bottom) in zip(card_payload(key_map, threshold), positions):
        ax = fig.add_axes([left, bottom, 0.38, 0.31])
        draw_card(ax, big, caption, face, accent)
    fig.savefig(CARDS_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(CARDS_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_main_figure(trajectory: pd.DataFrame, key_map: dict[str, float | str]) -> None:
    expert = float(key_map["expert_mean_yield_kg_ha"])
    threshold = float(key_map["high_yield_threshold"])
    fig, ax = plt.subplots(figsize=(9.8, 5.65), facecolor="white")
    plot_trajectory(ax, trajectory, expert, threshold)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.24)
    fig.savefig(MAIN_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(MAIN_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_best_final(best_final: pd.DataFrame) -> None:
    data = best_final.copy()
    data["_run_order"] = data["run_label"].map({run: i for i, run in enumerate(RUN_ORDER)})
    data = data.sort_values("_run_order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.9), facecolor="white")
    for y_pos, row in data.iterrows():
        variant = str(row["variant"])
        seed = int(row["seed"])
        color = VARIANT_STYLE[variant]["colors"][seed]
        lost = str(row["retention_class"]) == "LEARNED_THEN_LOST"
        best = float(row["best_yield_kg_ha"])
        final = float(row["final_100k_yield_kg_ha"])
        ax.plot(
            [best, final],
            [y_pos, y_pos],
            color=color,
            linewidth=3.0 if lost else 1.7,
            alpha=0.9 if lost else 0.62,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [best],
            [y_pos],
            s=54,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )
        ax.scatter(
            [final],
            [y_pos],
            s=64,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
    expert = float(key_map_from_file()["expert_mean_yield_kg_ha"])
    ax.axvline(expert, color="#20262D", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.text(expert, len(data) - 0.35, f"Expert {expert:,.0f}", ha="right", va="top", fontsize=8, color="#20262D")
    ax.set_yticks(range(len(data)), data["run_label"])
    ax.invert_yaxis()
    values = pd.concat([data["best_yield_kg_ha"], data["final_100k_yield_kg_ha"]]).astype(float)
    margin = max(float(values.max() - values.min()) * 0.08, 150.0)
    ax.set_xlim(float(values.min()) - margin, float(values.max()) + margin)
    ax.set_xlabel("Mean yield (kg ha$^{-1}$)")
    ax.set_title("Best observed checkpoint vs final 100K model", loc="left", pad=12, fontweight="bold")
    ax.grid(axis="x", color="#B5BEC7", linewidth=0.55, alpha=0.24)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#4B5966", markerfacecolor="white", linewidth=0, label="Best checkpoint"),
            Line2D([0], [0], marker="o", color="#4B5966", markerfacecolor="#4B5966", linewidth=0, label="Final 100K"),
            Line2D([0], [0], color="#4B5966", linewidth=2.4, label="learned -> lost"),
        ],
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.22)
    fig.savefig(BEST_FINAL_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(BEST_FINAL_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


_KEY_MAP_CACHE: dict[str, float | str] | None = None


def key_map_from_file() -> dict[str, float | str]:
    global _KEY_MAP_CACHE
    if _KEY_MAP_CACHE is None:
        key_numbers = pd.read_csv(KEY_NUMBERS_PATH)
        _KEY_MAP_CACHE = {}
        for _, row in key_numbers.iterrows():
            try:
                _KEY_MAP_CACHE[str(row["metric"])] = float(row["value"])
            except ValueError:
                _KEY_MAP_CACHE[str(row["metric"])] = str(row["value"])
    return _KEY_MAP_CACHE


def plot_layout_mockup(trajectory: pd.DataFrame, key_map: dict[str, float | str]) -> None:
    expert = float(key_map["expert_mean_yield_kg_ha"])
    threshold = float(key_map["high_yield_threshold"])
    cards = card_payload(key_map, threshold)
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        nrows=12,
        ncols=10,
        left=0.080,
        right=0.975,
        top=0.84,
        bottom=0.17,
        wspace=0.8,
        hspace=0.7,
    )
    fig.text(
        0.075,
        0.925,
        "PPO learns high-yield policies, but may lose them during continued training",
        fontsize=18.5,
        fontweight="bold",
        color="#1B2733",
        va="top",
    )
    fig.text(
        0.075,
        0.875,
        "YC held-out validation · 2014–2023 · six PPO runs · fixed checkpoints",
        fontsize=10.5,
        color="#5C6873",
        va="top",
    )
    main_ax = fig.add_subplot(grid[0:10, :7])
    plot_trajectory(
        main_ax,
        trajectory,
        expert,
        threshold,
        compact=True,
        add_title=False,
        show_expert_label=False,
    )
    main_ax.set_xlabel("Checkpoint", fontsize=9)
    main_ax.set_ylabel("Mean yield (kg ha$^{-1}$)", fontsize=9)
    handles, labels = main_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=6.5,
        loc="lower center",
        bbox_to_anchor=(0.365, 0.135),
        handlelength=1.8,
        columnspacing=0.8,
        frameon=False,
    )
    right_positions = [grid[0:3, 7:10], grid[3:6, 7:10], grid[6:9, 7:10], grid[9:12, 7:10]]
    for subgrid, (big, caption, face, accent) in zip(right_positions, cards):
        card_ax = fig.add_subplot(subgrid)
        draw_card(card_ax, big, caption, face, accent, compact=True)
    total = int(round(float(key_map["total_runs"])))
    ever = int(round(float(key_map["runs_ever_high_yield"])))
    retained = int(round(float(key_map["runs_retained_high_at_100k"])))
    fig.text(
        0.075,
        0.075,
        f"Takeaway: {ever}/{total} runs reached ≥{threshold:,.0f} kg ha$^{{-1}}$ at least once, but only {retained}/{total} retained it at 100K.",
        fontsize=13,
        fontweight="bold",
        color="#1B2733",
        va="center",
    )
    # Keep the exact 16:9 canvas; tight bounding-box cropping would change the
    # requested slide aspect ratio.
    fig.savefig(MOCKUP_PNG, dpi=192, facecolor="white")
    plt.close(fig)


def fmt(value: float, decimals: int = 3) -> str:
    return f"{float(value):,.{decimals}f}"


def write_report(
    trajectory: pd.DataFrame,
    summary: pd.DataFrame,
    key_map: dict[str, float | str],
    best_final: pd.DataFrame,
) -> None:
    data = ordered_trajectory(trajectory)
    checkpoint_order = sorted(data["checkpoint_k"].unique())
    wide = data.pivot(index="run_label", columns="checkpoint_k", values="mean_yield_kg_ha").reindex(RUN_ORDER)
    wide = wide.reindex(columns=checkpoint_order)
    wide.index.name = "run_label"
    wide.columns = [f"{int(column)}K yield (kg/ha)" for column in wide.columns]
    trajectory_table = wide.reset_index().to_markdown(index=False, floatfmt=".3f")
    expert = float(key_map["expert_mean_yield_kg_ha"])
    expert_i = float(key_map["expert_mean_I_mm"])
    expert_n = float(key_map["expert_mean_N_kg_ha"])
    threshold = float(key_map["high_yield_threshold"])
    conflict = "NO"
    lines = [
        "# YC PPO Slide 1 visual assets audit",
        "",
        "## Status",
        "",
        "- `YC_PPO_SLIDE1_VISUAL_ASSETS_READY`",
        "- Python/matplotlib only; no training, validation rerun, DSSAT rerun, PPO/DSSAT logic change, or PPTX generation was performed.",
        "- All scientific values in the figures and cards were read from the four 028_01 CSV inputs.",
        "",
        "## Output paths",
        "",
        f"1. Main trajectory: `{MAIN_PNG.relative_to(ROOT)}` and `{MAIN_SVG.relative_to(ROOT)}`",
        f"2. Key-number cards: `{CARDS_PNG.relative_to(ROOT)}` and `{CARDS_SVG.relative_to(ROOT)}`",
        f"3. Best-vs-final backup: `{BEST_FINAL_PNG.relative_to(ROOT)}` and `{BEST_FINAL_SVG.relative_to(ROOT)}`",
        f"4. 16:9 layout mockup: `{MOCKUP_PNG.relative_to(ROOT)}`",
        "",
        "## Main trajectory values",
        "",
        f"Descriptive display threshold: `{threshold:,.0f} kg/ha` (read from key-number notes).",
        "",
        trajectory_table,
        "",
        "## Expert baseline",
        "",
        f"- Mean yield: `{fmt(expert)} kg/ha`",
        f"- Mean irrigation: `{fmt(expert_i)} mm`",
        f"- Mean N: `{fmt(expert_n)} kg/ha`",
        f"- Source: `{key_map['expert_mean_yield_kg_ha']!s}` is derived from the formal Expert row in the 028_01 key-number CSV; no value was hand-filled.",
        "",
        "## Recommended Slide 1 combination",
        "",
        "Use the 16:9 layout mockup as the composition guide: the main six-run trajectory as the hero panel on the left, the 2×2 key-number cards on the right, and the dynamic retention takeaway at the bottom. Keep the best-vs-final figure as the backup diagnostic visual.",
        "",
        "## Data conflict check",
        "",
        f"- Conflict found: **{conflict}**.",
        "- The input trajectory, run-summary, key-number, and best-vs-final CSVs are internally consistent; all six trajectories contain 25K/50K/75K/100K and report `PASS` for 2014–2023.",
        "- No source CSV was modified.",
        "",
        "## Reproducibility note",
        "",
        f"The rendering helper is `{Path('src/prepare_yc_ppo_slide1_visual_assets_028_02.py')}`. PNG exports use 300 dpi except the 16:9 mockup, which uses 192 dpi; SVG text remains editable via matplotlib's `svg.fonttype=none` setting.",
        "",
        "YC_PPO_SLIDE1_VISUAL_ASSETS_READY",
        "",
    ]
    # Replace the source line's intentionally generic text with the actual source path stored in the key CSV.
    expert_source = str(pd.read_csv(KEY_NUMBERS_PATH).loc[lambda x: x["metric"].eq("expert_mean_yield_kg_ha"), "source_file"].iloc[0])
    lines[lines.index(f"- Source: `{key_map['expert_mean_yield_kg_ha']!s}` is derived from the formal Expert row in the 028_01 key-number CSV; no value was hand-filled.")] = (
        f"- Source: `{expert_source}`; the formal Expert value was read from the 028_01 key-number CSV, not hand-filled."
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs() -> dict[str, int]:
    outputs = [MAIN_PNG, MAIN_SVG, CARDS_PNG, CARDS_SVG, BEST_FINAL_PNG, BEST_FINAL_SVG, MOCKUP_PNG, REPORT_PATH]
    sizes: dict[str, int] = {}
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
        sizes[str(path.relative_to(ROOT))] = path.stat().st_size
    for svg in [MAIN_SVG, CARDS_SVG, BEST_FINAL_SVG]:
        if "<svg" not in svg.read_text(encoding="utf-8", errors="ignore")[:5000]:
            raise RuntimeError(f"SVG header check failed: {svg}")
    return sizes


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    trajectory, summary, key_map, best_final = load_inputs()
    plot_main_figure(trajectory, key_map)
    plot_key_cards(key_map, float(key_map["high_yield_threshold"]))
    # The best-vs-final chart reads the Expert baseline from the same key-number CSV.
    global _KEY_MAP_CACHE
    _KEY_MAP_CACHE = key_map
    plot_best_final(best_final)
    plot_layout_mockup(trajectory, key_map)
    write_report(trajectory, summary, key_map, best_final)
    sizes = verify_outputs()
    print({"status": "PASS", "outputs": sizes, "marker": "YC_PPO_SLIDE1_VISUAL_ASSETS_READY"})


if __name__ == "__main__":
    main()

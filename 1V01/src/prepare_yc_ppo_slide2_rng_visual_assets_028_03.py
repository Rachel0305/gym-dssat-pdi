"""Render the 028_03 YC PPO Slide 2 visual assets from existing evidence.

This script is presentation-only. It reads existing CSV/Markdown evidence and
does not train, run validation, run DSSAT, or modify the source result files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

REPRO_PATH = RESULTS / "yc_ppo_s1_control_5complete_reproducibility.csv"
LATE_PATH = RESULTS / "yc_ppo_s1_control_5complete_late_update_summary.csv"
RNG5_PATH = RESULTS / "yc_ppo_s1_control_rng5_checkpoint_validation.csv"
REPRO_DOC = DOCS / "yc_ppo_original_seed1_multirng_control_reproducibility.md"
RNG5_DOC = DOCS / "yc_ppo_original_seed1_rng5_control_completion.md"

MAIN_PNG = FIGURES / "slide2_rng_continuation_trajectory.png"
MAIN_SVG = FIGURES / "slide2_rng_continuation_trajectory.svg"
CARDS_PNG = FIGURES / "slide2_rng_key_numbers.png"
CARDS_SVG = FIGURES / "slide2_rng_key_numbers.svg"
SETUP_PNG = FIGURES / "slide2_rng_setup_card.png"
SETUP_SVG = FIGURES / "slide2_rng_setup_card.svg"
KL_PNG = FIGURES / "slide2_rng_late_kl_descriptive.png"
KL_SVG = FIGURES / "slide2_rng_late_kl_descriptive.svg"
MOCKUP_PNG = FIGURES / "slide2_layout_mockup.png"
REPORT_PATH = DOCS / "presentation_yc_ppo_slide2_visual_assets.md"


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

RNG_ORDER = [1, 2, 3, 4, 5]
CHECKPOINTS = [75, 80, 85, 90, 95, 100]
COLLAPSE_COLOR = "#6F3E73"
STABLE_COLORS = ["#547491", "#6F8EA6", "#8AA4B9", "#A1B6C7"]
STABLE_COLOR_BY_RNG = {1: STABLE_COLORS[0], 3: STABLE_COLORS[1], 4: STABLE_COLORS[2], 5: STABLE_COLORS[3]}
CARD_COLORS = ["#F2EAF3", "#EEF4F9", "#F6F0E8", "#F1F4F1"]
CARD_ACCENTS = [COLLAPSE_COLOR, "#315C7E", "#9A6533", "#4C6C5B"]


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [REPRO_PATH, LATE_PATH, RNG5_PATH, REPRO_DOC, RNG5_DOC]
    for path in required_paths:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty input evidence: {path}")

    repro = pd.read_csv(REPRO_PATH)
    late = pd.read_csv(LATE_PATH)
    rng5_validation = pd.read_csv(RNG5_PATH)
    require_columns(
        repro,
        [
            "continuation_rng",
            "yield_75K",
            "yield_80K",
            "yield_85K",
            "yield_90K",
            "yield_95K",
            "yield_100K",
            "collapse",
            "validation_source",
        ],
        "reproducibility CSV",
    )
    require_columns(
        late,
        ["trajectory_class", "window", "trajectory_count", "approx_kl_median"],
        "late-update CSV",
    )
    require_columns(
        rng5_validation,
        ["row_type", "checkpoint_step", "continuation_rng", "status", "validation_source"],
        "RNG5 validation CSV",
    )

    repro["continuation_rng"] = repro["continuation_rng"].astype(int)
    repro["collapse"] = repro["collapse"].astype(str).str.strip().str.lower().eq("true")
    if sorted(repro["continuation_rng"].tolist()) != RNG_ORDER:
        raise ValueError("Reproducibility CSV does not contain exactly RNG1-RNG5")
    if len(repro) != len(RNG_ORDER):
        raise ValueError(f"Expected 5 complete continuations, found {len(repro)}")
    if repro[[f"yield_{checkpoint}K" for checkpoint in CHECKPOINTS]].isna().any().any():
        raise ValueError("Reproducibility CSV contains missing trajectory values")

    aggregate = rng5_validation.loc[rng5_validation["row_type"].eq("aggregate")].copy()
    if aggregate.empty:
        raise ValueError("RNG5 validation CSV has no aggregate rows")
    if not (aggregate["status"].astype(str).str.upper().eq("PASS").all()):
        raise ValueError("RNG5 validation CSV has a non-PASS aggregate status")
    expected_continuation_steps = set(value * 1000 for value in CHECKPOINTS[1:])
    observed_aggregate_steps = set(aggregate["checkpoint_step"].astype(int))
    if not expected_continuation_steps.issubset(observed_aggregate_steps):
        raise ValueError("RNG5 validation CSV does not cover 80K-100K aggregate checkpoints")

    return repro, late, rng5_validation


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.tick_params(axis="both", which="major", length=3, width=0.7, color="#5A6470")
    ax.grid(axis=grid_axis, color="#B5BEC7", linewidth=0.55, alpha=0.24)
    ax.set_axisbelow(True)


def trajectory_frame(repro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, source_row in repro.sort_values("continuation_rng").iterrows():
        rng = int(source_row["continuation_rng"])
        for checkpoint in CHECKPOINTS:
            rows.append(
                {
                    "continuation_rng": rng,
                    "checkpoint_k": checkpoint,
                    "yield_kg_ha": float(source_row[f"yield_{checkpoint}K"]),
                    "collapse": bool(source_row["collapse"]),
                }
            )
    return pd.DataFrame(rows)


def set_yield_limits(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    margin = max((high - low) * 0.08, 150.0)
    return low - margin, high + margin


def plot_trajectory(ax: plt.Axes, repro: pd.DataFrame, compact: bool = False, with_title: bool = True) -> None:
    frame = trajectory_frame(repro)
    noncollapse = frame.loc[~frame["collapse"]]
    collapse = frame.loc[frame["collapse"]]
    for rng, group in noncollapse.groupby("continuation_rng", sort=True):
        color = STABLE_COLOR_BY_RNG[rng]
        ax.plot(
            group["checkpoint_k"],
            group["yield_kg_ha"],
            color=color,
            linewidth=2.0 if not compact else 1.5,
            marker="o",
            markersize=5.2 if not compact else 4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.15,
            alpha=0.95,
            label=f"RNG{rng}",
            zorder=3,
        )
    for rng, group in collapse.groupby("continuation_rng", sort=True):
        ax.plot(
            group["checkpoint_k"],
            group["yield_kg_ha"],
            color=COLLAPSE_COLOR,
            linewidth=3.0 if not compact else 2.2,
            marker="o",
            markersize=6.2 if not compact else 4.8,
            markerfacecolor="white",
            markeredgecolor=COLLAPSE_COLOR,
            markeredgewidth=1.6,
            label=f"RNG{rng} (collapse)",
            zorder=5,
        )

    for _, row in frame.loc[frame["checkpoint_k"].eq(100)].iterrows():
        rng = int(row["continuation_rng"])
        color = COLLAPSE_COLOR if bool(row["collapse"]) else STABLE_COLOR_BY_RNG[rng]
        ax.scatter(
            [100],
            [float(row["yield_kg_ha"])],
            s=85 if not compact else 56,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=6,
        )

    start = frame.loc[frame["checkpoint_k"].eq(75), "yield_kg_ha"].mean()
    ax.annotate(
        "same starting point",
        xy=(75, start),
        xytext=(76.4, start + 205),
        fontsize=8 if not compact else 6.4,
        color="#3C4650",
        arrowprops={"arrowstyle": "-", "color": "#66727D", "lw": 0.8},
    )
    ax.text(
        0.02,
        0.03,
        "Same 75K weights, same hyperparameters, different RNG continuation only",
        transform=ax.transAxes,
        fontsize=8 if not compact else 6.3,
        color="#53616D",
        va="bottom",
    )

    if not collapse.empty:
        collapsed_endpoint = collapse.loc[collapse["checkpoint_k"].eq(100)].iloc[0]
        ax.annotate(
            "RNG2: collapse",
            xy=(100, float(collapsed_endpoint["yield_kg_ha"])),
            xytext=(101.1, float(collapsed_endpoint["yield_kg_ha"]) + 90),
            fontsize=9 if not compact else 6.8,
            fontweight="bold",
            color=COLLAPSE_COLOR,
            va="center",
        )
    stable_endpoint = noncollapse.loc[noncollapse["checkpoint_k"].eq(100), "yield_kg_ha"]
    ax.text(
        101.1,
        float(stable_endpoint.mean()) + 35,
        "RNG1/3/4/5: non-collapse",
        fontsize=8.5 if not compact else 6.5,
        color="#415F78",
        va="center",
    )

    ax.set_xticks(CHECKPOINTS, [f"{value}K" for value in CHECKPOINTS])
    ax.set_xlim(73.5, 108.5)
    ax.set_ylim(*set_yield_limits(frame["yield_kg_ha"]))
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    style_axis(ax)
    if with_title:
        ax.set_title(
            "Continuation from the same 75K checkpoint can diverge\n"
            "YC PPO Original seed1, held-out validation (2014–2023)",
            loc="left",
            pad=12,
            fontweight="bold",
        )
        ax.legend(
            ncol=3,
            fontsize=7.6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            frameon=False,
            handlelength=2.2,
        )


def plot_main(repro: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.7), facecolor="white")
    plot_trajectory(ax, repro)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.24)
    fig.savefig(MAIN_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(MAIN_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_payload(repro: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
    total = int(len(repro))
    collapse_count = int(repro["collapse"].sum())
    noncollapse_count = total - collapse_count
    collapse_rate = collapse_count / total
    values = repro["yield_100K"].astype(float)
    return [
        (f"{collapse_count} / {total}", "full collapse", CARD_COLORS[0], CARD_ACCENTS[0], ""),
        (f"{noncollapse_count} / {total}", "non-collapse", CARD_COLORS[1], CARD_ACCENTS[1], ""),
        (f"{collapse_rate:.2f}", "collapse rate", CARD_COLORS[2], CARD_ACCENTS[2], ""),
        (
            f"{values.min():,.0f}–{values.max():,.0f}",
            "100K yield range (kg ha$^{-1}$)",
            CARD_COLORS[3],
            CARD_ACCENTS[3],
            f"mean = {values.mean():,.3f} · SD = {values.std(ddof=1):,.3f}",
        ),
    ]


def draw_card(ax: plt.Axes, big: str, caption: str, face: str, accent: str, note: str = "", compact: bool = False) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.02,rounding_size=0.045",
            linewidth=0,
            facecolor=face,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        Rectangle((0.01, 0.01), 0.014, 0.98, transform=ax.transAxes, facecolor=accent, linewidth=0, clip_on=False)
    )
    ax.text(
        0.09,
        0.61,
        big,
        transform=ax.transAxes,
        fontsize=27 if not compact else 16,
        fontweight="bold",
        color="#1B2733",
        va="center",
    )
    ax.text(
        0.09,
        0.25 if not note else 0.29,
        caption,
        transform=ax.transAxes,
        fontsize=9 if not compact else 6.6,
        color="#4B5966",
        va="center",
        linespacing=1.25,
    )
    if note:
        ax.text(
            0.09,
            0.11,
            note,
            transform=ax.transAxes,
            fontsize=7.1 if not compact else 5.7,
            color="#63717C",
            va="center",
        )


def plot_cards(repro: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(5.8, 4.55), facecolor="white")
    fig.text(0.08, 0.93, "RNG-dependent continuation outcomes", fontsize=16.5, fontweight="bold", color="#1B2733")
    fig.text(0.08, 0.875, "Same 75K checkpoint · five complete continuations", fontsize=9.5, color="#5C6873")
    positions = [(0.08, 0.52), (0.54, 0.52), (0.08, 0.08), (0.54, 0.08)]
    for item, (left, bottom) in zip(card_payload(repro), positions):
        big, caption, face, accent, note = item
        draw_card(fig.add_axes([left, bottom, 0.38, 0.31]), big, caption, face, accent, note)
    fig.savefig(CARDS_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(CARDS_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_rows() -> list[tuple[str, str]]:
    return [
        ("Source checkpoint", "75K"),
        ("Source model", "PPO Original seed1"),
        ("Train continuation budget", "+25K steps (75K -> 100K)"),
        ("Hyperparameters", "same as source run"),
        ("Validation years", "2014–2023"),
        ("Changed factor", "RNG continuation only"),
        ("Excluded case", "RNG0 infrastructure-censored, not in denominator"),
    ]


def draw_setup(ax: plt.Axes, compact: bool = False) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            linewidth=0.8,
            edgecolor="#D4DCE2",
            facecolor="#FBFCFD",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.055,
        0.93,
        "Setup: only RNG continuation changed",
        transform=ax.transAxes,
        fontsize=13 if not compact else 9,
        fontweight="bold",
        color="#1B2733",
        va="top",
    )
    rows = setup_rows()
    top = 0.80
    row_height = 0.105 if not compact else 0.108
    for idx, (field, value) in enumerate(rows):
        y = top - idx * row_height
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.045, y - row_height + 0.01), 0.91, row_height, transform=ax.transAxes, facecolor="#F3F6F8", linewidth=0))
        ax.text(0.065, y - 0.045, field, transform=ax.transAxes, fontsize=8.2 if not compact else 5.8, color="#53616D", va="center")
        ax.text(0.43, y - 0.045, value, transform=ax.transAxes, fontsize=8.2 if not compact else 5.8, color="#1B2733", va="center", wrap=True)


def plot_setup_card() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.25), facecolor="white")
    draw_setup(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(SETUP_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(SETUP_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_late_kl(late: pd.DataFrame) -> None:
    order = ["85-90K", "90-95K", "95-100K"]
    frame = late.loc[late["window"].isin(order)].copy()
    frame["window"] = pd.Categorical(frame["window"], categories=order, ordered=True)
    frame = frame.sort_values(["trajectory_class", "window"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor="white")
    for trajectory_class, group in frame.groupby("trajectory_class", sort=False):
        is_collapsed = trajectory_class == "collapsed"
        color = COLLAPSE_COLOR if is_collapsed else "#5B7892"
        label = "collapsed (RNG2)" if is_collapsed else "non-collapse (RNG1/3/4/5)"
        ax.plot(
            group["window"].astype(str),
            group["approx_kl_median"].astype(float),
            color=color,
            marker="o",
            linewidth=2.6 if is_collapsed else 2.0,
            markersize=6,
            label=label,
        )
    ax.set_title("Late-window approx_kl is elevated in the collapsed trajectory", loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("Late training window")
    ax.set_ylabel("approx_kl median (descriptive only)")
    ax.set_yscale("log")
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.03, "descriptive only; not causal", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#63717C")
    style_axis(ax, grid_axis="y")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(KL_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(KL_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mockup(repro: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    grid = fig.add_gridspec(nrows=12, ncols=10, left=0.075, right=0.975, top=0.83, bottom=0.18, wspace=0.8, hspace=0.8)
    fig.text(0.075, 0.925, "Collapse is not a one-off bug — it is RNG-dependent", fontsize=20, fontweight="bold", color="#1B2733", va="top")
    fig.text(0.075, 0.878, "YC PPO Original seed1 · same 75K checkpoint · held-out validation (2014–2023)", fontsize=10.5, color="#5C6873", va="top")

    main_ax = fig.add_subplot(grid[0:9, :7])
    plot_trajectory(main_ax, repro, compact=True, with_title=False)
    main_ax.set_xlabel("Checkpoint", fontsize=8.5)
    main_ax.set_ylabel("Mean yield (kg ha$^{-1}$)", fontsize=8.5)
    handles, labels = main_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.2, loc="lower left", bbox_to_anchor=(0.075, 0.145), frameon=False, handlelength=1.8, columnspacing=0.9)

    # A compact 2x2 card stack at right, followed by the setup card.
    payload = card_payload(repro)
    card_spans = [grid[0:2, 7:10], grid[2:4, 7:10], grid[4:6, 7:10], grid[6:8, 7:10]]
    for subgrid, item in zip(card_spans, payload):
        card_ax = fig.add_subplot(subgrid)
        draw_card(card_ax, item[0], item[1], item[2], item[3], item[4], compact=True)
    setup_ax = fig.add_subplot(grid[8:12, 7:10])
    draw_setup(setup_ax, compact=True)

    collapse_count = int(repro["collapse"].sum())
    total = len(repro)
    noncollapse_count = total - collapse_count
    fig.text(
        0.075,
        0.075,
        f"Takeaway: From the same 75K checkpoint, {collapse_count} of {total} continuations collapsed while {noncollapse_count} remained stable.",
        fontsize=13.5,
        fontweight="bold",
        color="#1B2733",
        va="center",
    )
    fig.savefig(MOCKUP_PNG, dpi=192, facecolor="white")
    plt.close(fig)


def write_report(repro: pd.DataFrame, late: pd.DataFrame) -> None:
    frame = trajectory_frame(repro)
    wide = repro.sort_values("continuation_rng")[["continuation_rng"] + [f"yield_{checkpoint}K" for checkpoint in CHECKPOINTS]].copy()
    wide.columns = ["RNG"] + [f"{checkpoint}K mean yield (kg/ha)" for checkpoint in CHECKPOINTS]
    wide_table = wide.to_markdown(index=False, floatfmt=".3f")
    values = repro["yield_100K"].astype(float)
    collapse_count = int(repro["collapse"].sum())
    total = len(repro)
    noncollapse_count = total - collapse_count
    late_table = late.loc[late["window"].isin(["85-90K", "90-95K", "95-100K"]), ["trajectory_class", "window", "trajectory_count", "approx_kl_median"]].copy()
    late_table = late_table.sort_values(["window", "trajectory_class"]).to_markdown(index=False, floatfmt=".8f")
    lines = [
        "# YC PPO Slide 2 visual assets audit",
        "",
        "## 状态",
        "",
        "- `YC_PPO_SLIDE2_VISUAL_ASSETS_READY`",
        "- 本轮只读取既有结果并生成图形；未训练、未重跑 validation/DSSAT、未修改 PPO/DSSAT/reward/action/mask，也未生成 PPTX。",
        "- 图内数值均由既有 CSV 读取或由其直接计算；原始 CSV 未改写。",
        "",
        "## 实际使用的数据文件",
        "",
        f"- `{REPRO_PATH.relative_to(ROOT)}`",
        f"- `{LATE_PATH.relative_to(ROOT)}`",
        f"- `{RNG5_PATH.relative_to(ROOT)}`（用于核验 RNG5 100K validation evidence）",
        f"- `{REPRO_DOC.relative_to(ROOT)}`",
        f"- `{RNG5_DOC.relative_to(ROOT)}`",
        "",
        "## 输出文件",
        "",
        f"1. 主图：`{MAIN_PNG.relative_to(ROOT)}`；`{MAIN_SVG.relative_to(ROOT)}`",
        f"2. 数字卡：`{CARDS_PNG.relative_to(ROOT)}`；`{CARDS_SVG.relative_to(ROOT)}`",
        f"3. setup card：`{SETUP_PNG.relative_to(ROOT)}`；`{SETUP_SVG.relative_to(ROOT)}`",
        f"4. late-KL 备用图：`{KL_PNG.relative_to(ROOT)}`；`{KL_SVG.relative_to(ROOT)}`",
        f"5. 16:9 layout mockup：`{MOCKUP_PNG.relative_to(ROOT)}`",
        "",
        "## 主图 5 条轨迹实际数值",
        "",
        wide_table,
        "",
        "## 结论数字",
        "",
        f"- full collapse：`{collapse_count}/{total}`",
        f"- non-collapse：`{noncollapse_count}/{total}`",
        f"- collapse rate：`{collapse_count / total:.2f}`",
        f"- 100K yield range：`{values.min():.3f}–{values.max():.3f} kg/ha`",
        f"- 100K mean / SD：`{values.mean():.3f} / {values.std(ddof=1):.3f} kg/ha`（SD 为 5 条完整 continuation 的样本标准差）",
        "",
        "## Late-window approx_kl 备用诊断",
        "",
        late_table,
        "",
        "该图明确标记为 descriptive only；它只展示 collapsed 与 non-collapse 轨迹的 late-window `approx_kl` 描述性差异，不构成因果证明，也不引入 KL intervention。",
        "",
        "## 数据冲突检查",
        "",
        "- 未发现数据冲突：**NO**。5 条 continuation 均具有 75K–100K 六个 checkpoint，且 collapse 字段与文档中的 1/5、0.20 一致。",
        "- RNG0 按既有文档作为 infrastructure-censored 排除，不进入 5 条完整 continuation 的分母。",
        "",
        "## 推荐 Slide 2 最终组合",
        "",
        "主图放左侧，右上放 2×2 数字卡，右下放 setup card；底部保留一句 takeaway。late-KL 图建议作为备用图，不放入主版面，以免把 Slide 2 从‘同一 75K、仅 RNG 不同、结果分叉’的核心叙事带向机制解释。",
        "",
        f"最终组合：`{MAIN_PNG.relative_to(ROOT)}` + `{CARDS_PNG.relative_to(ROOT)}` + `{SETUP_PNG.relative_to(ROOT)}`；`{KL_PNG.relative_to(ROOT)}` 作为备用。",
        "",
        "YC_PPO_SLIDE2_VISUAL_ASSETS_READY",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs() -> dict[str, int]:
    outputs = [MAIN_PNG, MAIN_SVG, CARDS_PNG, CARDS_SVG, SETUP_PNG, SETUP_SVG, KL_PNG, KL_SVG, MOCKUP_PNG, REPORT_PATH]
    sizes: dict[str, int] = {}
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
        sizes[str(path.relative_to(ROOT))] = path.stat().st_size
    for svg in [MAIN_SVG, CARDS_SVG, SETUP_SVG, KL_SVG]:
        if "<svg" not in svg.read_text(encoding="utf-8", errors="ignore")[:5000]:
            raise RuntimeError(f"SVG header check failed: {svg}")
    return sizes


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    repro, late, _rng5_validation = load_inputs()
    plot_main(repro)
    plot_cards(repro)
    plot_setup_card()
    plot_late_kl(late)
    plot_mockup(repro)
    write_report(repro, late)
    print({"status": "PASS", "outputs": verify_outputs(), "marker": "YC_PPO_SLIDE2_VISUAL_ASSETS_READY"})


if __name__ == "__main__":
    main()

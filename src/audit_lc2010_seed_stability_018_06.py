from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03"
OUT_DIR = BASE_DIR / "018_06_lc2010_seed_stability_audit"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_06_lc2010_dqn_seed_stability_audit_record.md"

YIELD_TOL = 20.0


def read_seed(seed: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "DSSAT_auto_validation" / "lc2010_baseline_relative_dqn_smoke_017_12" / f"seed{seed}_5000steps" / "checkpoint_summary.csv"
    df = pd.read_csv(path)
    df.insert(0, "seed", seed)
    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return df


def classify_checkpoint(row: pd.Series, auto_yield: float, ext_yield: float, ext_i: float, ext_n: float) -> dict:
    y = float(row["final_grain_kg_ha"])
    i = float(row["event_irrigation_total"])
    n = float(row["event_fertilizer_total"])
    return {
        "matches_auto_yield": y >= auto_yield - YIELD_TOL,
        "matches_extension_yield": y >= ext_yield - YIELD_TOL,
        "saves_water_vs_extension": i < ext_i,
        "saves_nitrogen_vs_extension": n < ext_n,
        "not_more_n_than_extension": n <= ext_n + 1e-6,
    }


def best_by_reward(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(["total_reward", "final_grain_kg_ha"], ascending=[False, False]).iloc[0]


def best_by_yield_then_resource(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["resource_sum"] = tmp["event_irrigation_total"] + tmp["event_fertilizer_total"]
    return tmp.sort_values(["final_grain_kg_ha", "resource_sum"], ascending=[False, True]).iloc[0]


def plot(checkpoints: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {0: "#117733", 1: "#CC6677"}
    for seed, sub in checkpoints.groupby("seed"):
        axes[0].plot(sub["checkpoint_step"], sub["final_grain_kg_ha"], marker="o", color=colors.get(seed), label=f"seed{seed}")
        axes[1].plot(sub["checkpoint_step"], sub["event_irrigation_total"], marker="o", color=colors.get(seed), label=f"seed{seed}")
        axes[2].plot(sub["checkpoint_step"], sub["event_fertilizer_total"], marker="o", color=colors.get(seed), label=f"seed{seed}")
    axes[0].axhline(8738, color="#DDCC77", lw=1.2, label="DSSAT auto")
    axes[0].axhline(8739, color="#4477AA", lw=1.2, ls="--", label="official expert")
    axes[1].axhline(198.8, color="#4477AA", lw=1.2, ls="--", label="official expert")
    axes[2].axhline(247.0, color="#4477AA", lw=1.2, ls="--", label="official expert")
    for ax, title, ylabel in [
        (axes[0], "Grain yield by checkpoint", "kg/ha"),
        (axes[1], "Irrigation by checkpoint", "mm"),
        (axes[2], "Nitrogen by checkpoint", "kg/ha"),
    ]:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E8E8E8", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("018_06 LC2010 DQN seed stability audit", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = pd.concat([read_seed(0), read_seed(1)], ignore_index=True)

    lc_comp = pd.read_csv(BASE_DIR / "018_04_lc2010_complete_comparison.csv")
    auto = lc_comp[lc_comp["scenario_label"].eq("DSSAT auto")].iloc[0]
    ext = lc_comp[lc_comp["scenario_label"].eq("Official extension expert fixed DAP")].iloc[0]
    auto_yield = float(auto["grain_yield_kg_ha"])
    ext_yield = float(ext["grain_yield_kg_ha"])
    ext_i = float(ext["irrigation_mm"])
    ext_n = float(ext["nitrogen_kg_ha"])

    best_rows = []
    for seed, sub in checkpoints.groupby("seed"):
        for selection, row in [
            ("best_reward", best_by_reward(sub)),
            ("best_yield_then_resource", best_by_yield_then_resource(sub)),
        ]:
            flags = classify_checkpoint(row, auto_yield, ext_yield, ext_i, ext_n)
            best_rows.append(
                {
                    "seed": int(seed),
                    "selection": selection,
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "final_grain_kg_ha": float(row["final_grain_kg_ha"]),
                    "final_biomass_kg_ha": float(row["final_biomass_kg_ha"]),
                    "irrigation_mm": float(row["event_irrigation_total"]),
                    "nitrogen_kg_ha": float(row["event_fertilizer_total"]),
                    "max_water_stress": float(row["max_water_stress"]),
                    "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                    "total_reward": float(row["total_reward"]),
                    **flags,
                }
            )
    best = pd.DataFrame(best_rows)

    # Stable success requires best-reward selections from both seeds to satisfy all core conditions.
    best_reward = best[best["selection"].eq("best_reward")]
    stable_yield = bool(best_reward["matches_auto_yield"].all() and best_reward["matches_extension_yield"].all())
    stable_water = bool(best_reward["saves_water_vs_extension"].all())
    stable_n = bool(best_reward["not_more_n_than_extension"].all())
    stable_success = stable_yield and stable_water and stable_n
    if stable_success and best_reward["saves_nitrogen_vs_extension"].all():
        conclusion = "LC2010 在两个 seed 的 best-reward checkpoint 上均追平关键产量基线，并且均节水节氮，可作为强稳定候选。"
    elif stable_success:
        conclusion = "LC2010 在两个 seed 的 best-reward checkpoint 上均追平关键产量基线，且均不超过官方推广氮投入并节水；但节氮幅度存在 seed 差异。"
    else:
        conclusion = "LC2010 产量较稳定，但资源效率跨 seed 不稳定；目前应作为 promising candidate，而不是稳定成功案例。"

    checkpoints.to_csv(OUT_DIR / "018_06_lc2010_seed_checkpoint_table.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "018_06_lc2010_seed_best_summary.csv", index=False, encoding="utf-8-sig")
    plot(checkpoints, best, OUT_DIR / "figures" / "018_06_lc2010_seed_stability.png")

    lines = [
        "# 018_06 LC2010 DQN 跨 seed 稳定性复核记录",
        "",
        "## 做了什么",
        "",
        "本轮没有训练、没有重跑 DSSAT。只读取 LC2010 现有 seed0/seed1 checkpoint summary，并与 LC2010 DSSAT auto 和 official extension expert 进行对照。",
        "",
        "## best checkpoint 汇总",
        "",
        md_table(best),
        "",
        "## 判定",
        "",
        f"- stable_yield = {stable_yield}",
        f"- stable_water_saving_vs_extension = {stable_water}",
        f"- stable_not_more_n_than_extension = {stable_n}",
        f"- stable_success = {stable_success}",
        "",
        conclusion,
        "",
        "## 解释",
        "",
        "seed0 的 best-reward checkpoint 非常理想：产量追平/略高于 auto 和官方推广 expert，同时灌溉、施氮都更少。seed1 的 best-reward checkpoint 产量也追平关键基线，但用水和用氮明显高于 seed0，说明 LC2010 当前更像“产量稳定、资源效率不稳定”。",
        "",
        "## 下一步建议",
        "",
        "如果要把 LC2010 写成稳定成功案例，需要继续做一个小规模 seed 稳定性补充：要么增加 seed2，要么延长/复核 seed1 的 checkpoint 选择。但在当前阶段，它已经足以作为 DQN 有潜力的代表案例之一。",
        "",
        "## 输出文件",
        "",
        f"- checkpoint table: `{(OUT_DIR / '018_06_lc2010_seed_checkpoint_table.csv').relative_to(PROJECT_ROOT)}`",
        f"- best summary: `{(OUT_DIR / '018_06_lc2010_seed_best_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- figure: `{(OUT_DIR / 'figures' / '018_06_lc2010_seed_stability.png').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(best.to_string(index=False))
    print(conclusion)


if __name__ == "__main__":
    main()


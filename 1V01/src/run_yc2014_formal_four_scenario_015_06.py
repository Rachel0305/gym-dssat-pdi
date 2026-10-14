from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_weather


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_015_06"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_06_yc2014_formal_four_scenario_record.md"

BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_four_scenario_smoke_013_03"
BASE_DAILY = BASE_DIR / "013_03_yc2014_four_scenario_daily.csv"
BASE_SUMMARY = BASE_DIR / "013_03_yc2014_four_scenario_summary.csv"

SEED0_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_diagnostic_015_04" / "seed0"
SEED1_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_seed1_015_05" / "seed1"
SEED0_SUMMARY = SEED0_DIR / "015_04_yc2014_unified_dqn_checkpoint_summary.csv"
SEED1_SUMMARY = SEED1_DIR / "015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv"

SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "dqn_best": "DQN best checkpoint",
}
SCENARIO_COLORS = {
    "null": "#404040",
    "recorded": "#B2182B",
    "dssat_auto": "#B8860B",
    "dqn_best": "#238B45",
}
SCENARIO_STYLES = {
    "null": "-",
    "recorded": "--",
    "dssat_auto": "-",
    "dqn_best": "-",
}
SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "dqn_best"]
WATER_COST = 1.0
N_COST = 5.0

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def checkpoint_daily_path(seed: int, checkpoint_step: int) -> Path:
    suffix = f"{int(checkpoint_step // 1000)}k"
    if seed == 0:
        return SEED0_DIR / f"015_04_yc2014_checkpoint_{suffix}_daily.csv"
    if seed == 1:
        return SEED1_DIR / f"015_05_yc2014_seed1_checkpoint_{suffix}_daily.csv"
    raise ValueError(f"Unsupported seed: {seed}")


def select_best_checkpoint(summary_path: Path) -> dict[str, float]:
    """Select by grain first, then lower fertilizer and lower irrigation."""
    df = pd.read_csv(summary_path)
    sort_cols = ["final_grain_kg_ha", "action_fertilizer_total", "action_irrigation_total"]
    ascending = [False, True, True]
    best = df.sort_values(sort_cols, ascending=ascending).iloc[0]
    return best.to_dict()


def parse_plantgro_dap_doy(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    in_table = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("@YEAR") and "DOY" in line and "DAP" in line:
                in_table = True
                continue
            if not in_table:
                continue
            if not line.strip() or line.startswith("*") or line.startswith("!"):
                in_table = False
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                year = int(parts[0])
                doy = int(parts[1])
                dap = int(parts[3])
            except ValueError:
                continue
            if year == 2014 and dap >= 0:
                rows.append({"dap": dap, "doy": doy})
    if not rows:
        raise RuntimeError(f"No DAP/DOY rows parsed from {path}")
    return pd.DataFrame(rows).drop_duplicates("dap").sort_values("dap")


def add_rain_by_dap(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    plantgro = BASE_DIR / "null" / "pdi_tmp_snapshot_eval" / "PlantGro.OUT"
    dap_doy = parse_plantgro_dap_doy(plantgro)
    weather = parse_weather("YC", 2014).rename(columns={"rain": "rain_mm"})
    rain_by_dap = dap_doy.merge(weather[["doy", "rain_mm"]], on="doy", how="left")
    rain_by_dap["rain_mm"] = rain_by_dap["rain_mm"].fillna(0.0)
    daily = daily.drop(columns=["rain"], errors="ignore")
    daily = daily.merge(rain_by_dap[["dap", "doy", "rain_mm"]], on="dap", how="left", suffixes=("", "_mapped"))
    if "doy_mapped" in daily.columns:
        daily["doy"] = daily["doy"].fillna(daily["doy_mapped"])
        daily = daily.drop(columns=["doy_mapped"])
    daily["rain"] = daily["rain_mm"].fillna(0.0)
    daily = daily.drop(columns=["rain_mm"], errors="ignore")
    return daily


def normalize_base_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["scenario"] = daily["scenario"].fillna("null")
    daily.loc[daily["scenario"].eq("recorded_shifted"), "scenario"] = "recorded"
    daily = daily[daily["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    daily["seed"] = np.nan
    daily["checkpoint_step"] = np.nan
    daily["source"] = "base_013_03"
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    return daily


def load_dqn_representative() -> pd.DataFrame:
    best0 = select_best_checkpoint(SEED0_SUMMARY)
    daily_path = checkpoint_daily_path(0, int(best0["checkpoint_step"]))
    dqn = pd.read_csv(daily_path).copy()
    dqn = dqn[dqn["step"].notna()].copy()
    dqn["scenario"] = "dqn_best"
    dqn["seed"] = 0
    dqn["checkpoint_step"] = int(best0["checkpoint_step"])
    dqn["source"] = f"seed0_checkpoint_{int(best0['checkpoint_step'])}"
    dqn["irrigation_mm"] = 0.0
    dqn["fertilizer_kg_ha"] = 0.0
    return dqn


def collapse_daily_crop_rows(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate DAP rows before injecting sparse management events."""
    preferred_cols = [
        "scenario",
        "seed",
        "checkpoint_step",
        "source",
        "dap",
        "step",
        "yrdoy",
        "doy",
        "grnwt",
        "topwt",
        "swfac",
        "nstres",
        "irrigation_mm",
        "fertilizer_kg_ha",
        "action_index",
        "reward",
        "rain",
    ]
    for col in preferred_cols:
        if col not in daily.columns:
            daily[col] = np.nan
    daily = daily[preferred_cols].copy()
    agg = {
        "step": "last",
        "yrdoy": "last",
        "doy": "last",
        "grnwt": "last",
        "topwt": "last",
        "swfac": "last",
        "nstres": "last",
        "irrigation_mm": "sum",
        "fertilizer_kg_ha": "sum",
        "action_index": "last",
        "reward": "last",
        "rain": "first",
    }
    return (
        daily.groupby(["scenario", "seed", "checkpoint_step", "source", "dap"], dropna=False, as_index=False)
        .agg(agg)
        .sort_values(["scenario", "seed", "dap"], na_position="first")
        .reset_index(drop=True)
    )


def inject_sparse_management_events(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    mgmt = events[events["operation"].isin(["Irrigation", "Fertilizer"])].copy()
    for _, event in mgmt.iterrows():
        mask = daily["scenario"].eq(event["scenario"]) & daily["dap"].round().eq(int(event["dap"]))
        if pd.notna(event.get("seed", np.nan)):
            mask = mask & daily["seed"].eq(float(event["seed"]))
        if event["operation"] == "Irrigation":
            daily.loc[mask, "irrigation_mm"] += float(event["amount"])
        elif event["operation"] == "Fertilizer":
            daily.loc[mask, "fertilizer_kg_ha"] += float(event["amount"])
    return daily


def add_cumulative_reward_proxy(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["reward_proxy"] = np.nan
    for scenario in SCENARIO_ORDER:
        mask = daily["scenario"].eq(scenario)
        sub = daily.loc[mask].sort_values(["dap", "step"], na_position="last").copy()
        if sub.empty:
            continue
        grain_delta = sub["grnwt"].fillna(0).diff().fillna(sub["grnwt"].fillna(0)).clip(lower=0)
        cost = WATER_COST * sub["irrigation_mm"].fillna(0) + N_COST * sub["fertilizer_kg_ha"].fillna(0)
        daily.loc[sub.index, "reward_proxy"] = (grain_delta - cost).cumsum()
    return daily


def assemble_daily() -> pd.DataFrame:
    base_daily = normalize_base_daily(pd.read_csv(BASE_DAILY))
    dqn_daily = load_dqn_representative()
    combined = pd.concat([base_daily, dqn_daily], ignore_index=True, sort=False)
    combined = add_rain_by_dap(combined)
    combined = collapse_daily_crop_rows(combined)
    combined = inject_sparse_management_events(combined, parse_management_events())
    combined = add_cumulative_reward_proxy(combined)
    return combined


def assemble_summary(daily: pd.DataFrame) -> pd.DataFrame:
    _, base_summary = pd.read_csv(BASE_DAILY), pd.read_csv(BASE_SUMMARY)
    base_summary = base_summary.copy()
    base_summary["scenario"] = base_summary["scenario"].fillna("null")
    base_summary.loc[base_summary["scenario"].eq("recorded_shifted"), "scenario"] = "recorded"
    base_summary = base_summary[base_summary["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()

    rows = []
    for scenario in ["null", "recorded", "dssat_auto"]:
        ssum = base_summary[base_summary["scenario"].eq(scenario)].iloc[0].to_dict()
        sub = daily[daily["scenario"].eq(scenario)]
        rows.append(
            {
                "scenario": scenario,
                "seed": np.nan,
                "checkpoint_step": np.nan,
                "irrigation_total_mm": float(ssum.get("event_irrigation_total", sub["irrigation_mm"].fillna(0).sum())),
                "fertilizer_total_kg_ha": float(ssum.get("event_fertilizer_total", sub["fertilizer_kg_ha"].fillna(0).sum())),
                "final_grain_kg_ha": float(ssum["final_grain_kg_ha"]),
                "final_biomass_kg_ha": float(ssum["final_biomass_kg_ha"]),
                "max_water_stress": float(ssum["max_water_stress"]),
                "max_nitrogen_stress": float(ssum["max_nitrogen_stress"]),
                "reward_proxy_final": float(sub["reward_proxy"].dropna().iloc[-1]) if sub["reward_proxy"].notna().any() else np.nan,
                "source": "base_013_03",
                "representative_for_figure": scenario != "dqn_best",
            }
        )

    for seed, summary_path in [(0, SEED0_SUMMARY), (1, SEED1_SUMMARY)]:
        best = select_best_checkpoint(summary_path)
        irrigation = float(best["mgmt_event_irrigation_total"])
        fertilizer = float(best["mgmt_event_fertilizer_total"])
        grain = float(best["final_grain_kg_ha"])
        rows.append(
            {
                "scenario": "dqn_best",
                "seed": seed,
                "checkpoint_step": int(best["checkpoint_step"]),
                "irrigation_total_mm": irrigation,
                "fertilizer_total_kg_ha": fertilizer,
                "final_grain_kg_ha": grain,
                "final_biomass_kg_ha": float(best["final_biomass_kg_ha"]),
                "max_water_stress": float(best["max_water_stress"]),
                "max_nitrogen_stress": float(best["max_nitrogen_stress"]),
                "reward_proxy_final": grain - WATER_COST * irrigation - N_COST * fertilizer,
                "source": f"seed{seed}_best_checkpoint",
                "representative_for_figure": seed == 0,
            }
        )
    return pd.DataFrame(rows)


def parse_management_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_event_dirs = {
        "null": BASE_DIR / "null" / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT",
        "recorded": BASE_DIR / "recorded" / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT",
        "dssat_auto": BASE_DIR / "dssat_auto" / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT",
    }
    event_re = re.compile(r"^\s*\d+\s+[A-Z]{3}\s+\d+,\s+\d{4}\s+(\d+)\s+\d+\s+(-?\d+).*?(Irrigation|Fertilizer|Harvest Yield)\s+([0-9.]+)")
    for scenario, path in base_event_dirs.items():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = event_re.match(line)
            if not m:
                continue
            doy, dap, operation, amount = m.groups()
            rows.append(
                {
                    "scenario": scenario,
                    "seed": np.nan,
                    "checkpoint_step": np.nan,
                    "doy": int(doy),
                    "dap": int(dap),
                    "operation": operation,
                    "amount": float(amount),
                    "source": str(path.relative_to(PROJECT_ROOT)),
                }
            )
    for seed, summary_path in [(0, SEED0_SUMMARY), (1, SEED1_SUMMARY)]:
        best = select_best_checkpoint(summary_path)
        d = pd.read_csv(checkpoint_daily_path(seed, int(best["checkpoint_step"])))
        for _, row in d[d["irrigation_mm"].fillna(0) > 1e-8].iterrows():
            rows.append(
                {
                    "scenario": "dqn_best",
                    "seed": seed,
                    "checkpoint_step": int(best["checkpoint_step"]),
                    "doy": row.get("doy", np.nan),
                    "dap": int(row["dap"]),
                    "operation": "Irrigation",
                    "amount": float(row["irrigation_mm"]),
                    "source": f"seed{seed}_checkpoint_{int(best['checkpoint_step'])}",
                }
            )
        for _, row in d[d["fertilizer_kg_ha"].fillna(0) > 1e-8].iterrows():
            rows.append(
                {
                    "scenario": "dqn_best",
                    "seed": seed,
                    "checkpoint_step": int(best["checkpoint_step"]),
                    "doy": row.get("doy", np.nan),
                    "dap": int(row["dap"]),
                    "operation": "Fertilizer",
                    "amount": float(row["fertilizer_kg_ha"]),
                    "source": f"seed{seed}_checkpoint_{int(best['checkpoint_step'])}",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["scenario", "seed", "checkpoint_step", "doy", "dap", "operation", "amount", "source"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["scenario", "seed", "checkpoint_step", "dap", "operation", "amount"], keep="first")
        .sort_values(["scenario", "seed", "dap", "operation"], na_position="first")
        .reset_index(drop=True)
    )


def plot_four_scenario(daily: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16.2, 14.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.22},
    )

    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("YC2014 formal four-scenario comparison", loc="left", fontsize=14, fontweight="bold")

    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        style = SCENARIO_STYLES[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, lw=2.0, ls=style, label=label)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, lw=2.0, ls=style, label=label)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linestyles=style, linewidth=2.2, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=58, color=color, edgecolor="#FFFFFF", linewidth=0.7, zorder=4)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, lw=2.0, ls=style)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, lw=1.5, ls=":", alpha=0.85)
        axes[5].plot(sub["dap"], sub["reward_proxy"], color=color, lw=2.0, ls=style, label=label)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[5].set_ylabel("Cum.\nproxy")
    axes[5].set_xlabel("DAP")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[4].set_title("Crop outcome: solid/dashed scenario style = grain; dotted = aboveground biomass", loc="left", fontsize=10)
    axes[5].set_title(f"Cumulative reward proxy: grain increment - {WATER_COST:g}×irrigation - {N_COST:g}×fertilizer", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(left=0)
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    handles = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2, ls=SCENARIO_STYLES[s], label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]
    axes[3].legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9)
    axes[5].legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.98, hspace=0.22)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_record(summary: pd.DataFrame, out_daily: Path, out_summary: Path, out_events: Path, fig_path: Path) -> None:
    dqn = summary[summary["scenario"].eq("dqn_best")].copy()
    seed0 = dqn[dqn["seed"].eq(0)].iloc[0]
    seed1 = dqn[dqn["seed"].eq(1)].iloc[0]
    recorded = summary[summary["scenario"].eq("recorded")].iloc[0]
    lines = [
        "# 015_06 YC2014 正式四情景对照与 best checkpoint 汇总记录",
        "",
        "## 结论",
        "",
        "YC2014 的正式四情景材料已补齐降雨、管理事件、产量/生物量和统一 reward proxy。DQN 的主图代表采用 seed0 best checkpoint，因为它在与专家策略相同 grain yield 下使用更少氮肥；seed1 作为跨 seed 稳定性验证保留在汇总表中。",
        "",
        "## 关键数字",
        "",
        f"- recorded expert: grain={recorded['final_grain_kg_ha']:.1f} kg/ha, I={recorded['irrigation_total_mm']:.1f} mm, N={recorded['fertilizer_total_kg_ha']:.1f} kg/ha, proxy={recorded['reward_proxy_final']:.1f}",
        f"- DQN seed0: checkpoint={int(seed0['checkpoint_step'])}, grain={seed0['final_grain_kg_ha']:.1f} kg/ha, I={seed0['irrigation_total_mm']:.1f} mm, N={seed0['fertilizer_total_kg_ha']:.1f} kg/ha, proxy={seed0['reward_proxy_final']:.1f}",
        f"- DQN seed1: checkpoint={int(seed1['checkpoint_step'])}, grain={seed1['final_grain_kg_ha']:.1f} kg/ha, I={seed1['irrigation_total_mm']:.1f} mm, N={seed1['fertilizer_total_kg_ha']:.1f} kg/ha, proxy={seed1['reward_proxy_final']:.1f}",
        "",
        "解释口径：如果只看产量，DQN 与专家策略持平；如果看同一经济 proxy，DQN seed0 因少用氮而优于专家策略。这不是声称 DQN 已经找到更高产策略，而是说明它已经找到更高资源效率的策略。",
        "",
        "## 输出文件",
        "",
        f"- 主图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        f"- 日值总表：`{out_daily.relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{out_summary.relative_to(PROJECT_ROOT)}`",
        f"- 管理事件表：`{out_events.relative_to(PROJECT_ROOT)}`",
        "",
        "## 下一步",
        "",
        "015_07 不直接加长训练，而是先做低成本 headroom 诊断：在当前动作空间、预算和操作窗口内，是否存在超过 recorded expert 9418 kg/ha 的调度组合。如果没有，继续训练也不可能证明产量超越；应换年份/站点或调整研究问题。如果有，再围绕这些候选调度继续训练或扩大动作空间。",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_dir = OUT_DIR / "seed0_seed1_best"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily = assemble_daily()
    summary = assemble_summary(daily)
    events = parse_management_events()

    daily_path = run_dir / "015_06_yc2014_formal_four_scenario_daily.csv"
    summary_path = run_dir / "015_06_yc2014_formal_four_scenario_summary.csv"
    events_path = run_dir / "015_06_yc2014_formal_four_scenario_management_events.csv"
    fig_path = fig_dir / "yc2014_formal_four_scenario.png"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    plot_four_scenario(daily, fig_path)
    write_record(summary, daily_path, summary_path, events_path, fig_path)
    print(summary.to_string(index=False))
    print(f"\nWrote: {fig_path}")


if __name__ == "__main__":
    main()

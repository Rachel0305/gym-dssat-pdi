from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gymnasium_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hla_baseline_relative_dqn_checkpoint_015_12 as hla_dqn
import run_yc2014_baseline_relative_dqn_015_10 as yc_br
from ppo_evaluate import latest_observation_dict, scalar


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_hla2010_dqn_cross_site_transfer_probe_015_21"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_015_21_hla2010_to_yc2014_dqn_cross_site_transfer_probe_record.md"

TRAIN_YEAR = 2010
TEST_SITE = "YC"
TEST_YEAR = 2014
TRANSFER_MODELS = [
    {"train_seed": 0, "checkpoint": 35000},
    {"train_seed": 1, "checkpoint": 25000},
]

BASELINE_DAILY_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_formal_four_scenario_015_06"
    / "seed0_seed1_best"
    / "015_06_yc2014_formal_four_scenario_daily.csv"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_formal_four_scenario_015_06"
    / "seed0_seed1_best"
    / "015_06_yc2014_formal_four_scenario_summary.csv"
)

COLORS = {
    "null": "#222222",
    "recorded": "#C9252D",
    "dssat_auto": "#B8860B",
    "local_dqn_best": "#2E8B57",
    "transfer_hla2010_seed0": "#255C99",
    "transfer_hla2010_seed1": "#7B3F98",
}

LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "local_dqn_best": "YC2014 local DQN best",
    "transfer_hla2010_seed0": "HLA2010-trained DQN seed0",
    "transfer_hla2010_seed1": "HLA2010-trained DQN seed1",
}


def model_path(train_seed: int, checkpoint: int) -> Path:
    return (
        hla_dqn.OUT_ROOT
        / str(TRAIN_YEAR)
        / f"baseline_relative_seed{train_seed}_50000steps"
        / "models"
        / f"dqn_baseline_relative_checkpoint_{checkpoint}.zip"
    )


def prepare_test_case(scenario: str) -> dict[str, Any]:
    run_dir = yc_br.yc.prepare_case_for_scenario(scenario)
    return json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))


def evaluate_transfer(model, env_args: dict[str, Any], train_seed: int, checkpoint: int, scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    null_baseline = yc_br.get_null_baseline_yield()
    eval_env = yc_br.make_train_env(env_args, null_baseline)
    rows: list[dict[str, Any]] = []
    snapshot_dir = OUT_DIR / scenario / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = eval_env.reset()
        for step in range(380):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(eval_env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "requested_year": TEST_YEAR,
                    "scenario": scenario,
                    "source": "hla2010_to_yc2014_transfer",
                    "train_year": TRAIN_YEAR,
                    "train_seed": train_seed,
                    "train_checkpoint_step": checkpoint,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(safe_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe_action.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp, snapshot_dir, dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    daily = attach_rain(daily)
    plantgro = yc_br.yc.parse_dssat_table(snapshot_dir / "PlantGro.OUT") if (snapshot_dir / "PlantGro.OUT").exists() else pd.DataFrame()
    events = yc_br.yc.parse_events_eval(snapshot_dir.parent, scenario)
    summary = {
        "requested_year": TEST_YEAR,
        "scenario": scenario,
        "label": LABELS[scenario],
        "train_year": TRAIN_YEAR,
        "train_seed": train_seed,
        "train_checkpoint_step": checkpoint,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "rain_total": float(daily[["dap", "rain"]].drop_duplicates("dap")["rain"].sum()) if not daily.empty and "rain" in daily.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
    }
    return daily, summary


def attach_rain(daily: pd.DataFrame) -> pd.DataFrame:
    rain = yc_br.yc.parse_weather(TEST_SITE, TEST_YEAR)
    if daily.empty:
        return daily.assign(rain=pd.Series(dtype=float))
    merged = daily.merge(rain, on="doy", how="left")
    merged["rain"] = merged["rain"].fillna(0.0)
    return merged


def load_baseline_and_local() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(BASELINE_DAILY_PATH)
    daily["scenario"] = daily["scenario"].fillna("null")
    daily = daily.rename(
        columns={
            "grnwt": "grnwt",
            "topwt": "topwt",
            "swfac": "swfac",
            "nstres": "nstres",
            "reward_proxy": "reward",
        }
    )
    daily["label"] = daily["scenario"].map(LABELS).fillna(daily["scenario"])
    # 只保留代表性的本地 DQN best（seed0）
    local_dqn = daily[(daily["scenario"].eq("dqn_best")) & (daily["seed"].fillna(-1).astype(int).eq(0))].copy()
    local_dqn["scenario"] = "local_dqn_best"
    keep = daily["scenario"].isin(["null", "recorded", "dssat_auto"])
    daily = pd.concat([daily[keep].copy(), local_dqn], ignore_index=True)

    summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    summary["scenario"] = summary["scenario"].fillna("null")
    summary = summary.rename(
        columns={
            "irrigation_total_mm": "irrigation_total",
            "fertilizer_total_kg_ha": "fertilizer_total",
            "final_grain_kg_ha": "final_gwad",
            "final_biomass_kg_ha": "final_cwad",
            "max_water_stress": "max_water_stress",
            "max_nitrogen_stress": "max_nitrogen_stress",
            "reward_proxy_final": "total_reward",
        }
    )
    local_sum = summary[(summary["scenario"].eq("dqn_best")) & (summary["seed"].fillna(-1).astype(int).eq(0))].copy()
    local_sum["scenario"] = "local_dqn_best"
    keep_sum = summary["scenario"].isin(["null", "recorded", "dssat_auto"])
    summary = pd.concat([summary[keep_sum].copy(), local_sum], ignore_index=True)
    summary["label"] = summary["scenario"].map(LABELS).fillna(summary["scenario"])
    return daily, summary


def plot_figure(all_daily: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    if all_daily.empty:
        return
    plot_order = ["null", "recorded", "dssat_auto", "local_dqn_best", "transfer_hla2010_seed0", "transfer_hla2010_seed1"]
    present = [x for x in plot_order if x in set(all_daily["scenario"].astype(str))]
    max_dap = int(np.nanmax(all_daily["dap"])) if not all_daily.empty else 130
    x_max = max(10, max_dap + 5)
    x_ticks = np.arange(0, x_max + 1, 25)
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True, gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.0, 1.1], "hspace": 0.24})

    rain = all_daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("YC2014 cross-site transfer probe: HLA2010-trained DQN on YC2014", loc="left", fontsize=14, weight="bold")

    for scenario in present:
        sub = all_daily[all_daily["scenario"].eq(scenario)].sort_values("dap")
        axes[1].plot(sub["dap"], sub["swfac"], lw=2.0, color=COLORS[scenario], label=LABELS[scenario])
        axes[2].plot(sub["dap"], sub["nstres"], lw=2.0, color=COLORS[scenario], label=LABELS[scenario])
        axes[4].plot(sub["dap"], sub["grnwt"], lw=2.0, color=COLORS[scenario], label=LABELS[scenario])
        axes[4].plot(sub["dap"], sub["topwt"], lw=1.5, ls="--", color=COLORS[scenario], alpha=0.9)

    for scenario in present:
        sub = all_daily[all_daily["scenario"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        irr = sub[sub["irrigation_mm"].fillna(0) > 0]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 0]
        if not irr.empty:
            axes[3].vlines(irr["dap"], 0, irr["irrigation_mm"], colors=color, lw=2.0, alpha=0.9)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], color=color, marker="^", s=36, zorder=3)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_xticks(x_ticks)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[1].legend(loc="upper left", ncol=2, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_record(smoke: bool, summary: pd.DataFrame, out_dir: Path) -> None:
    table_cols = [c for c in ["scenario", "label", "train_seed", "train_checkpoint_step", "final_gwad", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress", "total_reward"] if c in summary.columns]
    table = summary[table_cols].copy() if table_cols else summary.copy()
    md_lines = []
    if not table.empty:
        headers = list(table.columns)
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in table.iterrows():
            vals = []
            for col in headers:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{float(val):.3f}")
                else:
                    vals.append(str(val))
            md_lines.append("| " + " | ".join(vals) + " |")
    lines = [
        "# 015_21 HLA2010 -> YC2014 跨站点 DQN 迁移 probe",
        "",
        "## 目的",
        "",
        "- 不重新训练，只迁移评估 `HLA2010 baseline-relative DQN` 到 `YC2014`。",
        "- 判断这套 2010 训练模型是否只在 HLA 内有效，还是具备跨站点可迁移性。",
        "",
        "## 设置",
        "",
        f"- 训练年：HLA{TRAIN_YEAR}",
        f"- 测试站点年份：{TEST_SITE}{TEST_YEAR}",
        f"- 模式：{'smoke' if smoke else 'full'}",
        "- 奖励：baseline-relative，终止时 `max(0, GWAD_final - null_baseline)` 减去水/氮成本",
        "- 动作：9-action 离散动作，I∈{0,15,30}，N∈{0,50,100}",
        "- 约束：I<=120, N<=300, min_interval=7天",
        "",
        "## 结果摘要",
        "",
        *md_lines,
        "",
        "## 判读",
        "",
    ]
    if not summary.empty:
        transfer = summary[summary["scenario"].astype(str).str.startswith("transfer_hla2010_")].copy()
        if not transfer.empty:
            ok = transfer[transfer["final_gwad"].notna()].copy()
            fail = transfer[transfer["final_gwad"].isna()].copy()
            if not ok.empty:
                best = ok.sort_values("final_gwad", ascending=False).iloc[0]
                lines.append(
                    f"- 最佳迁移结果：{best['scenario']}，产量 {float(best['final_gwad']):.1f} kg/ha，灌溉 {float(best['irrigation_total']):.1f} mm，施氮 {float(best['fertilizer_total']):.1f} kg/ha。"
                )
            if not fail.empty and "failure_reason" in fail.columns:
                for _, row in fail.iterrows():
                    lines.append(f"- {row['scenario']} 未能直接迁移：{row['failure_reason']}")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def diagnostic_row(scenario: str, label: str, reason: str, train_seed: int | None = None, checkpoint: int | None = None) -> dict[str, Any]:
    return {
        "requested_year": TEST_YEAR,
        "scenario": scenario,
        "label": label,
        "train_year": TRAIN_YEAR if train_seed is not None else np.nan,
        "train_seed": train_seed if train_seed is not None else np.nan,
        "train_checkpoint_step": checkpoint if checkpoint is not None else np.nan,
        "final_gwad": np.nan,
        "final_cwad": np.nan,
        "irrigation_total": np.nan,
        "fertilizer_total": np.nan,
        "rain_total": np.nan,
        "max_water_stress": np.nan,
        "max_nitrogen_stress": np.nan,
        "total_reward": np.nan,
        "failure_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="只跑第一组迁移模型做链路验证")
    args = parser.parse_args()

    from stable_baselines3 import DQN

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_daily, baseline_summary = load_baseline_and_local()
    all_daily = [baseline_daily]
    all_summary = [baseline_summary]

    models = TRANSFER_MODELS[:1] if args.smoke else TRANSFER_MODELS
    for cfg in models:
        scenario = f"dqn_transfer_hla2010_seed{cfg['train_seed']}"
        output_scenario = f"transfer_hla2010_seed{cfg['train_seed']}"
        env_args = prepare_test_case(scenario)
        path = model_path(cfg["train_seed"], cfg["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        eval_env = yc_br.make_train_env(env_args, yc_br.get_null_baseline_yield())
        try:
            try:
                model = DQN.load(str(path), env=eval_env)
            except ValueError as exc:
                all_summary.append(pd.DataFrame([diagnostic_row(
                    output_scenario,
                    LABELS[output_scenario],
                    str(exc),
                    cfg["train_seed"],
                    cfg["checkpoint"],
                )]))
                continue
        finally:
            eval_env.close()
        daily, summary = evaluate_transfer(model, env_args, cfg["train_seed"], cfg["checkpoint"], output_scenario)
        all_daily.append(daily)
        all_summary.append(pd.DataFrame([summary]))

    daily_df = pd.concat(all_daily, ignore_index=True, sort=False)
    summary_df = pd.concat(all_summary, ignore_index=True, sort=False)

    daily_path = OUT_DIR / "015_21_yc2014_hla2010_transfer_daily.csv"
    summary_path = OUT_DIR / "015_21_yc2014_hla2010_transfer_summary.csv"
    events_rows = []
    for scenario in [x for x in summary_df["scenario"].dropna().astype(str).tolist() if x.startswith("transfer_hla2010_")]:
        events_path = OUT_DIR / scenario / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
        if events_path.exists():
            ev = yc_br.yc.parse_events_eval(OUT_DIR / scenario, scenario)
            if not ev.empty:
                events_rows.append(ev)
    events_df = pd.concat(events_rows, ignore_index=True) if events_rows else pd.DataFrame(columns=["scenario", "dap", "amount", "unit", "operation"])
    events_path = OUT_DIR / "015_21_yc2014_hla2010_transfer_management_events.csv"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    events_df.to_csv(events_path, index=False, encoding="utf-8-sig")

    fig_path = FIG_DIR / ("yc2014_hla2010_transfer_smoke.png" if args.smoke else "yc2014_hla2010_transfer_full.png")
    plot_figure(daily_df, summary_df, fig_path)
    write_record(args.smoke, summary_df, OUT_DIR)

    print(summary_df.to_string(index=False))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()

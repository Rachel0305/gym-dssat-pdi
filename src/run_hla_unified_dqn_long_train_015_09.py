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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events, prepare_case_at
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_unified_dqn_long_train_015_09"

ACTION_TABLE_9: dict[int, dict[str, float]] = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 15.0, "anfer": 0.0},
    2: {"amir": 30.0, "anfer": 0.0},
    3: {"amir": 0.0, "anfer": 50.0},
    4: {"amir": 15.0, "anfer": 50.0},
    5: {"amir": 30.0, "anfer": 50.0},
    6: {"amir": 0.0, "anfer": 100.0},
    7: {"amir": 15.0, "anfer": 100.0},
    8: {"amir": 30.0, "anfer": 100.0},
}

FREE_DAILY_WINDOWS = {
    "irrigation": [(1, 120)],
    "nitrogen": [(1, 120)],
}


def configure_shared_settings() -> None:
    yc.WATER_COST = 1.0
    yc.NITROGEN_COST = 5.0
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9


def make_env(env_args: dict[str, Any]):
    return yc.EconomicRewardWrapper(
        yc.YCDiscreteBudgetedWrapper(
            yc.make_raw_env(env_args),
            FREE_DAILY_WINDOWS["irrigation"],
            FREE_DAILY_WINDOWS["nitrogen"],
        )
    )


def evaluate_model(model, env_args: dict[str, Any], checkpoint_step: int, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_env = make_env(env_args)
    rows: list[dict[str, Any]] = []
    snapshot_dir = run_dir / f"checkpoint_{checkpoint_step}" / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": eval_env.last_safe_real_action.get("amir", 0.0),
                    "fertilizer_kg_ha": eval_env.last_safe_real_action.get("anfer", 0.0),
                    "action_index": eval_env.last_action_index,
                    "reward": float(reward),
                    "used_irrigation": float(eval_env.used_irrigation),
                    "used_nitrogen": float(eval_env.used_nitrogen),
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
    if not daily.empty:
        daily = yc.build_plot_df(daily)
    event_summary = parse_events(snapshot_dir / "MgmtEvent.OUT")
    plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT") if (snapshot_dir / "PlantGro.OUT").exists() else pd.DataFrame()
    summary = {
        "checkpoint_step": checkpoint_step,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        **event_summary,
    }
    return daily, summary


def plot_checkpoints(all_daily: pd.DataFrame, summary: pd.DataFrame, out_path: Path, year: int, seed: int, total_timesteps: int) -> None:
    steps = list(summary.sort_values("checkpoint_step")["checkpoint_step"])
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / max(1, len(steps) - 1)) for i, s in enumerate(steps)}
    fig, axes = plt.subplots(5, 1, figsize=(15.5, 13.5), sharex=False, gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.32})
    for checkpoint_step in steps:
        sub = all_daily[all_daily["checkpoint_step"].eq(checkpoint_step)].sort_values("dap")
        color = colors[checkpoint_step]
        label = f"{checkpoint_step // 1000}K"
        axes[0].plot(sub["dap"], sub["grnwt"], color=color, lw=1.6, label=label)
        axes[1].plot(sub["dap"], sub["nstres"], color=color, lw=1.6)
        axes[2].plot(sub["dap"], sub["swfac"], color=color, lw=1.6)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, lw=1.5, alpha=0.85)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=28, color=color, edgecolor="white", linewidth=0.4, zorder=3)

    summary = summary.sort_values("checkpoint_step")
    x = np.arange(len(summary))
    axes[4].bar(x - 0.22, summary["action_irrigation_total"], width=0.32, color="#74A9CF", label="I total")
    axes[4].bar(x + 0.10, summary["action_fertilizer_total"], width=0.32, color="#A1D99B", label="N total")
    ax2 = axes[4].twinx()
    ax2.plot(x, summary["final_grain_kg_ha"], color="#CB181D", marker="o", lw=1.9, label="Grain")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels([f"{int(v // 1000)}K" for v in summary["checkpoint_step"]])
    axes[4].set_xlabel("Checkpoint")
    axes[0].set_ylabel("GRNWT\nkg/ha")
    axes[1].set_ylabel("N stress")
    axes[2].set_ylabel("Water stress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("I/N total")
    ax2.set_ylabel("Final grain kg/ha")
    axes[0].set_title(f"HLA {year} unified DQN checkpoint diagnostic, seed{seed}, {total_timesteps // 1000}K training", loc="left", fontsize=13)
    axes[3].set_title("Irrigation as vertical lines; fertilization as triangles", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    axes[0].legend(loc="upper left", ncol=min(5, len(steps)), frameon=False, fontsize=8)
    axes[4].legend(loc="upper left", frameon=False, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(doc_path: Path, year: int, seed: int, total_timesteps: int, checkpoint_interval: int, daily_path: Path, summary_path: Path, fig_path: Path, summary: pd.DataFrame) -> None:
    view_cols = [
        "checkpoint_step",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
    ]
    view = summary[view_cols].sort_values("checkpoint_step").reset_index(drop=True)
    best_yield = view.loc[view["final_grain_kg_ha"].idxmax()]
    best_reward = view.loc[view["total_reward"].idxmax()]
    final = view.loc[view["checkpoint_step"].idxmax()]
    lines = [
        f"# 015_09 HLA{year} 统一 DQN 长训练记录",
        "",
        "## 设置",
        "",
        f"- 年份：HLA{year}",
        f"- 算法：DQN, seed={seed}",
        f"- 总步数：{total_timesteps}",
        f"- checkpoint 间隔：{checkpoint_interval}",
        "- 奖励：max(0, delta_grnwt) - 1.0 * irrigation - 5.0 * nitrogen",
        "- 动作空间：9-action，I in {0,15,30} mm, N in {0,50,100} kg/ha",
        "- 预算：I<=120 mm, N<=300 kg/ha",
        "- 单次上限：I<=30 mm, N<=100 kg/ha",
        "- 最小间隔：7 days",
        "- 管理模式：IRRIG=L, FERTI=L",
        "",
        "## 输出文件",
        "",
        f"- 日值：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 汇总：`{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- 过程图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## Checkpoint 结果",
        "",
        markdown_table(view),
        "",
        "## 关键判断",
        "",
        f"- 最高产量 checkpoint：{int(best_yield['checkpoint_step'])}，产量 {best_yield['final_grain_kg_ha']:.1f} kg/ha，I={best_yield['action_irrigation_total']:.1f} mm，N={best_yield['action_fertilizer_total']:.1f} kg/ha。",
        f"- 最高 reward checkpoint：{int(best_reward['checkpoint_step'])}，reward={best_reward['total_reward']:.1f}，产量 {best_reward['final_grain_kg_ha']:.1f} kg/ha。",
        f"- final checkpoint：{int(final['checkpoint_step'])}，产量 {final['final_grain_kg_ha']:.1f} kg/ha，I={final['action_irrigation_total']:.1f} mm，N={final['action_fertilizer_total']:.1f} kg/ha。",
        "- 如果 best checkpoint 明显优于 final model，则正式比较应采用 checkpoint selection，而不是只看 final。",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--tag", type=str, default="formal")
    args = parser.parse_args()

    configure_shared_settings()
    install_official_reward_module()

    year = int(args.year)
    timesteps = int(args.timesteps)
    seed = int(args.seed)
    checkpoint_interval = int(args.checkpoint_interval)

    run_dir = OUT_ROOT / str(year) / f"{args.tag}_seed{seed}_{timesteps}steps"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    prepare_case_at(year, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    env = make_env(env_args)
    model = None
    try:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=min(100, max(10, timesteps // 20)),
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )

        all_daily = []
        all_summary = []
        checkpoints = list(range(checkpoint_interval, timesteps + 1, checkpoint_interval))
        prev = 0
        for checkpoint in checkpoints:
            model.learn(total_timesteps=checkpoint - prev, reset_num_timesteps=False, progress_bar=False)
            prev = checkpoint
            ckpt_dir = run_dir / "models"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt_dir / f"dqn_checkpoint_{checkpoint}"))
            daily, summary = evaluate_model(model, env_args, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        env.close()

    all_daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)

    daily_path = run_dir / "dqn_eval_daily.csv"
    summary_path = run_dir / "checkpoint_summary.csv"
    fig_path = run_dir / "figures" / f"hla{year}_unified_dqn_seed{seed}_{timesteps}steps_checkpoint_diagnostic.png"
    doc_path = PROJECT_ROOT / "docs" / f"2026-07-01_015_09_hla{year}_unified_dqn_seed{seed}_{timesteps}steps_record.md"

    daily_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    all_daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_checkpoints(all_daily_df, summary_df, fig_path, year, seed, timesteps)
    write_record(doc_path, year, seed, timesteps, checkpoint_interval, daily_path, summary_path, fig_path, summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

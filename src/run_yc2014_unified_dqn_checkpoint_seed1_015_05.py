from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_seed1_015_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_05_yc2014_unified_dqn_checkpoint_seed1_record.md"
SEED0_SUMMARY_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_checkpoint_diagnostic_015_04"
    / "seed0"
    / "015_04_yc2014_unified_dqn_checkpoint_summary.csv"
)

SEED = 1
TOTAL_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL = 5_000
WATER_COST = 1.0
NITROGEN_COST = 5.0
SCENARIO = "dqn_unified_checkpoint_seed1"

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


def configure_yc_module() -> None:
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = SEED
    yc.TIMESTEPS = TOTAL_TIMESTEPS
    yc.WATER_COST = WATER_COST
    yc.NITROGEN_COST = NITROGEN_COST
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9
    yc.SCENARIO_ORDER = [SCENARIO]
    yc.SCENARIO_LABELS = {SCENARIO: "Unified DQN checkpoint seed1"}
    yc.SCENARIO_COLORS = {SCENARIO: "#386411"}


def make_train_env(env_args: dict):
    return yc.EconomicRewardWrapper(
        yc.YCDiscreteBudgetedWrapper(yc.make_raw_env(env_args), yc.FREE_DAILY_WINDOWS["irrigation"], yc.FREE_DAILY_WINDOWS["nitrogen"])
    )


def evaluate_model(model, env_args: dict, checkpoint_step: int) -> tuple[pd.DataFrame, dict]:
    eval_env = make_train_env(env_args)
    rows = []
    checkpoint_dir = OUT_DIR / f"seed{SEED}" / f"checkpoint_{checkpoint_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        obs, info = eval_env.reset()
        for step in range(240):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "scenario": f"checkpoint_{checkpoint_step}",
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
                    "reward": reward,
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, checkpoint_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    daily = yc.build_plot_df(daily)
    events = yc.parse_events_eval(checkpoint_dir, f"checkpoint_{checkpoint_step}")
    plantgro = parse_dssat_table(checkpoint_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "seed": SEED,
        "checkpoint_step": checkpoint_step,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
    }
    return daily, summary


def plot_checkpoints(all_daily: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    steps = list(summary.sort_values("checkpoint_step")["checkpoint_step"])
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / max(1, len(steps) - 1)) for i, s in enumerate(steps)}
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 13.5),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.32},
    )
    for checkpoint_step in steps:
        sub = all_daily[all_daily["checkpoint_step"].eq(checkpoint_step)].sort_values("dap")
        color = colors[checkpoint_step]
        label = f"{checkpoint_step // 1000}K"
        axes[0].plot(sub["dap"], sub["grnwt"], color=color, lw=1.6, label=label)
        axes[1].plot(sub["dap"], sub["nstres"], color=color, lw=1.6, label=label)
        axes[2].plot(sub["dap"], sub["swfac"], color=color, lw=1.6, label=label)
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
    axes[0].set_title("YC2014 unified DQN checkpoint diagnostic, seed1", loc="left", fontsize=13)
    axes[3].set_title("Irrigation as vertical lines; fertilization as triangles", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    axes[0].legend(loc="upper left", ncol=5, frameon=False, fontsize=8)
    axes[4].legend(loc="upper left", frameon=False, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
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


def load_seed0_best() -> dict[str, float] | None:
    if not SEED0_SUMMARY_PATH.exists():
        return None
    df = pd.read_csv(SEED0_SUMMARY_PATH)
    best = df.loc[df["final_grain_kg_ha"].idxmax()]
    return {
        "checkpoint_step": float(best["checkpoint_step"]),
        "final_grain_kg_ha": float(best["final_grain_kg_ha"]),
        "action_irrigation_total": float(best["action_irrigation_total"]),
        "action_fertilizer_total": float(best["action_fertilizer_total"]),
    }


def write_record(summary: pd.DataFrame, daily_path: Path, summary_path: Path, fig_path: Path) -> None:
    view_cols = [
        "seed",
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
    seed0_best = load_seed0_best()
    lines = [
        "# 015_05 YC2014 统一 DQN checkpoint selection seed1 复核记录",
        "",
        "## 目的",
        "",
        "015_04 证明 seed0 下 final model 不一定最好，checkpoint selection 能找到高产策略。本轮只将 seed 从 0 改为 1，其他设置完全不变，用于复核 YC2014 是否具备跨 seed 可复现性。",
        "",
        "## 固定设置",
        "",
        "- 站点年份：YC2014",
        "- 算法：DQN, seed=1",
        f"- 总步数：{TOTAL_TIMESTEPS}",
        f"- checkpoint 间隔：{CHECKPOINT_INTERVAL}",
        "- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - 5.0 × N_t",
        "- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha",
        "- 预算：I≤120 mm，N≤300 kg/ha",
        "- 最小操作间隔：7 days",
        "- 管理模式：IRRIG=L, FERTI=L",
        "",
        "## 输出文件",
        "",
        f"- 日值总表：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- 对比图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## seed1 汇总结果",
        "",
        markdown_table(view),
        "",
        "## 关键判断",
        "",
        f"- seed1 最高产量 checkpoint：{int(best_yield['checkpoint_step'])}，产量={best_yield['final_grain_kg_ha']:.1f} kg/ha，I={best_yield['action_irrigation_total']:.1f} mm，N={best_yield['action_fertilizer_total']:.1f} kg/ha。",
        f"- seed1 最高 reward checkpoint：{int(best_reward['checkpoint_step'])}，reward={best_reward['total_reward']:.1f}，产量={best_reward['final_grain_kg_ha']:.1f} kg/ha。",
        f"- seed1 final checkpoint：{int(final['checkpoint_step'])}，产量={final['final_grain_kg_ha']:.1f} kg/ha，I={final['action_irrigation_total']:.1f} mm，N={final['action_fertilizer_total']:.1f} kg/ha。",
    ]
    if seed0_best is not None:
        lines.extend(
            [
                "",
                "## 与 seed0 best checkpoint 对比",
                "",
                f"- seed0 best：checkpoint={int(seed0_best['checkpoint_step'])}，产量={seed0_best['final_grain_kg_ha']:.1f} kg/ha，I={seed0_best['action_irrigation_total']:.1f} mm，N={seed0_best['action_fertilizer_total']:.1f} kg/ha。",
                f"- seed1 best：checkpoint={int(best_yield['checkpoint_step'])}，产量={best_yield['final_grain_kg_ha']:.1f} kg/ha，I={best_yield['action_irrigation_total']:.1f} mm，N={best_yield['action_fertilizer_total']:.1f} kg/ha。",
            ]
        )
    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "- 如果 seed1 best 也接近 9400 kg/ha 且包含合理施氮，则 YC2014 可以进入正式 DQN 成功案例候选。",
            "- 如果 seed1 best 明显低于 seed0，则说明 seed0 仍可能存在偶然性，需要继续 seed2 或调整训练机制。",
            "",
        ]
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_yc_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_out_dir = OUT_DIR / f"seed{SEED}"
    if run_out_dir.exists():
        shutil.rmtree(run_out_dir)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import DQN

    base_run_dir = yc.prepare_case_for_scenario(SCENARIO)
    env_args = json.loads((base_run_dir / "env_args.json").read_text(encoding="utf-8"))
    train_env = make_train_env(env_args)
    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=50,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    daily_frames = []
    summaries = []
    try:
        completed = 0
        while completed < TOTAL_TIMESTEPS:
            target = min(CHECKPOINT_INTERVAL, TOTAL_TIMESTEPS - completed)
            print(f"[015_05] learn chunk {completed} -> {completed + target}", flush=True)
            model.learn(total_timesteps=target, reset_num_timesteps=(completed == 0), progress_bar=False)
            completed += target
            print(f"[015_05] evaluate checkpoint {completed}", flush=True)
            daily, summary = evaluate_model(model, env_args, completed)
            daily_path = run_out_dir / f"015_05_yc2014_seed1_checkpoint_{completed // 1000}k_daily.csv"
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            daily_frames.append(daily)
            summaries.append(summary)
    finally:
        train_env.close()

    all_daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    summary_df = pd.DataFrame(summaries).sort_values("checkpoint_step").reset_index(drop=True)
    daily_path = run_out_dir / "015_05_yc2014_unified_dqn_checkpoint_seed1_daily.csv"
    summary_path = run_out_dir / "015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv"
    fig_path = fig_dir / "yc2014_unified_dqn_checkpoint_seed1.png"
    all_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_checkpoints(all_daily, summary_df, fig_path)
    write_record(summary_df, daily_path, summary_path, fig_path)
    print(summary_df[["seed", "checkpoint_step", "action_irrigation_total", "action_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward"]].to_string(index=False))


if __name__ == "__main__":
    main()

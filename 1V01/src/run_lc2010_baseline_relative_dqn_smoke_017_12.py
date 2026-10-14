from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    parse_dssat_table,
    set_management_for_treatment,
)
from run_lc_fixed_input_year_screening_017_11 import (
    OUT_DIR as LC_SCREEN_DIR,
    fixed_source_text,
)
from run_yc2014_baseline_relative_dqn_015_10 import ACTION_TABLE_9, BaselineRelativeRewardWrapper
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


SITE = "LC"
STATION = "Luancheng"
YEAR = 2010
TRNO = 3
NULL_BASELINE = 8051.0
RECORDED_YIELD = 8732.0
AUTO_YIELD = 8738.0

OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "lc2010_baseline_relative_dqn_smoke_017_12"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-07_017_12_lc2010_baseline_relative_dqn_smoke_record.md"

WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


def configure_globals(seed: int, timesteps: int) -> None:
    yc_dqn.SEED = seed
    yc_dqn.TIMESTEPS = timesteps
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0
    yc_dqn.IRRIGATION_BUDGET = 120.0
    yc_dqn.NITROGEN_BUDGET = 300.0
    yc_dqn.DAILY_IRRIGATION_CAP = 30.0
    yc_dqn.DAILY_NITROGEN_CAP = 100.0
    yc_dqn.MIN_INTERVAL_DAYS = 7
    yc_dqn.ACTION_TABLE = ACTION_TABLE_9


def prepare_run_dir(seed: int, timesteps: int) -> Path:
    run_dir = OUT_DIR / f"seed{seed}_{timesteps}steps"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = fixed_source_text()
    text = set_management_for_treatment(text, TRNO, "L", "L")
    filex = input_dir / "CNLC1001_lc2010_dqn.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    source_input = LC_SCREEN_DIR / "runs" / "2010" / "null" / "input"
    if not source_input.exists():
        raise FileNotFoundError(f"LC 017_11 input not found: {source_input}")
    for src in source_input.iterdir():
        if src.is_file() and src.suffix.upper() != ".MZX":
            shutil.copyfile(src, input_dir / src.name)

    aux = [
        *sorted(input_dir.glob("CNLC*.WTH")),
        input_dir / "SOIL.SOL",
        input_dir / "MZCER048.CUL",
        input_dir / "CNLC.CLI",
        input_dir / "CNLC.PRM",
        input_dir / "CNLC.wdb",
    ]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TRNO,
        "auxiliary_file_paths": [str(p) for p in aux if p.exists()],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def make_env(env_args: dict[str, Any]):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(make_raw_env(env_args), WINDOWS["irrigation"], WINDOWS["nitrogen"])
    return BaselineRelativeRewardWrapper(linked, NULL_BASELINE)


def parse_events(snapshot: Path) -> pd.DataFrame:
    rows = []
    path = snapshot / "MgmtEvent.OUT"
    if not path.exists():
        return pd.DataFrame(columns=["dap", "operation", "amount", "unit"])
    import re

    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap = np.nan
        if len(parts) >= 7:
            try:
                dap = int(parts[6])
            except ValueError:
                pass
        amount = 0.0
        unit = ""
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
        rows.append({"dap": dap, "operation": raw.strip(), "amount": amount, "unit": unit})
    out = pd.DataFrame(rows)
    return out.drop_duplicates() if not out.empty else out


def evaluate_model(model, env_args: dict[str, Any], checkpoint: int, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    env = make_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(280):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": YEAR,
                    "scenario": "baseline_relative_dqn",
                    "checkpoint_step": checkpoint,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(safe_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe_action.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                    "used_irrigation": float(info.get("used_irrigation", np.nan)) if isinstance(info, dict) else np.nan,
                    "used_nitrogen": float(info.get("used_nitrogen", np.nan)) if isinstance(info, dict) else np.nan,
                    "yield_gain": float(info.get("yield_gain", 0.0)) if isinstance(info, dict) else 0.0,
                    "water_cost_term": float(info.get("water_cost_term", 0.0)) if isinstance(info, dict) else 0.0,
                    "nitrogen_cost_term": float(info.get("nitrogen_cost_term", 0.0)) if isinstance(info, dict) else 0.0,
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        snapshot = run_dir / f"pdi_tmp_snapshot_eval_{checkpoint}"
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    snapshot = run_dir / f"pdi_tmp_snapshot_eval_{checkpoint}"
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    events = parse_events(snapshot)
    event_i = float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0
    event_n = float(events.loc[events["unit"].astype(str).str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0
    summary = {
        "checkpoint_step": checkpoint,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": event_i,
        "event_fertilizer_total": event_n,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "yield_gain_vs_null": np.nan,
        "yield_diff_vs_recorded": np.nan,
        "yield_diff_vs_auto": np.nan,
    }
    if np.isfinite(summary["final_grain_kg_ha"]):
        summary["yield_gain_vs_null"] = summary["final_grain_kg_ha"] - NULL_BASELINE
        summary["yield_diff_vs_recorded"] = summary["final_grain_kg_ha"] - RECORDED_YIELD
        summary["yield_diff_vs_auto"] = summary["final_grain_kg_ha"] - AUTO_YIELD
    return daily, summary, events


def plot_checkpoint_summary(summary: pd.DataFrame, run_dir: Path, seed: int, timesteps: int) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = summary["checkpoint_step"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, summary["final_grain_kg_ha"], marker="o", color="#245A9B", label="DQN")
    axes[0].axhline(NULL_BASELINE, color="#222222", lw=1.2, label="null")
    axes[0].axhline(RECORDED_YIELD, color="#D04A3A", lw=1.2, ls="--", label="recorded")
    axes[0].axhline(AUTO_YIELD, color="#C49A00", lw=1.2, label="auto")
    axes[0].set_ylabel("GWAD kg/ha")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(x, summary["action_irrigation_total"], marker="o", color="#2E7D32", label="DQN I")
    axes[1].plot(x, summary["action_fertilizer_total"], marker="^", color="#7B1FA2", label="DQN N")
    axes[1].axhline(130.0, color="#2E7D32", lw=0.9, ls="--", alpha=0.5, label="recorded I")
    axes[1].axhline(250.0, color="#7B1FA2", lw=0.9, ls="--", alpha=0.5, label="recorded N")
    axes[1].set_ylabel("Resource total")
    axes[1].legend(frameon=False, ncol=4)
    axes[2].plot(x, summary["total_reward"], marker="o", color="#333333")
    axes[2].axhline(0, color="#888888", lw=1)
    axes[2].set_ylabel("Total reward")
    axes[2].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.grid(True, color="#E8EBF2", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"LC2010 baseline-relative DQN smoke seed{seed}", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lc2010_baseline_relative_seed{seed}_{timesteps}steps_checkpoint_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if np.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(seed: int, timesteps: int, checkpoint_interval: int, run_dir: Path, summary: pd.DataFrame) -> None:
    view_cols = [
        "checkpoint_step",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "action_irrigation_total",
        "action_fertilizer_total",
        "event_irrigation_total",
        "event_fertilizer_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
        "yield_diff_vs_recorded",
        "yield_diff_vs_auto",
    ]
    view = summary[view_cols].copy()
    best_reward = view.loc[view["total_reward"].idxmax()]
    best_yield = view.loc[view["final_grain_kg_ha"].idxmax()]
    lines = [
        "# 017_12 LC2010 baseline-relative DQN smoke 记录",
        "",
        "## 设置",
        "",
        f"- 站点年份：LC2010",
        f"- seed：{seed}",
        f"- timesteps：{timesteps}",
        f"- checkpoint interval：{checkpoint_interval}",
        "- 算法：Stable-Baselines3 DQN",
        "- 奖励：max(0, GWAD_final - LC2010_null) - 1.0*I - 5.0*N",
        "- null baseline：8051 kg/ha",
        "- recorded expert：8732 kg/ha, I=130 mm, N=250 kg/ha",
        "- DSSAT auto：8738 kg/ha, I=138.5 mm, N=0 kg/ha",
        "- 动作空间：I in {0,15,30}, N in {0,50,100}",
        "- 预算：I<=120 mm, N<=300 kg/ha, min interval=7 days",
        "",
        "## 输出",
        "",
        f"- Run dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- Summary: `{(run_dir / 'checkpoint_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- Daily: `{(run_dir / 'dqn_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- Events: `{(run_dir / 'dqn_eval_events.csv').relative_to(PROJECT_ROOT)}`",
        "",
        "## checkpoint 结果",
        "",
        markdown_table(view),
        "",
        "## 初步判断",
        "",
        f"- 最高产量 checkpoint: {int(best_yield['checkpoint_step'])}, GWAD={best_yield['final_grain_kg_ha']:.1f}, I={best_yield['action_irrigation_total']:.1f}, N={best_yield['action_fertilizer_total']:.1f}, reward={best_yield['total_reward']:.1f}.",
        f"- 最高奖励 checkpoint: {int(best_reward['checkpoint_step'])}, GWAD={best_reward['final_grain_kg_ha']:.1f}, I={best_reward['action_irrigation_total']:.1f}, N={best_reward['action_fertilizer_total']:.1f}, reward={best_reward['total_reward']:.1f}.",
        "- 本轮是 smoke，不作为跨 seed 稳定性结论；若出现候选 checkpoint，下一步才做 seed1。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    args = parser.parse_args()

    configure_globals(args.seed, args.timesteps)
    run_dir = prepare_run_dir(args.seed, args.timesteps)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    train_env = make_env(env_args)
    all_daily = []
    all_events = []
    all_summary = []
    try:
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=args.seed,
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
        prev = 0
        for checkpoint in range(args.checkpoint_interval, args.timesteps + 1, args.checkpoint_interval):
            model.learn(total_timesteps=checkpoint - prev, reset_num_timesteps=False, progress_bar=False)
            prev = checkpoint
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"dqn_baseline_relative_checkpoint_{checkpoint}"))
            daily, summary, events = evaluate_model(model, env_args, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
            if not events.empty:
                events = events.copy()
                events["checkpoint_step"] = checkpoint
                events["site"] = SITE
                events["year"] = YEAR
                all_events.append(events)
            print(
                f"[checkpoint {checkpoint}] GWAD={summary['final_grain_kg_ha']:.1f} "
                f"I={summary['action_irrigation_total']:.1f} N={summary['action_fertilizer_total']:.1f} "
                f"reward={summary['total_reward']:.1f}",
                flush=True,
            )
    finally:
        train_env.close()

    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    daily_df.to_csv(run_dir / "dqn_eval_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(run_dir / "dqn_eval_events.csv", index=False, encoding="utf-8-sig")
    plot_checkpoint_summary(summary_df, run_dir, args.seed, args.timesteps)
    write_record(args.seed, args.timesteps, args.checkpoint_interval, run_dir, summary_df)
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

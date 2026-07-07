from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    INPUT_ROOT,
    MZX_NAME,
    SITE,
    STATION,
    TEMPLATE_TRNO,
    parse_events,
    prepare_run_dir as prepare_fq_run_dir,
    prepare_text_for_shifted_scenario,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


YEAR = 2019
OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2019_baseline_relative_dqn_train_transfer_017_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_05_fq2019_train_then_cross_year_transfer_record.md"
BASE_SUMMARY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq2016_seed1_checkpoint30000_all_year_transfer_017_03"
    / "fq2016_seed1_ckpt30000_transfer_success_by_year.csv"
)

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}
WATER_COST = 1.0
NITROGEN_COST = 5.0


class BaselineRelativeRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env, null_baseline_yield: float):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.null_baseline_yield = float(null_baseline_yield)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, _old_reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        irrigation = float(getattr(self.env, "last_safe_real_action", {}).get("amir", 0.0))
        nitrogen = float(getattr(self.env, "last_safe_real_action", {}).get("anfer", 0.0))
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
        reward = float(yield_gain - WATER_COST * irrigation - NITROGEN_COST * nitrogen)
        info = info if isinstance(info, dict) else {}
        info.update(
            {
                "yield_gain": yield_gain,
                "water_cost_term": WATER_COST * irrigation,
                "nitrogen_cost_term": NITROGEN_COST * nitrogen,
                "baseline_relative_reward": reward,
            }
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def configure_globals(seed: int, timesteps: int) -> None:
    yc_dqn.SEED = seed
    yc_dqn.TIMESTEPS = timesteps
    yc_dqn.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_dqn.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_dqn.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_dqn.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_dqn.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_dqn.WATER_COST = WATER_COST
    yc_dqn.NITROGEN_COST = NITROGEN_COST


def load_baselines() -> pd.DataFrame:
    df = pd.read_csv(BASE_SUMMARY)
    for col in ["year", "null_zero", "recorded_shifted", "dssat_auto"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def null_yield_for_year(year: int, baselines: pd.DataFrame) -> float:
    row = baselines[baselines["year"].eq(year)]
    if row.empty:
        raise ValueError(f"No null baseline for FQ{year}")
    return float(row.iloc[0]["null_zero"])


def valid_transfer_years(baselines: pd.DataFrame) -> list[int]:
    years = baselines.loc[
        baselines[["null_zero", "recorded_shifted", "dssat_auto"]].notna().all(axis=1),
        "year",
    ].astype(int).tolist()
    return [y for y in sorted(years) if y not in {2007, 2008}]


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def make_env(env_args: dict[str, Any], null_baseline: float):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(make_raw_env(env_args), WINDOWS["irrigation"], WINDOWS["nitrogen"])
    return BaselineRelativeRewardWrapper(linked, null_baseline)


def prepare_train_dir(seed: int, timesteps: int) -> Path:
    run_dir = OUT_ROOT / f"seed{seed}_{timesteps}steps"
    if run_dir.exists():
        raise RuntimeError(f"Output directory already exists, refusing to overwrite: {run_dir}")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    text = prepare_text_for_shifted_scenario(YEAR, "dqn_linked_free_daily")
    filex = input_dir / f"CNFQ{YEAR}_baseline_relative_dqn.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)
    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TEMPLATE_TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def evaluate_model(model, env_args: dict[str, Any], null_baseline: float, checkpoint: int, run_dir: Path, year: int = YEAR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    env = make_env(env_args, null_baseline)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(380):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": year,
                    "scenario": "fq2019_baseline_relative_dqn",
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
    events = parse_events(run_dir, "fq2019_baseline_relative_dqn", snapshot_name=f"pdi_tmp_snapshot_eval_{checkpoint}")
    if not events.empty:
        events = events.assign(site=SITE, station=STATION, year=year, checkpoint_step=checkpoint)
    summary = {
        "year": year,
        "checkpoint_step": checkpoint,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "irrigation_total_mgmtevent": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "fertilizer_total_mgmtevent": float(events.loc[events["unit"].astype(str).str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
    }
    return daily, events, summary


def train(seed: int, timesteps: int, checkpoint_interval: int) -> tuple[Path, pd.DataFrame]:
    from stable_baselines3 import DQN

    configure_globals(seed, timesteps)
    baselines = load_baselines()
    null_baseline = null_yield_for_year(YEAR, baselines)
    run_dir = prepare_train_dir(seed, timesteps)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    train_env = make_env(env_args, null_baseline)
    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
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
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    try:
        trained = 0
        while trained < timesteps:
            step = min(checkpoint_interval, timesteps - trained)
            model.learn(total_timesteps=int(step), reset_num_timesteps=(trained == 0), progress_bar=False)
            trained += step
            model.save(str(models_dir / f"dqn_baseline_relative_checkpoint_{trained}.zip"))
            daily, events, summary = evaluate_model(model, env_args, null_baseline, trained, run_dir)
            all_daily.append(daily)
            if not events.empty:
                all_events.append(events)
            summaries.append(summary)
            print(f"[017_05][eval] ckpt={trained} GWAD={summary['final_grain_kg_ha']} I={summary['action_irrigation_total']} N={summary['action_fertilizer_total']} R={summary['total_reward']}", flush=True)
    finally:
        train_env.close()
    daily_df = pd.concat(all_daily, ignore_index=True, sort=False)
    events_df = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    daily_df.to_csv(run_dir / "fq2019_checkpoint_eval_daily.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(run_dir / "fq2019_checkpoint_eval_events.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "fq2019_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    plot_checkpoint_summary(summary_df, run_dir, baselines)
    return run_dir, summary_df


def plot_checkpoint_summary(summary: pd.DataFrame, run_dir: Path, baselines: pd.DataFrame) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ref = baselines[baselines["year"].eq(YEAR)].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.8), sharex=True, gridspec_kw={"hspace": 0.18})
    x = summary["checkpoint_step"]
    axes[0].plot(x, summary["final_grain_kg_ha"], marker="o", color="#2E8B57", label="DQN")
    axes[0].axhline(ref["null_zero"], color="#333333", lw=1.2, label="null")
    axes[0].axhline(ref["recorded_shifted"], color="#C73E3A", lw=1.2, ls="--", label="recorded")
    axes[0].axhline(ref["dssat_auto"], color="#B8860B", lw=1.2, label="auto")
    axes[0].axhline(8829, color="#999999", lw=1.0, ls=":", label="017_04 upper ref")
    axes[0].set_ylabel("GWAD\nkg/ha")
    axes[0].legend(ncol=5, fontsize=8, loc="upper left")
    axes[1].plot(x, summary["action_irrigation_total"], marker="o", color="#4C78A8", label="I")
    axes[1].plot(x, summary["action_fertilizer_total"], marker="^", color="#7B1FA2", label="N")
    axes[1].set_ylabel("Resource")
    axes[1].legend(loc="upper left")
    axes[2].plot(x, summary["total_reward"], marker="o", color="#222222")
    axes[2].axhline(0, color="#777777", lw=0.9)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.grid(True, color="#E6E8F0", alpha=0.9)
    stem = fig_dir / "fq2019_checkpoint_summary"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def prepare_transfer_run_dir(year: int, root: Path) -> tuple[Path, dict[str, Any]]:
    source_run = prepare_fq_run_dir(year, "dqn_linked_free_daily", seed=0)
    run_dir = root / str(year)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(source_run / "input", run_dir / "input", dirs_exist_ok=True)
    env_args = json.loads((source_run / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    env_args["fileX_template_path"] = str(run_dir / "input" / Path(env_args["fileX_template_path"]).name)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir, env_args


def transfer_best(run_dir: Path, best_checkpoint: int, seed: int, timesteps: int) -> pd.DataFrame:
    from stable_baselines3 import DQN

    baselines = load_baselines()
    years = valid_transfer_years(baselines)
    model_path = run_dir / "models" / f"dqn_baseline_relative_checkpoint_{best_checkpoint}.zip"
    out_dir = run_dir / f"transfer_ckpt{best_checkpoint}"
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for year in years:
        print(f"[017_05][transfer] year={year}", flush=True)
        year_run, env_args = prepare_transfer_run_dir(year, out_dir / "runs")
        null_baseline = null_yield_for_year(year, baselines)
        env = make_env(env_args, null_baseline)
        model = DQN.load(str(model_path), env=env)
        try:
            daily, events, summary = evaluate_model(model, env_args, null_baseline, best_checkpoint, year_run, year=year)
        finally:
            env.close()
        summary["transfer_from_year"] = YEAR
        summary["seed"] = seed
        summary["timesteps"] = timesteps
        all_daily.append(daily)
        if not events.empty:
            all_events.append(events)
        summaries.append(summary)
    daily_df = pd.concat(all_daily, ignore_index=True, sort=False)
    events_df = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    success = build_transfer_success(summary_df, baselines)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_dir / "fq2019_model_transfer_daily.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(out_dir / "fq2019_model_transfer_events.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "fq2019_model_transfer_summary.csv", index=False, encoding="utf-8-sig")
    success.to_csv(out_dir / "fq2019_model_transfer_success_by_year.csv", index=False, encoding="utf-8-sig")
    plot_transfer_summary(success, out_dir)
    return success


def build_transfer_success(summary: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    base = baselines[["year", "null_zero", "recorded_shifted", "dssat_auto"]].copy()
    out = base.merge(
        summary[["year", "final_grain_kg_ha", "action_irrigation_total", "action_fertilizer_total", "total_reward"]],
        on="year",
        how="inner",
    )
    out = out.rename(columns={"final_grain_kg_ha": "dqn_fq2019_transfer"})
    out["best_reference"] = out[["recorded_shifted", "dssat_auto"]].max(axis=1)
    out["reference_gain_over_null"] = out["best_reference"] - out["null_zero"]
    out["dqn_minus_null"] = out["dqn_fq2019_transfer"] - out["null_zero"]
    out["dqn_minus_recorded"] = out["dqn_fq2019_transfer"] - out["recorded_shifted"]
    out["dqn_minus_auto"] = out["dqn_fq2019_transfer"] - out["dssat_auto"]
    out["dqn_pct_auto"] = out["dqn_fq2019_transfer"] / out["dssat_auto"] * 100.0
    out["low_optimization_space"] = out["reference_gain_over_null"].abs().le(50)
    out["dqn_same_as_null"] = out["dqn_minus_null"].abs().le(30)
    out["dqn_resource_without_yield_gain"] = out["dqn_same_as_null"] & (out["action_irrigation_total"].fillna(0) + 5 * out["action_fertilizer_total"].fillna(0)).gt(0)
    out["strict_transfer_success"] = (
        out["reference_gain_over_null"].gt(100)
        & out["dqn_minus_null"].gt(100)
        & out["dqn_minus_recorded"].gt(0)
        & out["dqn_pct_auto"].ge(98)
    )
    return out.sort_values("year")


def plot_transfer_summary(success: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True, gridspec_kw={"hspace": 0.18})
    axes[0].plot(success["year"], success["null_zero"], color="#333333", lw=1.8, marker="o", label="null")
    axes[0].plot(success["year"], success["recorded_shifted"], color="#C73E3A", lw=1.8, ls="--", marker="s", label="recorded")
    axes[0].plot(success["year"], success["dssat_auto"], color="#B8860B", lw=1.8, marker="^", label="auto")
    axes[0].plot(success["year"], success["dqn_fq2019_transfer"], color="#2E8B57", lw=2.4, marker="D", label="FQ2019 DQN transfer")
    axes[0].set_ylabel("GWAD kg/ha")
    axes[0].legend(ncol=4, loc="upper left")
    axes[1].bar(success["year"] - 0.17, success["action_irrigation_total"], width=0.32, color="#4C78A8", label="DQN irrigation")
    axes[1].bar(success["year"] + 0.17, success["action_fertilizer_total"], width=0.32, color="#60BD68", label="DQN N")
    axes[1].set_ylabel("DQN resource")
    axes[1].set_xlabel("Year")
    axes[1].set_xticks(success["year"].astype(int).tolist())
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(True, color="#E6E8F0", alpha=0.9)
    stem = fig_dir / "fq2019_model_transfer_cross_year_summary"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_record(seed: int, timesteps: int, checkpoint_interval: int, run_dir: Path, ckpt_summary: pd.DataFrame, transfer_success: pd.DataFrame | None) -> None:
    best_reward = ckpt_summary.loc[ckpt_summary["total_reward"].idxmax()]
    best_yield = ckpt_summary.loc[ckpt_summary["final_grain_kg_ha"].idxmax()]

    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append("" if pd.isna(val) else f"{float(val):.3f}")
                else:
                    vals.append("" if pd.isna(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# 017_05 FQ2019 单年训练与跨年份迁移记录",
        "",
        "## 设置",
        "",
        f"- seed: {seed}",
        f"- timesteps: {timesteps}",
        f"- checkpoint interval: {checkpoint_interval}",
        "- reward: max(0, final_GWAD - local_null_GWAD) - 1*I - 5*N",
        "- constraints: I<=120, N<=300, daily I<=30, daily N<=100, min interval 7d",
        "",
        "## FQ2019 checkpoint 结果",
        "",
        md_table(ckpt_summary.round(3)),
        "",
        "## 最优 checkpoint",
        "",
        f"- best reward checkpoint: {int(best_reward['checkpoint_step'])}, GWAD={best_reward['final_grain_kg_ha']:.1f}, I={best_reward['action_irrigation_total']:.1f}, N={best_reward['action_fertilizer_total']:.1f}, reward={best_reward['total_reward']:.1f}",
        f"- best yield checkpoint: {int(best_yield['checkpoint_step'])}, GWAD={best_yield['final_grain_kg_ha']:.1f}, I={best_yield['action_irrigation_total']:.1f}, N={best_yield['action_fertilizer_total']:.1f}, reward={best_yield['total_reward']:.1f}",
    ]
    if transfer_success is not None:
        lines.extend(
            [
                "",
                "## 跨年份迁移摘要",
                "",
                md_table(
                    transfer_success[
                        [
                            "year",
                            "dqn_fq2019_transfer",
                            "null_zero",
                            "recorded_shifted",
                            "dssat_auto",
                            "dqn_minus_null",
                            "dqn_minus_recorded",
                            "dqn_minus_auto",
                            "action_irrigation_total",
                            "action_fertilizer_total",
                            "strict_transfer_success",
                        ]
                    ].round(3)
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- run dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()

    configure_style()
    run_dir, ckpt_summary = train(args.seed, args.timesteps, args.checkpoint_interval)
    transfer_success = None
    if args.transfer:
        best_checkpoint = int(ckpt_summary.loc[ckpt_summary["total_reward"].idxmax(), "checkpoint_step"])
        transfer_success = transfer_best(run_dir, best_checkpoint, args.seed, args.timesteps)
    write_record(args.seed, args.timesteps, args.checkpoint_interval, run_dir, ckpt_summary, transfer_success)
    print(ckpt_summary.to_string(index=False), flush=True)
    if transfer_success is not None:
        print(transfer_success.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


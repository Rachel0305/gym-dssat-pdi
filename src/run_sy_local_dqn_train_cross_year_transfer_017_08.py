from __future__ import annotations

import argparse
import json
import re
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

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table, prepare_text_for_scenario, set_management_for_treatment
from run_yc2014_baseline_relative_dqn_015_10 import ACTION_TABLE_9, BaselineRelativeRewardWrapper
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
MZX_NAME = "CNSY1201.MZX"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_08_sy_local_dqn_train_cross_year_transfer_record.md"

YEARS = {2012: 1, 2014: 2, 2015: 3}
SCENARIOS = ["null", "recorded", "dssat_auto"]
WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


def configure_globals() -> None:
    yc_dqn.IRRIGATION_BUDGET = 120.0
    yc_dqn.NITROGEN_BUDGET = 300.0
    yc_dqn.DAILY_IRRIGATION_CAP = 30.0
    yc_dqn.DAILY_NITROGEN_CAP = 100.0
    yc_dqn.MIN_INTERVAL_DAYS = 7
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0
    yc_dqn.ACTION_TABLE = ACTION_TABLE_9


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def make_eval_env(env_args: dict[str, Any], null_baseline_yield: float):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(make_raw_env(env_args), WINDOWS["irrigation"], WINDOWS["nitrogen"])
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def parse_weather(year: int) -> pd.DataFrame:
    wth = INPUT_ROOT / f"CNSY{year % 100:02d}01.WTH"
    rows = []
    if not wth.exists():
        return pd.DataFrame(columns=["doy", "rain"])
    header = []
    in_data = False
    for line in wth.read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("@"):
            header = line.replace("@", "", 1).split()
            in_data = "DATE" in header
            continue
        if in_data and line.strip() and not line.startswith("*") and not line.startswith("!"):
            parts = line.split()
            if len(parts) < len(header):
                continue
            rec = dict(zip(header, parts[: len(header)]))
            try:
                rows.append({"doy": int(str(rec["DATE"])[-3:]), "rain": float(rec.get("RAIN", 0.0))})
            except Exception:
                pass
    return pd.DataFrame(rows)


def parse_events(run_dir: Path, year: int, scenario: str) -> pd.DataFrame:
    import re

    path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["year", "scenario", "dap", "amount", "unit", "operation"])
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
        rows.append({"year": year, "scenario": scenario, "dap": dap, "amount": amount, "unit": unit, "operation": raw.strip()})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["year", "scenario", "dap", "amount", "unit", "operation"])
    return out


def prepare_text(year: int, scenario: str) -> str:
    source = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    trno = YEARS[year]
    if scenario == "recorded":
        return source
    if scenario == "null":
        return prepare_text_for_scenario(source, trno, "null")
    if scenario == "dssat_auto":
        return prepare_text_for_scenario(source, trno, "dssat_auto")
    if scenario.startswith("dqn") or scenario.startswith("transfer"):
        return set_management_for_treatment(source, trno, "L", "L")
    raise ValueError(scenario)


def prepare_run_dir(year: int, scenario: str, seed: int = 0, root: Path | None = None) -> Path:
    run_root = root or OUT_DIR / "runs"
    trno = YEARS[year]
    run_dir = run_root / str(year) / f"seed{seed}" / scenario
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / MZX_NAME).write_text(prepare_text(year, scenario), encoding="latin-1", errors="ignore")
    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)
    aux_candidates = [
        input_dir / f"CNSY{year % 100:02d}01.WTH",
        input_dir / "SOIL.SOL",
        input_dir / "MZCER048.CUL",
        input_dir / "CNSY.CLI",
        input_dir / "CNSY.PRM",
        input_dir / "CNSY.wdb",
    ]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(input_dir / MZX_NAME),
        "experiment_number": trno,
        "auxiliary_file_paths": [str(p) for p in aux_candidates if p.exists()],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def attach_rain_events(daily: pd.DataFrame, events: pd.DataFrame, year: int) -> pd.DataFrame:
    daily = daily.copy()
    rain = parse_weather(year)
    daily = daily.merge(rain, on="doy", how="left") if "doy" in daily.columns and not rain.empty else daily.assign(rain=0.0)
    daily["rain"] = daily["rain"].fillna(0.0)
    if "irrigation_mm" not in daily.columns:
        daily["irrigation_mm"] = 0.0
    if "fertilizer_kg_ha" not in daily.columns:
        daily["fertilizer_kg_ha"] = 0.0
    if not events.empty:
        tmp = events.copy()
        tmp["irrigation_event_mm"] = np.where(tmp["unit"].eq("mm"), tmp["amount"], 0.0)
        tmp["fertilizer_event_kg_ha"] = np.where(tmp["unit"].str.contains("kg", na=False), tmp["amount"], 0.0)
        tmp = tmp.groupby("dap", as_index=False)[["irrigation_event_mm", "fertilizer_event_kg_ha"]].sum()
        daily = daily.merge(tmp, on="dap", how="left")
        daily["irrigation_mm"] = daily["irrigation_event_mm"].fillna(daily["irrigation_mm"])
        daily["fertilizer_kg_ha"] = daily["fertilizer_event_kg_ha"].fillna(daily["fertilizer_kg_ha"])
        daily = daily.drop(columns=["irrigation_event_mm", "fertilizer_event_kg_ha"])
    return daily


def summarize(run_dir: Path, daily: pd.DataFrame, events: pd.DataFrame, year: int, scenario: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    event_i = float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0
    event_n = float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0
    action_i = float(daily["irrigation_mm"].sum()) if "irrigation_mm" in daily.columns and not daily.empty else 0.0
    action_n = float(daily["fertilizer_kg_ha"].sum()) if "fertilizer_kg_ha" in daily.columns and not daily.empty else 0.0
    total_i = event_i if event_i > 0 else action_i
    total_n = event_n if event_n > 0 else action_n
    rain_df = parse_weather(year)
    rain_total = float(rain_df["rain"].sum()) if not rain_df.empty else np.nan
    row = {
        "site": "SY",
        "station": "Shenyang",
        "year": year,
        "scenario": scenario,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "rain_total": rain_total,
        "irrigation_total": total_i,
        "fertilizer_total": total_n,
        "mgmt_event_irrigation_total": event_i,
        "mgmt_event_fertilizer_total": event_n,
        "max_water_stress": float(daily["swfac"].max()) if "swfac" in daily.columns and not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if "nstres" in daily.columns and not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if "reward" in daily.columns and not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    if extra:
        row.update(extra)
    return row


def run_zero_action(year: int, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = prepare_run_dir(year, scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(420):
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": 0.0,
                    "fertilizer_kg_ha": 0.0,
                    "action_index": np.nan,
                    "reward": scalar(reward),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        env.close()
    daily = pd.DataFrame(rows)
    events = parse_events(run_dir, year, scenario)
    daily = attach_rain_events(daily, events, year)
    summary = summarize(run_dir, daily, events, year, scenario)
    return daily, events, summary


def screen_baselines(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_daily, all_events, rows = [], [], []
    for year in years:
        for scenario in SCENARIOS:
            print(f"[baseline] SY{year} {scenario}", flush=True)
            daily, events, summary = run_zero_action(year, scenario)
            all_daily.append(daily)
            all_events.append(events)
            rows.append(summary)
    daily_df = pd.concat(all_daily, ignore_index=True)
    event_df = pd.concat(all_events, ignore_index=True)
    summary_df = pd.DataFrame(rows)
    base = summary_df.pivot_table(index="year", columns="scenario", values="final_gwad", aggfunc="first").reset_index()
    base["recorded_gain_vs_null"] = base.get("recorded", np.nan) - base.get("null", np.nan)
    base["auto_gain_vs_null"] = base.get("dssat_auto", np.nan) - base.get("null", np.nan)
    summary_df = summary_df.merge(base[["year", "recorded_gain_vs_null", "auto_gain_vs_null"]], on="year", how="left")
    return daily_df, event_df, summary_df


def choose_training_year(summary: pd.DataFrame) -> int:
    base = summary[summary["scenario"].isin(SCENARIOS)].pivot_table(index="year", columns="scenario", values="final_gwad", aggfunc="first")
    base["best_gain"] = base[["recorded", "dssat_auto"]].max(axis=1) - base["null"]
    base = base.sort_values("best_gain", ascending=False)
    if base.empty or float(base["best_gain"].iloc[0]) <= 100:
        raise RuntimeError("SY screened years do not show enough yield headroom for DQN training.")
    return int(base.index[0])


def train_checkpoint(year: int, seed: int, timesteps: int, interval: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback

    null_yield = float(run_zero_action(year, "null")[2]["final_gwad"])
    run_dir = prepare_run_dir(year, "dqn_train", seed=seed, root=OUT_DIR / "train_runs")
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    train_env = make_eval_env(env_args, null_yield)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(save_freq=interval, save_path=str(model_dir), name_prefix="dqn_baseline_relative_checkpoint")
    try:
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
            device="cpu",
        )
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    finally:
        train_env.close()

    daily_rows, event_rows, summaries = [], [], []
    for ckpt in sorted(model_dir.glob("dqn_baseline_relative_checkpoint_*.zip")):
        m = re.search(r"checkpoint_(\d+)", ckpt.stem)
        if not m:
            continue
        step = int(m.group(1))
        daily, events, summary = evaluate_model(year, ckpt, null_yield, f"dqn_ckpt{step}", seed=seed, checkpoint=step)
        daily_rows.append(daily)
        event_rows.append(events)
        summaries.append(summary)
    return pd.concat(daily_rows, ignore_index=True), pd.concat(event_rows, ignore_index=True), pd.DataFrame(summaries)


def evaluate_existing_checkpoints(year: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null_yield = float(run_zero_action(year, "null")[2]["final_gwad"])
    model_dir = OUT_DIR / "train_runs" / str(year) / f"seed{seed}" / "dqn_train" / "models"
    if not model_dir.exists():
        raise RuntimeError(f"No checkpoint directory: {model_dir}")
    daily_rows, event_rows, summaries = [], [], []
    for ckpt in sorted(model_dir.glob("dqn_baseline_relative_checkpoint_*.zip")):
        m = re.search(r"checkpoint_(\d+)", ckpt.stem)
        if not m:
            continue
        step = int(m.group(1))
        print(f"[eval existing] {ckpt.name}", flush=True)
        daily, events, summary = evaluate_model(year, ckpt, null_yield, f"dqn_ckpt{step}", seed=seed, checkpoint=step)
        daily_rows.append(daily)
        event_rows.append(events)
        summaries.append(summary)
    return pd.concat(daily_rows, ignore_index=True), pd.concat(event_rows, ignore_index=True), pd.DataFrame(summaries)


def evaluate_model(year: int, model_path: Path, null_yield: float, scenario: str, seed: int, checkpoint: int):
    from stable_baselines3 import DQN

    run_dir = prepare_run_dir(year, scenario, seed=seed, root=OUT_DIR / "eval_runs")
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_eval_env(env_args, null_yield)
    rows = []
    try:
        model = DQN.load(str(model_path), env=env, device="cpu")
        obs, info = env.reset()
        for step in range(420):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(safe.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        env.close()
    daily = pd.DataFrame(rows)
    events = parse_events(run_dir, year, scenario)
    daily = attach_rain_events(daily, events, year)
    summary = summarize(run_dir, daily, events, year, scenario, {"seed": seed, "checkpoint": checkpoint, "model_path": str(model_path.relative_to(PROJECT_ROOT))})
    return daily, events, summary


def transfer_best(train_year: int, best_model: Path, seed: int, checkpoint: int, baseline_summary: pd.DataFrame, years: list[int]):
    daily_rows, event_rows, summaries = [], [], []
    null_map = baseline_summary[baseline_summary["scenario"].eq("null")].set_index("year")["final_gwad"].to_dict()
    for year in years:
        null_yield = float(null_map[year])
        daily, events, summary = evaluate_model(year, best_model, null_yield, f"transfer_SY{train_year}_ckpt{checkpoint}", seed=seed, checkpoint=checkpoint)
        daily_rows.append(daily)
        event_rows.append(events)
        summaries.append(summary)
    return pd.concat(daily_rows, ignore_index=True), pd.concat(event_rows, ignore_index=True), pd.DataFrame(summaries)


def add_relative(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    stale_prefixes = tuple(f"{sc}_" for sc in SCENARIOS)
    stale_cols = [
        c
        for c in out.columns
        if c.startswith(stale_prefixes)
        or c.startswith("yield_diff_vs_")
        or c.startswith("irrigation_saving_vs_")
        or c.startswith("fertilizer_saving_vs_")
    ]
    if stale_cols:
        out = out.drop(columns=stale_cols)
    keys = ["year"]
    for sc in SCENARIOS:
        base = out[out["scenario"].eq(sc)][keys + ["final_gwad", "irrigation_total", "fertilizer_total"]].rename(
            columns={"final_gwad": f"{sc}_gwad", "irrigation_total": f"{sc}_irrigation", "fertilizer_total": f"{sc}_fertilizer"}
        )
        out = out.merge(base, on=keys, how="left")
        out[f"yield_diff_vs_{sc}"] = out["final_gwad"] - out[f"{sc}_gwad"]
        out[f"irrigation_saving_vs_{sc}"] = out[f"{sc}_irrigation"] - out["irrigation_total"]
        out[f"fertilizer_saving_vs_{sc}"] = out[f"{sc}_fertilizer"] - out["fertilizer_total"]
    return out


def plot_baseline(summary: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot_table(index="year", columns="scenario", values="final_gwad", aggfunc="first").sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot[SCENARIOS].plot(kind="bar", ax=ax, color=["#222222", "#C9252D", "#B8860B"])
    ax.set_ylabel("GWAD (kg/ha)")
    ax.set_title("SY baseline screen")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    out = FIG_DIR / "017_08_sy_baseline_screen.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_transfer(summary: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = [s for s in SCENARIOS + [x for x in summary["scenario"].unique() if str(x).startswith("transfer_")] if s in set(summary["scenario"])]
    colors = {"null": "#222222", "recorded": "#C9252D", "dssat_auto": "#B8860B"}
    x = np.arange(len(sorted(summary["year"].unique())))
    width = 0.8 / max(1, len(scenarios))
    for i, sc in enumerate(scenarios):
        sub = summary[summary["scenario"].eq(sc)].sort_values("year")
        ax.bar(x + (i - len(scenarios) / 2) * width + width / 2, sub["final_gwad"], width=width, label=sc, color=colors.get(sc, "#2E8B57"))
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(summary["year"].unique()))
    ax.set_ylabel("GWAD (kg/ha)")
    ax.set_title("SY local DQN train-year transfer")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    out = FIG_DIR / "017_08_sy_cross_year_transfer_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if np.isnan(v) else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(train_year: int | None, baseline: pd.DataFrame, ckpt: pd.DataFrame, transfer: pd.DataFrame, baseline_fig: Path, transfer_fig: Path | None) -> None:
    lines = [
        "# 017_08 SY 本地 DQN 训练与跨年迁移记录",
        "",
        "## 目的",
        "",
        "017_07 证明 HL/YC 模型无法直接迁移到 SY，因为 observation space 维度不一致。本轮改为同一方法在 SY 本地训练一年，再迁移到 SY 其他年份。",
        "",
        "## 基准筛选",
        "",
        markdown_table(baseline.sort_values(["year", "scenario"]), ["year", "scenario", "final_gwad", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress", "recorded_gain_vs_null", "auto_gain_vs_null"]),
        "",
        f"- 基准图：`{baseline_fig.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    if train_year is None:
        lines += ["## 训练未执行", "", "筛选结果没有足够优化空间，未进入 DQN 训练。"]
    else:
        lines += [
            "## 训练年选择",
            "",
            f"- 训练年：SY{train_year}",
            "- 选择理由：在已筛选年份中，管理基准相对 null 的增产空间最大。",
            "",
            "## checkpoint 训练结果",
            "",
            markdown_table(ckpt.sort_values("checkpoint"), ["year", "scenario", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_dssat_auto"]),
            "",
            "## 跨年迁移结果",
            "",
            markdown_table(transfer.sort_values(["year", "scenario"]), ["year", "scenario", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_recorded", "yield_diff_vs_dssat_auto"]),
            "",
            f"- 迁移总图：`{transfer_fig.relative_to(PROJECT_ROOT) if transfer_fig else ''}`",
        ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--years", nargs="*", type=int, default=sorted(YEARS))
    parser.add_argument("--reuse-checkpoints", action="store_true")
    args = parser.parse_args()

    configure_globals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[screen] SY baselines", flush=True)
    years = [y for y in args.years if y in YEARS]
    if not years:
        raise SystemExit("No valid SY years selected.")
    baseline_daily, baseline_events, baseline_summary = screen_baselines(years)
    baseline_summary = add_relative(baseline_summary)
    baseline_daily.to_csv(OUT_DIR / "017_08_sy_baseline_screen_daily.csv", index=False, encoding="utf-8-sig")
    baseline_events.to_csv(OUT_DIR / "017_08_sy_baseline_screen_events.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(OUT_DIR / "017_08_sy_baseline_screen_summary.csv", index=False, encoding="utf-8-sig")
    baseline_fig = plot_baseline(baseline_summary)
    if args.screen_only:
        write_record(None, baseline_summary, pd.DataFrame(), pd.DataFrame(), baseline_fig, None)
        print(baseline_summary.to_string(index=False))
        return

    train_year = choose_training_year(baseline_summary)
    if args.reuse_checkpoints:
        print(f"[reuse] SY{train_year} seed={args.seed}", flush=True)
        ckpt_daily, ckpt_events, ckpt_summary = evaluate_existing_checkpoints(train_year, args.seed)
    else:
        print(f"[train] SY{train_year} seed={args.seed} timesteps={args.timesteps}", flush=True)
        ckpt_daily, ckpt_events, ckpt_summary = train_checkpoint(train_year, args.seed, args.timesteps, args.checkpoint_interval)
    full_for_rel = pd.concat([baseline_summary, ckpt_summary], ignore_index=True)
    full_for_rel = add_relative(full_for_rel)
    ckpt_summary = full_for_rel[full_for_rel["scenario"].str.startswith("dqn_ckpt")].copy()
    ckpt_daily.to_csv(OUT_DIR / "017_08_sy_dqn_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
    ckpt_events.to_csv(OUT_DIR / "017_08_sy_dqn_checkpoint_events.csv", index=False, encoding="utf-8-sig")
    ckpt_summary.to_csv(OUT_DIR / "017_08_sy_dqn_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    best = ckpt_summary.sort_values(["total_reward", "final_gwad"], ascending=[False, False]).iloc[0]
    best_model = PROJECT_ROOT / str(best["model_path"])
    best_checkpoint = int(best["checkpoint"])
    print(f"[transfer] best checkpoint {best_checkpoint} model={best_model}", flush=True)
    transfer_daily, transfer_events, transfer_summary = transfer_best(train_year, best_model, args.seed, best_checkpoint, baseline_summary, years)
    combined_summary = add_relative(pd.concat([baseline_summary, transfer_summary], ignore_index=True))
    transfer_summary = combined_summary[combined_summary["scenario"].str.startswith("transfer_")].copy()
    transfer_daily.to_csv(OUT_DIR / "017_08_sy_dqn_transfer_daily.csv", index=False, encoding="utf-8-sig")
    transfer_events.to_csv(OUT_DIR / "017_08_sy_dqn_transfer_events.csv", index=False, encoding="utf-8-sig")
    transfer_summary.to_csv(OUT_DIR / "017_08_sy_dqn_transfer_summary.csv", index=False, encoding="utf-8-sig")
    combined_summary.to_csv(OUT_DIR / "017_08_sy_combined_summary.csv", index=False, encoding="utf-8-sig")
    transfer_fig = plot_transfer(combined_summary)
    write_record(train_year, baseline_summary, ckpt_summary, transfer_summary, baseline_fig, transfer_fig)
    print(transfer_summary.to_string(index=False))
    print(f"[done] {OUT_DIR}")


if __name__ == "__main__":
    main()

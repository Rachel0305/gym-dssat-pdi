from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
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
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    parse_weather,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_events as parse_standard_events
from run_yc2014_baseline_relative_dqn_015_10 import BaselineRelativeRewardWrapper
from run_yc2014_baseline_relative_dqn_015_10 import ACTION_TABLE_9
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_station_level3_true_model_transfer_016_04"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_016_04_yc2014_station_level3_true_model_transfer_record.md"

YC_MODEL_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_dqn_015_10"
SITE = "YC"
STATION = "Yucheng"
MZX_NAME = "CNYC0801.MZX"
TEMPLATE_TRNO = 2
TRAIN_YEAR = 2014
YEARS = list(range(2000, 2024))

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def configure_dqn_globals() -> None:
    yc_dqn.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_dqn.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_dqn.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_dqn.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_dqn.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0
    yc_dqn.ACTION_TABLE = ACTION_TABLE_9


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def make_env(env_args: dict[str, Any], null_baseline_yield: float):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(
        make_raw_env(env_args),
        WINDOWS["irrigation"],
        WINDOWS["nitrogen"],
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def shift_yc2014_template_to_year(source: str, year: int) -> str:
    yy = f"{year % 100:02d}"
    text = source
    text = text.replace("CNYC1401", f"CNYC{yy}01")
    text = text.replace("CNYC2014", f"CNYC{year}")
    text = text.replace("Sim2014", f"Sim{year}")
    text = re.sub(r"\b14(\d{3})\b", rf"{yy}\1", text)
    text = text.replace("08153", f"{yy}153")
    return text


def prepare_shifted_text(year: int, scenario: str) -> str:
    input_src = INPUT_ROOT / SITE
    source = (input_src / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    shifted = shift_yc2014_template_to_year(source, year)
    if scenario == "recorded_shifted":
        return shifted
    if scenario == "null":
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "null")
    if scenario == "dssat_auto":
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "dssat_auto")
    if scenario.startswith("transfer_"):
        return set_management_for_treatment(shifted, TEMPLATE_TRNO, "L", "L")
    raise ValueError(scenario)


def prepare_run_dir(year: int, scenario: str, seed: int = 0) -> Path:
    input_src = INPUT_ROOT / SITE
    run_dir = OUT_DIR / "runs" / str(year) / f"seed{seed}" / scenario
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_shifted_text(year, scenario)
    filex = input_dir / f"CNYC{year}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
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


def attach_events_to_daily(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if not events.empty:
        event_map = events.copy()
        event_map["irrigation_mm"] = np.where(event_map["unit"].eq("mm"), event_map["amount"], 0.0)
        event_map["fertilizer_kg_ha"] = np.where(event_map["unit"].str.contains("kg", na=False), event_map["amount"], 0.0)
        event_map = event_map.groupby("dap", as_index=False)[["irrigation_mm", "fertilizer_kg_ha"]].sum()
        daily = daily.merge(event_map, on="dap", how="left", suffixes=("", "_event"))
        daily["irrigation_mm"] = daily["irrigation_mm_event"].fillna(daily["irrigation_mm"])
        daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha_event"].fillna(daily["fertilizer_kg_ha"])
        daily = daily.drop(columns=["irrigation_mm_event", "fertilizer_kg_ha_event"])
    return daily


def run_zero_action_scenario(year: int, scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = prepare_run_dir(year, scenario, seed=0)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(340):
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": year,
                    "scenario": scenario,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
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
    events = parse_standard_events(run_dir, SITE, year, scenario)
    daily = attach_events_to_daily(daily, events)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": scenario,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def checkpoint_summary_path(seed: int) -> Path:
    return (
        YC_MODEL_ROOT
        / f"seed{seed}"
        / "dqn_baseline_relative_checkpoint"
        / "015_10_yc2014_baseline_relative_checkpoint_summary.csv"
    )


def checkpoint_model_path(seed: int, checkpoint: int) -> Path:
    return (
        YC_MODEL_ROOT
        / f"seed{seed}"
        / "dqn_baseline_relative_checkpoint"
        / "models"
        / f"dqn_baseline_relative_checkpoint_{checkpoint}.zip"
    )


def selected_checkpoints() -> list[dict[str, int | str]]:
    selections: list[dict[str, int | str]] = []
    for seed in [0, 1]:
        path = checkpoint_summary_path(seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing seed{seed} checkpoint summary: {path}")
        df = pd.read_csv(path)
        for col in ["checkpoint_step", "total_reward", "final_grain_kg_ha"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        best_reward = df.sort_values(["total_reward", "final_grain_kg_ha"], ascending=False).iloc[0]
        best_yield = df.sort_values(["final_grain_kg_ha", "total_reward"], ascending=False).iloc[0]
        selections.append({"seed": seed, "checkpoint": int(best_reward["checkpoint_step"]), "selection": "best_reward"})
        if int(best_yield["checkpoint_step"]) != int(best_reward["checkpoint_step"]):
            selections.append({"seed": seed, "checkpoint": int(best_yield["checkpoint_step"]), "selection": "best_yield_reference"})
    return selections


def run_transfer(year: int, seed: int, checkpoint: int, selection: str, null_yield: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    from stable_baselines3 import DQN

    configure_dqn_globals()
    scenario = f"transfer_seed{seed}_{selection}_ckpt{checkpoint}"
    run_dir = prepare_run_dir(year, scenario, seed=seed)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_env(env_args, null_yield)
    # Loading with a live PDI environment can block inside SB3 environment
    # validation.  The model only needs observations for deterministic predict.
    model = DQN.load(str(checkpoint_model_path(seed, checkpoint)))
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(340):
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
                    "scenario": scenario,
                    "train_year": TRAIN_YEAR,
                    "train_seed": seed,
                    "checkpoint_step": checkpoint,
                    "selection": selection,
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
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    events = parse_standard_events(run_dir, SITE, year, scenario)
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": scenario,
        "train_year": TRAIN_YEAR,
        "train_seed": seed,
        "checkpoint_step": checkpoint,
        "selection": selection,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def add_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    auto = out[out["scenario"].eq("dssat_auto")].set_index("year")
    null = out[out["scenario"].eq("null")].set_index("year")
    out = out.join(
        auto[["final_grain_kg_ha", "irrigation_total", "fertilizer_total"]].rename(
            columns={
                "final_grain_kg_ha": "auto_yield",
                "irrigation_total": "auto_irrigation",
                "fertilizer_total": "auto_fertilizer",
            }
        ),
        on="year",
    )
    out = out.join(null[["final_grain_kg_ha"]].rename(columns={"final_grain_kg_ha": "null_yield"}), on="year")
    out["yield_diff_vs_auto"] = out["final_grain_kg_ha"] - out["auto_yield"]
    out["yield_gain_vs_null"] = out["final_grain_kg_ha"] - out["null_yield"]
    out["irrigation_saving_vs_auto"] = out["auto_irrigation"] - out["irrigation_total"]
    out["fertilizer_saving_vs_auto"] = out["auto_fertilizer"] - out["fertilizer_total"]
    out["yield_ratio_vs_auto_pct"] = out["final_grain_kg_ha"] / out["auto_yield"] * 100.0
    return out


def success_table(summary: pd.DataFrame) -> pd.DataFrame:
    dqn = summary[summary["scenario"].str.startswith("transfer_")].copy()
    dqn["success_flag"] = (
        dqn["yield_ratio_vs_auto_pct"].ge(98.0)
        & dqn["irrigation_saving_vs_auto"].ge(-1e-6)
        & dqn["yield_gain_vs_null"].gt(100.0)
        & (dqn["irrigation_total"].gt(1e-6) | dqn["fertilizer_total"].gt(1e-6))
    )
    return dqn


def plot_summary(summary: pd.DataFrame) -> Path:
    dqn = summary[summary["scenario"].str.startswith("transfer_")].copy()
    dqn = dqn[dqn["selection"].eq("best_reward")].copy()
    years = sorted(dqn["year"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    colors = {0: "#255C99", 1: "#7B3F98"}
    for seed, sub in dqn.groupby("train_seed"):
        sub = sub.sort_values("year")
        axes[0].plot(sub["year"], sub["yield_ratio_vs_auto_pct"], marker="o", color=colors.get(int(seed), "#333333"), label=f"seed{int(seed)}")
        axes[1].plot(sub["year"], sub["irrigation_total"], marker="o", color=colors.get(int(seed), "#333333"))
        axes[2].plot(sub["year"], sub["fertilizer_total"], marker="o", color=colors.get(int(seed), "#333333"))
    axes[0].axhline(98, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].axhline(100, color="#333333", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Yield / auto (%)")
    axes[1].set_ylabel("Irrigation (mm)")
    axes[2].set_ylabel("Nitrogen (kg/ha)")
    axes[2].set_xlabel("Year")
    axes[0].set_title("YC2014-trained DQN model transfer across YC years", loc="left", fontweight="bold")
    axes[0].legend(loc="best")
    for ax in axes:
        ax.grid(True, color="#E8ECF2", linestyle="--", linewidth=0.7)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "yc2014_true_model_transfer_year_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    use = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.2f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame, successes: pd.DataFrame, fig_path: Path) -> None:
    best_reward = summary[
        summary["scenario"].str.startswith("transfer_") & summary["selection"].eq("best_reward")
    ].copy()
    seed_success = (
        successes[successes["selection"].eq("best_reward")]
        .groupby("train_seed")["success_flag"]
        .agg(["sum", "count"])
        .reset_index()
    )
    lines = [
        "# 016_04 YC2014 DQN 真模型跨年份迁移记录",
        "",
        "## 结论",
        "",
        "本轮加载 YC2014 训练得到的 DQN checkpoint zip，在 YC 2000-2023 天气年上逐年评估，没有在目标年份重新训练。",
        "",
        "## 输出文件",
        "",
        f"- 汇总表：`{(OUT_DIR / 'yc2014_true_model_transfer_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 日值表：`{(OUT_DIR / 'yc2014_true_model_transfer_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 成功判定表：`{(OUT_DIR / 'yc2014_transfer_success_by_year.csv').relative_to(PROJECT_ROOT)}`",
        f"- 总图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## Seed 成功数量",
        "",
        md_table(seed_success, ["train_seed", "sum", "count"]) if not seed_success.empty else "暂无。",
        "",
        "## Best-reward checkpoint 迁移结果",
        "",
        md_table(
            best_reward.sort_values(["train_seed", "year"]),
            [
                "year",
                "train_seed",
                "checkpoint_step",
                "final_grain_kg_ha",
                "yield_ratio_vs_auto_pct",
                "irrigation_total",
                "fertilizer_total",
                "yield_diff_vs_auto",
                "irrigation_saving_vs_auto",
                "yield_gain_vs_null",
            ],
        ),
        "",
        "## 判定口径",
        "",
        "- success_flag=True 表示：产量达到 auto 的 98% 以上、不比 auto 多用水、比 null 增产超过 100 kg/ha，并且不是完全无操作。",
        "- 这是同站点内部迁移，不等于跨站点泛化。",
        "- 如果某些年份 auto 本身不灌溉，DQN 很难同时满足节水判定；这类年份需要谨慎解释。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    configure_dqn_globals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_daily_path = OUT_DIR / "yc2014_baseline_daily_cache.csv"
    baseline_summary_path = OUT_DIR / "yc2014_baseline_summary_cache.csv"
    if baseline_daily_path.exists() and baseline_summary_path.exists():
        baseline_daily = [pd.read_csv(baseline_daily_path)]
        base_summary_df = pd.read_csv(baseline_summary_path)
    else:
        baseline_daily = []
        baseline_summary: list[dict[str, Any]] = []
        for year in YEARS:
            for scenario in ["null", "recorded_shifted", "dssat_auto"]:
                print(f"[YC baseline] year={year} scenario={scenario}", flush=True)
                daily, summary = run_zero_action_scenario(year, scenario)
                baseline_daily.append(daily)
                baseline_summary.append(summary)
        base_summary_df = pd.DataFrame(baseline_summary)
        pd.concat(baseline_daily, ignore_index=True, sort=False).to_csv(baseline_daily_path, index=False, encoding="utf-8-sig")
        base_summary_df.to_csv(baseline_summary_path, index=False, encoding="utf-8-sig")
    null_map = dict(
        zip(
            base_summary_df[base_summary_df["scenario"].eq("null")]["year"].astype(int),
            base_summary_df[base_summary_df["scenario"].eq("null")]["final_grain_kg_ha"].astype(float),
        )
    )

    dqn_daily: list[pd.DataFrame] = []
    dqn_summary: list[dict[str, Any]] = []
    selections = selected_checkpoints()
    pd.DataFrame(selections).to_csv(OUT_DIR / "yc2014_selected_checkpoints.csv", index=False, encoding="utf-8-sig")
    for item in selections:
        seed = int(item["seed"])
        checkpoint = int(item["checkpoint"])
        selection = str(item["selection"])
        for year in YEARS:
            print(f"[YC transfer] year={year} seed={seed} checkpoint={checkpoint} selection={selection}", flush=True)
            daily, summary = run_transfer(year, seed, checkpoint, selection, null_map[year])
            dqn_daily.append(daily)
            dqn_summary.append(summary)

    summary = add_comparisons(pd.concat([base_summary_df, pd.DataFrame(dqn_summary)], ignore_index=True, sort=False))
    successes = success_table(summary)
    daily = pd.concat(baseline_daily + dqn_daily, ignore_index=True, sort=False)

    summary.to_csv(OUT_DIR / "yc2014_true_model_transfer_summary.csv", index=False, encoding="utf-8-sig")
    successes.to_csv(OUT_DIR / "yc2014_transfer_success_by_year.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "yc2014_true_model_transfer_daily.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_summary(summary)
    write_record(summary, successes, fig_path)
    print(successes[successes["selection"].eq("best_reward")].to_string(index=False))
    print(f"Record: {DOC_PATH}")


if __name__ == "__main__":
    main()

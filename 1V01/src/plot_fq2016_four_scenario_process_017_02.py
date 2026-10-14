from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    INPUT_ROOT,
    MZX_NAME,
    SITE,
    STATION,
    TEMPLATE_TRNO,
    parse_events,
    parse_weather,
    prepare_text_for_shifted_scenario,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


YEAR = 2016
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_four_scenario_process_017_02"
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = OUT_DIR / "runs"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_017_02_fq2016_four_scenario_process_record.md"

DQN_SOURCE_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq2016_baseline_relative_dqn_checkpoint_015_14"
    / "seed1_50000steps"
)
DQN_CHECKPOINT = 30000

SCENARIO_ORDER = ["null_zero", "recorded_shifted", "dssat_auto", "dqn_seed1_best_reward"]
SCENARIO_LABELS = {
    "null_zero": "Null",
    "recorded_shifted": "Recorded shifted",
    "dssat_auto": "DSSAT auto",
    "dqn_seed1_best_reward": "DQN seed1 best reward",
}
SCENARIO_COLORS = {
    "null_zero": "#333333",
    "recorded_shifted": "#C73E3A",
    "dssat_auto": "#B8860B",
    "dqn_seed1_best_reward": "#2E8B57",
}
SCENARIO_LINESTYLES = {
    "null_zero": "-",
    "recorded_shifted": "--",
    "dssat_auto": "-",
    "dqn_seed1_best_reward": "-",
}
WATER_COST = 1.0
NITROGEN_COST = 5.0
NULL_BASELINE_YIELD = 7066.0
PLANTING_DOY = 162


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


class BaselineRelativeRewardForDisplay(gymnasium_base.Env):
    """Only used to expose a comparable reward column for non-DQN forward runs."""

    metadata = {"render_modes": []}

    def __init__(self, env, null_baseline_yield: float = NULL_BASELINE_YIELD):
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
        real_action = dict(getattr(self.env, "last_safe_real_action", {}) or {})
        irrigation = float(real_action.get("amir", 0.0) or 0.0)
        nitrogen = float(real_action.get("anfer", 0.0) or 0.0)
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
        reward = float(yield_gain - WATER_COST * irrigation - NITROGEN_COST * nitrogen)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def prepare_baseline_run_dir(scenario: str) -> Path:
    run_dir = RUN_DIR / scenario
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_text_for_shifted_scenario(YEAR, scenario)
    filex = input_dir / f"CNFQ{YEAR}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
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
    if events.empty:
        return daily
    event_map = events.copy()
    event_map["irrigation_mm"] = np.where(event_map["unit"].eq("mm"), event_map["amount"], 0.0)
    event_map["fertilizer_kg_ha"] = np.where(event_map["unit"].astype(str).str.contains("kg", na=False), event_map["amount"], 0.0)
    event_map = event_map.groupby("dap", as_index=False)[["irrigation_mm", "fertilizer_kg_ha"]].sum()
    daily = daily.merge(event_map, on="dap", how="left", suffixes=("", "_event"))
    daily["irrigation_mm"] = daily["irrigation_mm_event"].fillna(daily["irrigation_mm"])
    daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha_event"].fillna(daily["fertilizer_kg_ha"])
    return daily.drop(columns=["irrigation_mm_event", "fertilizer_kg_ha_event"])


def run_baseline_scenario(scenario: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    run_dir = prepare_baseline_run_dir(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = BaselineRelativeRewardForDisplay(make_raw_env(env_args))
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(380):
            action = {"amir": 0.0, "anfer": 0.0}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": YEAR,
                    "scenario": scenario,
                    "step": step,
                    "checkpoint_step": np.nan,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
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
    events = parse_events(run_dir, scenario)
    daily = attach_events_to_daily(daily, events)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = make_summary(scenario, daily, events, plantgro, run_dir)
    events = events.assign(site=SITE, station=STATION, year=YEAR)
    return daily, summary, events


def load_dqn_best_reward() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    daily_path = DQN_SOURCE_DIR / "dqn_eval_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    daily = pd.read_csv(daily_path)
    daily = daily[daily["checkpoint_step"].eq(DQN_CHECKPOINT)].copy()
    if daily.empty:
        raise ValueError(f"No DQN daily rows for checkpoint {DQN_CHECKPOINT}")
    daily["scenario"] = "dqn_seed1_best_reward"

    snapshot_name = f"pdi_tmp_snapshot_eval_{DQN_CHECKPOINT}"
    events = parse_events(DQN_SOURCE_DIR, "dqn_seed1_best_reward", snapshot_name=snapshot_name)
    plantgro = parse_dssat_table(DQN_SOURCE_DIR / snapshot_name / "PlantGro.OUT")
    daily = attach_events_to_daily(daily, events)
    summary = make_summary("dqn_seed1_best_reward", daily, events, plantgro, DQN_SOURCE_DIR / snapshot_name)
    events = events.assign(site=SITE, station=STATION, year=YEAR)
    return daily, summary, events


def make_summary(
    scenario: str,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    plantgro: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    fertilizer_mask = events["unit"].astype(str).str.contains("kg", na=False) if not events.empty else pd.Series(dtype=bool)
    irrigation_mask = events["unit"].eq("mm") if not events.empty else pd.Series(dtype=bool)
    final_grain = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else float(daily["grnwt"].dropna().iloc[-1])
    final_biomass = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else float(daily["topwt"].dropna().iloc[-1])
    return {
        "site": SITE,
        "station": STATION,
        "year": YEAR,
        "scenario": scenario,
        "final_grain_kg_ha": final_grain,
        "final_biomass_kg_ha": final_biomass,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "irrigation_total": float(events.loc[irrigation_mask, "amount"].sum()) if not events.empty else 0.0,
        "fertilizer_total": float(events.loc[fertilizer_mask, "amount"].sum()) if not events.empty else 0.0,
        "total_reward": float(daily["reward"].sum()) if "reward" in daily.columns else np.nan,
        "run_dir": str(run_dir),
    }


def add_rain(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["doy"] = pd.to_numeric(daily["doy"], errors="coerce")
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    missing_doy = daily["doy"].isna() & daily["dap"].notna()
    daily.loc[missing_doy, "doy"] = PLANTING_DOY + daily.loc[missing_doy, "dap"].round().astype(int)
    weather = parse_weather(YEAR).rename(columns={"rain": "rain_weather"})
    daily = daily.drop(columns=["rain"], errors="ignore")
    daily = daily.merge(weather[["doy", "rain_weather"]], on="doy", how="left")
    daily["rain"] = daily["rain_weather"].fillna(0.0)
    return daily.drop(columns=["rain_weather"])


def add_cumulative_reward_proxy(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["reward_proxy"] = 0.0
    daily["cumulative_reward_proxy"] = 0.0
    for scenario in daily["scenario"].dropna().unique():
        idx = daily["scenario"].eq(scenario)
        sub = daily[idx].sort_values("dap").copy()
        cost = WATER_COST * sub["irrigation_mm"].fillna(0) + NITROGEN_COST * sub["fertilizer_kg_ha"].fillna(0)
        terminal_bonus = pd.Series(0.0, index=sub.index)
        if not sub.empty:
            final_idx = sub.index[-1]
            terminal_bonus.loc[final_idx] = max(0.0, float(sub.loc[final_idx, "grnwt"]) - NULL_BASELINE_YIELD)
        proxy = terminal_bonus - cost
        daily.loc[sub.index, "reward_proxy"] = proxy
        daily.loc[sub.index, "cumulative_reward_proxy"] = proxy.cumsum()
    return daily


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.2, ls=SCENARIO_LINESTYLES[s], label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]


def plot_process(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.8, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0, 1.0, 1.15, 1.0], "hspace": 0.24},
    )

    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C9CED8", edgecolor="#AEB6C2", linewidth=0.4, width=0.9)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("FQ2016 four-scenario process plot", loc="left", fontsize=15, fontweight="bold", pad=8)
    axes[0].legend(handles=[Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall")], loc="upper left")

    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, ls=ls, lw=2.25)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, ls=":", lw=1.9, alpha=0.95)
        axes[5].plot(sub["dap"], sub["cumulative_reward_proxy"], color=color, ls=ls, lw=2.25)

        ev_i = events[(events["scenario"] == scenario) & (events["unit"] == "mm")]
        ev_n = events[(events["scenario"] == scenario) & (events["unit"].astype(str).str.contains("kg", na=False))]
        if not ev_i.empty:
            axes[3].vlines(ev_i["dap"], 0, ev_i["amount"], colors=color, linestyles=ls, linewidth=2.6, alpha=0.95)
        if not ev_n.empty:
            axes[3].scatter(ev_n["dap"], ev_n["amount"], marker="^", s=70, color=color, edgecolor="white", linewidth=0.7, zorder=5)

    handles = legend_handles()
    axes[1].set_ylabel("Water\nstress")
    axes[1].set_title("Water stress index by scenario", loc="left", fontsize=10)
    axes[1].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_title("Nitrogen stress index by scenario", loc="left", fontsize=10)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[3].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_title("Crop outcome: solid = grain yield, dotted = aboveground biomass", loc="left", fontsize=10)
    axes[5].set_ylabel("Cum.\nreward")
    axes[5].set_title("Cumulative reward proxy: terminal max(0, GWAD-null) - 1×irrigation - 5×fertilizer", loc="left", fontsize=10)
    axes[5].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[5].set_xlabel("DAP")

    max_dap = float(daily["dap"].max()) if not daily.empty else 120
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, max_dap + 2)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].set_ylim(bottom=-0.02)
    axes[3].set_ylim(bottom=-10)

    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.985, hspace=0.24)
    stem = FIG_DIR / "fq2016_four_scenario_process"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_record(summary: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append("" if pd.isna(val) else f"{float(val):.3f}")
                else:
                    vals.append(str(val))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    lines = [
        "# 017_02 FQ2016 四情景过程图整理记录",
        "",
        "## 目的",
        "",
        "在不新增训练的前提下，把 FQ2016 的 null、记录管理平移、DSSAT auto、DQN seed1 best-reward checkpoint 统一整理为可汇报的过程图和日值表。",
        "",
        "## 执行口径",
        "",
        "- 非 DQN 三个情景使用同一套 FQ2016 输入重新 forward；",
        f"- DQN 情景读取 `017_01` seed1 50K 训练中的 checkpoint {DQN_CHECKPOINT}，这是 seed1 的 best-reward checkpoint；",
        "- 本轮没有训练，没有修改奖励函数，没有改动作空间；",
        "- 奖励代理值仅用于图中对齐展示：terminal max(0, GWAD-null) - 1×灌溉 - 5×施氮。",
        "",
        "## 汇总结果",
        "",
        markdown_table(summary),
        "",
        "## 输出文件",
        "",
        f"- 日值表：`{(OUT_DIR / 'fq2016_four_scenario_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 管理事件：`{(OUT_DIR / 'fq2016_four_scenario_management_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{(OUT_DIR / 'fq2016_four_scenario_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 过程图 PNG：`{(FIG_DIR / 'fq2016_four_scenario_process.png').relative_to(PROJECT_ROOT)}`",
        f"- 过程图 SVG/PDF：同名 `.svg` / `.pdf`",
        "",
        "## 初步结论",
        "",
        "FQ2016 的 DQN seed1 best-reward 策略为约 I60/N0，符合 017_01 中 seed0/seed1 都倾向节水、零氮、接近高产的判断。该图可作为后续 FQ 跨年份迁移前的站点内代表过程图。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for scenario in ["null", "recorded_shifted", "dssat_auto"]:
        print(f"[forward] {scenario}", flush=True)
        daily, summary, events = run_baseline_scenario(scenario)
        if scenario == "null":
            daily["scenario"] = "null_zero"
            summary["scenario"] = "null_zero"
            if not events.empty:
                events["scenario"] = "null_zero"
        all_daily.append(daily)
        summaries.append(summary)
        if not events.empty:
            all_events.append(events)

    print(f"[load] DQN checkpoint {DQN_CHECKPOINT}", flush=True)
    dqn_daily, dqn_summary, dqn_events = load_dqn_best_reward()
    all_daily.append(dqn_daily)
    summaries.append(dqn_summary)
    if not dqn_events.empty:
        all_events.append(dqn_events)

    daily = pd.concat(all_daily, ignore_index=True, sort=False)
    events = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    daily = add_rain(daily)
    daily = add_cumulative_reward_proxy(daily)

    daily.to_csv(OUT_DIR / "fq2016_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "fq2016_four_scenario_management_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "fq2016_four_scenario_summary.csv", index=False, encoding="utf-8-sig")
    plot_process(daily, events)
    write_record(summary)
    print(summary[["scenario", "final_grain_kg_ha", "final_biomass_kg_ha", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress", "total_reward"]].to_string(index=False))


if __name__ == "__main__":
    main()

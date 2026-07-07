from __future__ import annotations

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    parse_weather,
    prepare_run_dir as prepare_fq_run_dir,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from run_fq2016_baseline_relative_dqn_checkpoint_015_14 import (
    BaselineRelativeRewardWrapper as FQBaselineRelativeRewardWrapper,
)
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_seed1_checkpoint30000_all_year_transfer_017_03"
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = OUT_DIR / "dqn_transfer_runs"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_017_03_fq2016_seed1_checkpoint30000_all_year_transfer_record.md"

BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01"
BASE_DAILY = BASE_DIR / "014_01_fq_all_year_screening_daily.csv"
BASE_EVENTS = BASE_DIR / "014_01_fq_all_year_screening_events.csv"
BASE_SUMMARY = BASE_DIR / "014_01_fq_all_year_screening_summary.csv"

DQN_MODEL = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq2016_baseline_relative_dqn_checkpoint_015_14"
    / "seed1_50000steps"
    / "models"
    / "dqn_baseline_relative_checkpoint_30000.zip"
)
DQN_CHECKPOINT = 30000
DQN_SCENARIO = "dqn_seed1_ckpt30000_transfer"
DAP0_DOY = 162

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
FREE_DAILY_WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}
WATER_COST = 1.0
NITROGEN_COST = 5.0

SCENARIO_ORDER = ["null_zero", "recorded_shifted", "dssat_auto", DQN_SCENARIO]
SCENARIO_LABELS = {
    "null_zero": "Null",
    "recorded_shifted": "Recorded shifted",
    "dssat_auto": "DSSAT auto",
    DQN_SCENARIO: "DQN seed1 ckpt30000 transfer",
}
SCENARIO_COLORS = {
    "null_zero": "#333333",
    "recorded_shifted": "#C73E3A",
    "dssat_auto": "#B8860B",
    DQN_SCENARIO: "#2E8B57",
}
SCENARIO_LINESTYLES = {
    "null_zero": "-",
    "recorded_shifted": "--",
    "dssat_auto": "-",
    DQN_SCENARIO: "-",
}


class BaselineRelativeRewardForDisplay(gymnasium_base.Env):
    """用于给迁移 evaluation 提供与训练一致的 reward 口径。"""

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
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def configure_dqn_globals() -> None:
    yc_dqn.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_dqn.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_dqn.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_dqn.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_dqn.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_dqn.WATER_COST = WATER_COST
    yc_dqn.NITROGEN_COST = NITROGEN_COST


def normalize_base_scenario(s: Any) -> str:
    if pd.isna(s) or str(s).strip() in {"", "null", "nan", "None"}:
        return "null_zero"
    return str(s)


def load_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(BASE_DAILY, keep_default_na=False)
    events = pd.read_csv(BASE_EVENTS, keep_default_na=False) if BASE_EVENTS.exists() else pd.DataFrame()
    summary = pd.read_csv(BASE_SUMMARY, keep_default_na=False)
    for df in [daily, events, summary]:
        if "scenario" in df.columns:
            df["scenario"] = df["scenario"].apply(normalize_base_scenario)
    for df in [daily, summary]:
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return daily, events, summary


def valid_years(summary: pd.DataFrame) -> list[int]:
    summary = summary.copy()
    summary["final_grain_kg_ha"] = pd.to_numeric(summary["final_grain_kg_ha"], errors="coerce")
    years: list[int] = []
    for year, group in summary.groupby("year"):
        have = set(group.loc[group["final_grain_kg_ha"].notna(), "scenario"])
        if {"null_zero", "recorded_shifted", "dssat_auto"}.issubset(have):
            years.append(int(year))
    # 014_01 已诊断 FQ2007/FQ2008 的 shifted IC 日期问题，继续排除。
    return [y for y in sorted(years) if y not in {2007, 2008}]


def null_yield_map(summary: pd.DataFrame) -> dict[int, float]:
    use = summary[summary["scenario"].eq("null_zero")].copy()
    use["final_grain_kg_ha"] = pd.to_numeric(use["final_grain_kg_ha"], errors="coerce")
    return dict(zip(use["year"].astype(int), use["final_grain_kg_ha"].astype(float)))


def prepare_dqn_run_dir(year: int) -> Path:
    source_run = prepare_fq_run_dir(year, "dqn_linked_free_daily", seed=0)
    run_dir = RUN_DIR / str(year)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(source_run / "input", run_dir / "input", dirs_exist_ok=True)
    env_args = json.loads((source_run / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    env_args["fileX_template_path"] = str(run_dir / "input" / Path(env_args["fileX_template_path"]).name)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def run_dqn_transfer(year: int, null_baseline_yield: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from stable_baselines3 import DQN

    configure_dqn_globals()
    run_dir = prepare_dqn_run_dir(year)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = FQBaselineRelativeRewardWrapper(
        yc_dqn.YCDiscreteBudgetedWrapper(
            make_raw_env(env_args),
            FREE_DAILY_WINDOWS["irrigation"],
            FREE_DAILY_WINDOWS["nitrogen"],
        ),
        null_baseline_yield,
    )
    model = DQN.load(str(DQN_MODEL), env=env)
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
                    "scenario": DQN_SCENARIO,
                    "checkpoint_step": DQN_CHECKPOINT,
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
    events = parse_events(run_dir, DQN_SCENARIO, snapshot_name="pdi_tmp_snapshot_eval")
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": DQN_SCENARIO,
        "checkpoint_step": DQN_CHECKPOINT,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    if not events.empty:
        events = events.assign(site=SITE, station=STATION, year=year)
    return daily, events, summary


def add_rain(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["doy"] = pd.to_numeric(daily["doy"], errors="coerce")
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    missing_doy = daily["doy"].isna() & daily["dap"].notna()
    daily.loc[missing_doy, "doy"] = DAP0_DOY + daily.loc[missing_doy, "dap"].round().astype(int)
    daily["rain"] = 0.0
    out = []
    for year, sub in daily.groupby("year", dropna=False):
        if pd.isna(year):
            out.append(sub)
            continue
        sub = sub.copy()
        weather = parse_weather(int(year)).rename(columns={"rain": "rain_weather"})
        if not weather.empty:
            sub = sub.drop(columns=["rain"], errors="ignore").merge(weather[["doy", "rain_weather"]], on="doy", how="left")
            sub["rain"] = sub["rain_weather"].fillna(0.0)
            sub = sub.drop(columns=["rain_weather"])
        out.append(sub)
    return pd.concat(out, ignore_index=True, sort=False)


def add_reward_proxy(daily: pd.DataFrame, nulls: dict[int, float]) -> pd.DataFrame:
    daily = daily.copy()
    daily["reward_proxy"] = 0.0
    daily["cumulative_reward_proxy"] = 0.0
    for (year, scenario), sub in daily.groupby(["year", "scenario"], dropna=False):
        sub = sub.sort_values("dap")
        baseline = nulls.get(int(year), 0.0) if not pd.isna(year) else 0.0
        cost = WATER_COST * pd.to_numeric(sub["irrigation_mm"], errors="coerce").fillna(0) + NITROGEN_COST * pd.to_numeric(sub["fertilizer_kg_ha"], errors="coerce").fillna(0)
        terminal = pd.Series(0.0, index=sub.index)
        if not sub.empty:
            final_idx = sub.index[-1]
            terminal.loc[final_idx] = max(0.0, float(sub.loc[final_idx, "grnwt"]) - baseline)
        proxy = terminal - cost
        daily.loc[sub.index, "reward_proxy"] = proxy
        daily.loc[sub.index, "cumulative_reward_proxy"] = proxy.cumsum()
    return daily


def build_combined_tables(base_daily: pd.DataFrame, base_events: pd.DataFrame, base_summary: pd.DataFrame, dqn_daily: pd.DataFrame, dqn_events: pd.DataFrame, dqn_summary: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_daily = base_daily[base_daily["year"].astype(int).isin(years) & base_daily["scenario"].isin(["null_zero", "recorded_shifted", "dssat_auto"])].copy()
    base_events = base_events[base_events["year"].astype(int).isin(years) & base_events["scenario"].isin(["recorded_shifted", "dssat_auto"])].copy() if not base_events.empty else pd.DataFrame()
    base_summary = base_summary[base_summary["year"].astype(int).isin(years) & base_summary["scenario"].isin(["null_zero", "recorded_shifted", "dssat_auto"])].copy()
    for df in [base_daily, dqn_daily]:
        for col in ["year", "doy", "dap", "grnwt", "topwt", "swfac", "nstres", "irrigation_mm", "fertilizer_kg_ha", "reward"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    daily = pd.concat([base_daily, dqn_daily], ignore_index=True, sort=False)
    events = pd.concat([base_events, dqn_events], ignore_index=True, sort=False) if not dqn_events.empty or not base_events.empty else pd.DataFrame()

    base_summary = base_summary.rename(columns={"event_irrigation_total": "irrigation_total", "event_fertilizer_total": "fertilizer_total"})
    summary = pd.concat([base_summary, dqn_summary], ignore_index=True, sort=False)
    for col in ["year", "final_grain_kg_ha", "final_biomass_kg_ha", "final_dap", "max_water_stress", "max_nitrogen_stress", "irrigation_total", "fertilizer_total", "total_reward"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return daily, events, summary


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.1, ls=SCENARIO_LINESTYLES[s], label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]


def plot_year(year: int, daily: pd.DataFrame, events: pd.DataFrame) -> None:
    sub_all = daily[daily["year"].eq(year)].copy()
    if sub_all.empty:
        return
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.8, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0, 1.0, 1.15, 1.0], "hspace": 0.24},
    )
    rain = sub_all[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C9CED8", edgecolor="#AEB6C2", linewidth=0.4, width=0.9)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title(f"FQ{year} four-scenario transfer process plot", loc="left", fontsize=15, fontweight="bold", pad=8)
    axes[0].legend(handles=[Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall")], loc="upper left")
    for scenario in SCENARIO_ORDER:
        sub = sub_all[sub_all["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, ls=ls, lw=2.25)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, ls=":", lw=1.9, alpha=0.95)
        axes[5].plot(sub["dap"], sub["cumulative_reward_proxy"], color=color, ls=ls, lw=2.25)
        ev_i = events[(events["year"].eq(year)) & (events["scenario"].eq(scenario)) & (events["unit"].eq("mm"))]
        ev_n = events[(events["year"].eq(year)) & (events["scenario"].eq(scenario)) & (events["unit"].astype(str).str.contains("kg", na=False))]
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
    axes[5].set_title("Cumulative reward proxy: terminal max(0, GWAD-null) - 1*irrigation - 5*fertilizer", loc="left", fontsize=10)
    axes[5].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[5].set_xlabel("DAP")
    max_dap = float(sub_all["dap"].max()) if not sub_all.empty else 120
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, max_dap + 2)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].set_ylim(bottom=-0.02)
    axes[3].set_ylim(bottom=-10)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / f"fq{year}_four_scenario_transfer_process"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_success_table(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot_table(index="year", columns="scenario", values="final_grain_kg_ha", aggfunc="first")
    for col in SCENARIO_ORDER:
        if col not in piv.columns:
            piv[col] = np.nan
    res = piv.reset_index()
    res["dqn_minus_null"] = res[DQN_SCENARIO] - res["null_zero"]
    res["dqn_minus_recorded"] = res[DQN_SCENARIO] - res["recorded_shifted"]
    res["dqn_minus_auto"] = res[DQN_SCENARIO] - res["dssat_auto"]
    res["dqn_pct_auto"] = res[DQN_SCENARIO] / res["dssat_auto"] * 100.0
    res["best_reference"] = res[["recorded_shifted", "dssat_auto"]].max(axis=1)
    res["reference_gain_over_null"] = res["best_reference"] - res["null_zero"]
    use = summary[summary["scenario"].eq(DQN_SCENARIO)].set_index("year")
    res["dqn_irrigation_total"] = res["year"].map(use["irrigation_total"])
    res["dqn_fertilizer_total"] = res["year"].map(use["fertilizer_total"])
    res["dqn_resource_total_weighted"] = res["dqn_irrigation_total"].fillna(0) + 5.0 * res["dqn_fertilizer_total"].fillna(0)
    res["low_optimization_space"] = res["reference_gain_over_null"].abs().le(50)
    res["dqn_same_as_null"] = res["dqn_minus_null"].abs().le(30)
    res["dqn_resource_without_yield_gain"] = res["dqn_same_as_null"] & res["dqn_resource_total_weighted"].gt(0)
    res["strict_transfer_success"] = (
        res["reference_gain_over_null"].gt(100)
        & res["dqn_minus_null"].gt(100)
        & res["dqn_minus_recorded"].gt(0)
        & res["dqn_pct_auto"].ge(98)
    )
    return res.sort_values("year")


def plot_cross_year(success: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True, gridspec_kw={"hspace": 0.20})
    for scenario in SCENARIO_ORDER:
        sub = summary[summary["scenario"].eq(scenario)].sort_values("year")
        axes[0].plot(sub["year"], sub["final_grain_kg_ha"], color=SCENARIO_COLORS[scenario], ls=SCENARIO_LINESTYLES[scenario], lw=2.2, marker="o", ms=3.5, label=SCENARIO_LABELS[scenario])
    axes[0].set_ylabel("GWAD\nkg/ha")
    axes[0].set_title("FQ cross-year transfer summary: yield and DQN resource use", loc="left", fontsize=14, fontweight="bold")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].bar(success["year"] - 0.18, success["dqn_irrigation_total"], width=0.34, color="#5DA5DA", label="DQN irrigation")
    axes[1].bar(success["year"] + 0.18, success["dqn_fertilizer_total"], width=0.34, color="#60BD68", label="DQN fertilizer")
    axes[1].set_ylabel("DQN resource\namount")
    axes[1].legend(loc="upper left")
    axes[2].axhline(0, color="#444444", lw=0.8)
    axes[2].plot(success["year"], success["dqn_minus_auto"], color="#B8860B", lw=2.0, marker="o", label="DQN - DSSAT auto")
    axes[2].plot(success["year"], success["dqn_minus_recorded"], color="#C73E3A", lw=2.0, ls="--", marker="s", label="DQN - recorded")
    axes[2].set_ylabel("Yield diff\nkg/ha")
    axes[2].set_xlabel("Year")
    axes[2].set_xticks(success["year"].astype(int).tolist())
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
    stem = FIG_DIR / "fq2016_seed1_ckpt30000_transfer_cross_year_summary"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_record(years: list[int], success: pd.DataFrame) -> None:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals: list[str] = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append("" if pd.isna(val) else f"{float(val):.3f}")
                else:
                    vals.append("" if pd.isna(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    preview = success.copy()
    keep = [
        "year",
        DQN_SCENARIO,
        "null_zero",
        "recorded_shifted",
        "dssat_auto",
        "reference_gain_over_null",
        "dqn_minus_null",
        "dqn_minus_recorded",
        "dqn_minus_auto",
        "dqn_irrigation_total",
        "dqn_fertilizer_total",
        "low_optimization_space",
        "dqn_resource_without_yield_gain",
        "strict_transfer_success",
    ]
    preview = preview[keep].round(3)
    lines = [
        "# 017_03 FQ2016 seed1 checkpoint30000 跨年份迁移记录",
        "",
        "## 执行内容",
        "",
        "- 不新增训练，只加载 FQ2016 seed1 的 checkpoint 30000 做 deterministic evaluation。",
        "- null / recorded_shifted / DSSAT auto 复用 014_01 的封丘站全年份 forward 结果。",
        "- DQN 情景使用同一套 baseline-relative reward 和预算 wrapper：I<=120 mm，N<=300 kg/ha，日上限 I30/N100，最小操作间隔 7 天。",
        "",
        "## 可用年份",
        "",
        ", ".join(map(str, years)),
        "",
        "FQ2007/FQ2008 沿用 014_01 诊断结论暂时排除，因为 shifted 初始条件日期问题会导致输出不可用或交互等待。",
        "",
        "## 输出文件",
        "",
        f"- 日值表：`{(OUT_DIR / 'fq2016_seed1_ckpt30000_transfer_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 管理事件：`{(OUT_DIR / 'fq2016_seed1_ckpt30000_transfer_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- summary：`{(OUT_DIR / 'fq2016_seed1_ckpt30000_transfer_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 跨年判读表：`{(OUT_DIR / 'fq2016_seed1_ckpt30000_transfer_success_by_year.csv').relative_to(PROJECT_ROOT)}`",
        f"- 图件目录：`{FIG_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 跨年份结果预览",
        "",
        simple_markdown_table(preview),
        "",
        "## 注意",
        "",
        "这一步是站点内跨年份迁移验证，不是重新训练；如果某些年份表现不好，不能直接解释为 FQ 站点不可优化，只能说明 FQ2016 seed1 checkpoint30000 的固定策略在这些年份的泛化有限。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not DQN_MODEL.exists():
        raise FileNotFoundError(DQN_MODEL)
    base_daily, base_events, base_summary = load_base_tables()
    years = valid_years(base_summary)
    nulls = null_yield_map(base_summary)
    print(f"[017_03] valid years: {years}", flush=True)

    daily_path = OUT_DIR / "fq2016_seed1_ckpt30000_transfer_daily.csv"
    events_path = OUT_DIR / "fq2016_seed1_ckpt30000_transfer_events.csv"
    summary_path = OUT_DIR / "fq2016_seed1_ckpt30000_transfer_summary.csv"
    success_path = OUT_DIR / "fq2016_seed1_ckpt30000_transfer_success_by_year.csv"

    if daily_path.exists() and events_path.exists() and summary_path.exists() and success_path.exists():
        print("[017_03] existing CSV found; skip DQN forward and regenerate figures/record only", flush=True)
        daily = pd.read_csv(daily_path)
        events = pd.read_csv(events_path)
        summary = pd.read_csv(summary_path)
        daily = add_rain(daily)
        daily = add_reward_proxy(daily, nulls)
        success = build_success_table(summary)
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        success.to_csv(success_path, index=False, encoding="utf-8-sig")
    else:
        dqn_daily_parts: list[pd.DataFrame] = []
        dqn_event_parts: list[pd.DataFrame] = []
        dqn_summaries: list[dict[str, Any]] = []
        for year in years:
            print(f"[017_03] DQN transfer year={year}", flush=True)
            daily, events, summary = run_dqn_transfer(year, nulls.get(year, 0.0))
            dqn_daily_parts.append(daily)
            if not events.empty:
                dqn_event_parts.append(events)
            dqn_summaries.append(summary)

        dqn_daily = pd.concat(dqn_daily_parts, ignore_index=True, sort=False)
        dqn_events = pd.concat(dqn_event_parts, ignore_index=True, sort=False) if dqn_event_parts else pd.DataFrame()
        dqn_summary = pd.DataFrame(dqn_summaries)

        daily, events, summary = build_combined_tables(base_daily, base_events, base_summary, dqn_daily, dqn_events, dqn_summary, years)
        daily = add_rain(daily)
        daily = add_reward_proxy(daily, nulls)
        success = build_success_table(summary)

        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        events.to_csv(events_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        success.to_csv(success_path, index=False, encoding="utf-8-sig")

    for year in years:
        plot_year(year, daily, events)
    plot_cross_year(success, summary)
    write_record(years, success)
    print(success.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

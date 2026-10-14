from __future__ import annotations

import json
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
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    parse_events as parse_fq_events,
    prepare_run_dir as prepare_fq_run_dir,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    parse_weather,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_events as parse_standard_events
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn
from run_fq2016_baseline_relative_dqn_checkpoint_015_14 import (
    BaselineRelativeRewardWrapper as FQBaselineRelativeRewardWrapper,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc_fq_station_level3_cross_year_seed_016_03"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_016_03_yc_fq_station_level3_cross_year_seed_record.md"

FQ_BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq_all_year_screen_and_dqn_transfer_014_01"
    / "014_01_fq_all_year_screening_summary.csv"
)
FQ_MODEL_ROOT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "fq2016_baseline_relative_dqn_checkpoint_015_14"
    / "seed0_50000steps"
    / "models"
)
FQ_CHECKPOINTS = [5000, 25000]
FQ_YEARS = list(range(2000, 2024))

YC_SEED_DAILY = {
    "yc2014_seed0_best5k": PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_checkpoint_diagnostic_015_04"
    / "seed0"
    / "015_04_yc2014_checkpoint_5k_daily.csv",
    "yc2014_seed1_best10k": PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_checkpoint_seed1_015_05"
    / "seed1"
    / "015_05_yc2014_seed1_checkpoint_10k_daily.csv",
}
YC_YEARS = [2008, 2014]

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
FREE_DAILY_WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


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
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0


def null_yield_by_year(site: str) -> dict[int, float]:
    if site == "FQ" and FQ_BASELINE_SUMMARY.exists():
        base = pd.read_csv(FQ_BASELINE_SUMMARY)
        base["scenario"] = base["scenario"].fillna("").replace({"": "null"})
        use = base[base["scenario"].eq("null")].copy()
        return dict(zip(use["year"].astype(int), use["final_grain_kg_ha"].astype(float)))
    return {}


def valid_fq_years() -> list[int]:
    """Only evaluate years with complete finite null and auto baselines.

    FQ2007/FQ2008 are present in the old screening table, but their outputs are
    blank because the shifted initial-condition date is before simulation start.
    Running those years interactively makes DSSAT wait for ENTER, so they must
    be excluded rather than forced into the transfer test.
    """
    if not FQ_BASELINE_SUMMARY.exists():
        return FQ_YEARS
    base = pd.read_csv(FQ_BASELINE_SUMMARY)
    base["scenario"] = base["scenario"].fillna("").replace({"": "null"})
    base["final_grain_kg_ha"] = pd.to_numeric(base["final_grain_kg_ha"], errors="coerce")
    ok_years: list[int] = []
    for year, group in base.groupby("year"):
        have = set(group.loc[group["final_grain_kg_ha"].notna(), "scenario"])
        if {"null", "dssat_auto"}.issubset(have):
            ok_years.append(int(year))
    return sorted(ok_years)


def run_fq_model_transfer(year: int, checkpoint: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from stable_baselines3 import DQN

    configure_dqn_globals()
    run_dir = OUT_DIR / "fq_model_transfer_runs" / str(year) / f"checkpoint_{checkpoint}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    source_run = prepare_fq_run_dir(year, "dqn_linked_free_daily", seed=0)
    shutil.copytree(source_run / "input", run_dir / "input", dirs_exist_ok=True)
    env_args = json.loads((source_run / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    env_args["fileX_template_path"] = str(run_dir / "input" / Path(env_args["fileX_template_path"]).name)
    (run_dir / "env_args.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")

    null_baseline = null_yield_by_year("FQ").get(year, 0.0)
    env = FQBaselineRelativeRewardWrapper(
        yc_dqn.YCDiscreteBudgetedWrapper(
            make_raw_env(env_args),
            FREE_DAILY_WINDOWS["irrigation"],
            FREE_DAILY_WINDOWS["nitrogen"],
        ),
        null_baseline,
    )
    model_path = FQ_MODEL_ROOT / f"dqn_baseline_relative_checkpoint_{checkpoint}.zip"
    model = DQN.load(str(model_path), env=env)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(320):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": "FQ",
                    "station": "Fengqiu",
                    "year": year,
                    "scenario": f"fq2016_model_ckpt{checkpoint}",
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
    events = parse_fq_events(run_dir, f"fq2016_model_ckpt{checkpoint}", snapshot_name="pdi_tmp_snapshot_eval")
    summary = {
        "site": "FQ",
        "station": "Fengqiu",
        "year": year,
        "scenario": f"fq2016_model_ckpt{checkpoint}",
        "checkpoint_step": checkpoint,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def extract_yc_events(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path)
    use = daily[(daily["irrigation_mm"].fillna(0) > 1e-6) | (daily["fertilizer_kg_ha"].fillna(0) > 1e-6)].copy()
    return use[["dap", "irrigation_mm", "fertilizer_kg_ha", "action_index"]].reset_index(drop=True)


def prepare_yc_replay_run(year: int, replay_name: str) -> Path:
    cfg = SITE_CONFIG["YC"]
    trno = cfg["treatments"][year]
    input_src = INPUT_ROOT / "YC"
    run_dir = OUT_DIR / "yc_action_replay_runs" / str(year) / replay_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = (input_src / cfg["mzx"]).read_text(encoding="latin-1", errors="ignore")
    text = set_management_for_treatment(source, trno, "L", "L")
    filex = input_dir / f"YC{year}_{replay_name}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
        if src.is_file() and src.name != cfg["mzx"]:
            shutil.copyfile(src, input_dir / src.name)
    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def run_yc_replay(year: int, replay_name: str, events_to_replay: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    configure_dqn_globals()
    run_dir = prepare_yc_replay_run(year, replay_name)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    event_map = {
        int(round(row.dap)): {"amir": float(row.irrigation_mm), "anfer": float(row.fertilizer_kg_ha)}
        for row in events_to_replay.itertuples()
    }
    try:
        obs, info = env.reset()
        for step in range(320):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", 0)) or 0))
            real_action = event_map.get(dap_before, {"amir": 0.0, "anfer": 0.0})
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, real_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": "YC",
                    "station": "Yucheng",
                    "year": year,
                    "scenario": replay_name,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": real_action["amir"],
                    "fertilizer_kg_ha": real_action["anfer"],
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
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    events = parse_standard_events(run_dir, "YC", year, replay_name)
    summary = {
        "site": "YC",
        "station": "Yucheng",
        "year": year,
        "scenario": replay_name,
        "evidence_type": "action_replay_not_model_transfer",
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def load_fq_baselines() -> pd.DataFrame:
    if not FQ_BASELINE_SUMMARY.exists():
        return pd.DataFrame()
    base = pd.read_csv(FQ_BASELINE_SUMMARY)
    base["scenario"] = base["scenario"].fillna("").replace({"": "null"})
    base["final_grain_kg_ha"] = pd.to_numeric(base["final_grain_kg_ha"], errors="coerce")
    base = base[base["year"].isin(valid_fq_years())].copy()
    keep = base.rename(
        columns={
            "event_irrigation_total": "irrigation_total",
            "event_fertilizer_total": "fertilizer_total",
        }
    )
    keep["checkpoint_step"] = np.nan
    keep["total_reward"] = np.nan
    return keep[
        [
            "site",
            "station",
            "year",
            "scenario",
            "checkpoint_step",
            "final_grain_kg_ha",
            "final_biomass_kg_ha",
            "irrigation_total",
            "fertilizer_total",
            "max_water_stress",
            "max_nitrogen_stress",
            "total_reward",
        ]
    ].copy()


def add_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    auto = out[out["scenario"].eq("dssat_auto")].set_index(["site", "year"])
    null = out[out["scenario"].eq("null")].set_index(["site", "year"])
    out = out.join(auto[["final_grain_kg_ha", "irrigation_total", "fertilizer_total"]].rename(columns={
        "final_grain_kg_ha": "auto_yield",
        "irrigation_total": "auto_irrigation",
        "fertilizer_total": "auto_fertilizer",
    }), on=["site", "year"])
    out = out.join(null[["final_grain_kg_ha"]].rename(columns={"final_grain_kg_ha": "null_yield"}), on=["site", "year"])
    out["yield_diff_vs_auto"] = out["final_grain_kg_ha"] - out["auto_yield"]
    out["yield_gain_vs_null"] = out["final_grain_kg_ha"] - out["null_yield"]
    out["irrigation_saving_vs_auto"] = out["auto_irrigation"] - out["irrigation_total"]
    out["fertilizer_saving_vs_auto"] = out["auto_fertilizer"] - out["fertilizer_total"]
    out["yield_ratio_vs_auto_pct"] = out["final_grain_kg_ha"] / out["auto_yield"] * 100.0
    return out


def plot_fq_summary(summary: pd.DataFrame) -> Path:
    dqn = summary[summary["scenario"].str.startswith("fq2016_model")].copy()
    if dqn.empty:
        raise RuntimeError("No FQ DQN rows to plot.")
    years = sorted(dqn["year"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    colors = {5000: "#2E7D32", 25000: "#7B1FA2"}
    for ckpt in sorted(dqn["checkpoint_step"].dropna().unique()):
        sub = dqn[dqn["checkpoint_step"].eq(ckpt)].sort_values("year")
        axes[0].plot(sub["year"], sub["yield_ratio_vs_auto_pct"], marker="o", color=colors.get(int(ckpt), "#333333"), label=f"ckpt {int(ckpt)}")
        axes[1].plot(sub["year"], sub["irrigation_total"], marker="o", color=colors.get(int(ckpt), "#333333"))
        axes[2].plot(sub["year"], sub["fertilizer_total"], marker="o", color=colors.get(int(ckpt), "#333333"))
    axes[0].axhline(100, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Yield / auto (%)")
    axes[1].set_ylabel("Irrigation (mm)")
    axes[2].set_ylabel("Nitrogen (kg/ha)")
    axes[2].set_xlabel("Year")
    axes[0].legend(loc="best")
    axes[0].set_title("FQ2016 DQN model transfer to FQ years", loc="left", fontweight="bold")
    for ax in axes:
        ax.grid(True, color="#E8ECF2", linestyle="--", linewidth=0.7)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fq2016_model_transfer_year_summary.png"
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


def write_record(fq_summary: pd.DataFrame, yc_summary: pd.DataFrame, fig_path: Path) -> None:
    fq_dqn = fq_summary[fq_summary["scenario"].str.startswith("fq2016_model")].copy()
    success = fq_dqn[
        (fq_dqn["yield_ratio_vs_auto_pct"].ge(98.0))
        & (fq_dqn["irrigation_saving_vs_auto"].ge(-1e-6))
    ].copy()
    lines = [
        "# 016_03 YC/FQ 第3层同站点跨年份诊断记录",
        "",
        "## 结论先写清楚",
        "",
        "- FQ：本次是**真模型迁移**，因为加载了 FQ2016 已保存的 DQN checkpoint zip，并在其他年份不重新训练地评估。",
        "- YC：本次不是模型迁移，而是**动作策略回放诊断**，因为 YC2014 当时没有保存 DQN 模型 zip；它只能说明动作时间表是否可移植。",
        "- 因此 FQ 和 YC 的证据等级不同，不能混在一起说。",
        "",
        "## 输出文件",
        "",
        f"- FQ 汇总表：`{(OUT_DIR / 'fq_model_transfer_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- FQ 日值表：`{(OUT_DIR / 'fq_model_transfer_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- YC 回放汇总表：`{(OUT_DIR / 'yc_action_replay_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- YC 回放日值表：`{(OUT_DIR / 'yc_action_replay_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- FQ 总图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## FQ 真模型迁移：达到 98% auto 且不多用水的年份",
        "",
        md_table(
            success.sort_values(["checkpoint_step", "year"]),
            ["year", "scenario", "checkpoint_step", "final_grain_kg_ha", "yield_ratio_vs_auto_pct", "irrigation_total", "fertilizer_total", "irrigation_saving_vs_auto", "yield_diff_vs_auto"],
        )
        if not success.empty
        else "暂无达到该标准的年份。",
        "",
        "## FQ DQN 全部迁移结果",
        "",
        md_table(
            fq_dqn.sort_values(["checkpoint_step", "year"]),
            ["year", "scenario", "checkpoint_step", "final_grain_kg_ha", "yield_ratio_vs_auto_pct", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress"],
        ),
        "",
        "## YC 动作回放结果（不是模型泛化）",
        "",
        md_table(
            yc_summary.sort_values(["scenario", "year"]),
            ["year", "scenario", "evidence_type", "final_grain_kg_ha", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress"],
        ),
        "",
        "## 下一步建议",
        "",
        "1. 如果 FQ 迁移结果好：再补 FQ seed1 模型训练并保存 checkpoint，验证跨 seed。",
        "2. 如果 YC 需要进入第3层：先重跑 YC2014，保存 DQN checkpoint zip；否则只能停留在动作回放层级。",
        "3. 不建议在没有模型 zip 的情况下宣称 YC 已实现模型跨年泛化。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fq_daily_all: list[pd.DataFrame] = []
    fq_summary_rows: list[dict[str, Any]] = []
    for checkpoint in FQ_CHECKPOINTS:
        for year in valid_fq_years():
            print(f"[FQ transfer] year={year} checkpoint={checkpoint}", flush=True)
            daily, summary = run_fq_model_transfer(year, checkpoint)
            fq_daily_all.append(daily)
            fq_summary_rows.append(summary)

    fq_dqn_summary = pd.DataFrame(fq_summary_rows)
    fq_baselines = load_fq_baselines()
    fq_summary = add_comparisons(pd.concat([fq_baselines, fq_dqn_summary], ignore_index=True, sort=False))
    fq_daily = pd.concat(fq_daily_all, ignore_index=True, sort=False)

    yc_daily_all: list[pd.DataFrame] = []
    yc_summary_rows: list[dict[str, Any]] = []
    for replay_name, daily_path in YC_SEED_DAILY.items():
        events = extract_yc_events(daily_path)
        events.to_csv(OUT_DIR / f"{replay_name}_extracted_events.csv", index=False, encoding="utf-8-sig")
        for year in YC_YEARS:
            print(f"[YC replay] year={year} replay={replay_name}", flush=True)
            daily, summary = run_yc_replay(year, replay_name, events)
            yc_daily_all.append(daily)
            yc_summary_rows.append(summary)
    yc_daily = pd.concat(yc_daily_all, ignore_index=True, sort=False)
    yc_summary = pd.DataFrame(yc_summary_rows)

    fq_summary.to_csv(OUT_DIR / "fq_model_transfer_summary.csv", index=False, encoding="utf-8-sig")
    fq_daily.to_csv(OUT_DIR / "fq_model_transfer_daily.csv", index=False, encoding="utf-8-sig")
    yc_summary.to_csv(OUT_DIR / "yc_action_replay_summary.csv", index=False, encoding="utf-8-sig")
    yc_daily.to_csv(OUT_DIR / "yc_action_replay_daily.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_fq_summary(fq_summary)
    write_record(fq_summary, yc_summary, fig_path)
    print(fq_summary[fq_summary["scenario"].str.startswith("fq2016_model")].to_string(index=False))
    print(yc_summary.to_string(index=False))
    print(f"Record: {DOC_PATH}")


if __name__ == "__main__":
    main()

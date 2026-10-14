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
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
from run_yc2014_baseline_relative_dqn_015_10 import ACTION_TABLE_9, BaselineRelativeRewardWrapper
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_07_lc_sy_hl_yc_dqn_cross_site_transfer_probe_record.md"

SITE_CONFIG: dict[str, dict[str, Any]] = {
    "LC": {
        "station": "Luancheng",
        "mzx": "CNLC0801.MZX",
        "years": {2008: 1, 2009: 2, 2010: 3, 2011: 4},
    },
    "SY": {
        "station": "Shenyang",
        "mzx": "CNSY1201.MZX",
        "years": {2012: 1, 2014: 2, 2015: 3},
    },
}

TRANSFER_MODELS = [
    {
        "source": "HLA2010",
        "source_seed": 0,
        "checkpoint": 35000,
        "kind": "hla_seed0_high_reward",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "hla_baseline_relative_dqn_checkpoint_015_12"
        / "2010"
        / "baseline_relative_seed0_50000steps"
        / "models"
        / "dqn_baseline_relative_checkpoint_35000.zip",
    },
    {
        "source": "HLA2010",
        "source_seed": 1,
        "checkpoint": 25000,
        "kind": "hla_seed1_high_reward",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "hla_baseline_relative_dqn_checkpoint_015_12"
        / "2010"
        / "baseline_relative_seed1_50000steps"
        / "models"
        / "dqn_baseline_relative_checkpoint_25000.zip",
    },
    {
        "source": "YC2014",
        "source_seed": 0,
        "checkpoint": 5000,
        "kind": "yc_seed0_high_yield",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "yc2014_baseline_relative_checkpoint_refresh_016_08"
        / "seed0"
        / "dqn_baseline_relative_checkpoint"
        / "models"
        / "dqn_baseline_relative_checkpoint_5000.zip",
    },
    {
        "source": "YC2014",
        "source_seed": 0,
        "checkpoint": 25000,
        "kind": "yc_seed0_high_reward",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "yc2014_baseline_relative_checkpoint_refresh_016_08"
        / "seed0"
        / "dqn_baseline_relative_checkpoint"
        / "models"
        / "dqn_baseline_relative_checkpoint_25000.zip",
    },
    {
        "source": "YC2014",
        "source_seed": 1,
        "checkpoint": 30000,
        "kind": "yc_seed1_low_water_high_yield",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "yc2014_baseline_relative_checkpoint_refresh_016_08"
        / "seed1"
        / "dqn_baseline_relative_checkpoint"
        / "models"
        / "dqn_baseline_relative_checkpoint_30000.zip",
    },
    {
        "source": "YC2014",
        "source_seed": 1,
        "checkpoint": 50000,
        "kind": "yc_seed1_high_reward",
        "path": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "yc2014_baseline_relative_checkpoint_refresh_016_08"
        / "seed1"
        / "dqn_baseline_relative_checkpoint"
        / "models"
        / "dqn_baseline_relative_checkpoint_50000.zip",
    },
]

WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}
SCENARIOS = ["null", "recorded", "dssat_auto"]


def configure_dqn_globals() -> None:
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
    linked = yc_dqn.YCDiscreteBudgetedWrapper(
        make_raw_env(env_args),
        WINDOWS["irrigation"],
        WINDOWS["nitrogen"],
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def parse_weather(site: str, year: int) -> pd.DataFrame:
    yy = year % 100
    wth = INPUT_ROOT / site / f"CN{site}{yy:02d}01.WTH"
    if not wth.exists():
        return pd.DataFrame(columns=["doy", "rain"])
    rows = []
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
                continue
    return pd.DataFrame(rows)


def parse_events(run_dir: Path, site: str, year: int, scenario: str) -> pd.DataFrame:
    path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["site", "requested_year", "scenario", "dap", "amount", "unit", "operation"])
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
        rows.append(
            {
                "site": site,
                "requested_year": year,
                "scenario": scenario,
                "dap": dap,
                "amount": amount,
                "unit": unit,
                "operation": raw.strip(),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["site", "requested_year", "scenario", "dap", "amount", "unit", "operation"])
    return out


def attach_rain_and_events(daily: pd.DataFrame, events: pd.DataFrame, site: str, year: int) -> pd.DataFrame:
    daily = daily.copy()
    rain = parse_weather(site, year)
    if "doy" in daily.columns and not rain.empty:
        daily = daily.merge(rain, on="doy", how="left")
    else:
        daily["rain"] = 0.0
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


def prepare_text(site: str, trno: int, scenario: str) -> str:
    cfg = SITE_CONFIG[site]
    input_src = INPUT_ROOT / site
    source = (input_src / cfg["mzx"]).read_text(encoding="latin-1", errors="ignore")
    if scenario == "recorded":
        return source
    if scenario == "null":
        return prepare_text_for_scenario(source, trno, "null")
    if scenario == "dssat_auto":
        return prepare_text_for_scenario(source, trno, "dssat_auto")
    if scenario.startswith("transfer_"):
        return set_management_for_treatment(source, trno, "L", "L")
    return source


def normalize_auto_management_dates(text: str) -> str:
    """Fix clearly invalid automatic-management date tokens in temporary run copies.

    LC/SY input packages may contain multiple simulation-control blocks.  Some
    LC automatic-management rows use 00001 or a template year's YY001 for all
    treatments.  Windows DSSAT can be permissive, but the PDI/gym parser may
    read the full file during environment construction even when recorded
    management is selected.  This function only edits the generated temporary
    MZX text, not the source input package.
    """
    lines = text.splitlines()
    sim_year: dict[str, str] = {}
    in_general = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@N GENERAL"):
            in_general = True
            continue
        if in_general and stripped.startswith("@"):
            in_general = False
        if in_general and re.match(r"^\s*\d+\s+GE\b", line):
            parts = line.split()
            if len(parts) >= 6 and re.match(r"\d{5}", parts[5]):
                sim_year[parts[0]] = parts[5][:2]

    out: list[str] = []
    auto_section: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@N PLANTING"):
            auto_section = "planting"
            out.append(line)
            continue
        if stripped.startswith("@N HARVEST"):
            auto_section = "harvest"
            out.append(line)
            continue
        if stripped.startswith("@N ") and not stripped.startswith("@N PLANTING") and not stripped.startswith("@N HARVEST"):
            auto_section = None
            out.append(line)
            continue
        if auto_section and re.match(r"^\s*\d+\s+", line):
            parts = line.split()
            yy = sim_year.get(parts[0])
            if yy:
                if auto_section == "planting" and len(parts) >= 4:
                    parts[2] = f"{yy}001"
                    parts[3] = f"{yy}001"
                    out.append(" ".join(parts))
                    continue
                if auto_section == "harvest" and len(parts) >= 4:
                    parts[3] = f"{yy}001"
                    out.append(" ".join(parts))
                    continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_run_dir(site: str, year: int, scenario: str, seed: int = 0) -> Path:
    cfg = SITE_CONFIG[site]
    trno = int(cfg["years"][year])
    input_src = INPUT_ROOT / site
    run_dir = OUT_DIR / "runs" / site / str(year) / f"seed{seed}" / scenario
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_text(site, trno, scenario)
    # Keep original DSSAT experiment file names.  PDI/DSSAT can be sensitive
    # to FILEX basename conventions; long ad-hoc names may hang during
    # gym.make before reset.
    filex = input_dir / cfg["mzx"]
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
        if src.is_file() and src.name != cfg["mzx"]:
            shutil.copyfile(src, input_dir / src.name)
    # Do not pass every WTH file as an auxiliary file.  Some station folders
    # contain large historical/multi-year weather files that are useful for
    # archiving but can make gym-DSSAT/PDI environment construction hang or
    # become unnecessarily slow.  The experiment only needs the requested
    # year's station weather file plus genotype/soil/experiment auxiliary
    # files.
    station_prefix = f"CN{site}"
    target_wth = input_dir / f"{station_prefix}{year % 100:02d}01.WTH"
    aux_candidates = [
        target_wth,
        input_dir / "SOIL.SOL",
        input_dir / "MZCER048.CUL",
        input_dir / f"{station_prefix}.CLI",
        input_dir / f"{station_prefix}.PRM",
        input_dir / f"{station_prefix}.wdb",
    ]
    aux = [str(p) for p in aux_candidates if p.exists()]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def summarize_from_snapshot(run_dir: Path, daily: pd.DataFrame, events: pd.DataFrame, site: str, year: int, scenario: str) -> dict[str, Any]:
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    final_gwad = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan
    final_cwad = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan
    return {
        "site": site,
        "station": SITE_CONFIG[site]["station"],
        "requested_year": year,
        "scenario": scenario,
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "rain_total": float(daily[["dap", "rain"]].drop_duplicates("dap")["rain"].sum()) if not daily.empty and "rain" in daily.columns else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty and "irrigation_mm" in daily.columns else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty and "fertilizer_kg_ha" in daily.columns else 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty and "swfac" in daily.columns else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty and "nstres" in daily.columns else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty and "reward" in daily.columns else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def run_zero_action(site: str, year: int, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = prepare_run_dir(site, year, scenario, seed=0)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(380):
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": site,
                    "station": SITE_CONFIG[site]["station"],
                    "requested_year": year,
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
    events = parse_events(run_dir, site, year, scenario)
    daily = attach_rain_and_events(daily, events, site, year)
    summary = summarize_from_snapshot(run_dir, daily, events, site, year, scenario)
    return daily, events, summary


def run_transfer_model(site: str, year: int, model_meta: dict[str, Any], null_baseline_yield: float):
    from stable_baselines3 import DQN

    scenario = f"transfer_{model_meta['source']}_seed{model_meta['source_seed']}_ckpt{model_meta['checkpoint']}_{model_meta['kind']}"
    run_dir = prepare_run_dir(site, year, scenario, seed=int(model_meta["source_seed"]))
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    eval_env = make_eval_env(env_args, null_baseline_yield)
    rows = []
    try:
        model = DQN.load(str(model_meta["path"]), env=eval_env, device="cpu")
        obs, info = eval_env.reset()
        for step in range(380):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(eval_env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": site,
                    "station": SITE_CONFIG[site]["station"],
                    "requested_year": year,
                    "scenario": scenario,
                    "source_model": model_meta["source"],
                    "source_seed": model_meta["source_seed"],
                    "source_checkpoint": model_meta["checkpoint"],
                    "source_kind": model_meta["kind"],
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
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        eval_env.close()
    daily = pd.DataFrame(rows)
    events = parse_events(run_dir, site, year, scenario)
    daily = attach_rain_and_events(daily, events, site, year)
    summary = summarize_from_snapshot(run_dir, daily, events, site, year, scenario)
    summary.update(
        {
            "source_model": model_meta["source"],
            "source_seed": model_meta["source_seed"],
            "source_checkpoint": model_meta["checkpoint"],
            "source_kind": model_meta["kind"],
            "model_path": str(model_meta["path"].relative_to(PROJECT_ROOT)),
        }
    )
    return daily, events, summary


def add_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    keys = ["site", "requested_year"]
    for baseline in ["null", "recorded", "dssat_auto"]:
        base = (
            out[out["scenario"].eq(baseline)][keys + ["final_gwad", "irrigation_total", "fertilizer_total"]]
            .rename(
                columns={
                    "final_gwad": f"{baseline}_gwad",
                    "irrigation_total": f"{baseline}_irrigation",
                    "fertilizer_total": f"{baseline}_fertilizer",
                }
            )
        )
        out = out.merge(base, on=keys, how="left")
        out[f"yield_diff_vs_{baseline}"] = out["final_gwad"] - out[f"{baseline}_gwad"]
        out[f"irrigation_saving_vs_{baseline}"] = out[f"{baseline}_irrigation"] - out["irrigation_total"]
        out[f"fertilizer_saving_vs_{baseline}"] = out[f"{baseline}_fertilizer"] - out["fertilizer_total"]
    out["strict_success_vs_auto"] = (
        out["scenario"].str.startswith("transfer_")
        & out["yield_diff_vs_null"].gt(0)
        & out["yield_diff_vs_dssat_auto"].ge(-100)
        & (out["irrigation_saving_vs_dssat_auto"].gt(0) | out["fertilizer_saving_vs_dssat_auto"].gt(0))
    )
    return out


def plot_summary(summary: pd.DataFrame) -> Path:
    transfer = summary[summary["scenario"].str.startswith("transfer_")].copy()
    if transfer.empty:
        return FIG_DIR / "017_07_lc_sy_cross_site_summary.png"
    transfer["label"] = transfer["site"] + transfer["requested_year"].astype(str) + "\n" + transfer["source_kind"].fillna("")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    x = np.arange(len(transfer))
    colors = np.where(transfer["source_model"].eq("HLA2010"), "#255C99", "#2E8B57")
    axes[0].bar(x, transfer["yield_diff_vs_dssat_auto"], color=colors)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Yield diff vs auto\nkg/ha")
    axes[1].bar(x, transfer["irrigation_total"], color=colors)
    axes[1].set_ylabel("Irrigation\nmm")
    axes[2].bar(x, transfer["fertilizer_total"], color=colors)
    axes[2].set_ylabel("Nitrogen\nkg/ha")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(transfer["label"], rotation=90, fontsize=7)
    for ax in axes:
        ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("017_07 LC/SY direct cross-site DQN transfer probe", x=0.01, ha="left", weight="bold")
    out = FIG_DIR / "017_07_lc_sy_cross_site_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, (float, np.floating)):
                vals.append("" if np.isnan(v) else f"{float(v):.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame, status: pd.DataFrame, fig_path: Path, smoke: bool) -> None:
    cols = [
        "site",
        "requested_year",
        "scenario",
        "source_model",
        "source_seed",
        "source_checkpoint",
        "final_gwad",
        "irrigation_total",
        "fertilizer_total",
        "yield_diff_vs_null",
        "yield_diff_vs_dssat_auto",
        "strict_success_vs_auto",
    ]
    existing_cols = [c for c in cols if c in summary.columns]
    transfer = summary[summary["scenario"].str.startswith("transfer_")].copy()
    sort_cols = [c for c in ["site", "requested_year", "source_model", "source_seed", "source_checkpoint"] if c in transfer.columns]
    if sort_cols and not transfer.empty:
        transfer = transfer.sort_values(sort_cols)
    base_cols = ["site", "requested_year", "scenario", "final_gwad", "irrigation_total", "fertilizer_total", "max_water_stress", "max_nitrogen_stress"]
    baseline = summary[summary["scenario"].isin(SCENARIOS)].sort_values(["site", "requested_year", "scenario"])
    lines = [
        "# 017_07 LC/SY 跨站点 DQN 策略迁移 probe 记录",
        "",
        "## 运行设置",
        "",
        f"- smoke 模式：{smoke}",
        "- 本轮不训练新模型，只加载 HLA2010 / YC2014 已有 DQN checkpoint。",
        "- 统一动作/约束：9-action，I≤120 mm，N≤300 kg/ha，单次 I≤30，单次 N≤100，最小间隔 7 天。",
        "- 评估奖励：`max(0, GWAD_final - local_null_GWAD) - 1*I - 5*N`。",
        "",
        "## 输出",
        "",
        f"- 汇总表：`{(OUT_DIR / '017_07_lc_sy_cross_site_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 日值表：`{(OUT_DIR / '017_07_lc_sy_cross_site_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 事件表：`{(OUT_DIR / '017_07_lc_sy_cross_site_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- 状态表：`{(OUT_DIR / '017_07_lc_sy_cross_site_status.csv').relative_to(PROJECT_ROOT)}`",
        f"- 总图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 本地基准",
        "",
        markdown_table(baseline, [c for c in base_cols if c in baseline.columns]) if not baseline.empty else "无。",
        "",
        "## 跨站点 DQN 迁移结果",
        "",
        markdown_table(transfer, existing_cols) if not transfer.empty else "无。",
        "",
        "## 状态/失败记录",
        "",
        markdown_table(status, list(status.columns)) if not status.empty else "无失败。",
        "",
        "## 初步判定",
        "",
        "- 如果 direct transfer 表现不好，不代表 DQN 框架在 LC/SY 不可用，只代表已经学到的 HLA/YC 策略不能直接照搬。",
        "- 下一步应根据 LC/SY 本地基准是否存在优化空间，决定是否进入本地重新训练，而不是继续盲目跨站点迁移。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(smoke: bool, sites: list[str] | None = None) -> None:
    configure_dqn_globals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    selected_sites = sites or list(SITE_CONFIG)
    for site in selected_sites:
        cfg = SITE_CONFIG[site]
        years = list(cfg["years"].keys())
        if smoke:
            years = years[:1]
        for year in years:
            print(f"[baseline] {site}{year}", flush=True)
            null_yield = np.nan
            for scenario in SCENARIOS:
                try:
                    daily, events, summary = run_zero_action(site, year, scenario)
                    all_daily.append(daily)
                    all_events.append(events)
                    summaries.append(summary)
                    statuses.append({"site": site, "year": year, "scenario": scenario, "status": "ok", "error": ""})
                    if scenario == "null":
                        null_yield = float(summary["final_gwad"])
                except Exception as exc:
                    statuses.append({"site": site, "year": year, "scenario": scenario, "status": "failed", "error": repr(exc)[-500:]})
                    print(f"[failed] {site}{year} {scenario}: {exc}", flush=True)
            if not np.isfinite(null_yield):
                print(f"[skip transfer] {site}{year}: null baseline missing", flush=True)
                continue
            for model_meta in TRANSFER_MODELS:
                scenario_name = f"transfer_{model_meta['source']}_seed{model_meta['source_seed']}_ckpt{model_meta['checkpoint']}_{model_meta['kind']}"
                try:
                    print(f"[transfer] {site}{year} {scenario_name}", flush=True)
                    daily, events, summary = run_transfer_model(site, year, model_meta, null_yield)
                    all_daily.append(daily)
                    all_events.append(events)
                    summaries.append(summary)
                    statuses.append({"site": site, "year": year, "scenario": scenario_name, "status": "ok", "error": ""})
                except Exception as exc:
                    statuses.append({"site": site, "year": year, "scenario": scenario_name, "status": "failed", "error": repr(exc)[-700:]})
                    print(f"[failed] {site}{year} {scenario_name}: {exc}", flush=True)

    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    event_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    status_df = pd.DataFrame(statuses)
    if not summary_df.empty:
        summary_df = add_relative_metrics(summary_df)
    daily_df.to_csv(OUT_DIR / "017_07_lc_sy_cross_site_daily.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUT_DIR / "017_07_lc_sy_cross_site_events.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "017_07_lc_sy_cross_site_summary.csv", index=False, encoding="utf-8-sig")
    status_df.to_csv(OUT_DIR / "017_07_lc_sy_cross_site_status.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_summary(summary_df) if not summary_df.empty else FIG_DIR / "017_07_lc_sy_cross_site_summary.png"
    write_record(summary_df, status_df, fig_path, smoke)
    print(summary_df[["site", "requested_year", "scenario", "final_gwad", "irrigation_total", "fertilizer_total"]].to_string(index=False))
    print(f"[done] {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sites", nargs="*", default=None, choices=sorted(SITE_CONFIG))
    args = parser.parse_args()
    run(smoke=bool(args.smoke), sites=args.sites)


if __name__ == "__main__":
    main()

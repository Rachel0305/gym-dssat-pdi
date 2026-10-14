from __future__ import annotations

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

import run_hla_baseline_relative_dqn_checkpoint_015_12 as hla_dqn
import run_hla_unified_dqn_long_train_015_09 as base
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events, prepare_case_at


OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2010_to_2007_dqn_transfer_eval_015_18"
)
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-01_015_18_hla2010_to_2007_dqn_transfer_eval_record.md"

TRAIN_YEAR = 2010
TEST_YEAR = 2007
TRANSFER_MODELS = [
    {"train_seed": 0, "checkpoint": 35000},
    {"train_seed": 1, "checkpoint": 25000},
]

BASELINE_DAILY_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_2007_2009_management_space_check"
    / "hla_2007_2009_management_space_daily.csv"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_2007_2009_management_space_check"
    / "hla_2007_2009_management_space_summary_corrected.csv"
)
BASELINE_EVENTS_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_2007_2009_management_space_check"
    / "hla_2007_2009_management_space_events_deduplicated.csv"
)
BASELINE_WTH_PATH = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_2007_2009_management_space_check"
    / "runs"
    / f"{TEST_YEAR}_null"
    / "input"
    / f"CNHL{str(TEST_YEAR)[-2:]}01.WTH"
)


COLORS = {
    "null": "#222222",
    "recorded": "#C9252D",
    "dssat_auto": "#B8860B",
    "transfer_2010_seed0": "#255C99",
    "transfer_2010_seed1": "#7B3F98",
}

LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "transfer_2010_seed0": "2010-trained DQN seed0",
    "transfer_2010_seed1": "2010-trained DQN seed1",
}


def configure() -> None:
    base.configure_shared_settings()
    install_official_reward_module()


def model_path(train_seed: int, checkpoint: int) -> Path:
    return (
        hla_dqn.OUT_ROOT
        / str(TRAIN_YEAR)
        / f"baseline_relative_seed{train_seed}_50000steps"
        / "models"
        / f"dqn_baseline_relative_checkpoint_{checkpoint}.zip"
    )


def prepare_test_case(run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_case_at(TEST_YEAR, run_dir)
    return json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))


def evaluate_transfer(model, env_args: dict[str, Any], train_seed: int, checkpoint: int, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    null_baseline = get_test_null_baseline_yield()
    eval_env = hla_dqn.make_env(env_args, null_baseline)
    rows: list[dict[str, Any]] = []
    scenario = f"transfer_2010_seed{train_seed}"
    snapshot_dir = run_dir / scenario / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = eval_env.reset()
        for step in range(280):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(eval_env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "requested_year": TEST_YEAR,
                    "scenario": scenario,
                    "source": "hla2010_to_hla2015_transfer",
                    "train_year": TRAIN_YEAR,
                    "train_seed": train_seed,
                    "train_checkpoint_step": checkpoint,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grain_kg_ha": scalar(latest.get("grnwt")),
                    "biomass_kg_ha": scalar(latest.get("topwt")),
                    "water_stress": scalar(latest.get("swfac")),
                    "nitrogen_stress": scalar(latest.get("nstres")),
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
            snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp, snapshot_dir, dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    daily = attach_rain(daily)
    plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT") if (snapshot_dir / "PlantGro.OUT").exists() else pd.DataFrame()
    event_summary = parse_events(snapshot_dir / "MgmtEvent.OUT")
    summary = {
        "requested_year": TEST_YEAR,
        "scenario": scenario,
        "label": LABELS[scenario],
        "train_year": TRAIN_YEAR,
        "train_seed": train_seed,
        "train_checkpoint_step": checkpoint,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "rain_total": float(daily[["dap", "rain"]].drop_duplicates("dap")["rain"].sum()) if not daily.empty and "rain" in daily.columns else np.nan,
        "max_water_stress": float(daily["water_stress"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nitrogen_stress"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        **event_summary,
    }
    return daily, summary


def get_test_null_baseline_yield() -> float:
    summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    if "requested_year" not in summary.columns and "year" in summary.columns:
        summary["requested_year"] = summary["year"]
    scenario = summary["scenario"].fillna("null").replace({"": "null"})
    mask = summary["requested_year"].astype(int).eq(TEST_YEAR) & scenario.eq("null")
    if not mask.any():
        raise RuntimeError(f"Cannot find HLA{TEST_YEAR} null baseline in {BASELINE_SUMMARY_PATH}")
    row = summary.loc[mask].iloc[0]
    if "final_gwad" in row:
        return float(row["final_gwad"])
    return float(row["gwad"])


def load_baseline_and_local() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(BASELINE_DAILY_PATH)
    if "requested_year" not in daily.columns and "year" in daily.columns:
        daily["requested_year"] = daily["year"]
    daily = daily[daily["requested_year"].astype(int).eq(TEST_YEAR)].copy()
    daily["scenario"] = daily["scenario"].fillna("null").replace({"": "null"})
    daily = daily[daily["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    daily = daily.rename(
        columns={
            "gwad": "grain_kg_ha",
            "cwad": "biomass_kg_ha",
            "wspd": "water_stress",
            "nstd": "nitrogen_stress",
        }
    )
    daily["source"] = "hla_2007_2009_management_space_check"
    daily["rain"] = daily["doy"].map(load_weather_rain_by_doy()).fillna(0.0)
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if BASELINE_EVENTS_PATH.exists():
        events = pd.read_csv(BASELINE_EVENTS_PATH)
        events = events[events["year"].astype(int).eq(TEST_YEAR)].copy()
        events["scenario"] = events["scenario"].fillna("null").replace({"": "null"})
        for _, event in events.iterrows():
            mask = daily["scenario"].eq(event["scenario"]) & daily["dap"].astype(float).eq(float(event["dap"]))
            op = str(event["operation"]).lower()
            if "irrig" in op:
                daily.loc[mask, "irrigation_mm"] += float(event["amount"])
            elif "fert" in op:
                daily.loc[mask, "fertilizer_kg_ha"] += float(event["amount"])

    summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    if "requested_year" not in summary.columns and "year" in summary.columns:
        summary["requested_year"] = summary["year"]
    summary = summary[summary["requested_year"].astype(int).eq(TEST_YEAR)].copy()
    summary["scenario"] = summary["scenario"].fillna("null").replace({"": "null"})
    summary = summary[summary["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    summary = summary.rename(
        columns={
            "gwad": "final_gwad",
            "cwad": "final_cwad",
            "max_wspd": "max_water_stress",
            "max_nstd": "max_nitrogen_stress",
            "total_irrigation_mm": "irrigation_total",
            "total_n_kg_ha": "fertilizer_total",
        }
    )
    if "label" not in summary.columns:
        summary["label"] = summary["scenario"].map(LABELS)
    rain_total = float(daily.loc[daily["scenario"].eq("null"), ["dap", "rain"]].drop_duplicates("dap")["rain"].sum())
    summary["rain_total"] = rain_total
    return daily, summary


def load_weather_rain_by_doy() -> dict[int, float]:
    rain: dict[int, float] = {}
    if not BASELINE_WTH_PATH.exists():
        return rain
    for line in BASELINE_WTH_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("@") or text.startswith("$"):
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        try:
            yrdoy = int(float(parts[0]))
            doy = yrdoy % 1000
            rain[doy] = float(parts[4])
        except ValueError:
            continue
    return rain


def attach_rain(daily: pd.DataFrame) -> pd.DataFrame:
    rain_by_dap = load_baseline_rain_by_dap()
    out = daily.drop(columns=["rain"], errors="ignore").copy()
    out["rain"] = out["dap"].map(rain_by_dap).fillna(0.0)
    return out


def load_baseline_rain_by_dap() -> dict[float, float]:
    base = pd.read_csv(BASELINE_DAILY_PATH)
    if "requested_year" not in base.columns and "year" in base.columns:
        base["requested_year"] = base["year"]
    base["scenario"] = base["scenario"].fillna("null").replace({"": "null"})
    base = base[base["requested_year"].astype(int).eq(TEST_YEAR) & base["scenario"].eq("null")].copy()
    base["rain"] = base["doy"].map(load_weather_rain_by_doy()).fillna(0.0)
    return {float(row["dap"]): float(row["rain"]) for _, row in base.iterrows()}


def plot_compare(daily: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rain = daily[daily["scenario"].eq("null")][["dap", "rain"]].drop_duplicates().sort_values("dap")
    order = [
        "null",
        "recorded",
        "dssat_auto",
        "transfer_2010_seed0",
        "transfer_2010_seed1",
    ]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(9.0, 9.3),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0, 0.9, 1.25]},
    )
    fig.suptitle("HLA2010-trained DQN transfer evaluation on HLA2007", x=0.06, y=0.995, ha="left", fontsize=10, fontweight="bold")
    axes[0].bar(rain["dap"], rain["rain"], width=0.9, color="#BFC5CF", edgecolor="#87909C", linewidth=0.25)
    axes[0].set_ylabel("Rain\n(mm)")

    for scenario in order:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        label = LABELS[scenario]
        ls = "--" if scenario in {"recorded", "transfer_2010_seed1"} else "-"
        axes[1].plot(sub["dap"], sub["water_stress"], color=color, linestyle=ls, linewidth=1.5, label=label)
        axes[2].plot(sub["dap"], sub["nitrogen_stress"], color=color, linestyle=ls, linewidth=1.5)
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=color, linestyle=ls, linewidth=1.6)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=color, linestyle=":", linewidth=1.25)
        irrig = sub[sub["irrigation_mm"].fillna(0) > 0]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 0]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], colors=color, linestyles=ls, linewidth=1.9, alpha=0.9)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], s=26, color=color, edgecolor="white", linewidth=0.35, zorder=4)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[1].legend(loc="upper left", ncol=2, fontsize=6.8)
    axes[3].set_title("Irrigation: vertical lines; fertilization: markers", loc="left", fontsize=8)
    axes[4].set_title("Solid/dashed lines: grain yield; dotted lines: aboveground biomass", loc="left", fontsize=8)
    for ax in axes:
        ax.grid(True, axis="x", color="#E2E7EF", linewidth=0.55)
        ax.grid(True, axis="y", color="#EDF1F5", linewidth=0.45, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f"{out_path}.png", dpi=350, bbox_inches="tight")
    fig.savefig(f"{out_path}.svg", bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "scenario",
        "label",
        "train_year",
        "train_seed",
        "train_checkpoint_step",
        "final_gwad",
        "final_cwad",
        "irrigation_total",
        "fertilizer_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
    ]
    use = df[[c for c in cols if c in df.columns]].copy()
    lines = ["| " + " | ".join(use.columns) + " |", "| " + " | ".join(["---"] * len(use.columns)) + " |"]
    for _, row in use.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.3f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame) -> None:
    lines = [
        "# 015_18 HLA2010-trained DQN transfer evaluation on HLA2007",
        "",
        "## Purpose",
        "",
        "Evaluate whether DQN checkpoints trained on HLA2010 can produce reasonable water-nitrogen decisions on HLA2007 without additional training.",
        "",
        "## Tested models",
        "",
        "- HLA2010 seed0 checkpoint 35000 -> HLA2007",
        "- HLA2010 seed1 checkpoint 25000 -> HLA2007",
        "",
        "## Outputs",
        "",
        f"- Daily CSV: `{(OUT_DIR / 'hla2010_to_2007_transfer_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- Summary CSV: `{(OUT_DIR / 'hla2010_to_2007_transfer_eval_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- Figure: `{(FIG_DIR / 'hla2010_to_2007_transfer_eval_process.png').relative_to(PROJECT_ROOT)}`",
        "",
        "## Results",
        "",
        markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "- If transfer models are close to local DQN, the policy has useful cross-year transfer.",
        "- If transfer models are closer to null or use resources poorly, current DQN is mainly year-specific.",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    env_args = prepare_test_case(OUT_DIR / "test_case_hla2015")
    from stable_baselines3 import DQN

    transfer_daily = []
    transfer_summary = []
    for spec in TRANSFER_MODELS:
        path = model_path(spec["train_seed"], spec["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(path)
        model = DQN.load(str(path))
        daily, summary = evaluate_transfer(model, env_args, spec["train_seed"], spec["checkpoint"], OUT_DIR)
        transfer_daily.append(daily)
        transfer_summary.append(summary)

    baseline_daily, baseline_summary = load_baseline_and_local()
    all_daily = pd.concat([baseline_daily, *transfer_daily], ignore_index=True, sort=False)
    all_daily["scenario"] = all_daily["scenario"].fillna("null").replace({"": "null"})
    all_summary = pd.concat([baseline_summary, pd.DataFrame(transfer_summary)], ignore_index=True, sort=False)

    all_daily.to_csv(OUT_DIR / "hla2010_to_2007_transfer_eval_daily.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT_DIR / "hla2010_to_2007_transfer_eval_summary.csv", index=False, encoding="utf-8-sig")
    plot_compare(all_daily, FIG_DIR / "hla2010_to_2007_transfer_eval_process")
    write_record(all_summary)
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()

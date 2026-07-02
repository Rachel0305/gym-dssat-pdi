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
import run_hla_new_cultivar_candidate_year_screening as screen
import run_hla_unified_dqn_long_train_015_09 as base
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from run_hla_official_reward_restart_smoke import (
    NEW_CUL,
    install_official_reward_module,
    parse_events,
    replace_zero_management_rows,
    set_pdi_jinja_placeholders,
    set_treatment_mi_mf,
)


OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2010_to_2016_2022_dqn_transfer_eval_015_19"
)
BASELINE_DIR = OUT_DIR / "baseline_null_auto"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-01_015_19_hla2010_to_2016_2022_dqn_transfer_eval_record.md"

TRAIN_YEAR = 2010
TEST_YEARS = [2016, 2022]
TRANSFER_MODELS = [
    {"train_seed": 0, "checkpoint": 35000},
    {"train_seed": 1, "checkpoint": 25000},
]

COLORS = {
    "null": "#222222",
    "dssat_auto": "#B8860B",
    "transfer_2010_seed0": "#255C99",
    "transfer_2010_seed1": "#7B3F98",
}

LABELS = {
    "null": "Null",
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


def prepare_test_case(year: int, run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    src_dir = screen.SOURCE_ROOT / "null" / str(year) / "input"
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")
    filex = next(input_dir.glob("*.MZX"))
    text = filex.read_text(encoding="latin1", errors="ignore")
    text = set_treatment_mi_mf(text, "1", "1")
    text = set_pdi_jinja_placeholders(text)
    text = replace_zero_management_rows(text)
    filex.write_text(text, encoding="latin1")
    aux = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(run_dir / f"transfer_eval_{year}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))


def build_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    if BASELINE_DIR.exists():
        shutil.rmtree(BASELINE_DIR)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    old_out = screen.OUT_DIR
    old_years = list(screen.YEARS)
    old_scenarios = list(screen.SCENARIOS)
    try:
        screen.OUT_DIR = BASELINE_DIR
        screen.YEARS = TEST_YEARS
        screen.SCENARIOS = ["null", "auto_irrig"]
        for scenario in screen.SCENARIOS:
            for year in screen.YEARS:
                screen.prepare_case(scenario, year)
                screen.child_run(scenario, year)
        daily, events, summary = screen.collect()
    finally:
        screen.OUT_DIR = old_out
        screen.YEARS = old_years
        screen.SCENARIOS = old_scenarios

    daily = daily.copy()
    summary = summary.copy()
    daily["scenario"] = daily["scenario"].replace({"auto_irrig": "dssat_auto"})
    summary["scenario"] = summary["scenario"].replace({"auto_irrig": "dssat_auto"})
    daily = daily.rename(
        columns={
            "gwad": "grain_kg_ha",
            "cwad": "biomass_kg_ha",
            "wspd": "water_stress",
            "nstd": "nitrogen_stress",
        }
    )
    daily = (
        daily.sort_values(["scenario", "requested_year", "dap"])
        .drop_duplicates(subset=["scenario", "requested_year", "dap"], keep="last")
        .reset_index(drop=True)
    )
    summary = (
        daily.sort_values(["scenario", "requested_year", "dap"])
        .groupby(["scenario", "requested_year"], as_index=False)
        .agg(
            days=("dap", "size"),
            final_dap=("dap", "last"),
            final_gwad=("grain_kg_ha", "last"),
            final_cwad=("biomass_kg_ha", "last"),
            rain_total=("rain", "sum"),
            irrigation_total=("irrigation_mm", "sum"),
            fertilizer_total=("fertilizer_kg_ha", "sum"),
            max_water_stress=("water_stress", "max"),
            mean_wspd=("water_stress", "mean"),
            max_nitrogen_stress=("nitrogen_stress", "max"),
            mean_nstd=("nitrogen_stress", "mean"),
        )
    )
    summary["label"] = summary["scenario"].map(LABELS)
    return daily, summary


def get_null_baseline_yield(baseline_summary: pd.DataFrame, year: int) -> float:
    row = baseline_summary[
        baseline_summary["requested_year"].astype(int).eq(year)
        & baseline_summary["scenario"].eq("null")
    ]
    if row.empty:
        raise RuntimeError(f"Missing null baseline for HLA{year}")
    return float(row.iloc[0]["final_gwad"])


def evaluate_transfer(
    model,
    env_args: dict[str, Any],
    year: int,
    null_baseline: float,
    train_seed: int,
    checkpoint: int,
    run_dir: Path,
    rain_by_dap: dict[float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_env = hla_dqn.make_env(env_args, null_baseline)
    rows: list[dict[str, Any]] = []
    scenario = f"transfer_2010_seed{train_seed}"
    snapshot_dir = run_dir / str(year) / scenario / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = eval_env.reset()
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(eval_env.env, "last_safe_real_action", {}) or {})
            dap = scalar(latest.get("dap"))
            rows.append(
                {
                    "requested_year": year,
                    "scenario": scenario,
                    "source": "hla2010_transfer",
                    "train_year": TRAIN_YEAR,
                    "train_seed": train_seed,
                    "train_checkpoint_step": checkpoint,
                    "step": step,
                    "dap": dap,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "rain": rain_by_dap.get(float(dap), 0.0),
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
    plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT") if (snapshot_dir / "PlantGro.OUT").exists() else pd.DataFrame()
    event_summary = parse_events(snapshot_dir / "MgmtEvent.OUT")
    summary = {
        "requested_year": year,
        "scenario": scenario,
        "label": LABELS[scenario],
        "train_year": TRAIN_YEAR,
        "train_seed": train_seed,
        "train_checkpoint_step": checkpoint,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "rain_total": float(daily[["dap", "rain"]].drop_duplicates("dap")["rain"].sum()) if not daily.empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "max_water_stress": float(daily["water_stress"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nitrogen_stress"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        **event_summary,
    }
    return daily, summary


def rain_lookup(baseline_daily: pd.DataFrame, year: int) -> dict[float, float]:
    sub = baseline_daily[
        baseline_daily["requested_year"].astype(int).eq(year)
        & baseline_daily["scenario"].eq("null")
    ][["dap", "rain"]].drop_duplicates("dap")
    return {float(r["dap"]): float(r["rain"]) for _, r in sub.iterrows()}


def plot_year(daily: pd.DataFrame, year: int, out_base: Path) -> None:
    sub_all = daily[daily["requested_year"].astype(int).eq(year)].copy()
    rain = sub_all[sub_all["scenario"].eq("null")][["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    order = ["null", "dssat_auto", "transfer_2010_seed0", "transfer_2010_seed1"]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.8, 9.1),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0, 0.9, 1.25]},
    )
    fig.suptitle(f"HLA2010-trained DQN transfer evaluation on HLA{year}", x=0.06, y=0.995, ha="left", fontsize=10, fontweight="bold")
    axes[0].bar(rain["dap"], rain["rain"], width=0.9, color="#BFC5CF", edgecolor="#87909C", linewidth=0.25)
    axes[0].set_ylabel("Rain\n(mm)")
    for scenario in order:
        sub = sub_all[sub_all["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        label = LABELS[scenario]
        ls = "--" if scenario == "transfer_2010_seed1" else "-"
        axes[1].plot(sub["dap"], sub["water_stress"], color=color, linestyle=ls, linewidth=1.6, label=label)
        axes[2].plot(sub["dap"], sub["nitrogen_stress"], color=color, linestyle=ls, linewidth=1.6)
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=color, linestyle=ls, linewidth=1.7)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=color, linestyle=":", linewidth=1.25)
        irrig = sub[sub["irrigation_mm"].fillna(0) > 0]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 0]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], colors=color, linestyles=ls, linewidth=1.8, alpha=0.9)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], s=25, color=color, edgecolor="white", linewidth=0.35, zorder=4)
    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    axes[3].set_title("Irrigation: vertical lines; fertilization: markers", loc="left", fontsize=8)
    axes[4].set_title("Solid/dashed lines: grain yield; dotted lines: aboveground biomass", loc="left", fontsize=8)
    for ax in axes:
        ax.grid(True, axis="x", color="#E2E7EF", linewidth=0.55)
        ax.grid(True, axis="y", color="#EDF1F5", linewidth=0.45, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=350, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "requested_year",
        "scenario",
        "label",
        "train_seed",
        "train_checkpoint_step",
        "final_gwad",
        "final_cwad",
        "rain_total",
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
        "# 015_19 HLA2010-trained DQN transfer evaluation on HLA2016 and HLA2022",
        "",
        "## Purpose",
        "",
        "Evaluate whether HLA2010 DQN checkpoints transfer to HLA2016 and HLA2022 without additional training.",
        "",
        "## Outputs",
        "",
        f"- Daily CSV: `{(OUT_DIR / 'hla2010_to_2016_2022_transfer_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- Summary CSV: `{(OUT_DIR / 'hla2010_to_2016_2022_transfer_eval_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- Figures: `{FIG_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## Results",
        "",
        markdown_table(summary),
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_daily, baseline_summary = build_baselines()
    from stable_baselines3 import DQN

    all_transfer_daily: list[pd.DataFrame] = []
    all_transfer_summary: list[dict[str, Any]] = []
    for year in TEST_YEARS:
        env_args = prepare_test_case(year, OUT_DIR / "test_cases" / str(year))
        null_yield = get_null_baseline_yield(baseline_summary, year)
        rain_by_dap = rain_lookup(baseline_daily, year)
        for spec in TRANSFER_MODELS:
            path = model_path(spec["train_seed"], spec["checkpoint"])
            if not path.exists():
                raise FileNotFoundError(path)
            model = DQN.load(str(path))
            daily, summary = evaluate_transfer(
                model,
                env_args,
                year,
                null_yield,
                spec["train_seed"],
                spec["checkpoint"],
                OUT_DIR,
                rain_by_dap,
            )
            all_transfer_daily.append(daily)
            all_transfer_summary.append(summary)

    all_daily = pd.concat([baseline_daily, *all_transfer_daily], ignore_index=True, sort=False)
    all_summary = pd.concat([baseline_summary, pd.DataFrame(all_transfer_summary)], ignore_index=True, sort=False)
    all_daily.to_csv(OUT_DIR / "hla2010_to_2016_2022_transfer_eval_daily.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT_DIR / "hla2010_to_2016_2022_transfer_eval_summary.csv", index=False, encoding="utf-8-sig")
    for year in TEST_YEARS:
        plot_year(all_daily, year, FIG_DIR / f"hla2010_to_{year}_transfer_eval_process")
    write_record(all_summary)
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()

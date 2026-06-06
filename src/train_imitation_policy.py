from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from evaluate_imitation_policy import evaluate_imitation_policy, profit_score
from imitation_policy_models import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    ConstantScheduleBaseline,
    TwoStageClassifierRegressor,
    make_mlp_regressor,
    make_random_forest_regressor,
    postprocess_actions,
    supervised_metrics,
)
from ppo_safe_rendering import PROJECT_ROOT, load_yaml


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_imitation_learning_prior.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-06_imitation_learning_prior_report.md"
DOC_PPT = PROJECT_ROOT / "docs" / "2026-06-06_imitation_learning_prior_report.pptx"


def ensure_dirs() -> None:
    for sub in ["configs", "datasets", "models", "logs", "daily_outputs", "evaluation", "figures", "reports"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for st in ["HLA", "SYA", "LCA"]:
        (OUTPUT_ROOT / "daily_outputs" / st).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "figures" / st).mkdir(parents=True, exist_ok=True)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    for col in work.select_dtypes(include=["float", "int"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    for col in work.columns:
        work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(work.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, separator, *rows])


def parse_state_variables(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def best_schedules(config: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for station, path in config["imitation"]["expert_rankings"].items():
        ranking = pd.read_csv(PROJECT_ROOT / path)
        ranking = ranking.sort_values("overall_score", ascending=False)
        result[station] = str(ranking["schedule_id"].iloc[0])
    return result


def load_expert_cross_summary(config: dict, best: dict[str, str]) -> pd.DataFrame:
    rows = []
    for station, path in config["imitation"]["cross_year_summaries"].items():
        cross = pd.read_csv(PROJECT_ROOT / path)
        cross = cross[cross["schedule_id"].astype(str).eq(best[station])].copy()
        rows.append(cross)
    return pd.concat(rows, ignore_index=True)


def build_dataset_from_daily_outputs(config: dict, best: dict[str, str], cross: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in cross.iterrows():
        station = str(row["station"])
        year = int(row["eval_year"])
        schedule_id = str(row["schedule_id"])
        path = PROJECT_ROOT / str(row["daily_csv_path"])
        daily = pd.read_csv(path)
        daily = daily.reset_index(drop=True)
        daily["sim_day"] = np.arange(1, len(daily) + 1)
        daily["station"] = station
        daily["year"] = year
        daily["schedule_id"] = schedule_id
        daily["expert_policy_type"] = "best_offline_schedule_cross_year"
        daily["expert_action_irrigation"] = pd.to_numeric(daily["real_action_amir"], errors="coerce").fillna(0.0)
        daily["expert_action_n"] = pd.to_numeric(daily["real_action_anfer"], errors="coerce").fillna(0.0)
        if "state_variables" not in daily.columns:
            state_cols = ["sim_day", "doy", "topwt", "grnwt", "xlai", "totir", "tofer", "swfac", "nstres"]
            daily["state_variables"] = daily[state_cols].apply(lambda s: json.dumps(s.to_dict(), ensure_ascii=False), axis=1)
        for col in FEATURE_COLUMNS:
            if col not in daily.columns:
                daily[col] = 0.0
        frames.append(daily)
    data = pd.concat(frames, ignore_index=True)
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        if col != "station":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def clean_existing_imitation_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    states = data.get("state_variables", pd.Series(["{}"] * len(data))).map(parse_state_variables)
    for key in ["topwt", "grnwt", "xlai", "totir", "tofer", "swfac", "nstres", "doy", "sim_day"]:
        if key not in data.columns:
            data[key] = states.map(lambda item: item.get(key, np.nan))
    if "sim_day" not in data.columns or data["sim_day"].isna().all():
        data["sim_day"] = data.groupby(["station", "year"]).cumcount() + 1
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        if col != "station" and col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def dataset_check_markdown(original: pd.DataFrame, clean: pd.DataFrame, rebuilt: pd.DataFrame, best: dict[str, str]) -> str:
    lines = ["# Imitation Dataset Check", "", "Generated at: 2026-06-06", ""]
    lines += [
        f"- Original imitation_dataset.csv exists: {not original.empty}",
        f"- Original rows: {len(original)}",
        f"- Original stations: {', '.join(sorted(original['station'].dropna().astype(str).unique())) if not original.empty and 'station' in original else 'none'}",
        f"- Rebuilt rows from best HLA/SYA/LCA daily outputs: {len(rebuilt)}",
        f"- Best schedules: {best}",
        f"- State variables used: {', '.join(FEATURE_COLUMNS)}",
        f"- Targets used: {', '.join(TARGET_COLUMNS)}",
        f"- NaN cells after cleaning: {int(clean[FEATURE_COLUMNS + TARGET_COLUMNS].isna().sum().sum())}",
        f"- Inf cells after cleaning: {int(np.isinf(clean[FEATURE_COLUMNS[1:] + TARGET_COLUMNS].to_numpy(dtype=float)).sum())}",
    ]
    action = clean[TARGET_COLUMNS].fillna(0.0)
    lines += [
        f"- Irrigation nonzero samples: {int((action['expert_action_irrigation'] > 0).sum())}",
        f"- Nitrogen nonzero samples: {int((action['expert_action_n'] > 0).sum())}",
        f"- Zero-action ratio: {float(((action <= 0).all(axis=1)).mean()):.4f}",
        "- Conclusion: actions are extremely sparse; two-stage classifier-regressor is trained in addition to regression baselines.",
        "",
    ]
    if not clean.empty:
        counts = clean.groupby(["station", "year"]).size().reset_index(name="rows")
        lines += ["## Rows by Station-Year", "", df_to_markdown(counts), ""]
    return "\n".join(lines)


def prepare_splits(config: dict, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for station, train_year in config["imitation"]["train_years"].items():
        train_parts.append(data[(data["station"].eq(station)) & (data["year"].astype(int).eq(int(train_year)))])
    for station, years in config["imitation"]["validation_years"].items():
        val_parts.append(data[(data["station"].eq(station)) & (data["year"].astype(int).isin([int(y) for y in years]))])
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(val_parts, ignore_index=True)
    return train, validation


def train_models(config: dict, train: pd.DataFrame, all_data: pd.DataFrame, best: dict[str, str]) -> dict[str, object]:
    schedule_table = (
        train[["station", "sim_day", "expert_action_irrigation", "expert_action_n"]]
        .groupby(["station", "sim_day"], as_index=False)
        .agg({"expert_action_irrigation": "max", "expert_action_n": "max"})
    )
    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMNS].to_numpy(dtype=float)
    models: dict[str, object] = {
        "BC_constant_schedule_baseline": ConstantScheduleBaseline(schedule_table=schedule_table),
        "BC_random_forest_regressor": make_random_forest_regressor(int(config.get("seed", 0))).fit(x_train, y_train),
        "BC_mlp_regressor": make_mlp_regressor(int(config.get("seed", 0))).fit(x_train, y_train),
        "BC_two_stage_classifier_regressor": TwoStageClassifierRegressor(
            threshold=float(config.get("action_postprocess", {}).get("event_threshold", 25.0)),
            random_state=int(config.get("seed", 0)),
        ).fit(x_train, y_train),
    }
    model_dir = OUTPUT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    return models


def export_policy_action_tables(config: dict, models: dict[str, object], train: pd.DataFrame) -> None:
    """Export learned station x simulation-day schedules for dependency-light DSSAT replay."""
    out_dir = OUTPUT_ROOT / "evaluation" / "policy_action_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config.get("action_postprocess", {}).get("event_threshold", 25.0))
    for name, model in models.items():
        rows: list[pd.DataFrame] = []
        for station, part in train.groupby("station"):
            canonical = part.sort_values("sim_day").copy()
            features = canonical[FEATURE_COLUMNS]
            pred = model.predict(features)
            if name in {"BC_random_forest_regressor", "BC_mlp_regressor"}:
                pred = postprocess_actions(pred, threshold=threshold)
            pred = np.asarray(pred, dtype=float).reshape(-1, 2)
            table = canonical[["station", "year", "sim_day", "doy"]].copy()
            table["policy_name"] = name
            table["model_type"] = name.replace("BC_", "")
            table["source_train_year"] = table["year"]
            table["real_action_amir"] = pred[:, 0]
            table["real_action_anfer"] = pred[:, 1]
            rows.append(table[["policy_name", "model_type", "station", "source_train_year", "sim_day", "doy", "real_action_amir", "real_action_anfer"]])
        pd.concat(rows, ignore_index=True).to_csv(out_dir / f"{name}_action_table.csv", index=False, encoding="utf-8-sig")


def evaluate_supervised(config: dict, models: dict[str, object], train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    threshold = float(config.get("action_postprocess", {}).get("event_threshold", 25.0))
    for split_name, split in [("train", train), ("validation", validation)]:
        x = split[FEATURE_COLUMNS]
        y = split[TARGET_COLUMNS].to_numpy(dtype=float)
        for name, model in models.items():
            pred = model.predict(x)
            if name in {"BC_random_forest_regressor", "BC_mlp_regressor"}:
                pred = postprocess_actions(pred, threshold=threshold)
            row = {"model_name": name, "split": split_name, "n_rows": len(split)}
            row.update(supervised_metrics(y, pred))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_ROOT / "evaluation" / "imitation_supervised_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def run_dssat_evaluation(config: dict, models: dict[str, object], expert_lookup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    ref_yield = float(config.get("ppo_reference", {}).get("mean_yield", np.nan))
    for name, model in models.items():
        model_type = name.replace("BC_", "")
        for station, years in config["imitation"]["evaluation_years"].items():
            for year in years:
                print(f"[imitation] DSSAT eval {name} {station} {year}", flush=True)
                try:
                    rows.append(evaluate_imitation_policy(model, config, station, int(year), name, model_type, expert_lookup, ref_yield))
                except Exception as exc:
                    rows.append(
                        {
                            "station": station,
                            "policy_name": name,
                            "model_type": model_type,
                            "eval_year": int(year),
                            "run_status": "failed",
                            "episode_completed": False,
                            "error_message": f"{type(exc).__name__}: {exc}",
                        }
                    )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def make_policy_comparison(config: dict, expert_lookup: pd.DataFrame, dssat_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    ref = config.get("ppo_reference", {})
    ppo_yield = float(ref.get("mean_yield", np.nan))
    ppo_irrigation = float(ref.get("total_irrigation", 300.0))
    ppo_n = float(ref.get("total_n", 450.0))
    for _, row in expert_lookup.iterrows():
        rows.append(
            {
                "station": row["station"],
                "year": int(row["eval_year"]),
                "policy_type": "expert_schedule",
                "policy_name": row["schedule_id"],
                "final_grnwt": row["final_grnwt"],
                "total_irrigation": row["total_irrigation"],
                "total_n_fertilizer": row["total_n_fertilizer"],
                "profit_score": row["profit_score"],
                "mean_swfac": row["mean_swfac"],
                "mean_nstres": row["mean_nstres"],
                "yield_loss_vs_ppo": (ppo_yield - row["final_grnwt"]) / ppo_yield if ppo_yield else np.nan,
                "input_reduction_vs_ppo": 1.0 - ((row["total_irrigation"] / ppo_irrigation + row["total_n_fertilizer"] / ppo_n) / 2.0),
                "notes": "best_offline_schedule",
            }
        )
    for _, row in dssat_summary[dssat_summary["run_status"].eq("ok")].iterrows():
        rows.append(
            {
                "station": row["station"],
                "year": int(row["eval_year"]),
                "policy_type": "imitation_policy",
                "policy_name": row["policy_name"],
                "final_grnwt": row["final_grnwt"],
                "total_irrigation": row["total_irrigation"],
                "total_n_fertilizer": row["total_n_fertilizer"],
                "profit_score": row["profit_score"],
                "mean_swfac": row["mean_swfac"],
                "mean_nstres": row["mean_nstres"],
                "yield_loss_vs_ppo": row["yield_loss_vs_ppo_baseline"],
                "input_reduction_vs_ppo": 1.0 - ((row["total_irrigation"] / ppo_irrigation + row["total_n_fertilizer"] / ppo_n) / 2.0),
                "notes": "behavior_cloning_policy",
            }
        )
    for station, years in config["imitation"]["evaluation_years"].items():
        for year in years:
            rows.append(
                {
                    "station": station,
                    "year": int(year),
                    "policy_type": "ppo_cap_saturated_baseline",
                    "policy_name": "reference_300mm_450kgN",
                    "final_grnwt": ppo_yield,
                    "total_irrigation": ppo_irrigation,
                    "total_n_fertilizer": ppo_n,
                    "profit_score": profit_score(ppo_yield, ppo_irrigation, ppo_n, config),
                    "mean_swfac": np.nan,
                    "mean_nstres": np.nan,
                    "yield_loss_vs_ppo": 0.0,
                    "input_reduction_vs_ppo": 0.0,
                    "notes": f"reference from {ref.get('source', '')}; same yield reference used for cross-policy normalization",
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_ROOT / "evaluation" / "policy_comparison_expert_bc_ppo.csv", index=False, encoding="utf-8-sig")
    return comparison


def make_figures(config: dict, clean: pd.DataFrame, metrics: pd.DataFrame, dssat: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig_dir = OUTPUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    actions = clean[TARGET_COLUMNS].fillna(0.0)
    plt.figure(figsize=(8, 4.8))
    plt.hist(actions["expert_action_irrigation"], bins=20, alpha=0.6, label="irrigation")
    plt.hist(actions["expert_action_n"], bins=20, alpha=0.6, label="nitrogen")
    plt.xlabel("Action amount")
    plt.ylabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "imitation_action_distribution.png", dpi=180)
    plt.close()

    best_policy = "BC_two_stage_classifier_regressor"
    if dssat[dssat["policy_name"].eq(best_policy)].empty:
        best_policy = "BC_constant_schedule_baseline"
    plt.figure(figsize=(9, 5))
    sub = comparison[comparison["policy_type"].isin(["expert_schedule", "imitation_policy"])]
    for policy_type, marker in [("expert_schedule", "o"), ("imitation_policy", "s")]:
        part = sub[sub["policy_type"].eq(policy_type)]
        plt.scatter(part["total_n_fertilizer"], part["final_grnwt"], label=policy_type, marker=marker)
    plt.xlabel("Total N fertilizer (kg/ha)")
    plt.ylabel("Final grain yield")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_comparison_yield_vs_input.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    comp = comparison.groupby(["policy_type", "policy_name"], dropna=False)["profit_score"].mean().reset_index()
    comp = comp.sort_values("profit_score", ascending=False).head(12)
    plt.barh(comp["policy_name"], comp["profit_score"], color="#4F7ECF")
    plt.xlabel("Mean profit score")
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_comparison_profit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    ok = dssat[dssat["run_status"].eq("ok")]
    site = ok.groupby(["station", "policy_name"])[["final_grnwt", "total_irrigation", "total_n_fertilizer"]].mean().reset_index()
    for station, part in site.groupby("station"):
        plt.plot(part["policy_name"], part["final_grnwt"], marker="o", label=station)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean yield")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_comparison_by_site.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    gen = ok.groupby(["eval_year", "policy_name"])["final_grnwt"].mean().reset_index()
    for name, part in gen.groupby("policy_name"):
        plt.plot(part["eval_year"].astype(str), part["final_grnwt"], marker="o", label=name.replace("BC_", ""))
    plt.ylabel("Final grain yield")
    plt.xlabel("Evaluation year")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "imitation_generalization_by_year.png", dpi=180)
    plt.close()

    daily_paths = ok[ok["policy_name"].eq(best_policy)]["daily_csv_path"].dropna().head(3)
    if len(daily_paths):
        plt.figure(figsize=(9, 5))
        for rel in daily_paths:
            daily = pd.read_csv(PROJECT_ROOT / rel)
            label = f"{daily['station'].iloc[0]} {int(daily['eval_year'].iloc[0])}"
            plt.step(daily["sim_day"], daily["real_action_anfer"], where="post", label=f"{label} N")
        plt.xlabel("Simulation day")
        plt.ylabel("Nitrogen action")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "expert_vs_imitation_daily_actions.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        for rel in daily_paths:
            daily = pd.read_csv(PROJECT_ROOT / rel)
            label = f"{daily['station'].iloc[0]} {int(daily['eval_year'].iloc[0])}"
            plt.plot(daily["sim_day"], daily["real_action_anfer"].cumsum(), label=f"{label} cumulative N")
        plt.xlabel("Simulation day")
        plt.ylabel("Cumulative N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "expert_vs_imitation_cumulative_inputs.png", dpi=180)
        plt.close()


def write_report(config: dict, best: dict[str, str], metrics: pd.DataFrame, dssat: pd.DataFrame, comparison: pd.DataFrame) -> None:
    ok = dssat[dssat["run_status"].eq("ok")]
    policy_summary = (
        ok.groupby("policy_name")
        .agg(
            eval_count=("eval_year", "count"),
            mean_yield=("final_grnwt", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n=("total_n_fertilizer", "mean"),
            mean_yield_loss_vs_expert=("yield_loss_vs_expert_schedule", "mean"),
            mean_yield_loss_vs_ppo=("yield_loss_vs_ppo_baseline", "mean"),
        )
        .reset_index()
    )
    pass_gate = policy_summary[
        (policy_summary["mean_irrigation"] <= 100)
        & (policy_summary["mean_n"] <= 200)
        & (policy_summary["mean_yield_loss_vs_ppo"] <= 0.15)
    ].copy()
    text = f"""# Imitation learning prior report

Generated at: 2026-06-06

## Goal

This stage trains supervised behavior cloning priors from offline deterministic expert schedules. It does not train PPO, does not run multi-seed PPO, does not enter rainfall scaling, and does not modify my_data or the original reward files.

## Expert schedules

- HLA: {best.get('HLA')}
- SYA: {best.get('SYA')}
- LCA: {best.get('LCA')}

The original 006_08 imitation dataset contained HLA only, so this stage rebuilt the clean BC dataset from HLA/SYA/LCA best-schedule daily outputs.

## Supervised metrics

{df_to_markdown(metrics)}

## DSSAT/gym-DSSAT evaluation summary

{df_to_markdown(policy_summary)}

## Policy comparison

{df_to_markdown(comparison.groupby(['policy_type','policy_name']).agg(final_grnwt=('final_grnwt','mean'), total_irrigation=('total_irrigation','mean'), total_n_fertilizer=('total_n_fertilizer','mean'), profit_score=('profit_score','mean')).reset_index())}

## Cap saturation check

No successful imitation policy exceeded the 300 mm irrigation / 450 kg ha-1 N safety cap. The main practical threshold for the next stage is stricter: mean irrigation <= 100 mm, mean N <= 200 kg/ha, and mean yield loss vs PPO reference <= 15%.

Policies passing the constrained PPO fine-tuning gate:

{df_to_markdown(pass_gate) if len(pass_gate) else 'None.'}

## Interpretation

BC_constant_schedule_baseline is an expert replay baseline, not a learned general policy. Random forest, MLP, and the two-stage classifier-regressor are supervised priors. Because action samples are extremely sparse and irrigation is zero in the selected expert schedules, these results should be treated as an interpretable policy prior rather than a final RL policy.

## Recommendation

If at least one learned imitation policy passes the gate, the next step can be constrained PPO fine-tuning from imitation prior. If only the constant schedule passes, the next step should be expert dataset augmentation before RL fine-tuning.
"""
    DOC_MD.write_text(text, encoding="utf-8")
    shutil.copy2(DOC_MD, OUTPUT_ROOT / "reports" / DOC_MD.name)


def add_title(slide, title: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.4), Inches(0.45))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Microsoft YaHei"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)


def add_bullets(slide, lines: list[str], left=0.7, top=1.0, width=12.0, height=5.6) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(15)
        p.font.color.rgb = RGBColor(0, 0, 0)


def add_table(slide, df: pd.DataFrame, left=0.5, top=1.05, width=12.3, height=5.6, max_rows=8) -> None:
    data = df.head(max_rows).copy()
    rows, cols = len(data) + 1, len(data.columns)
    table = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    for j, col in enumerate(data.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(79, 126, 207)
        for p in cell.text_frame.paragraphs:
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(9)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER
    for i, (_, row) in enumerate(data.iterrows(), start=1):
        for j, col in enumerate(data.columns):
            cell = table.cell(i, j)
            value = row[col]
            if isinstance(value, float):
                value = round(value, 4)
            cell.text = "" if pd.isna(value) else str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(235, 240, 250) if i % 2 == 0 else RGBColor(255, 255, 255)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(8)
                p.font.color.rgb = RGBColor(0, 0, 0)


def write_ppt(best: dict[str, str], metrics: pd.DataFrame, dssat: pd.DataFrame, comparison: pd.DataFrame) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    add_title(slide, "006_09 模仿学习先验实验记录")
    add_bullets(
        slide,
        [
            "目标：用 006_08 离线 expert schedules 训练 behavior cloning prior。",
            "约束：不训练 PPO，不做 multi-seed，不进入 rainfall scaling，不修改 my_data 或原始 reward。",
            "方法：构建 HLA/SYA/LCA 三站点 BC 数据集，训练 constant/RF/MLP/two-stage 模型，并回到 DSSAT 评估。",
        ],
    )

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Expert schedules 来源")
    add_table(slide, pd.DataFrame([{"station": k, "best_schedule": v} for k, v in best.items()]))

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Supervised metrics")
    add_table(slide, metrics.round(4), max_rows=10)

    policy_summary = (
        dssat[dssat["run_status"].eq("ok")]
        .groupby("policy_name")
        .agg(eval_count=("eval_year", "count"), mean_yield=("final_grnwt", "mean"), mean_irrigation=("total_irrigation", "mean"), mean_n=("total_n_fertilizer", "mean"), mean_yield_loss_vs_ppo=("yield_loss_vs_ppo_baseline", "mean"))
        .reset_index()
    )
    slide = prs.slides.add_slide(blank)
    add_title(slide, "DSSAT evaluation summary")
    add_table(slide, policy_summary.round(4), max_rows=8)

    comp = comparison.groupby(["policy_type", "policy_name"]).agg(final_grnwt=("final_grnwt", "mean"), total_irrigation=("total_irrigation", "mean"), total_n_fertilizer=("total_n_fertilizer", "mean"), profit_score=("profit_score", "mean")).reset_index()
    slide = prs.slides.add_slide(blank)
    add_title(slide, "Expert / BC / PPO reference 对比")
    add_table(slide, comp.round(4), max_rows=10)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "结论与下一步")
    add_bullets(
        slide,
        [
            "BC 结果只作为可解释先验，不作为最终 RL 策略。",
            "若 learned BC 达到低投入且产量损失 <=15%，可进入 constrained PPO fine-tuning。",
            "若只有 constant expert replay 达标，应先扩展 expert schedule 数据，再做 RL fine-tuning。",
        ],
    )

    prs.save(DOC_PPT)
    shutil.copy2(DOC_PPT, OUTPUT_ROOT / "reports" / DOC_PPT.name)


def load_training_context(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    old_path = PROJECT_ROOT / "Leave_One_experiments" / "offline_schedule_search" / "expert_policy" / "imitation_dataset.csv"
    original = clean_existing_imitation_dataset(old_path)
    best = best_schedules(config)
    cross = load_expert_cross_summary(config, best)
    cross.to_csv(OUTPUT_ROOT / "evaluation" / "expert_schedule_lookup_for_bc.csv", index=False, encoding="utf-8-sig")
    rebuilt = build_dataset_from_daily_outputs(config, best, cross)
    clean = rebuilt.copy()
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        if col != "station":
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    numeric_cols = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col != "station"]
    clean[numeric_cols] = clean[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clean.to_csv(OUTPUT_ROOT / "datasets" / "imitation_dataset_clean.csv", index=False, encoding="utf-8-sig")
    check = dataset_check_markdown(original, clean, rebuilt, best)
    (OUTPUT_ROOT / "evaluation" / "imitation_dataset_check.md").write_text(check, encoding="utf-8")
    train, validation = prepare_splits(config, clean)
    train.to_csv(OUTPUT_ROOT / "datasets" / "train_split.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_ROOT / "datasets" / "validation_split.csv", index=False, encoding="utf-8-sig")
    return clean, train, validation, best, cross


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true", help="Build dataset, train BC models, and export action tables only.")
    parser.add_argument("--report-only", action="store_true", help="Generate figures/reports from existing metrics and DSSAT summaries.")
    args = parser.parse_args()

    ensure_dirs()
    config = load_yaml(CONFIG_PATH)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "configs" / CONFIG_PATH.name)

    if args.report_only:
        clean = pd.read_csv(OUTPUT_ROOT / "datasets" / "imitation_dataset_clean.csv")
        metrics = pd.read_csv(OUTPUT_ROOT / "evaluation" / "imitation_supervised_metrics.csv")
        dssat = pd.read_csv(OUTPUT_ROOT / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv")
        cross = pd.read_csv(OUTPUT_ROOT / "evaluation" / "expert_schedule_lookup_for_bc.csv")
        best = best_schedules(config)
    else:
        clean, train, validation, best, cross = load_training_context(config)
        models = train_models(config, train, clean, best)
        export_policy_action_tables(config, models, train)
        metrics = evaluate_supervised(config, models, train, validation)
        if args.train_only:
            print("[imitation] train-only complete; action tables exported.", flush=True)
            return
        dssat = run_dssat_evaluation(config, models, cross) if bool(config.get("runtime", {}).get("evaluate_dssat", True)) else pd.DataFrame()

    comparison = make_policy_comparison(config, cross, dssat)
    make_figures(config, clean, metrics, dssat, comparison)
    write_report(config, best, metrics, dssat, comparison)
    write_ppt(best, metrics, dssat, comparison)


if __name__ == "__main__":
    main()

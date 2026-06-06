from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_safe_rendering import PROJECT_ROOT, load_yaml
from ppo_strategy_selection import select_best_policies
from ppo_train import train_one_policy
from run_action_safe_site_training import gate_row


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_ppo_action_safe_two_year_cv.yaml"
SUMMARY_BASENAME = "two_year_cv_ppo_evaluation_summary.csv"
STATUS_BASENAME = "two_year_cv_training_status.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ordered_jobs(config: dict) -> list[dict]:
    rows: list[dict] = []
    seed = int(config.get("seed", 0))
    cv_type = config.get("training", {}).get("cross_validation_type", "limited_two_year_cross_validation")
    for station in config.get("training", {}).get("stations_in_order", ["FQA", "YCA"]):
        years = config["observed_years"][station]
        for train in years:
            train_year = int(train["year"])
            validations = [item for item in years if int(item["year"]) != train_year]
            rows.append(
                {
                    "station": station,
                    "policy_name": f"{station}_train{train_year}_seed{seed}_action_safe",
                    "train_year": train_year,
                    "train_year_label": train["label"],
                    "seed": seed,
                    "validation_years": ",".join(str(int(v["year"])) for v in validations),
                    "validation_year_labels": ",".join(str(v["label"]) for v in validations),
                    "num_observed_years": len(years),
                    "cross_validation_type": cv_type,
                    "action_safety_enabled": True,
                    "notes": "limited_data_two_year_cv",
                }
            )
    return rows


def write_plan(config: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    out = output_root / "configs" / "two_year_cv_training_plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered_jobs(config)).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def update_quality_gates(config: dict, policy_name: str) -> pd.DataFrame:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    summary_path = output_root / "evaluation" / "ppo_evaluation_summary.csv"
    df = pd.read_csv(summary_path)
    if "quality_gate_pass" not in df.columns:
        df["quality_gate_pass"] = False
    if "quality_gate_reason" not in df.columns:
        df["quality_gate_reason"] = ""
    df["quality_gate_pass"] = df["quality_gate_pass"].fillna(False).astype(bool)
    df["quality_gate_reason"] = df["quality_gate_reason"].fillna("").astype(str)
    mask = df["policy_name"].eq(policy_name)
    for idx, row in df[mask].iterrows():
        ok, reason = gate_row(row, config)
        df.loc[idx, "quality_gate_pass"] = bool(ok)
        df.loc[idx, "quality_gate_reason"] = reason
        df.loc[idx, "notes"] = "limited_two_year_cross_validation"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    two_year_summary = output_root / "evaluation" / SUMMARY_BASENAME
    df.to_csv(two_year_summary, index=False, encoding="utf-8-sig")
    return df[mask].copy()


def dominant_rule(series: pd.Series) -> str:
    values: list[str] = []
    for item in series.fillna("").astype(str):
        for part in item.split(";"):
            part = part.strip()
            if part:
                values.append(part)
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def write_saturation_check(config: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    summary_path = output_root / "evaluation" / SUMMARY_BASENAME
    out = output_root / "evaluation" / "action_safety_saturation_check.csv"
    if not summary_path.exists():
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        return out
    summary = pd.read_csv(summary_path)
    rows: list[dict] = []
    season_i = float(config["action_safety"]["season_irrigation_soft_limit"])
    season_n = float(config["action_safety"]["season_n_soft_limit"])
    for row in summary.to_dict("records"):
        daily_path = PROJECT_ROOT / str(row.get("daily_csv_path", ""))
        daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
        triggers = daily.get("safety_rule_triggered", pd.Series(dtype=str)).fillna("").astype(str) if not daily.empty else pd.Series(dtype=str)
        rows.append(
            {
                "station": row.get("station"),
                "policy_name": row.get("policy_name"),
                "train_year": row.get("train_year"),
                "eval_year": row.get("eval_year"),
                "total_irrigation": row.get("total_irrigation"),
                "total_n_fertilizer": row.get("total_n_fertilizer"),
                "irrigation_at_season_limit": abs(float(row.get("total_irrigation", 0.0)) - season_i) <= 1e-6,
                "n_at_season_limit": abs(float(row.get("total_n_fertilizer", 0.0)) - season_n) <= 1e-6,
                "num_irrigation_clipped_days": int(pd.to_numeric(daily.get("action_clipped_amir", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else 0,
                "num_n_clipped_days": int(pd.to_numeric(daily.get("action_clipped_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else 0,
                "num_safety_trigger_days": int(triggers.str.len().gt(0).sum()) if not daily.empty else 0,
                "dominant_safety_rule": dominant_rule(triggers),
                "notes": "at_200_300_cap" if abs(float(row.get("total_irrigation", 0.0)) - season_i) <= 1e-6 and abs(float(row.get("total_n_fertilizer", 0.0)) - season_n) <= 1e-6 else "below_cap_or_missing",
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def plot_site_figures(config: dict) -> None:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    summary_path = output_root / "evaluation" / SUMMARY_BASENAME
    ranking_path = output_root / "strategy_selection" / "policy_ranking_by_site.csv"
    saturation_path = output_root / "evaluation" / "action_safety_saturation_check.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
    saturation = pd.read_csv(saturation_path) if saturation_path.exists() else pd.DataFrame()
    for station, group in summary.groupby("station"):
        out_dir = output_root / "figures" / station / "site_level"
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = [
            ("final_grnwt", f"{station}_final_grnwt_by_policy_eval_year.png", "Final grain weight"),
            ("total_irrigation", f"{station}_total_irrigation_by_policy_eval_year.png", "Total irrigation"),
            ("total_n_fertilizer", f"{station}_total_n_by_policy_eval_year.png", "Total nitrogen fertilizer"),
            ("mean_reward", f"{station}_mean_reward_by_policy_eval_year.png", "Mean reward"),
        ]
        for metric, filename, title in specs:
            pivot = group.pivot(index="policy_name", columns="eval_year", values=metric)
            ax = pivot.plot(kind="bar", figsize=(9, 4.8))
            ax.set_title(title)
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=150)
            plt.close()
        if not ranking.empty and station in set(ranking["station"]):
            rg = ranking[ranking["station"].eq(station)].sort_values("rank")
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(rg["policy_name"], pd.to_numeric(rg["stability_score"], errors="coerce"))
            ax.set_title("Stability score by policy")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"{station}_stability_score_by_policy.png", dpi=150)
            plt.close(fig)
        if not saturation.empty and station in set(saturation["station"]):
            sg = saturation[saturation["station"].eq(station)].copy()
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            x = np.arange(len(sg))
            ax.bar(x - 0.18, pd.to_numeric(sg["total_irrigation"], errors="coerce"), width=0.36, label="total_irrigation")
            ax.bar(x + 0.18, pd.to_numeric(sg["total_n_fertilizer"], errors="coerce"), width=0.36, label="total_n_fertilizer")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{p}\neval{e}" for p, e in zip(sg["policy_name"], sg["eval_year"])], rotation=20, ha="right")
            ax.set_title("Action safety saturation")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / f"{station}_action_safety_saturation.png", dpi=150)
            plt.close(fig)


def write_all_site_summary(config: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    prev_best_path = PROJECT_ROOT / "Leave_One_experiments" / "ppo_action_safe_site_training" / "strategy_selection" / "best_policy_by_site.csv"
    two_best_path = output_root / "strategy_selection" / "best_policy_by_site.csv"
    out = PROJECT_ROOT / "Leave_One_experiments" / "ppo_action_safe_summary" / "all_site_best_policy_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if prev_best_path.exists():
        prev = pd.read_csv(prev_best_path)
        for row in prev.to_dict("records"):
            rows.append(
                {
                    "station": row["station"],
                    "best_policy": row["best_policy_name"],
                    "train_year": row["best_train_year"],
                    "validation_years": row["validation_years"],
                    "cross_validation_type": "observed_year_leave_one",
                    "mean_yield": row["mean_yield"],
                    "mean_reward": row["mean_reward"],
                    "mean_irrigation": row["mean_irrigation"],
                    "mean_n": row["mean_n_fertilizer"],
                    "stability_score": row["stability_score"],
                    "quality_gate_pass": True,
                    "limited_data_flag": False,
                    "notes": row.get("reason", ""),
                }
            )
    if two_best_path.exists():
        two = pd.read_csv(two_best_path)
        for row in two.to_dict("records"):
            rows.append(
                {
                    "station": row["station"],
                    "best_policy": row["best_policy_name"],
                    "train_year": row["best_train_year"],
                    "validation_years": row["validation_years"],
                    "cross_validation_type": "limited_two_year_cross_validation",
                    "mean_yield": row["mean_yield"],
                    "mean_reward": row["mean_reward"],
                    "mean_irrigation": row["mean_irrigation"],
                    "mean_n": row["mean_n_fertilizer"],
                    "stability_score": row["stability_score"],
                    "quality_gate_pass": True,
                    "limited_data_flag": True,
                    "notes": "limited_two_year_cross_validation; not full dry_normal_wet stability validation",
                }
            )
    pd.DataFrame(rows).sort_values("station").to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_status(config: dict, status: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    path = output_root / "reports" / STATUS_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    config = load_yaml(CONFIG_PATH)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    for subdir in ["configs", "rendered_inputs", "smoke_checks", "models", "logs", "tensorboard", "daily_outputs", "evaluation", "figures", "strategy_selection", "reports"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    plan_path = write_plan(config)
    jobs = ordered_jobs(config)
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 5000))
    status = {
        "started_at": now_text(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "plan_path": str(plan_path.relative_to(PROJECT_ROOT)),
        "total_timesteps_per_model": total_timesteps,
        "cross_validation_type": "limited_two_year_cross_validation",
        "jobs_total": len(jobs),
        "jobs_completed": 0,
        "stopped": False,
        "stop_reason": "",
        "last_policy": "",
    }
    write_status(config, status)
    completed_jobs: list[dict] = []
    print(f"[{now_text()}] plan written: {plan_path.relative_to(PROJECT_ROOT)}", flush=True)
    try:
        for job in jobs:
            policy_name = job["policy_name"]
            station = job["station"]
            train_year = int(job["train_year"])
            seed = int(job["seed"])
            status["last_policy"] = policy_name
            write_status(config, status)
            print(f"[{now_text()}] START {policy_name}: train_year={train_year}, validation={job['validation_years']}", flush=True)
            result = train_one_policy(
                station=station,
                train_year=train_year,
                seed=seed,
                total_timesteps=total_timesteps,
                config_path=CONFIG_PATH,
                debug=False,
            )
            gated = update_quality_gates(config, policy_name)
            write_saturation_check(config)
            failed = gated[gated["quality_gate_pass"].astype(str).str.lower().ne("true")]
            completed_jobs.append({**job, "model_path": str(result["model_path"].relative_to(PROJECT_ROOT)), "quality_gate_pass": failed.empty})
            status["jobs_completed"] = len(completed_jobs)
            write_status(config, status)
            if failed.empty:
                print(f"[{now_text()}] PASS {policy_name}: {len(gated)} evaluations passed quality gate", flush=True)
            else:
                reason = " | ".join(f"eval{int(r.eval_year)}: {r.quality_gate_reason}" for r in failed.itertuples())
                status["stopped"] = True
                status["stop_reason"] = f"{policy_name} failed quality gate: {reason}"
                write_status(config, status)
                print(f"[{now_text()}] STOP {status['stop_reason']}", flush=True)
                break
    except Exception as exc:
        status["stopped"] = True
        status["stop_reason"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        write_status(config, status)
        print(status["traceback"], file=sys.stderr, flush=True)
        return 1
    pd.DataFrame(completed_jobs).to_csv(output_root / "reports" / "completed_policy_jobs.csv", index=False, encoding="utf-8-sig")
    best_path = select_best_policies(CONFIG_PATH)
    saturation_path = write_saturation_check(config)
    plot_site_figures(config)
    all_site_path = write_all_site_summary(config)
    status["finished_at"] = now_text()
    status["best_policy_by_site_path"] = str(best_path.relative_to(PROJECT_ROOT))
    status["action_safety_saturation_check_path"] = str(saturation_path.relative_to(PROJECT_ROOT))
    status["all_site_best_policy_summary_path"] = str(all_site_path.relative_to(PROJECT_ROOT))
    write_status(config, status)
    print(f"[{now_text()}] strategy selection: {best_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[{now_text()}] all-site summary: {all_site_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0 if not status.get("stopped") else 2


if __name__ == "__main__":
    raise SystemExit(main())

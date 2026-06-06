from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_experiment_plan import cross_validation_type, validation_years_for
from ppo_safe_rendering import PROJECT_ROOT, load_yaml
from ppo_strategy_selection import select_best_policies
from ppo_train import train_one_policy


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_ppo_action_safe_site_training.yaml"
SUMMARY_BASENAME = "action_safe_site_ppo_evaluation_summary.csv"
STATUS_BASENAME = "action_safe_site_training_status.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ordered_jobs(config: dict) -> list[dict]:
    rows: list[dict] = []
    seed = int(config.get("seed", 0))
    for station in config.get("training", {}).get("stations_in_order", ["HLA", "SYA", "LCA"]):
        years = config["observed_years"][station]
        for item in years:
            train_year = int(item["year"])
            validations = validation_years_for(config, station, train_year)
            rows.append(
                {
                    "station": station,
                    "policy_name": f"{station}_train{train_year}_seed{seed}_action_safe",
                    "train_year": train_year,
                    "train_year_label": item["label"],
                    "seed": seed,
                    "validation_years": ",".join(str(int(v["year"])) for v in validations),
                    "validation_year_labels": ",".join(str(v["label"]) for v in validations),
                    "num_observed_years": len(years),
                    "cross_validation_type": cross_validation_type(len(years)),
                    "action_safety_enabled": True,
                }
            )
    return rows


def write_plan(config: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    out = output_root / "configs" / "action_safe_site_training_plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered_jobs(config)).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def gate_row(row: pd.Series, config: dict) -> tuple[bool, str]:
    gates = config.get("quality_gates", {})
    reasons: list[str] = []
    if gates.get("require_run_status_ok", True) and str(row.get("run_status", "")) != "ok":
        reasons.append(f"run_status={row.get('run_status')}")
    if gates.get("require_episode_completed", True) and not bool(row.get("episode_completed", False)):
        reasons.append("episode_not_completed")
    max_irrig = float(gates.get("max_total_irrigation", np.inf))
    max_n = float(gates.get("max_total_n_fertilizer", np.inf))
    total_irrig = float(pd.to_numeric(pd.Series([row.get("total_irrigation")]), errors="coerce").iloc[0])
    total_n = float(pd.to_numeric(pd.Series([row.get("total_n_fertilizer")]), errors="coerce").iloc[0])
    if total_irrig > max_irrig:
        reasons.append(f"total_irrigation>{max_irrig}: {total_irrig:.3f}")
    if total_n > max_n:
        reasons.append(f"total_n_fertilizer>{max_n}: {total_n:.3f}")
    if gates.get("require_daily_csv", True):
        daily = PROJECT_ROOT / str(row.get("daily_csv_path", ""))
        if not daily.exists():
            reasons.append(f"missing_daily_csv: {daily}")
    fig_dir = PROJECT_ROOT / str(row.get("figure_dir", ""))
    for figure_name in gates.get("required_figures", []):
        if not (fig_dir / figure_name).exists():
            reasons.append(f"missing_figure: {figure_name}")
    return (len(reasons) == 0, "; ".join(reasons))


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
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    site_summary = output_root / "evaluation" / SUMMARY_BASENAME
    df.to_csv(site_summary, index=False, encoding="utf-8-sig")
    return df[mask].copy()


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
    total_timesteps = int(config.get("training", {}).get("total_timesteps", config.get("ppo", {}).get("total_timesteps_full", 5000)))
    status = {
        "started_at": now_text(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "plan_path": str(plan_path.relative_to(PROJECT_ROOT)),
        "total_timesteps_per_model": total_timesteps,
        "jobs_total": len(jobs),
        "jobs_completed": 0,
        "stopped": False,
        "stop_reason": "",
        "last_policy": "",
    }
    write_status(config, status)
    print(f"[{now_text()}] plan written: {plan_path.relative_to(PROJECT_ROOT)}", flush=True)
    completed_jobs: list[dict] = []
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
    status["finished_at"] = now_text()
    status["best_policy_by_site_path"] = str(best_path.relative_to(PROJECT_ROOT))
    write_status(config, status)
    print(f"[{now_text()}] strategy selection: {best_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0 if not status.get("stopped") else 2


if __name__ == "__main__":
    raise SystemExit(main())

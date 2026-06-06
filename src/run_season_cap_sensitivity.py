from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_experiment_plan import find_year, validation_years_for
from ppo_safe_rendering import PROJECT_ROOT, load_yaml
from ppo_train import train_one_policy


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_season_cap_sensitivity.yaml"
SUMMARY_BASENAME = "season_cap_sensitivity_evaluation_summary.csv"
STATUS_BASENAME = "season_cap_sensitivity_status.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def output_root(config: dict) -> Path:
    return PROJECT_ROOT / config["paths"]["output_root"]


def cap_sort_key(cap_name: str, config: dict) -> int:
    order = [item["cap_name"] for item in config["cap_levels"]]
    return order.index(cap_name)


def build_jobs(config: dict) -> list[dict]:
    rows: list[dict] = []
    run_order = 1
    seed = int(config.get("seed", 0))
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 5000))
    for station in config["training"]["stations_in_order"]:
        train_year = int(config["training"]["best_train_years"][station])
        train_info = find_year(config, station, train_year)
        eval_years = [train_year] + [int(item["year"]) for item in validation_years_for(config, station, train_year)]
        for cap in config["cap_levels"]:
            rows.append(
                {
                    "station": station,
                    "train_year": train_year,
                    "train_year_label": train_info["label"],
                    "eval_years": ",".join(str(year) for year in eval_years),
                    "cv_type": config["training"]["cv_types"][station],
                    "cap_name": cap["cap_name"],
                    "season_irrigation_cap": float(cap["season_irrigation_cap"]),
                    "season_n_cap": float(cap["season_n_cap"]),
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "run_order": run_order,
                    "notes": "HLA_pilot" if station == "HLA" else "season_cap_sensitivity",
                }
            )
            run_order += 1
    return rows


def write_plan(config: dict) -> Path:
    out = output_root(config) / "configs" / "season_cap_sensitivity_plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build_jobs(config)).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_job_config(config: dict, job: dict) -> Path:
    job_config = deepcopy(config)
    job_config["action_safety"]["season_irrigation_soft_limit"] = float(job["season_irrigation_cap"])
    job_config["action_safety"]["season_n_soft_limit"] = float(job["season_n_cap"])
    job_config["training"]["active_cap_name"] = job["cap_name"]
    path = output_root(config) / "configs" / "job_configs" / f"{job['station']}_train{job['train_year']}_{job['cap_name']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(job_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def gate_row(row: pd.Series, config: dict) -> tuple[bool, str]:
    reasons: list[str] = []
    gates = config.get("quality_gates", {})
    tol = float(gates.get("cap_tolerance", 1e-6))
    if gates.get("require_run_status_ok", True) and str(row.get("run_status", "")) != "ok":
        reasons.append(f"run_status={row.get('run_status')}")
    if gates.get("require_episode_completed", True) and not bool(row.get("episode_completed", False)):
        reasons.append("episode_not_completed")
    total_i = float(pd.to_numeric(pd.Series([row.get("total_irrigation")]), errors="coerce").iloc[0])
    total_n = float(pd.to_numeric(pd.Series([row.get("total_n_fertilizer")]), errors="coerce").iloc[0])
    cap_i = float(row.get("season_irrigation_cap"))
    cap_n = float(row.get("season_n_cap"))
    if total_i > cap_i + tol:
        reasons.append(f"total_irrigation>{cap_i}: {total_i:.6f}")
    if total_n > cap_n + tol:
        reasons.append(f"total_n_fertilizer>{cap_n}: {total_n:.6f}")
    if gates.get("require_daily_csv", True):
        daily = PROJECT_ROOT / str(row.get("daily_csv_path", ""))
        if not daily.exists():
            reasons.append(f"missing_daily_csv: {daily}")
    fig_dir = PROJECT_ROOT / str(row.get("figure_dir", ""))
    for figure_name in gates.get("required_figures", []):
        if not (fig_dir / figure_name).exists():
            reasons.append(f"missing_figure: {figure_name}")
    return len(reasons) == 0, "; ".join(reasons)


def update_quality_gates(config: dict, policy_name: str) -> pd.DataFrame:
    summary_path = output_root(config) / "evaluation" / "ppo_evaluation_summary.csv"
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
    out = output_root(config) / "evaluation" / SUMMARY_BASENAME
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df[mask].copy()


def dominant_rule(series: pd.Series) -> str:
    values: list[str] = []
    for item in series.fillna("").astype(str):
        for part in item.split(";"):
            part = part.strip()
            if part:
                values.append(part)
    return Counter(values).most_common(1)[0][0] if values else ""


def write_cap_saturation_summary(config: dict) -> Path:
    summary_path = output_root(config) / "evaluation" / SUMMARY_BASENAME
    out = output_root(config) / "evaluation" / "cap_saturation_summary.csv"
    if not summary_path.exists():
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        return out
    summary = pd.read_csv(summary_path)
    rows: list[dict] = []
    for row in summary.to_dict("records"):
        daily_path = PROJECT_ROOT / str(row.get("daily_csv_path", ""))
        daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
        triggers = daily.get("safety_rule_triggered", pd.Series(dtype=str)).fillna("").astype(str) if not daily.empty else pd.Series(dtype=str)
        cap_i = float(row["season_irrigation_cap"])
        cap_n = float(row["season_n_cap"])
        total_i = float(row["total_irrigation"])
        total_n = float(row["total_n_fertilizer"])
        i_ratio = total_i / cap_i if cap_i else np.nan
        n_ratio = total_n / cap_n if cap_n else np.nan
        rows.append(
            {
                "station": row["station"],
                "cap_name": row["cap_name"],
                "season_irrigation_cap": cap_i,
                "season_n_cap": cap_n,
                "train_year": row["train_year"],
                "eval_year": row["eval_year"],
                "total_irrigation": total_i,
                "total_n_fertilizer": total_n,
                "irrigation_saturation_ratio": i_ratio,
                "n_saturation_ratio": n_ratio,
                "irrigation_at_cap": bool(i_ratio >= 0.98),
                "n_at_cap": bool(n_ratio >= 0.98),
                "num_irrigation_clipped_days": int(pd.to_numeric(daily.get("action_clipped_amir", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else 0,
                "num_n_clipped_days": int(pd.to_numeric(daily.get("action_clipped_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else 0,
                "num_safety_trigger_days": int(triggers.str.len().gt(0).sum()) if not daily.empty else 0,
                "dominant_safety_rule": dominant_rule(triggers),
                "notes": "at_or_near_cap" if i_ratio >= 0.98 and n_ratio >= 0.98 else "below_cap",
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_marginal_response(config: dict) -> Path:
    summary_path = output_root(config) / "evaluation" / SUMMARY_BASENAME
    out = output_root(config) / "evaluation" / "marginal_response_by_cap.csv"
    if not summary_path.exists():
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        return out
    df = pd.read_csv(summary_path)
    agg = (
        df.groupby(["station", "cap_name", "season_irrigation_cap", "season_n_cap", "train_year", "cv_type"], as_index=False)
        .agg(
            mean_yield=("final_grnwt", "mean"),
            mean_reward=("mean_reward", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n=("total_n_fertilizer", "mean"),
            mean_swfac=("mean_swfac", "mean"),
            mean_nstres=("mean_nstres", "mean"),
            quality_gate_pass=("quality_gate_pass", lambda s: bool(s.astype(str).str.lower().eq("true").all())),
        )
    )
    rows: list[dict] = []
    for station, group in agg.groupby("station"):
        group = group.sort_values("season_irrigation_cap")
        previous = None
        for row in group.to_dict("records"):
            if previous is None:
                delta_yield = delta_reward = delta_i = delta_n = np.nan
            else:
                delta_yield = row["mean_yield"] - previous["mean_yield"]
                delta_reward = row["mean_reward"] - previous["mean_reward"]
                delta_i = row["mean_irrigation"] - previous["mean_irrigation"]
                delta_n = row["mean_n"] - previous["mean_n"]
            rows.append(
                {
                    **row,
                    "delta_yield_from_previous_cap": delta_yield,
                    "delta_reward_from_previous_cap": delta_reward,
                    "delta_irrigation_from_previous_cap": delta_i,
                    "delta_n_from_previous_cap": delta_n,
                    "yield_gain_per_100mm_irrigation": (delta_yield / delta_i * 100.0) if previous is not None and abs(delta_i) > 1e-9 else np.nan,
                    "yield_gain_per_100kg_n": (delta_yield / delta_n * 100.0) if previous is not None and abs(delta_n) > 1e-9 else np.nan,
                    "reward_gain_per_cap_step": delta_reward,
                    "limited_data_flag": row["cv_type"] == "limited_two_year_cross_validation",
                }
            )
            previous = row
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def plot_metric(ax, group: pd.DataFrame, metric: str, title: str) -> None:
    for station, sub in group.groupby("station"):
        sub = sub.sort_values("season_irrigation_cap")
        ax.plot(sub["season_irrigation_cap"], pd.to_numeric(sub[metric], errors="coerce"), marker="o", label=station)
    ax.set_title(title)
    ax.set_xlabel("Season irrigation cap (mm)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def write_figures(config: dict) -> None:
    marginal_path = output_root(config) / "evaluation" / "marginal_response_by_cap.csv"
    saturation_path = output_root(config) / "evaluation" / "cap_saturation_summary.csv"
    if not marginal_path.exists():
        return
    marginal = pd.read_csv(marginal_path)
    saturation = pd.read_csv(saturation_path) if saturation_path.exists() else pd.DataFrame()
    for station, group in marginal.groupby("station"):
        fig_dir = output_root(config) / "figures" / station
        fig_dir.mkdir(parents=True, exist_ok=True)
        for metric, filename, title in [
            ("mean_yield", f"{station}_yield_vs_cap.png", "Yield vs cap"),
            ("mean_reward", f"{station}_reward_vs_cap.png", "Reward vs cap"),
            ("yield_gain_per_100mm_irrigation", f"{station}_marginal_yield_gain_vs_cap.png", "Marginal yield gain vs cap"),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            sub = group.sort_values("season_irrigation_cap")
            ax.plot(sub["season_irrigation_cap"], pd.to_numeric(sub[metric], errors="coerce"), marker="o")
            ax.set_title(title)
            ax.set_xlabel("Season irrigation cap (mm)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_dir / filename, dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        sub = group.sort_values("season_irrigation_cap")
        ax.plot(sub["season_irrigation_cap"], sub["mean_irrigation"], marker="o", label="mean_irrigation")
        ax.plot(sub["season_irrigation_cap"], sub["mean_n"], marker="o", label="mean_n")
        ax.set_title("Irrigation and nitrogen vs cap")
        ax.set_xlabel("Season irrigation cap (mm)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{station}_irrigation_n_vs_cap.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(sub["season_irrigation_cap"], sub["mean_swfac"], marker="o", label="mean_swfac")
        ax.plot(sub["season_irrigation_cap"], sub["mean_nstres"], marker="o", label="mean_nstres")
        ax.set_title("Stress indicators vs cap")
        ax.set_xlabel("Season irrigation cap (mm)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{station}_swfac_nstres_vs_cap.png", dpi=150)
        plt.close(fig)
        if not saturation.empty:
            sat = saturation[saturation["station"].eq(station)].groupby(["cap_name", "season_irrigation_cap"], as_index=False).agg(
                irrigation_saturation_ratio=("irrigation_saturation_ratio", "mean"),
                n_saturation_ratio=("n_saturation_ratio", "mean"),
            ).sort_values("season_irrigation_cap")
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.plot(sat["season_irrigation_cap"], sat["irrigation_saturation_ratio"], marker="o", label="irrigation_ratio")
            ax.plot(sat["season_irrigation_cap"], sat["n_saturation_ratio"], marker="o", label="n_ratio")
            ax.axhline(0.98, color="gray", linestyle="--", linewidth=1)
            ax.set_title("Saturation ratio vs cap")
            ax.set_xlabel("Season irrigation cap (mm)")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{station}_saturation_ratio_vs_cap.png", dpi=150)
            plt.close(fig)
    all_dir = output_root(config) / "figures" / "all_sites"
    all_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, title in [
        ("mean_yield", "all_sites_yield_vs_cap.png", "All sites yield vs cap"),
        ("mean_reward", "all_sites_reward_vs_cap.png", "All sites reward vs cap"),
        ("yield_gain_per_100mm_irrigation", "all_sites_marginal_yield_gain_vs_cap.png", "All sites marginal yield gain vs cap"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_metric(ax, marginal, metric, title)
        fig.tight_layout()
        fig.savefig(all_dir / filename, dpi=150)
        plt.close(fig)
    if not saturation.empty:
        sat_agg = saturation.groupby(["station", "cap_name", "season_irrigation_cap"], as_index=False).agg(
            irrigation_saturation_ratio=("irrigation_saturation_ratio", "mean")
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_metric(ax, sat_agg, "irrigation_saturation_ratio", "All sites saturation ratio vs cap")
        ax.axhline(0.98, color="gray", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(all_dir / "all_sites_saturation_ratio_vs_cap.png", dpi=150)
        plt.close(fig)


def write_status(config: dict, status: dict) -> Path:
    path = output_root(config) / "reports" / STATUS_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    config = load_yaml(CONFIG_PATH)
    root = output_root(config)
    for subdir in ["configs", "rendered_inputs", "smoke_checks", "models", "logs", "tensorboard", "daily_outputs", "evaluation", "figures", "strategy_selection", "reports"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    plan_path = write_plan(config)
    jobs = build_jobs(config)
    status = {
        "started_at": now_text(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "plan_path": str(plan_path.relative_to(PROJECT_ROOT)),
        "jobs_total": len(jobs),
        "jobs_completed": 0,
        "stopped": False,
        "stop_reason": "",
        "last_policy": "",
    }
    write_status(config, status)
    print(f"[{now_text()}] plan written: {plan_path.relative_to(PROJECT_ROOT)}", flush=True)
    completed: list[dict] = []
    try:
        for job in jobs:
            policy_tag = f"_{job['cap_name']}"
            policy_name = f"{job['station']}_train{job['train_year']}_{job['cap_name']}_seed{job['seed']}_action_safe"
            status["last_policy"] = policy_name
            write_status(config, status)
            print(f"[{now_text()}] START {policy_name}: eval_years={job['eval_years']}", flush=True)
            job_config_path = write_job_config(config, job)
            row_metadata = {
                "cap_name": job["cap_name"],
                "season_irrigation_cap": float(job["season_irrigation_cap"]),
                "season_n_cap": float(job["season_n_cap"]),
                "cv_type": job["cv_type"],
            }
            result = train_one_policy(
                station=job["station"],
                train_year=int(job["train_year"]),
                seed=int(job["seed"]),
                total_timesteps=int(job["total_timesteps"]),
                config_path=job_config_path,
                debug=False,
                policy_tag=policy_tag,
                row_metadata=row_metadata,
            )
            gated = update_quality_gates(config, policy_name)
            write_cap_saturation_summary(config)
            write_marginal_response(config)
            failed = gated[gated["quality_gate_pass"].astype(str).str.lower().ne("true")]
            completed.append({**job, "model_path": str(result["model_path"].relative_to(PROJECT_ROOT)), "quality_gate_pass": failed.empty})
            status["jobs_completed"] = len(completed)
            write_status(config, status)
            if failed.empty:
                print(f"[{now_text()}] PASS {policy_name}: {len(gated)} evaluations passed cap gate", flush=True)
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
    pd.DataFrame(completed).to_csv(root / "reports" / "completed_cap_jobs.csv", index=False, encoding="utf-8-sig")
    saturation = write_cap_saturation_summary(config)
    marginal = write_marginal_response(config)
    write_figures(config)
    status["finished_at"] = now_text()
    status["cap_saturation_summary_path"] = str(saturation.relative_to(PROJECT_ROOT))
    status["marginal_response_by_cap_path"] = str(marginal.relative_to(PROJECT_ROOT))
    write_status(config, status)
    print(f"[{now_text()}] saturation summary: {saturation.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[{now_text()}] marginal response: {marginal.relative_to(PROJECT_ROOT)}", flush=True)
    return 0 if not status.get("stopped") else 2


if __name__ == "__main__":
    raise SystemExit(main())

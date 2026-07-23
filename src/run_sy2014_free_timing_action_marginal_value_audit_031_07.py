from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_action_safety import ActionSafetyState
from run_free_timing_reward_v2_ppo_dqn_031_06 import timing_reward


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_06_free_timing_reward_v2_ppo.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
OUT = ROOT / "benchmark_results" / "031_07_sy2014_free_timing_action_marginal_value_audit"
DOC = ROOT / "docs" / "031_07_sy2014_free_timing_action_marginal_value_audit_record.md"


BASE_SCHEDULE: dict[int, tuple[float, float]] = {
    1: (30.0, 50.0),
    30: (30.0, 50.0),
    50: (30.0, 50.0),
    65: (30.0, 50.0),
    85: (20.0, 50.0),
    110: (20.0, 0.0),
}
TARGET_DAPS = [1, 30, 50, 65, 85, 100]
ACTION_GRID = [(i, n) for i in [0.0, 20.0, 40.0] for n in [0.0, 40.0, 80.0]]


def candidates_for_target(target_dap: int) -> list[tuple[float, float]]:
    candidates = list(ACTION_GRID)
    background = BASE_SCHEDULE.get(int(target_dap), (0.0, 0.0))
    if background not in candidates:
        candidates.append(background)
    return candidates


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "evaluation", "figures"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_07_free_timing_action_marginal_value_SY2014"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def normalized_action(config: dict, amir: float, anfer: float) -> np.ndarray:
    i_max = float(config["action_scale"]["daily_irrigation_max"])
    n_max = float(config["action_scale"]["daily_n_max"])
    return np.asarray(
        [
            np.clip(2.0 * (float(amir) / i_max) - 1.0, -1.0, 1.0),
            np.clip(2.0 * (float(anfer) / n_max) - 1.0, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def branch_action(target_dap: int, candidate: tuple[float, float], dap: int) -> tuple[float, float]:
    if int(dap) == int(target_dap):
        return candidate
    return BASE_SCHEDULE.get(int(dap), (0.0, 0.0))


def run_branch(
    *,
    config: dict,
    env_config: dict,
    weather: pd.DataFrame,
    target_dap: int,
    candidate_i: float,
    candidate_n: float,
    seed: int = 0,
) -> dict[str, Any]:
    station = "SYA"
    year = 2014
    label = f"dap{target_dap:03d}_I{candidate_i:.0f}_N{candidate_n:.0f}"
    env = direct_ppo.make_training_env(config, env_config, station, year, seed, f"{station}_{year}_{label}")
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = direct_ppo.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        latest_prev = direct_ppo.latest_observation_dict(env, obs, info)

        while not done and step_count < int(config["runtime"]["max_steps"]):
            dap_raw = scalar(latest_prev.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            requested_i, requested_n = branch_action(target_dap, (candidate_i, candidate_n), dap)
            safety_before = ActionSafetyState(**getattr(env, "safety_state").__dict__)
            action = normalized_action(config, requested_i, requested_n)
            obs, env_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest_after = direct_ppo.latest_observation_dict(env, obs, info)
            action_info = dict(getattr(env, "last_action_info", {}))
            safe_i = scalar(action_info.get("safe_action_amir"), 0.0)
            safe_n = scalar(action_info.get("safe_action_anfer"), 0.0)
            reward_v2, reward_parts = timing_reward(config, latest_prev, latest_after, safety_before, safe_i, safe_n, dap)

            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "target_dap": target_dap,
                    "candidate_i": candidate_i,
                    "candidate_n": candidate_n,
                    "branch_label": label,
                    "station_code": station,
                    "year": year,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "xlai": scalar(latest_after.get("xlai")),
                    "env_reward": float(env_reward),
                    "requested_i": requested_i,
                    "requested_n": requested_n,
                    **action_info,
                    **reward_parts,
                    "reward_v2_recomputed": reward_v2,
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            latest_prev = latest_after
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_path = OUT / "daily_outputs" / station / f"2014_{label}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nfer = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce")
    topwt = pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce")
    reward_v2 = pd.to_numeric(daily.get("reward_v2", pd.Series(dtype=float)), errors="coerce")
    timing_penalty = pd.to_numeric(daily.get("timing_shaping_penalty", pd.Series(dtype=float)), errors="coerce")

    target_rows = daily[daily["dap"].astype(int).eq(int(target_dap))]
    requested_i_at_target = float(target_rows["requested_i"].iloc[0]) if len(target_rows) else np.nan
    requested_n_at_target = float(target_rows["requested_n"].iloc[0]) if len(target_rows) else np.nan
    safe_i_at_target = float(target_rows["safe_action_amir"].iloc[0]) if len(target_rows) else np.nan
    safe_n_at_target = float(target_rows["safe_action_anfer"].iloc[0]) if len(target_rows) else np.nan
    safety_triggered_at_target = str(target_rows["action_safety_triggered"].iloc[0]) if len(target_rows) else ""

    final_grnwt = float(grnwt.iloc[-1]) if len(grnwt) else np.nan
    final_topwt = float(topwt.iloc[-1]) if len(topwt) else np.nan
    total_i = float(irr.sum())
    total_n = float(nfer.sum())
    simple_profit = final_grnwt - total_i - 5.0 * total_n if np.isfinite(final_grnwt) else np.nan
    return {
        "branch_label": label,
        "target_dap": int(target_dap),
        "candidate_i": float(candidate_i),
        "candidate_n": float(candidate_n),
        "background_i": float(BASE_SCHEDULE.get(int(target_dap), (0.0, 0.0))[0]),
        "background_n": float(BASE_SCHEDULE.get(int(target_dap), (0.0, 0.0))[1]),
        "requested_i_at_target": requested_i_at_target,
        "requested_n_at_target": requested_n_at_target,
        "safe_i_at_target": safe_i_at_target,
        "safe_n_at_target": safe_n_at_target,
        "safety_triggered_at_target": safety_triggered_at_target,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "final_topwt": final_topwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "simple_profit": simple_profit,
        "reward_v2_sum": float(reward_v2.sum()) if len(reward_v2) else np.nan,
        "timing_penalty_sum": float(timing_penalty.sum()) if len(timing_penalty) else np.nan,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((nfer > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[nfer > 0, "dap"].iloc[0]) if (nfer > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def add_delta_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["background_i"] = out["target_dap"].astype(int).map(lambda d: float(BASE_SCHEDULE.get(d, (0.0, 0.0))[0]))
    out["background_n"] = out["target_dap"].astype(int).map(lambda d: float(BASE_SCHEDULE.get(d, (0.0, 0.0))[1]))
    no_op = out[(out["candidate_i"].eq(0.0)) & (out["candidate_n"].eq(0.0))][
        ["target_dap", "final_grnwt", "simple_profit", "reward_v2_sum", "total_irrigation", "total_n"]
    ].rename(
        columns={
            "final_grnwt": "noop_final_grnwt",
            "simple_profit": "noop_simple_profit",
            "reward_v2_sum": "noop_reward_v2_sum",
            "total_irrigation": "noop_total_irrigation",
            "total_n": "noop_total_n",
        }
    )
    out = out.merge(no_op, on="target_dap", how="left")
    for metric in ["final_grnwt", "simple_profit", "reward_v2_sum", "total_irrigation", "total_n"]:
        out[f"delta_vs_noop_{metric}"] = out[metric] - out[f"noop_{metric}"]

    background_rows = []
    for dap in TARGET_DAPS:
        bi, bn = BASE_SCHEDULE.get(int(dap), (0.0, 0.0))
        background_rows.append({"target_dap": int(dap), "candidate_i": float(bi), "candidate_n": float(bn)})
    bg = pd.DataFrame(background_rows)
    if not bg.empty:
        bg_metrics = out.merge(bg, on=["target_dap", "candidate_i", "candidate_n"], how="inner")[
            ["target_dap", "final_grnwt", "simple_profit", "reward_v2_sum"]
        ].rename(
            columns={
                "final_grnwt": "background_final_grnwt",
                "simple_profit": "background_simple_profit",
                "reward_v2_sum": "background_reward_v2_sum",
            }
        )
        out = out.merge(bg_metrics, on="target_dap", how="left")
        for metric in ["final_grnwt", "simple_profit", "reward_v2_sum"]:
            out[f"delta_vs_background_{metric}"] = out[metric] - out[f"background_{metric}"]
    return out


def make_heatmap(summary: pd.DataFrame) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    pivot = summary.pivot_table(
        index="target_dap",
        columns=["candidate_i", "candidate_n"],
        values="delta_vs_noop_simple_profit",
        aggfunc="mean",
    ).sort_index()
    labels = [f"I{int(i)}N{int(n)}" for i, n in pivot.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("candidate action at target DAP")
    ax.set_ylabel("target DAP")
    ax.set_title("031_07 SY2014 marginal simple-profit delta vs target-DAP no-op")
    fig.colorbar(im, ax=ax, label="delta simple profit (kg/ha equivalent)")
    fig.tight_layout()
    path = OUT / "figures" / "031_07_sy2014_action_marginal_profit_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_record(summary: pd.DataFrame, heatmap_path: str) -> None:
    best_by_dap = summary.sort_values(["target_dap", "simple_profit"], ascending=[True, False]).groupby("target_dap").head(1)
    worst_by_dap = summary.sort_values(["target_dap", "simple_profit"], ascending=[True, True]).groupby("target_dap").head(1)
    lines = [
        "# 031_07 SY2014 free-timing action marginal value audit record",
        "",
        "## Scope",
        "",
        "- SYA2014 only.",
        "- No RL training.",
        "- Deterministic DSSAT branches under a fixed `uniform_spread` / `expert_window_budget` background schedule.",
        "- At each target DAP, only that day's action is replaced by one 3x3 candidate action.",
        "",
        "## Background schedule",
        "",
        "| DAP | irrigation | nitrogen |",
        "|---:|---:|---:|",
    ]
    for dap, (i, n) in BASE_SCHEDULE.items():
        lines.append(f"| {dap} | {i:.1f} | {n:.1f} |")
    lines.extend(
        [
            "",
            "## Best action by target DAP, ranked by simple profit",
            "",
            markdown_table(best_by_dap[
                [
                    "target_dap",
                    "candidate_i",
                    "candidate_n",
                    "safe_i_at_target",
                    "safe_n_at_target",
                    "final_grnwt",
                    "total_irrigation",
                    "total_n",
                    "simple_profit",
                    "delta_vs_noop_simple_profit",
                    "delta_vs_background_simple_profit",
                    "swfac_stress_days_gt_0p05",
                    "nstres_days_gt_0p05",
                ]
            ]),
            "",
            "## Worst action by target DAP, ranked by simple profit",
            "",
            markdown_table(worst_by_dap[
                [
                    "target_dap",
                    "candidate_i",
                    "candidate_n",
                    "safe_i_at_target",
                    "safe_n_at_target",
                    "final_grnwt",
                    "total_irrigation",
                    "total_n",
                    "simple_profit",
                    "delta_vs_noop_simple_profit",
                    "delta_vs_background_simple_profit",
                    "swfac_stress_days_gt_0p05",
                    "nstres_days_gt_0p05",
                ]
            ]),
            "",
            "## Full summary CSV",
            "",
            "`benchmark_results/031_07_sy2014_free_timing_action_marginal_value_audit/evaluation/031_07_action_marginal_value_summary.csv`",
            "",
            "## Figure",
            "",
            f"`{heatmap_path}`" if heatmap_path else "Heatmap skipped because matplotlib was unavailable.",
            "",
            "## Interpretation boundary",
            "",
            "This is not an RL result. It is a counterfactual marginal-value map for deciding how to redesign free-timing RL reward and diagnostics.",
        ]
    )
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / "031_07_sy2014_free_timing_action_marginal_value_audit_record.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = direct_ppo.load_yaml(CONFIG)
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT))
    direct_ppo.OUTPUT_ROOT = OUT
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_07_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_07_resolved_env_config.yaml")
    weather = direct_ppo.weather_for_daily(config)

    summary_path = OUT / "evaluation" / "031_07_action_marginal_value_summary.csv"
    rows: list[dict[str, Any]] = []
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not existing.empty:
        rows.extend(existing.drop(columns=[c for c in existing.columns if c.startswith("delta_vs_") or c.startswith("noop_") or c in {"background_i", "background_n", "background_final_grnwt", "background_simple_profit", "background_reward_v2_sum"}], errors="ignore").to_dict(orient="records"))
    existing_keys = set()
    if not existing.empty:
        for row in existing.itertuples(index=False):
            existing_keys.add((int(getattr(row, "target_dap")), float(getattr(row, "candidate_i")), float(getattr(row, "candidate_n"))))

    for target_dap in TARGET_DAPS:
        for candidate_i, candidate_n in candidates_for_target(target_dap):
            key = (int(target_dap), float(candidate_i), float(candidate_n))
            if key in existing_keys:
                continue
            rows.append(
                run_branch(
                    config=config,
                    env_config=env_config,
                    weather=weather,
                    target_dap=target_dap,
                    candidate_i=candidate_i,
                    candidate_n=candidate_n,
                    seed=0,
                )
            )

    summary = add_delta_columns(pd.DataFrame(rows))
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    heatmap_path = make_heatmap(summary)
    write_record(summary, heatmap_path)

    result = {
        "task": "031_07_sy2014_free_timing_action_marginal_value_audit",
        "training_run": False,
        "dssat_branch_runs": int(len(summary)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
        "heatmap": heatmap_path,
        "target_daps": TARGET_DAPS,
        "action_grid": [{"i": i, "n": n} for i, n in ACTION_GRID],
        "background_actions_added_when_missing": True,
        "interpretation": "DSSAT-only counterfactual marginal value audit; not an RL success claim.",
    }
    (OUT / "031_07_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(summary.sort_values(["target_dap", "simple_profit"], ascending=[True, False]).groupby("target_dap").head(3).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19 as dqn19
import run_literature_aligned_ppo_dqn_ddqn_sy2014_smoke_031_22 as ddqn22
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_24_ddqn_training_length_50k_100k_sy_crossyear.yaml"
OUT = ROOT / "benchmark_results" / "031_24_ddqn_training_length_50k_100k_sy_crossyear"
DOC = ROOT / "docs" / "031_24_ddqn_training_length_50k_100k_sy_crossyear_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "daily_outputs/SYA", "runs", "logs", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_selection(meta: dict) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    years = [int(meta["train_year"])] + [int(y) for y in meta["transfer_years"]]
    rows = pool[(pool["station_code"].eq(meta["train_site"])) & (pool["year"].astype(int).isin(years))].copy()
    if sorted(rows["year"].astype(int).tolist()) != sorted(years):
        raise RuntimeError(f"Expected years {years}, got {rows['year'].tolist()}")
    rows["selected_for_train"] = rows["year"].astype(int).eq(int(meta["train_year"]))
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "031_24_ddqn_training_length_sensitivity"
    return rows


def clean_scenario(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    if text.strip() == "" or text.lower() == "nan":
        return "null"
    return text


def load_baselines(meta: dict) -> pd.DataFrame:
    path = ROOT / meta["four_baseline_summary_csv"]
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].map(clean_scenario)
    years = [int(y) for y in meta["transfer_years"]]
    scenarios = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]
    rows = df[(df["site"].eq("SY")) & (df["year"].astype(int).isin(years)) & (df["scenario"].isin(scenarios))].copy()
    rows = rows.drop_duplicates(["site", "year", "scenario"], keep="first")
    expected = len(years) * len(scenarios)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} baseline rows, found {len(rows)}")
    return rows


def model_config(base_config: dict, meta: dict, seed: int, steps: int) -> dict:
    cfg = json.loads(json.dumps(base_config))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = int(steps)
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    return cfg


def action_sequence(daily: pd.DataFrame) -> str:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nonzero = daily[(irr > 0) | (n > 0)]
    return "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )


def evaluate_model(
    config: dict,
    env_config: dict,
    model: ddqn22.MaskAwareDoubleDuelingDQN,
    model_path: Path,
    seed: int,
    steps: int,
    year: int,
) -> dict[str, Any]:
    station = "SYA"
    run_tag = f"SYA_{year}_031_24_seed{seed}_{steps}_eval"
    env = dqn19.make_env(config, env_config, station, year, seed, run_tag, evaluation=True)
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = dqn19.latest_observation_dict(env, obs, info)
            dap_raw = dqn19.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = env.action_masks().copy()
            action = model.select_action(np.asarray(obs, dtype=np.float32), mask, step_count, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = dqn19.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "site": "SY",
                    "year": int(year),
                    "seed": int(seed),
                    "step_budget": int(steps),
                    "split": "eval",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": dqn19.scalar(wrow.get("rain"), np.nan),
                    "srad": dqn19.scalar(wrow.get("srad"), np.nan),
                    "tmax": dqn19.scalar(wrow.get("tmax"), np.nan),
                    "tmin": dqn19.scalar(wrow.get("tmin"), np.nan),
                    "swfac": dqn19.scalar(latest.get("swfac")),
                    "nstres": dqn19.scalar(latest.get("nstres")),
                    "topwt": dqn19.scalar(latest.get("topwt")),
                    "grnwt": dqn19.scalar(latest.get("grnwt")),
                    "xlai": dqn19.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Evaluation did not finish for year={year}, seed={seed}, steps={steps}")
        daily_path = OUT / "daily_outputs" / station / f"{year}_seed{seed}_{steps}_ddqn_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0)
        n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        total_i = float(irr.sum())
        total_n = float(n.sum())
        snapshot = OUT / "runs" / str(year) / f"seed{seed}_{steps}" / "snapshot"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(siteppo.snapshot_from_env(env), snapshot, dirs_exist_ok=True)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        swfac = pd.to_numeric(daily["swfac"], errors="coerce")
        nstres = pd.to_numeric(daily["nstres"], errors="coerce")
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        return {
            "algorithm": "mask_aware_double_dueling_DQN",
            "station_code": station,
            "site": "SY",
            "year": int(year),
            "seed": int(seed),
            "step_budget": int(steps),
            "source_train_year": 2014,
            "scenario": "rl_candidate_031_24",
            "run_status": "ok",
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path),
            "final_grain_kg_ha": final_y,
            "final_biomass_kg_ha": final_b,
            "total_irrigation": total_i,
            "total_n": total_n,
            "profit_simple": final_y - total_i - 5.0 * total_n,
            "PFP_N": final_y / total_n if total_n > 0 else math.nan,
            "irrigation_event_total_mm": total_i,
            "nitrogen_event_total_kg_ha": total_n,
            "etcp_mm": metrics["etcp_mm"],
            "wp_et_kg_m3": metrics["WP_ET_kg_m3"],
            "pfp_n_kg_kg": metrics["PFP_N_kg_kg"],
            "max_water_stress_wspd": float(swfac.max()),
            "max_nitrogen_stress_nstd": float(nstres.max()),
            "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()),
            "nstres_days_gt_0p05": int((nstres > 0.05).sum()),
            "early_dap1_10_irrigation": float(irr[dap <= 10].sum()),
            "early_dap1_10_n": float(n[dap <= 10].sum()),
            "irrigation_event_count": int((irr > 0).sum()),
            "n_event_count": int((n > 0).sum()),
            "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
            "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
            "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "action_sequence": action_sequence(daily),
            "summary_match_score": metrics["summary_match_score"],
        }
    finally:
        env.close()


def train_one(config: dict, env_config: dict, seed: int, steps: int) -> tuple[dict[str, Any], list[dict[str, Any]], ddqn22.MaskAwareDoubleDuelingDQN, Path]:
    station = "SYA"
    year = 2014
    cfg = config["ddqn"]
    env = dqn19.make_env(config, env_config, station, year, seed, f"SYA_2014_031_24_seed{seed}_{steps}_train", evaluation=False)
    updates: list[dict[str, Any]] = []
    model_path = OUT / "models" / station / f"free_timing_mask_aware_double_dueling_dqn_seed{seed}_{steps}.pt"
    try:
        obs, info = env.reset()
        model = ddqn22.MaskAwareDoubleDuelingDQN(int(np.asarray(obs).shape[0]), int(env.action_space.n), config, seed=seed)
        done = False
        for global_step in range(1, int(steps) + 1):
            mask = env.action_masks().copy()
            action = model.select_action(np.asarray(obs, dtype=np.float32), mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_mask = np.zeros(model.action_dim, dtype=bool) if done else env.action_masks().copy()
            model.replay.add(
                observation=np.asarray(obs, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_observation=np.asarray(next_obs, dtype=np.float32),
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            if global_step >= int(cfg["learning_starts"]) and global_step % int(cfg["train_freq"]) == 0:
                metrics = model.train_step()
                if global_step % 500 == 0:
                    updates.append({"seed": seed, "step_budget": steps, "global_step": global_step, **metrics})
            if global_step % int(cfg["target_update_interval"]) == 0:
                model.sync_target()
            if done and global_step < int(steps):
                obs, info = env.reset()
            else:
                obs = next_obs
        model.save(model_path, global_step=steps)
        summary = {
            "algorithm": "mask_aware_double_dueling_DQN",
            "station_code": station,
            "train_year": year,
            "seed": int(seed),
            "step_budget": int(steps),
            "run_status": "ok",
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path),
            "optimizer_updates": model.optimizer_updates,
            "target_updates": model.target_updates,
            "online_hash": model.module_hash(model.online),
            "target_hash": model.module_hash(model.target),
            "notes": "",
        }
        return summary, updates, model, model_path
    finally:
        env.close()


def add_transfer_gaps(eval_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in eval_df[eval_df["year"].astype(int).isin(sorted(baselines["year"].astype(int).unique()))].itertuples(index=False):
        d = row._asdict()
        base = baselines[baselines["year"].astype(int).eq(int(d["year"]))]
        max_y = float(pd.to_numeric(base["final_grain_kg_ha"], errors="coerce").max())
        max_wp = float(pd.to_numeric(base["wp_et_kg_m3"], errors="coerce").max())
        pfp_values = pd.to_numeric(base["pfp_n_kg_kg"], errors="coerce").dropna()
        max_pfp = float(pfp_values.max()) if not pfp_values.empty else math.nan
        y = float(d["final_grain_kg_ha"])
        wp = float(d["wp_et_kg_m3"])
        pfp = float(d["pfp_n_kg_kg"]) if pd.notna(d["pfp_n_kg_kg"]) else math.nan
        d.update(
            {
                "baseline_max_yield": max_y,
                "baseline_max_wp_et": max_wp,
                "baseline_max_pfp_n_positive_n": max_pfp,
                "gap_yield": y - max_y,
                "gap_wp_et": wp - max_wp,
                "gap_pfp_n": pfp - max_pfp if math.isfinite(pfp) and math.isfinite(max_pfp) else math.nan,
                "yield_strict_win": bool(y > max_y),
                "wp_et_strict_win": bool(wp > max_wp),
                "pfp_n_strict_win": bool(math.isfinite(pfp) and math.isfinite(max_pfp) and pfp > max_pfp),
            }
        )
        d["advisor_any_metric_strict_winner"] = bool(d["yield_strict_win"] or d["wp_et_strict_win"] or d["pfp_n_strict_win"])
        d["winning_metrics"] = ";".join(
            name for name, flag in [("yield", d["yield_strict_win"]), ("WP_ET", d["wp_et_strict_win"]), ("PFP_N", d["pfp_n_strict_win"])] if flag
        )
        rows.append(d)
    return pd.DataFrame(rows)


def write_record(train_df: pd.DataFrame, eval_df: pd.DataFrame, gap_df: pd.DataFrame, context_20k: pd.DataFrame) -> None:
    lines = [
        "# 031_24 Double-Dueling DQN 50k vs 100k training-length sensitivity",
        "",
        "## Scope",
        "",
        "- SYA2014 training, seeds 0/1/2.",
        "- Step budgets: 50k and 100k.",
        "- Frozen deterministic evaluation on SYA2014, SYA2012, SYA2015.",
        "- Same reward/action/constraint settings as 031_22/031_23.",
        "- Final models only; no checkpoint selection.",
        "",
        "## Training summary",
        "",
        train_df.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_df[
            [
                "year",
                "seed",
                "step_budget",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "irrigation_event_total_mm",
                "nitrogen_event_total_kg_ha",
                "early_dap1_10_n",
                "nstres_days_gt_0p05",
                "action_sequence",
            ]
        ].to_string(index=False),
        "",
        "## SY2012/SY2015 gap versus four baselines",
        "",
        gap_df[
            [
                "year",
                "seed",
                "step_budget",
                "gap_yield",
                "gap_wp_et",
                "gap_pfp_n",
                "advisor_any_metric_strict_winner",
                "winning_metrics",
            ]
        ].to_string(index=False) if not gap_df.empty else "No gap rows.",
        "",
        "## 20k DDQN context",
        "",
        context_20k.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- Better SYA2014 performance alone is not success; SYA2012/SYA2015 frozen transfer must also be checked.",
        "- If 100k worsens transfer relative to 50k/20k, treat it as possible overfitting to SYA2014.",
        "- No hyperparameter or reward conclusion should be drawn from this task alone.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    base_config = direct_ppo.load_yaml(base_config_path)
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)
    selection = make_selection(meta)
    selection.to_csv(OUT / "configs" / "031_24_sy_selection.csv", index=False, encoding="utf-8-sig")
    baselines = load_baselines(meta)
    baselines.to_csv(OUT / "evaluation" / "031_24_transfer_four_baselines_reused_from_028_05.csv", index=False, encoding="utf-8-sig")

    train_rows = []
    update_rows = []
    eval_rows = []
    for steps in [int(x) for x in meta["step_budgets"]]:
        for seed in [int(s) for s in meta["seeds"]]:
            config = model_config(base_config, meta, seed, steps)
            direct_ppo.OUTPUT_ROOT = OUT
            env_config = direct_ppo.build_env_config(config, selection)
            direct_ppo.write_yaml(env_config, OUT / "configs" / f"031_24_resolved_env_seed{seed}_{steps}.yaml")
            try:
                train_summary, updates, model, model_path = train_one(config, env_config, seed, steps)
                train_rows.append(train_summary)
                update_rows.extend(updates)
                for year in [int(meta["train_year"])] + [int(y) for y in meta["transfer_years"]]:
                    eval_rows.append(evaluate_model(config, env_config, model, model_path, seed, steps, year))
            except Exception:
                train_rows.append(
                    {
                        "algorithm": "mask_aware_double_dueling_DQN",
                        "station_code": "SYA",
                        "train_year": int(meta["train_year"]),
                        "seed": seed,
                        "step_budget": steps,
                        "run_status": "failed",
                        "notes": traceback.format_exc()[-2500:],
                    }
                )
    train_df = pd.DataFrame(train_rows)
    updates_df = pd.DataFrame(update_rows)
    eval_df = pd.DataFrame(eval_rows)
    train_df.to_csv(OUT / "evaluation" / "031_24_training_summary.csv", index=False, encoding="utf-8-sig")
    updates_df.to_csv(OUT / "logs" / "031_24_update_sample_log.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT / "evaluation" / "031_24_eval_summary.csv", index=False, encoding="utf-8-sig")
    gap_df = add_transfer_gaps(eval_df, baselines) if not eval_df.empty else pd.DataFrame()
    gap_df.to_csv(OUT / "evaluation" / "031_24_transfer_gap_summary.csv", index=False, encoding="utf-8-sig")
    context_path = ROOT / "benchmark_results" / "031_23_double_dueling_dqn_sy2014_seeds1_2_repeat" / "evaluation" / "031_23_ddqn_seed0_1_2_comparison.csv"
    context_20k = pd.read_csv(context_path)[
        ["comparison_label", "seed", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "PFP_N", "early_dap1_10_n", "nstres_days_gt_0p05"]
    ]
    context_20k.to_csv(OUT / "evaluation" / "031_24_20k_context_from_031_23.csv", index=False, encoding="utf-8-sig")
    write_record(train_df, eval_df, gap_df, context_20k)
    result = {
        "task": "031_24_ddqn_training_length_50k_100k_sy_crossyear",
        "record_md": str(DOC.relative_to(ROOT)),
        "training_summary": str((OUT / "evaluation" / "031_24_training_summary.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "031_24_eval_summary.csv").relative_to(ROOT)),
        "transfer_gap_summary": str((OUT / "evaluation" / "031_24_transfer_gap_summary.csv").relative_to(ROOT)),
    }
    (OUT / "031_24_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        print(
            eval_df[
                [
                    "year",
                    "seed",
                    "step_budget",
                    "final_grain_kg_ha",
                    "wp_et_kg_m3",
                    "pfp_n_kg_kg",
                    "irrigation_event_total_mm",
                    "nitrogen_event_total_kg_ha",
                    "early_dap1_10_n",
                    "nstres_days_gt_0p05",
                ]
            ].to_string(index=False)
        )
    if not gap_df.empty:
        print("TRANSFER GAPS")
        print(gap_df[["year", "seed", "step_budget", "gap_yield", "gap_wp_et", "gap_pfp_n", "advisor_any_metric_strict_winner", "winning_metrics"]].to_string(index=False))


if __name__ == "__main__":
    main()


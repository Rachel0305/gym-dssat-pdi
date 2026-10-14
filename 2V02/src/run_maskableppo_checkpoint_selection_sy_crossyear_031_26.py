from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_18 as ppo18
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_26_maskableppo_checkpoint_selection_sy_crossyear.yaml"
OUT = ROOT / "benchmark_results" / "031_26_maskableppo_checkpoint_selection_sy_crossyear"
DOC = ROOT / "docs" / "031_26_maskableppo_checkpoint_selection_sy_crossyear_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "daily_outputs/SYA", "runs", "logs", "evaluation", "tensorboard/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_selection(meta: dict[str, Any]) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    years = [int(meta["train_year"]), int(meta["validation_year"]), int(meta["test_year"])]
    rows = pool[(pool["station_code"].eq(meta["train_site"])) & (pool["year"].astype(int).isin(years))].copy()
    if sorted(rows["year"].astype(int).tolist()) != sorted(years):
        raise RuntimeError(f"Expected SY years {years}, got {rows['year'].tolist()}")
    rows["selected_for_train"] = rows["year"].astype(int).eq(int(meta["train_year"]))
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "031_26_maskableppo_checkpoint_selection"
    return rows


def clean_scenario(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    if text.strip() == "" or text.lower() == "nan":
        return "null"
    return text


def load_baselines(meta: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / meta["four_baseline_summary_csv"]
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].map(clean_scenario)
    years = [int(meta["validation_year"]), int(meta["test_year"])]
    scenarios = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]
    rows = df[(df["site"].eq("SY")) & (df["year"].astype(int).isin(years)) & (df["scenario"].isin(scenarios))].copy()
    rows = rows.drop_duplicates(["site", "year", "scenario"], keep="first")
    expected = len(years) * len(scenarios)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} baseline rows, found {len(rows)}")
    return rows


def model_config(base_config: dict[str, Any], meta: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = int(meta["max_train_steps"])
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    return cfg


def checkpoint_path(seed: int, step: int) -> Path:
    return OUT / "models" / "SYA" / f"free_timing_discrete_maskableppo_seed{seed}_ckpt{step}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, seed: int, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step and x not in self.outer.saved_steps]
                for target in pending:
                    path = checkpoint_path(self.outer.seed, target)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(target)
                return True

        self.seed = int(seed)
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


def train_one_seed(config: dict[str, Any], env_config: dict[str, Any], meta: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sb3_contrib import MaskablePPO

    station = "SYA"
    year = int(meta["train_year"])
    env = None
    train_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    try:
        env = ppo18.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_26_seed{seed}_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **ppo18.ppo_kwargs(config),
        )
        ckpt_cb = FixedStepCheckpointCallback(seed, [int(x) for x in meta["checkpoint_steps"]])
        model.learn(total_timesteps=int(meta["max_train_steps"]), reset_num_timesteps=True, progress_bar=False, callback=ckpt_cb.callback)
        for step in [int(x) for x in meta["checkpoint_steps"]]:
            path = checkpoint_path(seed, step)
            checkpoint_rows.append(
                {
                    "algorithm": "free_timing_discrete_MaskablePPO",
                    "station_code": station,
                    "train_year": year,
                    "seed": int(seed),
                    "checkpoint_step": int(step),
                    "run_status": "ok" if path.exists() else "missing",
                    "model_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "",
                    "model_sha256": sha256_file(path) if path.exists() else "",
                }
            )
        train_rows.append(
            {
                "algorithm": "free_timing_discrete_MaskablePPO",
                "station_code": station,
                "train_year": year,
                "seed": int(seed),
                "max_train_steps": int(meta["max_train_steps"]),
                "run_status": "ok",
                "checkpoint_count": int(sum((checkpoint_path(seed, int(x))).exists() for x in meta["checkpoint_steps"])),
            }
        )
    except Exception:
        train_rows.append(
            {
                "algorithm": "free_timing_discrete_MaskablePPO",
                "station_code": station,
                "train_year": year,
                "seed": int(seed),
                "max_train_steps": int(meta["max_train_steps"]),
                "run_status": "failed",
                "notes": traceback.format_exc()[-2500:],
            }
        )
    finally:
        if env is not None:
            env.close()
    return train_rows, checkpoint_rows


def action_sequence(daily: pd.DataFrame) -> str:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nonzero = daily[(irr > 0) | (n > 0)]
    return "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], seed: int, checkpoint_step: int, year: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = "SYA"
    model_path = checkpoint_path(seed, checkpoint_step)
    model = MaskablePPO.load(str(model_path), device="cpu")
    env = ppo18.make_env(config, env_config, station, int(year), seed, f"SYA_{year}_031_26_seed{seed}_ckpt{checkpoint_step}_eval", evaluation=True)
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = ppo18.latest_observation_dict(env, obs, info)
            dap_raw = ppo18.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = ppo18.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "site": "SY",
                    "year": int(year),
                    "seed": int(seed),
                    "checkpoint_step": int(checkpoint_step),
                    "selection_role": "train_year" if int(year) == 2014 else ("validation_year" if int(year) == 2012 else "test_year"),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": ppo18.scalar(wrow.get("rain"), np.nan),
                    "srad": ppo18.scalar(wrow.get("srad"), np.nan),
                    "tmax": ppo18.scalar(wrow.get("tmax"), np.nan),
                    "tmin": ppo18.scalar(wrow.get("tmin"), np.nan),
                    "swfac": ppo18.scalar(latest.get("swfac")),
                    "nstres": ppo18.scalar(latest.get("nstres")),
                    "topwt": ppo18.scalar(latest.get("topwt")),
                    "grnwt": ppo18.scalar(latest.get("grnwt")),
                    "xlai": ppo18.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Evaluation did not finish for year={year}, seed={seed}, ckpt={checkpoint_step}")
        daily_path = OUT / "daily_outputs" / station / f"{year}_seed{seed}_ckpt{checkpoint_step}_maskableppo_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0)
        n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        total_i = float(irr.sum())
        total_n = float(n.sum())
        snapshot = OUT / "runs" / str(year) / f"seed{seed}_ckpt{checkpoint_step}" / "snapshot"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(siteppo.snapshot_from_env(env), snapshot, dirs_exist_ok=True)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        swfac = pd.to_numeric(daily["swfac"], errors="coerce")
        nstres = pd.to_numeric(daily["nstres"], errors="coerce")
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        return {
            "algorithm": "free_timing_discrete_MaskablePPO",
            "station_code": station,
            "site": "SY",
            "year": int(year),
            "seed": int(seed),
            "checkpoint_step": int(checkpoint_step),
            "source_train_year": 2014,
            "scenario": f"rl_candidate_031_26_ckpt{checkpoint_step}",
            "selection_role": "train_year" if int(year) == 2014 else ("validation_year" if int(year) == 2012 else "test_year"),
            "run_status": "ok",
            "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
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


def baseline_maxima(baselines: pd.DataFrame, year: int) -> dict[str, float]:
    base = baselines[baselines["year"].astype(int).eq(int(year))]
    max_y = float(pd.to_numeric(base["final_grain_kg_ha"], errors="coerce").max())
    max_wp = float(pd.to_numeric(base["wp_et_kg_m3"], errors="coerce").max())
    pfp_values = pd.to_numeric(base["pfp_n_kg_kg"], errors="coerce").dropna()
    max_pfp = float(pfp_values.max()) if not pfp_values.empty else math.nan
    return {"max_yield": max_y, "max_wp_et": max_wp, "max_pfp_n": max_pfp}


def add_gaps(eval_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in eval_df.itertuples(index=False):
        d = row._asdict()
        year = int(d["year"])
        if year not in set(baselines["year"].astype(int)):
            rows.append(d)
            continue
        maxima = baseline_maxima(baselines, year)
        y = float(d["final_grain_kg_ha"])
        wp = float(d["wp_et_kg_m3"])
        pfp = float(d["pfp_n_kg_kg"]) if pd.notna(d["pfp_n_kg_kg"]) else math.nan
        gap_y = y - maxima["max_yield"]
        gap_wp = wp - maxima["max_wp_et"]
        gap_pfp = pfp - maxima["max_pfp_n"] if math.isfinite(pfp) and math.isfinite(maxima["max_pfp_n"]) else math.nan
        score_terms = [
            gap_y / maxima["max_yield"] if maxima["max_yield"] else -math.inf,
            gap_wp / maxima["max_wp_et"] if maxima["max_wp_et"] else -math.inf,
            gap_pfp / maxima["max_pfp_n"] if math.isfinite(gap_pfp) and maxima["max_pfp_n"] else -math.inf,
        ]
        d.update(
            {
                "baseline_max_yield": maxima["max_yield"],
                "baseline_max_wp_et": maxima["max_wp_et"],
                "baseline_max_pfp_n_positive_n": maxima["max_pfp_n"],
                "gap_yield": gap_y,
                "gap_wp_et": gap_wp,
                "gap_pfp_n": gap_pfp,
                "yield_strict_win": bool(gap_y > 0),
                "wp_et_strict_win": bool(gap_wp > 0),
                "pfp_n_strict_win": bool(math.isfinite(gap_pfp) and gap_pfp > 0),
                "validation_score": float(max(score_terms)),
            }
        )
        d["advisor_any_metric_strict_winner"] = bool(d["yield_strict_win"] or d["wp_et_strict_win"] or d["pfp_n_strict_win"])
        d["winning_metrics"] = ";".join(
            name
            for name, flag in [
                ("yield", d["yield_strict_win"]),
                ("WP_ET", d["wp_et_strict_win"]),
                ("PFP_N", d["pfp_n_strict_win"]),
            ]
            if flag
        )
        rows.append(d)
    return pd.DataFrame(rows)


def select_checkpoints(gap_df: pd.DataFrame, baselines: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    val_year = int(meta["validation_year"])
    min_yield_fraction = float(meta["selection"]["min_yield_fraction_of_validation_baseline_max"])
    max_y = baseline_maxima(baselines, val_year)["max_yield"]
    rows = []
    val = gap_df[gap_df["year"].astype(int).eq(val_year)].copy()
    for seed, group in val.groupby("seed"):
        g = group.copy()
        g["guardrail_positive_n"] = pd.to_numeric(g["nitrogen_event_total_kg_ha"], errors="coerce") > 0
        g["guardrail_yield_fraction"] = pd.to_numeric(g["final_grain_kg_ha"], errors="coerce") >= min_yield_fraction * max_y
        g["selected_guardrail_pass"] = g["guardrail_positive_n"] & g["guardrail_yield_fraction"] & g["run_status"].eq("ok")
        g["selected_pass"] = g["advisor_any_metric_strict_winner"] & g["selected_guardrail_pass"]
        if bool(g["selected_pass"].any()):
            pool = g[g["selected_pass"]].copy()
        elif bool(g["selected_guardrail_pass"].any()):
            pool = g[g["selected_guardrail_pass"]].copy()
        else:
            pool = g.copy()
        pool = pool.sort_values(
            by=["validation_score", "gap_yield", "nitrogen_event_total_kg_ha", "irrigation_event_total_mm", "checkpoint_step"],
            ascending=[False, False, True, True, True],
        )
        selected = pool.iloc[0].to_dict()
        selected["selection_reason"] = (
            "validation_pass_highest_score"
            if bool(selected.get("selected_pass"))
            else ("no_validation_pass_highest_guardrail_score" if bool(selected.get("selected_guardrail_pass")) else "no_guardrail_pass_highest_score")
        )
        rows.append(selected)
    return pd.DataFrame(rows)


def load_ddqn_comparison(meta: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / meta["ddqn_031_25_selected_final_eval_csv"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["comparison_algorithm"] = "DDQN_031_25"
    return df


def write_record(
    train_df: pd.DataFrame,
    ckpt_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    final_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    lines = [
        "# 031_26 MaskablePPO checkpoint selection / early-stopping test on SY cross-year transfer",
        "",
        "## Scope",
        "",
        "- Train: SYA2014, seeds 0/1/2.",
        "- Checkpoints: 10k, 20k, 30k, 50k, 75k, 100k.",
        "- Selection year: SYA2012 only.",
        "- Held-out test year: SYA2015; not used for selection.",
        "- Reward/action/constraints unchanged from 031_17/031_18 and matched to 031_25.",
        "",
        "## Training summary",
        "",
        train_df.to_string(index=False) if not train_df.empty else "No training rows.",
        "",
        "## Checkpoint inventory",
        "",
        ckpt_df.to_string(index=False) if not ckpt_df.empty else "No checkpoint rows.",
        "",
        "## Validation/test gap summary",
        "",
        gap_df[
            [
                "year",
                "seed",
                "checkpoint_step",
                "selection_role",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "irrigation_event_total_mm",
                "nitrogen_event_total_kg_ha",
                "gap_yield",
                "gap_wp_et",
                "gap_pfp_n",
                "validation_score",
                "advisor_any_metric_strict_winner",
                "winning_metrics",
            ]
        ].to_string(index=False) if not gap_df.empty else "No gap rows.",
        "",
        "## Selected checkpoints from SYA2012",
        "",
        selected_df[
            [
                "seed",
                "checkpoint_step",
                "selected_pass",
                "selected_guardrail_pass",
                "selection_reason",
                "validation_score",
                "gap_yield",
                "gap_wp_et",
                "gap_pfp_n",
                "winning_metrics",
            ]
        ].to_string(index=False) if not selected_df.empty else "No selected checkpoints.",
        "",
        "## Final selected checkpoint evaluation",
        "",
        final_df[
            [
                "year",
                "seed",
                "checkpoint_step",
                "selection_role",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "irrigation_event_total_mm",
                "nitrogen_event_total_kg_ha",
                "gap_yield",
                "gap_wp_et",
                "gap_pfp_n",
                "advisor_any_metric_strict_winner",
                "winning_metrics",
                "action_sequence",
            ]
        ].to_string(index=False) if not final_df.empty else "No final rows.",
        "",
        "## PPO vs DDQN selected-checkpoint comparison",
        "",
        comparison_df.to_string(index=False) if not comparison_df.empty else "031_25 DDQN comparison file not found or comparison empty.",
        "",
        "## Interpretation boundary",
        "",
        "- This is a fair checkpoint-selection comparison against 031_25 DDQN.",
        "- SYA2015 is held out from checkpoint selection.",
        "- Do not declare algorithm superiority from one metric only; report yield/WP_ET/PFP_N/input tradeoffs.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    base_config = direct_ppo.load_yaml(base_config_path)
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)
    selection = make_selection(meta)
    selection.to_csv(OUT / "configs" / "031_26_sy_selection.csv", index=False, encoding="utf-8-sig")
    baselines = load_baselines(meta)
    baselines.to_csv(OUT / "evaluation" / "031_26_transfer_four_baselines_reused_from_028_05.csv", index=False, encoding="utf-8-sig")

    train_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for seed in [int(s) for s in meta["seeds"]]:
        config = model_config(base_config, meta, seed)
        direct_ppo.OUTPUT_ROOT = OUT
        env_config = direct_ppo.build_env_config(config, selection)
        direct_ppo.write_yaml(env_config, OUT / "configs" / f"031_26_resolved_env_seed{seed}.yaml")
        t_rows, c_rows = train_one_seed(config, env_config, meta, seed)
        train_rows.extend(t_rows)
        checkpoint_rows.extend(c_rows)
        for ckpt in [int(x) for x in meta["checkpoint_steps"]]:
            if not checkpoint_path(seed, ckpt).exists():
                continue
            for year in [int(meta["train_year"]), int(meta["validation_year"])]:
                eval_rows.append(evaluate_checkpoint(config, env_config, seed, ckpt, year))

    train_df = pd.DataFrame(train_rows)
    ckpt_df = pd.DataFrame(checkpoint_rows)
    eval_df = pd.DataFrame(eval_rows)
    train_df.to_csv(OUT / "evaluation" / "031_26_training_summary.csv", index=False, encoding="utf-8-sig")
    ckpt_df.to_csv(OUT / "evaluation" / "031_26_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT / "evaluation" / "031_26_checkpoint_eval_train_validation_summary.csv", index=False, encoding="utf-8-sig")
    gap_df = add_gaps(eval_df, baselines) if not eval_df.empty else pd.DataFrame()
    gap_df.to_csv(OUT / "evaluation" / "031_26_checkpoint_gap_train_validation_summary.csv", index=False, encoding="utf-8-sig")
    selected_df = select_checkpoints(gap_df, baselines, meta) if not gap_df.empty else pd.DataFrame()
    selected_df.to_csv(OUT / "evaluation" / "031_26_selected_checkpoints_from_validation.csv", index=False, encoding="utf-8-sig")

    final_rows: list[dict[str, Any]] = []
    for selected in selected_df.itertuples(index=False):
        seed = int(selected.seed)
        ckpt = int(selected.checkpoint_step)
        config = model_config(base_config, meta, seed)
        env_config = direct_ppo.load_yaml(OUT / "configs" / f"031_26_resolved_env_seed{seed}.yaml")
        for year in [int(meta["train_year"]), int(meta["validation_year"]), int(meta["test_year"])]:
            row = evaluate_checkpoint(config, env_config, seed, ckpt, year)
            row["selected_pass"] = bool(selected.selected_pass)
            row["selected_guardrail_pass"] = bool(selected.selected_guardrail_pass)
            row["selection_reason"] = str(selected.selection_reason)
            final_rows.append(row)
    final_df = pd.DataFrame(final_rows)
    final_gap_df = add_gaps(final_df, baselines) if not final_df.empty else pd.DataFrame()
    final_gap_df.to_csv(OUT / "evaluation" / "031_26_selected_final_eval_summary.csv", index=False, encoding="utf-8-sig")

    ddqn_df = load_ddqn_comparison(meta)
    ppo_comp = final_gap_df.copy()
    if not ppo_comp.empty:
        ppo_comp["comparison_algorithm"] = "MaskablePPO_031_26"
    comparison_df = pd.concat([ppo_comp, ddqn_df], ignore_index=True, sort=False) if not ddqn_df.empty else ppo_comp
    if not comparison_df.empty:
        cols = [
            "comparison_algorithm",
            "year",
            "seed",
            "checkpoint_step",
            "final_grain_kg_ha",
            "wp_et_kg_m3",
            "pfp_n_kg_kg",
            "irrigation_event_total_mm",
            "nitrogen_event_total_kg_ha",
            "gap_yield",
            "gap_wp_et",
            "gap_pfp_n",
            "advisor_any_metric_strict_winner",
            "winning_metrics",
        ]
        comparison_df[[c for c in cols if c in comparison_df.columns]].to_csv(OUT / "evaluation" / "031_26_ppo_vs_ddqn_selected_comparison.csv", index=False, encoding="utf-8-sig")

    write_record(train_df, ckpt_df, gap_df, selected_df, final_gap_df, comparison_df)
    result = {
        "task": "031_26_maskableppo_checkpoint_selection_sy_crossyear",
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "training_summary": str((OUT / "evaluation" / "031_26_training_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "selected_checkpoints": str((OUT / "evaluation" / "031_26_selected_checkpoints_from_validation.csv").relative_to(ROOT)).replace("\\", "/"),
        "final_eval_summary": str((OUT / "evaluation" / "031_26_selected_final_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "ppo_vs_ddqn_comparison": str((OUT / "evaluation" / "031_26_ppo_vs_ddqn_selected_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "031_26_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not selected_df.empty:
        print("SELECTED")
        print(selected_df[["seed", "checkpoint_step", "selected_pass", "selected_guardrail_pass", "selection_reason", "validation_score", "gap_yield", "gap_wp_et", "gap_pfp_n", "winning_metrics"]].to_string(index=False))
    if not final_gap_df.empty:
        print("FINAL SELECTED EVAL")
        print(final_gap_df[["year", "seed", "checkpoint_step", "gap_yield", "gap_wp_et", "gap_pfp_n", "advisor_any_metric_strict_winner", "winning_metrics"]].to_string(index=False))


if __name__ == "__main__":
    main()


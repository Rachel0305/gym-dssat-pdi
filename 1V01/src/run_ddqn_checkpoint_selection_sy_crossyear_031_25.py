from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19 as dqn19
import run_literature_aligned_ppo_dqn_ddqn_sy2014_smoke_031_22 as ddqn22
import run_ddqn_training_length_50k_100k_sy_crossyear_031_24 as ddqn24


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_25_ddqn_checkpoint_selection_sy_crossyear.yaml"
OUT = ROOT / "benchmark_results" / "031_25_ddqn_checkpoint_selection_sy_crossyear"
DOC = ROOT / "docs" / "031_25_ddqn_checkpoint_selection_sy_crossyear_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "daily_outputs/SYA", "runs", "logs", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_selection(meta: dict[str, Any]) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    years = [int(meta["train_year"]), int(meta["validation_year"]), int(meta["test_year"])]
    rows = pool[(pool["station_code"].eq(meta["train_site"])) & (pool["year"].astype(int).isin(years))].copy()
    if sorted(rows["year"].astype(int).tolist()) != sorted(years):
        raise RuntimeError(f"Expected SY years {years}, got {rows['year'].tolist()}")
    rows["selected_for_train"] = rows["year"].astype(int).eq(int(meta["train_year"]))
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "031_25_ddqn_checkpoint_selection"
    return rows


def model_config(base_config: dict[str, Any], meta: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = int(meta["max_train_steps"])
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    return cfg


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


def checkpoint_path(seed: int, step: int) -> Path:
    return OUT / "models" / "SYA" / f"free_timing_mask_aware_double_dueling_dqn_seed{seed}_ckpt{step}.pt"


def train_one_seed(
    config: dict[str, Any],
    env_config: dict[str, Any],
    meta: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    station = "SYA"
    year = int(meta["train_year"])
    cfg = config["ddqn"]
    max_steps = int(meta["max_train_steps"])
    checkpoint_steps = set(int(x) for x in meta["checkpoint_steps"])
    env = dqn19.make_env(config, env_config, station, year, seed, f"SYA_2014_031_25_seed{seed}_train", evaluation=False)
    train_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        model = ddqn22.MaskAwareDoubleDuelingDQN(int(np.asarray(obs).shape[0]), int(env.action_space.n), config, seed=seed)
        done = False
        for global_step in range(1, max_steps + 1):
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
                if global_step % 1000 == 0:
                    update_rows.append({"seed": seed, "global_step": global_step, **metrics})
            if global_step % int(cfg["target_update_interval"]) == 0:
                model.sync_target()
            if global_step in checkpoint_steps:
                path = checkpoint_path(seed, global_step)
                model.save(path, global_step=global_step)
                checkpoint_rows.append(
                    {
                        "algorithm": "mask_aware_double_dueling_DQN",
                        "station_code": station,
                        "train_year": year,
                        "seed": int(seed),
                        "checkpoint_step": int(global_step),
                        "run_status": "ok",
                        "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "model_sha256": ddqn24.sha256_file(path),
                        "optimizer_updates": model.optimizer_updates,
                        "target_updates": model.target_updates,
                        "online_hash": model.module_hash(model.online),
                        "target_hash": model.module_hash(model.target),
                    }
                )
            if done and global_step < max_steps:
                obs, info = env.reset()
            else:
                obs = next_obs
        train_rows.append(
            {
                "algorithm": "mask_aware_double_dueling_DQN",
                "station_code": station,
                "train_year": year,
                "seed": int(seed),
                "max_train_steps": max_steps,
                "run_status": "ok",
                "checkpoint_count": len(checkpoint_rows),
                "optimizer_updates": model.optimizer_updates,
                "target_updates": model.target_updates,
            }
        )
        return train_rows, update_rows, checkpoint_rows
    except Exception:
        train_rows.append(
            {
                "algorithm": "mask_aware_double_dueling_DQN",
                "station_code": station,
                "train_year": year,
                "seed": int(seed),
                "max_train_steps": max_steps,
                "run_status": "failed",
                "notes": traceback.format_exc()[-2500:],
            }
        )
        return train_rows, update_rows, checkpoint_rows
    finally:
        env.close()


def load_model(config: dict[str, Any], seed: int, model_path: Path) -> ddqn22.MaskAwareDoubleDuelingDQN:
    payload = torch.load(model_path, map_location="cpu")
    model = ddqn22.MaskAwareDoubleDuelingDQN(int(payload["observation_dim"]), int(payload["action_dim"]), config, seed=seed)
    model.online.load_state_dict(payload["online"])
    model.target.load_state_dict(payload["target"])
    model.optimizer_updates = int(payload.get("optimizer_updates", 0))
    model.target_updates = int(payload.get("target_updates", 0))
    return model


def evaluate_checkpoint(
    config: dict[str, Any],
    env_config: dict[str, Any],
    seed: int,
    checkpoint_step: int,
    year: int,
) -> dict[str, Any]:
    model_path = checkpoint_path(seed, checkpoint_step)
    model = load_model(config, seed, model_path)
    row = ddqn24.evaluate_model(config, env_config, model, model_path, seed, checkpoint_step, int(year))
    row["scenario"] = f"rl_candidate_031_25_ckpt{checkpoint_step}"
    row["checkpoint_step"] = int(checkpoint_step)
    row["selection_role"] = "train_year" if int(year) == 2014 else ("validation_year" if int(year) == 2012 else "test_year")
    return row


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


def write_record(
    train_df: pd.DataFrame,
    ckpt_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> None:
    lines = [
        "# 031_25 DDQN checkpoint selection / early-stopping test on SY cross-year transfer",
        "",
        "## Scope",
        "",
        "- Train: SYA2014, seeds 0/1/2.",
        "- Checkpoints: 10k, 20k, 30k, 50k, 75k, 100k.",
        "- Selection year: SYA2012 only.",
        "- Held-out test year: SYA2015; not used for selection.",
        "- Reward/action/constraints unchanged from 031_22-031_24.",
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
        "## Interpretation boundary",
        "",
        "- This tests checkpoint selection, not new reward or new action constraints.",
        "- SYA2015 is held out from checkpoint selection.",
        "- If selected checkpoints pass validation but fail SYA2015, report validation overfitting rather than tuning on SYA2015.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ddqn24.OUT = OUT
    ddqn24.DOC = DOC
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    base_config = direct_ppo.load_yaml(base_config_path)
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)
    selection = make_selection(meta)
    selection.to_csv(OUT / "configs" / "031_25_sy_selection.csv", index=False, encoding="utf-8-sig")
    baselines = load_baselines(meta)
    baselines.to_csv(OUT / "evaluation" / "031_25_transfer_four_baselines_reused_from_028_05.csv", index=False, encoding="utf-8-sig")

    train_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for seed in [int(s) for s in meta["seeds"]]:
        config = model_config(base_config, meta, seed)
        direct_ppo.OUTPUT_ROOT = OUT
        env_config = direct_ppo.build_env_config(config, selection)
        direct_ppo.write_yaml(env_config, OUT / "configs" / f"031_25_resolved_env_seed{seed}.yaml")
        t_rows, u_rows, c_rows = train_one_seed(config, env_config, meta, seed)
        train_rows.extend(t_rows)
        update_rows.extend(u_rows)
        checkpoint_rows.extend(c_rows)
        for ckpt in [int(x) for x in meta["checkpoint_steps"]]:
            if not checkpoint_path(seed, ckpt).exists():
                continue
            for year in [int(meta["train_year"]), int(meta["validation_year"])]:
                eval_rows.append(evaluate_checkpoint(config, env_config, seed, ckpt, year))

    train_df = pd.DataFrame(train_rows)
    update_df = pd.DataFrame(update_rows)
    ckpt_df = pd.DataFrame(checkpoint_rows)
    eval_df = pd.DataFrame(eval_rows)
    train_df.to_csv(OUT / "evaluation" / "031_25_training_summary.csv", index=False, encoding="utf-8-sig")
    update_df.to_csv(OUT / "logs" / "031_25_update_sample_log.csv", index=False, encoding="utf-8-sig")
    ckpt_df.to_csv(OUT / "evaluation" / "031_25_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT / "evaluation" / "031_25_checkpoint_eval_train_validation_summary.csv", index=False, encoding="utf-8-sig")
    gap_df = add_gaps(eval_df, baselines) if not eval_df.empty else pd.DataFrame()
    gap_df.to_csv(OUT / "evaluation" / "031_25_checkpoint_gap_train_validation_summary.csv", index=False, encoding="utf-8-sig")
    selected_df = select_checkpoints(gap_df, baselines, meta) if not gap_df.empty else pd.DataFrame()
    selected_df.to_csv(OUT / "evaluation" / "031_25_selected_checkpoints_from_validation.csv", index=False, encoding="utf-8-sig")

    final_rows: list[dict[str, Any]] = []
    for selected in selected_df.itertuples(index=False):
        seed = int(selected.seed)
        ckpt = int(selected.checkpoint_step)
        config = model_config(base_config, meta, seed)
        env_config_path = OUT / "configs" / f"031_25_resolved_env_seed{seed}.yaml"
        env_config = direct_ppo.load_yaml(env_config_path)
        for year in [int(meta["train_year"]), int(meta["validation_year"]), int(meta["test_year"])]:
            row = evaluate_checkpoint(config, env_config, seed, ckpt, year)
            row["selected_pass"] = bool(selected.selected_pass)
            row["selected_guardrail_pass"] = bool(selected.selected_guardrail_pass)
            row["selection_reason"] = str(selected.selection_reason)
            final_rows.append(row)
    final_df = pd.DataFrame(final_rows)
    final_gap_df = add_gaps(final_df, baselines) if not final_df.empty else pd.DataFrame()
    final_gap_df.to_csv(OUT / "evaluation" / "031_25_selected_final_eval_summary.csv", index=False, encoding="utf-8-sig")

    write_record(train_df, ckpt_df, gap_df, selected_df, final_gap_df)
    result = {
        "task": "031_25_ddqn_checkpoint_selection_sy_crossyear",
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "training_summary": str((OUT / "evaluation" / "031_25_training_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "selected_checkpoints": str((OUT / "evaluation" / "031_25_selected_checkpoints_from_validation.csv").relative_to(ROOT)).replace("\\", "/"),
        "final_eval_summary": str((OUT / "evaluation" / "031_25_selected_final_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "031_25_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not selected_df.empty:
        print("SELECTED")
        print(selected_df[["seed", "checkpoint_step", "selected_pass", "selected_guardrail_pass", "selection_reason", "validation_score", "gap_yield", "gap_wp_et", "gap_pfp_n", "winning_metrics"]].to_string(index=False))
    if not final_gap_df.empty:
        print("FINAL SELECTED EVAL")
        print(final_gap_df[["year", "seed", "checkpoint_step", "gap_yield", "gap_wp_et", "gap_pfp_n", "advisor_any_metric_strict_winner", "winning_metrics"]].to_string(index=False))


if __name__ == "__main__":
    main()


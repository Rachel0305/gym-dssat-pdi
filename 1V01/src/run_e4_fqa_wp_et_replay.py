"""Replay the completed FQ E4 smoke checkpoints and calculate WP_ET/PFP_N."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import forecast_engineered_observation_056_057 as forecast
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_e4_fqa_warmstart_weather_residual_2k as e4
from run_152E3_sya_dual_branch_noforecast_control import make_zero_weather_factory
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_e4_fqa_warmstart_weather_residual_2k import (
    CFG_PATHS,
    ROOT,
    SEED,
    STATION,
    configure_runtime,
    output_root,
    read_json,
)
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env

REPLAY_BASENAME = "182_183_FQA_e4_wp_et_replay"


def replay(kind: str, runtime_cfg: dict, env_cfg: dict, years: list[int]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    cfg = read_json(CFG_PATHS[kind])
    out = output_root(kind)
    inventory = pd.read_csv(out / "evaluation" / "training_checkpoint_inventory.csv", keep_default_na=False)
    model_path = ROOT / str(inventory.iloc[0]["model_path"])
    eval_df = pd.read_csv(out / "evaluation" / f"{cfg['task_id']}_checkpoint_validation_summary.csv", keep_default_na=False)
    rows = []
    # Reuse a completed snapshot replay when available. This keeps the report
    # rerunnable without spending another 20 DSSAT episodes.
    for year in years:
        target = out / "snapshots" / STATION / str(year) / "e4_policy"
        expected = eval_df[pd.to_numeric(eval_df["year"], errors="coerce").eq(int(year))].iloc[0]
        if target.exists() and (target / "Summary.OUT").exists():
            final_yield = float(expected["final_grnwt"])
            metrics = baseline.metrics_from_snapshot(target, final_yield)
            rows.append({"station_code": STATION, "site": "FQ", "year": int(year), "kind": kind, "final_grnwt": final_yield, "steps": 0, **metrics, "snapshot_path": str(target.relative_to(ROOT))})
    cached_years = {int(row["year"]) for row in rows}
    remaining_years = [int(year) for year in years if int(year) not in cached_years]
    if not remaining_years:
        result = pd.DataFrame(rows)
        result.to_csv(out / "evaluation" / f"{cfg['task_id']}_wp_et_replay_summary.csv", index=False, encoding="utf-8-sig")
        return result
    old_make_env = engine.base03222.base.make_env
    exp_cfg = read_json(CFG_PATHS["forecast"])
    make_env = forecast.make_forecast_env_factory(old_make_env, exp_cfg) if kind == "forecast" else make_zero_weather_factory(old_make_env)
    try:
        engine.base03222.base.make_env = make_env
        model = MaskablePPO.load(str(model_path), device="cpu")
        for year in remaining_years:
            env = make_env(runtime_cfg, env_cfg, STATION, int(year), SEED, f"E4_{kind}_{year}_wp_et_replay", evaluation=True)
            try:
                obs, info = env.reset()
                done = False
                steps = 0
                while not done and steps < int(runtime_cfg["runtime"]["max_steps"]):
                    action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
                    obs, _reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    steps += 1
                if not done:
                    raise RuntimeError(f"{kind} FQA{year} replay did not terminate")
                source_snapshot = snapshot_from_env(env)
                target = out / "snapshots" / STATION / str(year) / "e4_policy"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_snapshot, target)
                expected = eval_df[pd.to_numeric(eval_df["year"], errors="coerce").eq(int(year))].iloc[0]
                final_yield = float(expected["final_grnwt"])
                metrics = baseline.metrics_from_snapshot(target, final_yield)
                rows.append({"station_code": STATION, "site": "FQ", "year": int(year), "kind": kind, "final_grnwt": final_yield, "steps": steps, **metrics, "snapshot_path": str(target.relative_to(ROOT))})
            finally:
                env.close()
    finally:
        engine.base03222.base.make_env = old_make_env
    result = pd.DataFrame(rows)
    result.to_csv(out / "evaluation" / f"{cfg['task_id']}_wp_et_replay_summary.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    patch_contract()
    runtime_cfg, env_cfg, _selection, _train_years, valid_years = configure_runtime()
    all_rows = []
    for kind in ["forecast", "noforecast"]:
        df = replay(kind, runtime_cfg, env_cfg, valid_years)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ROOT / "benchmark_results" / f"{REPLAY_BASENAME}_summary.csv", index=False, encoding="utf-8-sig")
    means = combined.groupby("kind", as_index=False)[["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]].mean()
    means.to_csv(ROOT / "benchmark_results" / f"{REPLAY_BASENAME}_means.csv", index=False, encoding="utf-8-sig")
    result_path = ROOT / "benchmark_results" / f"{e4.RESULT_BASENAME}.json"
    if result_path.exists():
        smoke = json.loads(result_path.read_text(encoding="utf-8"))
        from sb3_contrib import MaskablePPO
        post_branch = {}
        for kind in ["forecast", "noforecast"]:
            inv = pd.read_csv(output_root(kind) / "evaluation" / "training_checkpoint_inventory.csv", keep_default_na=False)
            model = MaskablePPO.load(str(ROOT / inv.iloc[0]["model_path"]), device="cpu")
            weight = model.policy.features_extractor.weather_encoder[0].weight.detach()
            post_branch[kind] = {
                "weather_linear_weight_l2_after_adaptation": float(torch.linalg.vector_norm(weight).cpu()),
                "weather_linear_nonzero": bool(float(weight.norm().cpu()) > 1e-8),
            }
        smoke["post_adaptation_weather_branch"] = post_branch
        mean_records = means.to_dict(orient="records")
        mean_map = {str(row["kind"]): row for row in mean_records}
        smoke["wp_et_replay"] = {
            "means": mean_records,
            "delta_forecast_minus_noforecast": {
                col: float(mean_map["forecast"][col] - mean_map["noforecast"][col])
                for col in ["final_grnwt", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]
            },
            "summary_path": f"benchmark_results/{REPLAY_BASENAME}_summary.csv",
        }
        smoke["performance_gate"] = {
            "forecast_not_worse_than_paired_control_yield": bool(mean_map["forecast"]["final_grnwt"] >= mean_map["noforecast"]["final_grnwt"] - 1e-6),
            "forecast_not_worse_than_paired_control_wp_et": bool(mean_map["forecast"]["WP_ET_kg_m3"] >= mean_map["noforecast"]["WP_ET_kg_m3"] - 1e-6),
            "forecast_not_worse_than_paired_control_pfp_n": bool(mean_map["forecast"]["PFP_N_kg_kg"] >= mean_map["noforecast"]["PFP_N_kg_kg"] - 1e-6),
            "passed": bool(
                mean_map["forecast"]["final_grnwt"] >= mean_map["noforecast"]["final_grnwt"] - 1e-6
                and mean_map["forecast"]["WP_ET_kg_m3"] >= mean_map["noforecast"]["WP_ET_kg_m3"] - 1e-6
                and mean_map["forecast"]["PFP_N_kg_kg"] >= mean_map["noforecast"]["PFP_N_kg_kg"] - 1e-6
            ),
        }
        smoke["five_k_allowed"] = False
        result_path.write_text(json.dumps(smoke, indent=2, ensure_ascii=False), encoding="utf-8")
    print(means.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()

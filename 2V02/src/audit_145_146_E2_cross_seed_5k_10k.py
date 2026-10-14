"""Cross-seed E2 audit at 5K and 10K; no training and no reward changes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import ppo_safe_rendering
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map, collect_states, counterfactual_decisions
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import DualBranchBaseWeatherExtractor  # noqa: F401

RUNS = {
    0: ("144E2", ROOT / "benchmark_results/144E2_sya_originIC_dual_branch_weather_reader_10k_seed0"),
    1: ("145E2S1", ROOT / "benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1"),
    2: ("146E2S2", ROOT / "benchmark_results/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2"),
}
OUT = ROOT / "benchmark_results/147E2_sya_dual_branch_weather_reader_cross_seed_5k_10k_validation"
DOC = ROOT / "docs/2026-08-16_sya_E2_cross_seed_5k_10k_validation.md"


def metrics(frame: pd.DataFrame, step: int) -> dict:
    g = frame.loc[frame["checkpoint_step"].eq(step)].copy()
    return {
        "years": int(g["year"].nunique()),
        "mean_yield": float(g["final_grnwt"].mean()),
        "mean_irrigation": float(g["total_irrigation"].mean()),
        "mean_n": float(g["total_n"].mean()),
        "mean_pfp_n": float(g["PFP_N"].mean()),
        "unique_sequences": int(g["action_sequence"].nunique()),
    }


def strict(frame: pd.DataFrame, step: int) -> dict:
    g = frame.loc[frame["checkpoint_step"].eq(step)]
    mismatch = off_grid = after_dap1 = daily_rows = 0
    pairs: set[tuple[float, float]] = set()
    for _, row in g.iterrows():
        daily = pd.read_csv(ROOT / row["daily_csv_path"])
        cols = ["safe_action_amir", "safe_action_anfer", "season_cumulative_irrigation", "season_cumulative_n", "dap"]
        for col in cols:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        di = daily["season_cumulative_irrigation"].diff().fillna(daily["season_cumulative_irrigation"])
        dn = daily["season_cumulative_n"].diff().fillna(daily["season_cumulative_n"])
        mismatch += int((~np.isclose(di, daily["safe_action_amir"])).sum())
        mismatch += int((~np.isclose(dn, daily["safe_action_anfer"])).sum())
        off_grid += int((~daily["safe_action_amir"].isin([0, 15, 30, 45])).sum())
        off_grid += int((~daily["safe_action_anfer"].isin([0, 40, 80, 120])).sum())
        positive = daily["safe_action_amir"].gt(0) | daily["safe_action_anfer"].gt(0)
        after_dap1 += int((positive & daily["dap"].gt(1)).sum())
        pairs.update(zip(daily.loc[positive, "safe_action_amir"], daily.loc[positive, "safe_action_anfer"]))
        daily_rows += len(daily)
    return {
        "validation_rows": int(len(g)),
        "daily_rows": int(daily_rows),
        "transmission_mismatch": int(mismatch),
        "off_grid": int(off_grid),
        "positive_rows_after_dap1": int(after_dap1),
        "unique_nonzero_pairs": int(len(pairs)),
        "pairs": [f"I{i:g}/N{n:g}" for i, n in sorted(pairs)],
        "passed": bool(len(g) == 10 and mismatch == 0 and off_grid == 0 and after_dap1 > 0 and len(pairs) >= 2),
    }


def response(decisions: pd.DataFrame) -> dict:
    swap = decisions["swap_changed"].astype(bool)
    shuffled = decisions["shuffled_changed"].astype(bool)
    swap_rate = float(swap.mean())
    shuffled_rate = float(shuffled.mean())
    return {
        "states": int(len(decisions)),
        "swap_rows": int(swap.sum()),
        "swap_rate": swap_rate,
        "shuffled_rows": int(shuffled.sum()),
        "shuffled_rate": shuffled_rate,
        "irrigation_changed": int((~np.isclose(decisions["real_i"], decisions["swap_i"])).sum()),
        "nitrogen_changed": int((~np.isclose(decisions["real_n"], decisions["swap_n"])).sum()),
        "passed_gt_1pct": bool(max(swap_rate, shuffled_rate) > 0.01),
    }


def model_path(run: Path, seed: int, step: int) -> Path:
    return run / f"models/SYA/SYA_half_split_stress_aware_maskableppo_seed{seed}_ckpt{step}.zip"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    patch_contract()

    rows: list[dict] = []
    details: dict[str, dict] = {}
    for seed, (task, run) in RUNS.items():
        csv_path = run / f"evaluation/{task}_checkpoint_validation_summary.csv"
        frame = pd.read_csv(csv_path)
        if set(frame["seed"].astype(int)) != {seed} or len(frame) != 30:
            raise RuntimeError(f"seed/row mismatch: {csv_path}")
        details[str(seed)] = {}
        for step in (5000, 10000):
            met = metrics(frame, step)
            act = strict(frame, step)
            details[str(seed)][str(step)] = {"metrics": met, "strict_action": act}
            rows.append({"seed": seed, "checkpoint": step, **met, **{f"strict_{k}": v for k, v in act.items() if k != "pairs"}})

    old = {k: getattr(forecast.engine, k) for k in ["LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make = forecast.engine.base03222.base.make_env
    old_seed = forecast.engine.base03222.SEED
    try:
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.STATION = "SYA"
        forecast.engine.SITES = ["SYA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = [0, 15, 30, 45]
        forecast.engine.BINARY_NITROGEN_LEVELS = [0, 40, 80, 120]
        forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        for seed in (1, 2):
            task, run = RUNS[seed]
            cfg = forecast.read_json(ROOT / f"configs/{task}_sya_originIC_dual_branch_weather_reader_10k_seed{seed}.json")
            forecast.engine.base03222.SEED = seed
            make_env = forecast.make_forecast_env_factory(old_make, cfg)
            probe = make_env(runtime_config, env_config, "SYA", 2014, seed, f"{task}_probe", evaluation=True)
            try:
                grid = action_map(probe)
            finally:
                probe.close()
            for step in (5000, 10000):
                model = MaskablePPO.load(str(model_path(run, seed, step)), device="cpu")
                states = collect_states(model, make_env, runtime_config, env_config, "SYA", seed)
                decisions = counterfactual_decisions(states, model, grid, seed)
                decisions.to_csv(OUT / f"{task}_seed{seed}_ckpt{step}_counterfactual.csv", index=False, encoding="utf-8-sig")
                details[str(seed)][str(step)]["weather_response"] = response(decisions)
    finally:
        forecast.engine.base03222.SEED = old_seed
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make
        forecast.engine.base03222.SITE_NAMES.clear()
        forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items():
            setattr(forecast.engine, key, value)

    seed0_5 = json.loads((ROOT / "benchmark_results/143E2_sya_dual_branch_weather_reader_5k_validation/143E2_5k_validation_summary.json").read_text(encoding="utf-8"))["five_k_weather_response"]
    seed0_10 = json.loads((ROOT / "benchmark_results/144E2_sya_dual_branch_weather_reader_10k_seed0_validation/144E2_seed0_10k_summary.json").read_text(encoding="utf-8"))["weather_response_10k"]
    details["0"]["5000"]["weather_response"] = {
        "states": seed0_5["decision_states"], "swap_rows": seed0_5["swap_changed_rows"], "swap_rate": seed0_5["swap_action_change_rate"],
        "shuffled_rows": seed0_5["shuffled_changed_rows"], "shuffled_rate": seed0_5["shuffled_action_change_rate"],
        "irrigation_changed": seed0_5["swap_irrigation_changed_rows"], "nitrogen_changed": seed0_5["swap_nitrogen_changed_rows"],
        "passed_gt_1pct": seed0_5["response_passed_gt_1pct"],
    }
    details["0"]["10000"]["weather_response"] = seed0_10

    summary_rows = []
    collapse_by_seed = {}
    for seed in (0, 1, 2):
        m5 = details[str(seed)]["5000"]["metrics"]
        m10 = details[str(seed)]["10000"]["metrics"]
        for step in (5000, 10000):
            item = details[str(seed)][str(step)]
            summary_rows.append({
                "seed": seed, "checkpoint": step, **item["metrics"],
                "unique_nonzero_pairs": item["strict_action"]["unique_nonzero_pairs"],
                "strict_action_passed": item["strict_action"]["passed"],
                "weather_swap_rate": item["weather_response"]["swap_rate"],
                "weather_shuffled_rate": item["weather_response"]["shuffled_rate"],
                "weather_response_passed_gt_1pct": item["weather_response"]["passed_gt_1pct"],
            })
        change = (m10["mean_yield"] - m5["mean_yield"]) / m5["mean_yield"]
        r10 = details[str(seed)]["10000"]["weather_response"]
        a10 = details[str(seed)]["10000"]["strict_action"]
        collapse_by_seed[str(seed)] = {
            "weather_response_collapsed": not r10["passed_gt_1pct"],
            "action_sequence_collapsed": bool(m10["unique_sequences"] <= 1 or a10["unique_nonzero_pairs"] < 2),
            "yield_collapsed_gt_5pct": bool(change < -0.05),
            "yield_change_5k_to_10k_fraction": float(change),
        }
        collapse_by_seed[str(seed)]["any_collapse"] = any(collapse_by_seed[str(seed)][k] for k in ["weather_response_collapsed", "action_sequence_collapsed", "yield_collapsed_gt_5pct"])

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(OUT / "147E2_cross_seed_5k_10k_summary.csv", index=False, encoding="utf-8-sig")
    stability = {
        "five_k_weather_response_pass_seeds": int(summary_frame.query("checkpoint == 5000")["weather_response_passed_gt_1pct"].sum()),
        "five_k_action_diversity_pass_seeds": int(summary_frame.query("checkpoint == 5000")["strict_action_passed"].sum()),
        "ten_k_weather_response_pass_seeds": int(summary_frame.query("checkpoint == 10000")["weather_response_passed_gt_1pct"].sum()),
        "ten_k_action_diversity_pass_seeds": int(summary_frame.query("checkpoint == 10000")["strict_action_passed"].sum()),
        "ten_k_any_collapse_seeds": int(sum(v["any_collapse"] for v in collapse_by_seed.values())),
        "five_k_fully_cross_seed_stable": bool(summary_frame.query("checkpoint == 5000")["weather_response_passed_gt_1pct"].all() and summary_frame.query("checkpoint == 5000")["strict_action_passed"].all()),
        "ten_k_collapse_reproduced_in_majority": bool(sum(v["any_collapse"] for v in collapse_by_seed.values()) >= 2),
    }
    summary = {"status": "completed_cross_seed_5k_10k_audit", "details": details, "collapse_by_seed": collapse_by_seed, "stability": stability, "wp_et": "unavailable_without_valid_ETCP", "no_forecast_scope": "seed0 comparison only; no seed1/2 no-forecast models were trained"}
    (OUT / "147E2_cross_seed_5k_10k_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E2 5K/10K 跨-seed验证", "", "| seed | checkpoint | yield | I | N | PFP_N | sequences | pairs | weather swap | weather shuffle |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['seed']} | {row['checkpoint']} | {row['mean_yield']:.2f} | {row['mean_irrigation']:.1f} | {row['mean_n']:.1f} | {row['mean_pfp_n']:.3f} | {row['unique_sequences']} | {row['unique_nonzero_pairs']} | {row['weather_swap_rate']:.3%} | {row['weather_shuffled_rate']:.3%} |")
    lines += ["", f"- 5K天气响应通过seed数：{stability['five_k_weather_response_pass_seeds']}/3。", f"- 10K任一坍缩seed数：{stability['ten_k_any_collapse_seeds']}/3。", f"- 5K完全跨seed稳定：{stability['five_k_fully_cross_seed_stable']}。", f"- 10K坍缩是否在多数seed复现：{stability['ten_k_collapse_reproduced_in_majority']}。", "", "no-forecast只存在seed0对照，本审计不把它外推到seed1/2。WP_ET unavailable。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize existing E2/no-forecast 5K cross-seed evidence, including WP_ET."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASK = "154E2_sya_cross_seed_5k_wp_et_stability"
OUT = ROOT / "benchmark_results" / TASK
PROMPT = ROOT / "prompts/2026-08-16_sya_E2_cross_seed_5k_wp_et_stability.md"
WP_ET = ROOT / "benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_by_policy_year.csv"
WP_ET_CHECK = ROOT / "benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_reproducibility_check.csv"
CROSS_SEED = ROOT / "benchmark_results/147E2_sya_dual_branch_weather_reader_cross_seed_5k_10k_validation/147E2_cross_seed_5k_10k_summary.json"
SUMMARY_FILES = {
    "e2_s0": ROOT / "benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k/evaluation/143E2_checkpoint_validation_summary.csv",
    "e2_s1": ROOT / "benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1/evaluation/145E2S1_checkpoint_validation_summary.csv",
    "e2_s2": ROOT / "benchmark_results/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2/evaluation/146E2S2_checkpoint_validation_summary.csv",
    "nof_s0": ROOT / "benchmark_results/140N_sya_originIC_paired_noforecast_paired5k/evaluation/140N_checkpoint_validation_summary.csv",
    "nof_s1": ROOT / "benchmark_results/148N1_sya_originIC_paired_noforecast_5k_seed1/evaluation/148N1_checkpoint_validation_summary.csv",
    "nof_s2": ROOT / "benchmark_results/149N2_sya_originIC_paired_noforecast_5k_seed2/evaluation/149N2_checkpoint_validation_summary.csv",
}


def load_validation(policy: str) -> pd.DataFrame:
    path = SUMMARY_FILES[policy]
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["checkpoint_step"].eq(5000)].copy()
    if len(frame) != 10:
        raise RuntimeError(f"{policy}: expected 10 checkpoint=5000 rows, got {len(frame)}")
    return frame


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "years": int(frame["year"].nunique()),
        "mean_yield": float(frame["final_grnwt"].mean()),
        "mean_irrigation": float(frame["total_irrigation"].mean()),
        "mean_n": float(frame["total_n"].mean()),
        "mean_pfp_n": float(frame["PFP_N"].mean()),
        "unique_sequences": int(frame["action_sequence"].nunique()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    if not PROMPT.exists() or not WP_ET.exists() or not WP_ET_CHECK.exists() or not CROSS_SEED.exists():
        raise FileNotFoundError("E2 cross-seed evidence file missing")

    validation = {policy: load_validation(policy) for policy in SUMMARY_FILES}
    replay = pd.read_csv(WP_ET, keep_default_na=False)
    replay = replay.loc[replay["checkpoint_step"].eq(5000)].copy()
    if set(replay["policy"].unique()) != set(SUMMARY_FILES):
        raise RuntimeError(f"WP_ET policy set mismatch: {sorted(replay['policy'].unique())}")
    check = pd.read_csv(WP_ET_CHECK, keep_default_na=False)
    check_cols = [c for c in check.columns if c.endswith("_abs_diff_vs_original")]
    check_summary = {
        "rows": int(len(check)),
        "max_abs_diff": {c: float(pd.to_numeric(check[c], errors="coerce").max()) for c in check_cols},
        "action_sequence_all_match": bool(check["action_sequence_match"].astype(str).str.lower().eq("true").all()),
    }

    cross = json.loads(CROSS_SEED.read_text(encoding="utf-8"))
    five_k_details = cross["details"]
    rows = []
    pair_rows = []
    for seed in (0, 1, 2):
        e2_key, nof_key = f"e2_s{seed}", f"nof_s{seed}"
        e2v, nofv = validation[e2_key], validation[nof_key]
        e2r = replay.loc[replay["policy"].eq(e2_key)].copy()
        nofr = replay.loc[replay["policy"].eq(nof_key)].copy()
        e2m, nofm = metrics(e2v), metrics(nofv)
        e2wp = {"mean_etcp_mm": float(e2r["etcp_mm"].mean()), "mean_wp_et": float(e2r["WP_ET_kg_m3"].mean()), "weighted_wp_et": float(e2r["final_grnwt"].sum() / (e2r["etcp_mm"].sum() * 10.0))}
        nofwp = {"mean_etcp_mm": float(nofr["etcp_mm"].mean()), "mean_wp_et": float(nofr["WP_ET_kg_m3"].mean()), "weighted_wp_et": float(nofr["final_grnwt"].sum() / (nofr["etcp_mm"].sum() * 10.0))}
        e2_details = five_k_details[str(seed)]["5000"]
        rows.extend([
            {"policy": e2_key, **e2m, **{k: e2wp[k] for k in ["mean_etcp_mm", "mean_wp_et", "weighted_wp_et"]}, "weather_response_passed": bool(e2_details["weather_response"]["passed_gt_1pct"]), "action_diversity_passed": bool(e2_details["strict_action"]["passed"])},
            {"policy": nof_key, **nofm, **{k: nofwp[k] for k in ["mean_etcp_mm", "mean_wp_et", "weighted_wp_et"]}, "weather_response_passed": None, "action_diversity_passed": None},
        ])
        e2_by_year = e2r[["year", "final_grnwt", "PFP_N_kg_kg", "WP_ET_kg_m3"]].rename(columns={"final_grnwt": "e2_yield", "PFP_N_kg_kg": "e2_pfp_n", "WP_ET_kg_m3": "e2_wp_et"})
        nof_by_year = nofr[["year", "final_grnwt", "PFP_N_kg_kg", "WP_ET_kg_m3"]].rename(columns={"final_grnwt": "nof_yield", "PFP_N_kg_kg": "nof_pfp_n", "WP_ET_kg_m3": "nof_wp_et"})
        pair = e2_by_year.merge(nof_by_year, on="year", validate="one_to_one")
        pair["seed"] = seed
        pair["yield_delta_e2_minus_nof"] = pair["e2_yield"] - pair["nof_yield"]
        pair["pfp_n_delta_e2_minus_nof"] = pair["e2_pfp_n"] - pair["nof_pfp_n"]
        pair["wp_et_delta_e2_minus_nof"] = pair["e2_wp_et"] - pair["nof_wp_et"]
        pair_rows.append(pair)

    summary = pd.DataFrame(rows)
    pairs = pd.concat(pair_rows, ignore_index=True)
    pooled = {
        "e2_mean_yield": float(summary.loc[summary.policy.str.startswith("e2_"), "mean_yield"].mean()),
        "nof_mean_yield": float(summary.loc[summary.policy.str.startswith("nof_"), "mean_yield"].mean()),
        "e2_mean_pfp_n": float(summary.loc[summary.policy.str.startswith("e2_"), "mean_pfp_n"].mean()),
        "nof_mean_pfp_n": float(summary.loc[summary.policy.str.startswith("nof_"), "mean_pfp_n"].mean()),
        "e2_mean_wp_et": float(summary.loc[summary.policy.str.startswith("e2_"), "mean_wp_et"].mean()),
        "nof_mean_wp_et": float(summary.loc[summary.policy.str.startswith("nof_"), "mean_wp_et"].mean()),
        "e2_weighted_wp_et": float(replay.loc[replay.policy.str.startswith("e2_"), "final_grnwt"].sum() / (replay.loc[replay.policy.str.startswith("e2_"), "etcp_mm"].sum() * 10.0)),
        "nof_weighted_wp_et": float(replay.loc[replay.policy.str.startswith("nof_"), "final_grnwt"].sum() / (replay.loc[replay.policy.str.startswith("nof_"), "etcp_mm"].sum() * 10.0)),
        "wp_et_winning_seeds": int(sum(float(pairs.loc[pairs.seed.eq(seed), "wp_et_delta_e2_minus_nof"].mean()) > 0 for seed in (0, 1, 2))),
        "yield_winning_seeds": int(sum(float(pairs.loc[pairs.seed.eq(seed), "yield_delta_e2_minus_nof"].mean()) > 0 for seed in (0, 1, 2))),
        "pfp_n_winning_seeds": int(sum(float(pairs.loc[pairs.seed.eq(seed), "pfp_n_delta_e2_minus_nof"].mean()) > 0 for seed in (0, 1, 2))),
    }
    pair_summary = pairs.groupby("seed", as_index=False).agg(
        mean_yield_delta=("yield_delta_e2_minus_nof", "mean"),
        mean_pfp_n_delta=("pfp_n_delta_e2_minus_nof", "mean"),
        mean_wp_et_delta=("wp_et_delta_e2_minus_nof", "mean"),
        yield_win_years=("yield_delta_e2_minus_nof", lambda s: int((s > 0).sum())),
        pfp_n_win_years=("pfp_n_delta_e2_minus_nof", lambda s: int((s > 0).sum())),
        wp_et_win_years=("wp_et_delta_e2_minus_nof", lambda s: int((s > 0).sum())),
    )
    summary.to_csv(OUT / "154E2_policy_summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "154E2_paired_by_year.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(OUT / "154E2_paired_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    check_summary_path = OUT / "154E2_wp_et_reproducibility_check.json"
    check_summary_path.write_text(json.dumps(check_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    stability = {
        "five_k_weather_response_pass_seeds": int(sum(bool(five_k_details[str(seed)]["5000"]["weather_response"]["passed_gt_1pct"]) for seed in (0, 1, 2))),
        "five_k_action_diversity_pass_seeds": int(sum(bool(five_k_details[str(seed)]["5000"]["strict_action"]["passed"]) for seed in (0, 1, 2))),
        "five_k_full_weather_and_action_stability": bool(all(bool(five_k_details[str(seed)]["5000"]["weather_response"]["passed_gt_1pct"]) and bool(five_k_details[str(seed)]["5000"]["strict_action"]["passed"]) for seed in (0, 1, 2))),
        "wp_et_replay_rows": int(len(replay)),
        "wp_et_replay_failures": int(replay["run_status"].astype(str).str.startswith("ok").eq(False).sum()) if "run_status" in replay.columns else 0,
    }
    result = {"task": TASK, "prompt": str(PROMPT.relative_to(ROOT)).replace("\\", "/"), "scope": "existing frozen E2/no-forecast 5K checkpoints; no training", "pooled": pooled, "stability": stability, "wp_et_reproducibility": check_summary}
    (OUT / "154E2_cross_seed_5k_wp_et_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# E2 跨 seed 5K 稳定性与 WP_ET 核验",
        "",
        "本报告复用既有冻结 checkpoint，不重新训练。",
        "",
        "## 每个 seed 的 E2 与 no-forecast 配对",
        "",
        pair_summary.round(4).to_markdown(index=False),
        "",
        "## pooled",
        "",
        f"- E2 平均 WP_ET：{pooled['e2_mean_wp_et']:.4f}；no-forecast：{pooled['nof_mean_wp_et']:.4f}。",
        f"- E2 加权 WP_ET：{pooled['e2_weighted_wp_et']:.4f}；no-forecast：{pooled['nof_weighted_wp_et']:.4f}。",
        f"- E2 平均产量胜出 seed：{pooled['yield_winning_seeds']}/3；PFP-N：{pooled['pfp_n_winning_seeds']}/3；WP_ET：{pooled['wp_et_winning_seeds']}/3。",
        "",
        "## 稳定性门禁",
        "",
        f"- 5K 天气响应通过：{stability['five_k_weather_response_pass_seeds']}/3。",
        f"- 5K 动作多样性通过：{stability['five_k_action_diversity_pass_seeds']}/3。",
        f"- 5K 天气响应和动作多样性同时 3/3：{stability['five_k_full_weather_and_action_stability']}。",
        "",
        "## 结论",
        "",
        "E2 是有希望的 forecast 候选：部分 seed 的产量/PFP-N 优于 no-forecast，且 5K 动作多样性通过。但 WP_ET 没有跨 seed 稳定占优，天气响应也只有 2/3 seed 通过，因此不能宣称 E2 已经全面、稳定优于 no-forecast。",
        "",
        "- 逐年配对：`154E2_paired_by_year.csv`",
        "- 每 seed 汇总：`154E2_paired_summary_by_seed.csv`",
        "- WP_ET 闭合核验：`154E2_wp_et_reproducibility_check.json`",
    ]
    (OUT / "2026-08-16_sya_E2_cross_seed_5k_wp_et_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

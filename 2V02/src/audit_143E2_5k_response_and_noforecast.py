"""Audit 143E2 2K reproducibility, 5K weather response, and no-forecast gap."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import ppo_safe_rendering
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map, collect_states, counterfactual_decisions
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import DualBranchBaseWeatherExtractor
from run_143E2_sya_dual_branch_weather_reader_5k import CONFIG, PROMPT, isolation_audit, verify_142e2_gate


RUN = ROOT / "benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k"
SMOKE = ROOT / "benchmark_results/142E2_sya_originIC_dual_branch_weather_reader_smoke2k"
NOF = ROOT / "benchmark_results/140N_sya_originIC_paired_noforecast_paired5k"
OUT = ROOT / "benchmark_results/143E2_sya_dual_branch_weather_reader_5k_validation"
DOC = ROOT / "docs/2026-08-16_sya_E2_dual_branch_weather_reader_5k_validation.md"
EVAL = RUN / "evaluation/143E2_checkpoint_validation_summary.csv"
SMOKE_EVAL = SMOKE / "evaluation/142E2_checkpoint_validation_summary.csv"
NOF_EVAL = NOF / "evaluation/140N_checkpoint_validation_summary.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_model(path: Path) -> MaskablePPO:
    model = MaskablePPO.load(str(path), device="cpu")
    extractor = model.policy.features_extractor
    if not (isinstance(extractor, DualBranchBaseWeatherExtractor) or type(extractor).__name__ == "DualBranchBaseWeatherExtractor"):
        raise RuntimeError(f"wrong extractor: {type(extractor)}")
    return model


def policy_reproducibility(a: MaskablePPO, b: MaskablePPO) -> dict[str, Any]:
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    same_keys = list(sa) == list(sb)
    exact = same_keys and all(torch.equal(sa[key].cpu(), sb[key].cpu()) for key in sa)
    max_delta = max((float(torch.max(torch.abs(sa[key].cpu() - sb[key].cpu())).item()) for key in sa), default=0.0) if same_keys else float("inf")
    smoke_eval = pd.read_csv(SMOKE_EVAL)
    run_eval = pd.read_csv(EVAL)
    smoke_2k = smoke_eval[smoke_eval["checkpoint_step"].eq(2000)].sort_values("year")
    run_2k = run_eval[run_eval["checkpoint_step"].eq(2000)].sort_values("year")
    metric_cols = ["final_grnwt", "total_irrigation", "total_n", "PFP_N"]
    metrics_exact = len(smoke_2k) == len(run_2k) == 10 and all(np.allclose(pd.to_numeric(smoke_2k[c]), pd.to_numeric(run_2k[c]), atol=0.0, rtol=0.0) for c in metric_cols)
    sequences_exact = smoke_2k["action_sequence"].tolist() == run_2k["action_sequence"].tolist()
    return {"policy_state_keys_match": same_keys, "policy_tensors_exact": exact, "max_policy_parameter_abs_delta": max_delta, "ten_year_metrics_exact": metrics_exact, "ten_year_action_sequences_exact": sequences_exact, "passed": bool(exact and metrics_exact and sequences_exact)}


def strict_action_audit(frame: pd.DataFrame, checkpoint: int) -> dict[str, Any]:
    final = frame[frame["checkpoint_step"].eq(checkpoint)]
    pairs: set[tuple[float, float]] = set()
    mismatch = off_grid = after_dap1 = rows = 0
    i_grid, n_grid = {0.0, 15.0, 30.0, 45.0}, {0.0, 40.0, 80.0, 120.0}
    for _, record in final.iterrows():
        daily = pd.read_csv(ROOT / str(record["daily_csv_path"]))
        for col in ["safe_action_amir", "safe_action_anfer", "season_cumulative_irrigation", "season_cumulative_n", "dap"]:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        rows += len(daily)
        di = daily["season_cumulative_irrigation"].diff().fillna(daily["season_cumulative_irrigation"])
        dn = daily["season_cumulative_n"].diff().fillna(daily["season_cumulative_n"])
        mismatch += int((~np.isclose(di, daily["safe_action_amir"], atol=1e-6)).sum() + (~np.isclose(dn, daily["safe_action_anfer"], atol=1e-6)).sum())
        off_grid += int((~daily["safe_action_amir"].isin(i_grid)).sum() + (~daily["safe_action_anfer"].isin(n_grid)).sum())
        positive = daily["safe_action_amir"].gt(0) | daily["safe_action_anfer"].gt(0)
        after_dap1 += int((positive & daily["dap"].gt(1)).sum())
        pairs.update((float(i), float(n)) for i, n in zip(daily.loc[positive, "safe_action_amir"], daily.loc[positive, "safe_action_anfer"]))
    return {"checkpoint_step": checkpoint, "validation_rows": int(len(final)), "daily_rows": rows, "transmission_mismatch_fields": mismatch, "off_grid_fields": off_grid, "positive_rows_after_dap1": after_dap1, "unique_nonzero_pairs": len(pairs), "nonzero_pairs": [f"I{i:g}/N{n:g}" for i, n in sorted(pairs)], "passed": bool(len(final) == 10 and mismatch == 0 and off_grid == 0 and after_dap1 > 0 and len(pairs) >= 2)}


def endpoint_comparison() -> tuple[pd.DataFrame, dict[str, Any]]:
    e2 = pd.read_csv(EVAL)
    nof = pd.read_csv(NOF_EVAL)
    e2_5 = e2[e2["checkpoint_step"].eq(5000)].copy()
    nof_5 = nof[nof["checkpoint_step"].eq(5000)].copy()
    paired = e2_5[["year", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "action_sequence", "daily_csv_path"]].merge(nof_5[["year", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "action_sequence", "daily_csv_path"]], on="year", suffixes=("_e2", "_nof"), validate="one_to_one")
    paired["yield_delta_e2_minus_nof"] = paired["final_grnwt_e2"] - paired["final_grnwt_nof"]
    paired["pfp_n_delta_e2_minus_nof"] = paired["PFP_N_e2"] - paired["PFP_N_nof"]
    changed = []
    for _, row in paired.iterrows():
        a = pd.read_csv(ROOT / row["daily_csv_path_e2"])
        b = pd.read_csv(ROOT / row["daily_csv_path_nof"])
        aligned = a[["dap", "safe_action_amir", "safe_action_anfer"]].merge(b[["dap", "safe_action_amir", "safe_action_anfer"]], on="dap", suffixes=("_e2", "_nof"))
        diff = ~np.isclose(aligned["safe_action_amir_e2"], aligned["safe_action_amir_nof"]) | ~np.isclose(aligned["safe_action_anfer_e2"], aligned["safe_action_anfer_nof"])
        changed.append(float(diff.mean()))
    paired["action_change_rate_e2_vs_nof"] = changed
    summary = {"years": int(len(paired)), "mean_e2_yield": float(paired["final_grnwt_e2"].mean()), "mean_nof_yield": float(paired["final_grnwt_nof"].mean()), "mean_yield_delta": float(paired["yield_delta_e2_minus_nof"].mean()), "yield_win_years": int(paired["yield_delta_e2_minus_nof"].gt(0).sum()), "mean_e2_irrigation": float(paired["total_irrigation_e2"].mean()), "mean_nof_irrigation": float(paired["total_irrigation_nof"].mean()), "mean_e2_n": float(paired["total_n_e2"].mean()), "mean_nof_n": float(paired["total_n_nof"].mean()), "mean_e2_pfp_n": float(paired["PFP_N_e2"].mean()), "mean_nof_pfp_n": float(paired["PFP_N_nof"].mean()), "pfp_n_win_years": int(paired["pfp_n_delta_e2_minus_nof"].gt(0).sum()), "e2_unique_sequences": int(paired["action_sequence_e2"].nunique()), "nof_unique_sequences": int(paired["action_sequence_nof"].nunique()), "mean_action_change_rate_e2_vs_nof": float(paired["action_change_rate_e2_vs_nof"].mean())}
    return paired, summary


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    OUT.mkdir(parents=True)
    patch_contract()
    cfg = forecast.read_json(CONFIG)
    model_2k = load_model(RUN / "models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip")
    smoke_model_2k = load_model(SMOKE / "models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip")
    model_5k = load_model(RUN / "models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt5000.zip")
    reproduction = policy_reproducibility(smoke_model_2k, model_2k)
    run_eval = pd.read_csv(EVAL)
    strict_2k, strict_5k = strict_action_audit(run_eval, 2000), strict_action_audit(run_eval, 5000)
    paired, endpoint = endpoint_comparison()
    paired.to_csv(OUT / "143E2_5k_vs_noforecast_by_year.csv", index=False, encoding="utf-8-sig")

    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root, old_make_env = ppo_safe_rendering.MULTISITE_INPUT_ROOT, forecast.engine.base03222.base.make_env
    try:
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.STATION, forecast.engine.SITES = "SYA", ["SYA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        states = collect_states(model_5k, make_env, runtime_config, env_config, "SYA", int(cfg["seed"]))
        states.to_json(OUT / "143E2_5k_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", 2014, int(cfg["seed"]), "143E2_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        decisions = counterfactual_decisions(states, model_5k, grid, int(cfg["seed"]))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT, forecast.engine.base03222.base.make_env = old_root, old_make_env
        forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items(): setattr(forecast.engine, key, value)

    decisions.to_csv(OUT / "143E2_5k_counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
    swap = decisions["swap_changed"]
    shuf = decisions["shuffled_changed"]
    swap_i = ~np.isclose(decisions["real_i"], decisions["swap_i"])
    swap_n = ~np.isclose(decisions["real_n"], decisions["swap_n"])
    response = {"decision_states": int(len(decisions)), "swap_changed_rows": int(swap.sum()), "swap_action_change_rate": float(swap.mean()), "shuffled_changed_rows": int(shuf.sum()), "shuffled_action_change_rate": float(shuf.mean()), "swap_irrigation_changed_rows": int(swap_i.sum()), "swap_nitrogen_changed_rows": int(swap_n.sum()), "all_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()), "response_passed_gt_1pct": bool(max(float(swap.mean()), float(shuf.mean())) > 0.01)}
    retain = bool(isolation_audit()["passed"] and verify_142e2_gate({})["next_step_allowed"] and reproduction["passed"] and strict_5k["passed"] and response["response_passed_gt_1pct"] and endpoint["mean_yield_delta"] >= 0 and endpoint["mean_e2_pfp_n"] >= endpoint["mean_nof_pfp_n"] and endpoint["e2_unique_sequences"] > 1)
    summary = {"status": "completed_143E2_5k_validation", "isolation_audit": isolation_audit(), "authorization_gate": verify_142e2_gate({}), "two_k_reproducibility": reproduction, "strict_action_audit": {"ckpt2000": strict_2k, "ckpt5000": strict_5k}, "five_k_weather_response": response, "five_k_vs_noforecast": endpoint, "wp_et": "unavailable_without_valid_Summary.OUT_ETCP; not inferred", "retain_E2_for_independent_seed_validation": retain, "longer_training_executed": False}
    (OUT / "143E2_5k_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text("\n".join(["# 143E2 SY 双通道天气读取 5K 验证记录", "", "## 核心结论", "", f"- 新运行2K精确复现：`{reproduction['passed']}`；5K天气响应通过：`{response['response_passed_gt_1pct']}`。", f"- E2 5K相对no-forecast平均产量差：`{endpoint['mean_yield_delta']:.2f}` kg/ha；逐年产量胜出：`{endpoint['yield_win_years']}/10`。", f"- E2/no-forecast PFP_N：`{endpoint['mean_e2_pfp_n']:.4f}` / `{endpoint['mean_nof_pfp_n']:.4f}`；动作序列种类：`{endpoint['e2_unique_sequences']}` / `{endpoint['nof_unique_sequences']}`。", f"- 保留进入独立seed验证：`{retain}`；未运行10K或更长训练。", "", "## 5K天气响应", "", "```json", json.dumps(response, ensure_ascii=False, indent=2), "```", "", "## 严格动作审计", "", "```json", json.dumps(strict_5k, ensure_ascii=False, indent=2), "```", "", "## 解释边界", "", "5K仍为单seed诊断。天气交换只评估同状态策略敏感性，反事实动作未执行进DSSAT。WP_ET因缺少有效ETCP保持unavailable。", ""]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

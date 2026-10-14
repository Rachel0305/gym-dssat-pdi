"""Validate 142E2 architecture, smoke outputs, and same-state weather response."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

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
from run_142E2_sya_dual_branch_weather_reader_smoke2k import (
    CONFIG,
    PROMPT,
    DualBranchBaseWeatherExtractor,
    isolation_audit,
)


SMOKE = ROOT / "benchmark_results/142E2_sya_originIC_dual_branch_weather_reader_smoke2k"
MODEL = SMOKE / "models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
OUT = ROOT / "benchmark_results/142E2_sya_dual_branch_weather_reader_validation"
DOC = ROOT / "docs/2026-08-16_sya_E2_dual_branch_weather_reader_validation.md"
SUMMARY_PATHS = {
    "E2_dual_branch": SMOKE / "evaluation/142E2_checkpoint_validation_summary.csv",
    "E1_single_reader": ROOT / "benchmark_results/141E1_sya_originIC_actionable_weather_encoding_smoke2k/evaluation/141E1_checkpoint_validation_summary.csv",
    "140F_flat28": ROOT / "benchmark_results/140F_sya_originIC_paired_forecast_paired2k/evaluation/140F_checkpoint_validation_summary.csv",
    "140N_no_forecast": ROOT / "benchmark_results/140N_sya_originIC_paired_noforecast_paired2k/evaluation/140N_checkpoint_validation_summary.csv",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def metric_comparison() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch, path in SUMMARY_PATHS.items():
        frame = pd.read_csv(path)
        for checkpoint, group in frame.groupby("checkpoint_step"):
            rows.append(
                {
                    "branch": branch,
                    "checkpoint_step": int(checkpoint),
                    "validation_years": int(group["year"].nunique()),
                    "mean_yield_kg_ha": float(pd.to_numeric(group["final_grnwt"], errors="coerce").mean()),
                    "mean_irrigation_mm": float(pd.to_numeric(group["total_irrigation"], errors="coerce").mean()),
                    "mean_n_kg_ha": float(pd.to_numeric(group["total_n"], errors="coerce").mean()),
                    "mean_pfp_n": float(pd.to_numeric(group["PFP_N"], errors="coerce").mean()),
                    "mean_irrigation_events": float(pd.to_numeric(group["irrigation_event_count"], errors="coerce").mean()),
                    "mean_n_events": float(pd.to_numeric(group["n_event_count"], errors="coerce").mean()),
                    "unique_action_sequences": int(group["action_sequence"].astype(str).nunique()),
                }
            )
    return pd.DataFrame(rows)


def strict_action_audit() -> dict[str, Any]:
    summary = pd.read_csv(SUMMARY_PATHS["E2_dual_branch"])
    final = summary[summary["checkpoint_step"].eq(2000)].copy()
    i_grid, n_grid = {0.0, 15.0, 30.0, 45.0}, {0.0, 40.0, 80.0, 120.0}
    pairs: set[tuple[float, float]] = set()
    raw_safe_mismatch = cumulative_mismatch = off_grid = positive_after_dap1 = daily_rows = 0
    for _, record in final.iterrows():
        daily = pd.read_csv(ROOT / str(record["daily_csv_path"]))
        cols = ["raw_action_amir", "raw_action_anfer", "safe_action_amir", "safe_action_anfer", "season_cumulative_irrigation", "season_cumulative_n", "dap"]
        for col in cols:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        daily_rows += len(daily)
        raw_safe_mismatch += int((~np.isclose(daily["raw_action_amir"], daily["safe_action_amir"]) | ~np.isclose(daily["raw_action_anfer"], daily["safe_action_anfer"])).sum())
        di = daily["season_cumulative_irrigation"].diff().fillna(daily["season_cumulative_irrigation"])
        dn = daily["season_cumulative_n"].diff().fillna(daily["season_cumulative_n"])
        cumulative_mismatch += int((~np.isclose(di, daily["safe_action_amir"], atol=1e-6)).sum() + (~np.isclose(dn, daily["safe_action_anfer"], atol=1e-6)).sum())
        off_grid += int((~daily["safe_action_amir"].isin(i_grid)).sum() + (~daily["safe_action_anfer"].isin(n_grid)).sum())
        positive = daily["safe_action_amir"].gt(0) | daily["safe_action_anfer"].gt(0)
        positive_after_dap1 += int((positive & daily["dap"].gt(1)).sum())
        pairs.update((float(i), float(n)) for i, n in zip(daily.loc[positive, "safe_action_amir"], daily.loc[positive, "safe_action_anfer"]))
    novel = any(i in {15.0, 30.0} or n in {40.0, 120.0} for i, n in pairs)
    return {
        "checkpoint_step": 2000,
        "validation_rows": int(len(final)),
        "daily_rows": int(daily_rows),
        "raw_safe_mismatch_rows": int(raw_safe_mismatch),
        "safe_action_cumulative_transmission_mismatch_fields": int(cumulative_mismatch),
        "off_grid_action_fields": int(off_grid),
        "positive_action_rows_after_dap1": int(positive_after_dap1),
        "unique_nonzero_action_pairs": int(len(pairs)),
        "nonzero_action_pairs": [f"I{i:g}/N{n:g}" for i, n in sorted(pairs)],
        "new_action_levels_used": bool(novel),
        "strict_gate_passed": bool(len(final) == 10 and cumulative_mismatch == 0 and off_grid == 0 and positive_after_dap1 > 0 and len(pairs) >= 2 and novel),
    }


def architecture_audit(model: MaskablePPO) -> dict[str, Any]:
    extractor = model.policy.features_extractor
    checks = {
        "extractor_class_is_dual_branch": isinstance(extractor, DualBranchBaseWeatherExtractor) or type(extractor).__name__ == "DualBranchBaseWeatherExtractor",
        "features_dim_is_96": int(extractor.features_dim) == 96,
        "base_dim_is_25": int(extractor.base_observation_dim) == 25,
        "weather_dim_is_12": int(extractor.weather_observation_dim) == 12,
        "downstream_net_arch_is_64_64": list(model.policy.net_arch) == [64, 64],
    }
    return {"loaded_extractor": f"{type(extractor).__module__}.{type(extractor).__name__}", "checks": checks, "passed": all(checks.values())}


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    required = [CONFIG, PROMPT, MODEL, SMOKE / "142E2_smoke_result.json", *SUMMARY_PATHS.values()]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    iso = isolation_audit()
    smoke_result = json.loads((SMOKE / "142E2_smoke_result.json").read_text(encoding="utf-8"))
    metrics = metric_comparison()
    strict = strict_action_audit()
    patch_contract()
    cfg = forecast.read_json(CONFIG)
    preflight = forecast.preflight(cfg, PROMPT)
    model = MaskablePPO.load(str(MODEL), device="cpu")
    architecture = architecture_audit(model)
    if not architecture["passed"]:
        raise RuntimeError(json.dumps(architecture, ensure_ascii=False))

    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    try:
        forecast.engine.TASK_ID = "142E2"
        forecast.engine.TASK_NAME = "sya_dual_branch_weather_reader_validation"
        forecast.engine.BASE_OUT = OUT
        forecast.engine.PROMPT = PROMPT
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
        states = collect_states(model, make_env, runtime_config, env_config, "SYA", int(cfg["seed"]))
        states.to_json(OUT / "142E2_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", 2014, int(cfg["seed"]), "142E2_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        decisions = counterfactual_decisions(states, model, grid, int(cfg["seed"]))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.SITE_NAMES.clear()
        forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items():
            setattr(forecast.engine, key, value)

    decisions.to_csv(OUT / "142E2_counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "142E2_metric_comparison.csv", index=False, encoding="utf-8-sig")
    swap_rate = float(decisions["swap_changed"].mean()) if len(decisions) else 0.0
    shuffled_rate = float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0
    response_passed = bool(len(decisions) and decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all() and max(swap_rate, shuffled_rate) > 0.01)
    e2 = metrics[(metrics["branch"] == "E2_dual_branch") & metrics["checkpoint_step"].eq(2000)].iloc[0]
    e1 = metrics[(metrics["branch"] == "E1_single_reader") & metrics["checkpoint_step"].eq(2000)].iloc[0]
    performance_passed = bool(e2["mean_yield_kg_ha"] >= 0.98 * e1["mean_yield_kg_ha"])
    five_k_candidate = bool(iso["passed"] and architecture["passed"] and smoke_result["smoke_gate"]["next_step_allowed"] and strict["strict_gate_passed"] and response_passed and performance_passed)
    summary = {
        "status": "completed_E2_smoke_and_read_only_response_validation",
        "as_of": "2026-08-16 Asia/Shanghai",
        "model": rel(MODEL),
        "isolation_audit": iso,
        "architecture_audit": architecture,
        "smoke_gate": smoke_result["smoke_gate"],
        "strict_action_audit": strict,
        "counterfactual": {
            "decision_states": int(len(decisions)),
            "swap_action_change_rate": swap_rate,
            "shuffled_action_change_rate": shuffled_rate,
            "swap_changes_after_dap1": int(decisions.loc[decisions["dap"].gt(1) & decisions["swap_changed"]].shape[0]),
            "shuffled_changes_after_dap1": int(decisions.loc[decisions["dap"].gt(1) & decisions["shuffled_changed"]].shape[0]),
            "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()),
            "counterfactual_actions_executed_in_dssat": False,
            "response_threshold": "max(swap, shuffled) > 1%",
            "response_passed": response_passed,
        },
        "checkpoint_2k": {
            "E2": e2.to_dict(),
            "E1": e1.to_dict(),
            "E2_minus_E1_yield_kg_ha": float(e2["mean_yield_kg_ha"] - e1["mean_yield_kg_ha"]),
            "E2_minus_E1_pfp_n": float(e2["mean_pfp_n"] - e1["mean_pfp_n"]),
            "performance_floor": "E2 yield >= 98% of E1 yield",
            "performance_passed": performance_passed,
        },
        "wp_et": "unavailable_without_valid_Summary.OUT_ETCP; not inferred",
        "five_k_candidate": five_k_candidate,
        "five_k_executed": False,
    }
    (OUT / "142E2_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 142E2 SY 双通道天气读取 2K 验证记录",
        "",
        "## 结论边界",
        "",
        "E2 保留 E1 的全部观测、reward、动作和训练设置，只把 25 维基础状态与 12 维天气分别编码后合并。天气交换审计固定基础状态、DAP 和 mask，只替换其他年份同 DAP 的天气向量；反事实动作未执行进 DSSAT。",
        "",
        "## 1K/2K 指标",
        "",
        metrics.to_markdown(index=False),
        "",
        "## 架构与动作审计",
        "",
        "```json",
        json.dumps({"architecture": architecture, "strict_action": strict}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 天气响应",
        "",
        f"- 决策状态：{len(decisions)}；swap 动作变化率：{swap_rate:.4%}；shuffled 动作变化率：{shuffled_rate:.4%}。",
        f"- 天气响应闸门：`{response_passed}`；所有变体动作合法：`{bool(decisions[['real_legal', 'swap_legal', 'shuffled_legal']].all().all())}`。",
        "",
        "## 5K 判定",
        "",
        f"- 性能保护闸门：`{performance_passed}`；five_k_candidate：`{five_k_candidate}`；5K 未执行。",
        "- WP_ET 缺少有效 ETCP，保持 unavailable。2K 结果只用于诊断，不能写成稳定优势。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

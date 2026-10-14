"""Validate 141E1 smoke outputs and run the existing same-state swap audit.

This is read-only with respect to the trained policy: it reuses the established
counterfactual action evaluator, never trains, and never executes swapped
actions in DSSAT.
"""

from __future__ import annotations

import json
import shutil
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
from audit_111_112_sy_lc_rolling7d_counterfactual import (
    action_map,
    collect_states,
    counterfactual_decisions,
)
from run_141E1_sya_actionable_weather_encoding_smoke2k import (
    CONFIG,
    PROMPT,
    isolation_audit,
    patch_contract,
)


SMOKE = ROOT / "benchmark_results/141E1_sya_originIC_actionable_weather_encoding_smoke2k"
REFERENCE_SUMMARIES = {
    "E1": SMOKE / "evaluation/141E1_checkpoint_validation_summary.csv",
    "140F_flat28": ROOT / "benchmark_results/140F_sya_originIC_paired_forecast_paired2k/evaluation/140F_checkpoint_validation_summary.csv",
    "140N_no_forecast": ROOT / "benchmark_results/140N_sya_originIC_paired_noforecast_paired2k/evaluation/140N_checkpoint_validation_summary.csv",
}
OUT = ROOT / "benchmark_results/141E1_sya_actionable_weather_encoding_validation"
DOC = ROOT / "docs/2026-08-16_sya_E1_actionable_weather_encoding_validation.md"
YEARS = list(range(2014, 2024))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def metric_comparison() -> pd.DataFrame:
    rows: list[dict] = []
    for branch, path in REFERENCE_SUMMARIES.items():
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


def strict_action_audit() -> dict:
    summary = pd.read_csv(REFERENCE_SUMMARIES["E1"])
    final = summary[summary["checkpoint_step"].eq(2000)].copy()
    irrigation_grid = {0.0, 15.0, 30.0, 45.0}
    nitrogen_grid = {0.0, 40.0, 80.0, 120.0}
    pairs: set[tuple[float, float]] = set()
    raw_safe_mismatch = 0
    cumulative_mismatch = 0
    off_grid = 0
    positive_after_dap1 = 0
    daily_rows = 0
    for _, record in final.iterrows():
        daily = pd.read_csv(ROOT / str(record["daily_csv_path"]))
        for col in [
            "raw_action_amir",
            "raw_action_anfer",
            "safe_action_amir",
            "safe_action_anfer",
            "season_cumulative_irrigation",
            "season_cumulative_n",
            "dap",
        ]:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        daily_rows += len(daily)
        raw_safe_mismatch += int(
            (
                ~np.isclose(daily["raw_action_amir"], daily["safe_action_amir"])
                | ~np.isclose(daily["raw_action_anfer"], daily["safe_action_anfer"])
            ).sum()
        )
        irrigation_delta = daily["season_cumulative_irrigation"].diff().fillna(daily["season_cumulative_irrigation"])
        n_delta = daily["season_cumulative_n"].diff().fillna(daily["season_cumulative_n"])
        cumulative_mismatch += int(
            (~np.isclose(irrigation_delta, daily["safe_action_amir"], atol=1e-6)).sum()
            + (~np.isclose(n_delta, daily["safe_action_anfer"], atol=1e-6)).sum()
        )
        off_grid += int((~daily["safe_action_amir"].isin(irrigation_grid)).sum() + (~daily["safe_action_anfer"].isin(nitrogen_grid)).sum())
        positive = daily["safe_action_amir"].gt(0) | daily["safe_action_anfer"].gt(0)
        positive_after_dap1 += int((positive & daily["dap"].gt(1)).sum())
        pairs.update(
            (float(i), float(n))
            for i, n in zip(daily.loc[positive, "safe_action_amir"], daily.loc[positive, "safe_action_anfer"])
        )
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
        "strict_gate_passed": bool(
            len(final) == 10
            and cumulative_mismatch == 0
            and off_grid == 0
            and positive_after_dap1 > 0
            and len(pairs) >= 2
            and novel
        ),
        "raw_safe_note": "raw/safe differences are reported but are not transmission failures when the safety layer intentionally changes a request",
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    for path in [CONFIG, PROMPT, SMOKE / "141E1_smoke_result.json", *REFERENCE_SUMMARIES.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    iso = isolation_audit()
    smoke_result = json.loads((SMOKE / "141E1_smoke_result.json").read_text(encoding="utf-8"))
    metrics = metric_comparison()
    strict = strict_action_audit()

    patch_contract()
    cfg = forecast.read_json(CONFIG)
    preflight = forecast.preflight(cfg, PROMPT)
    model_path = next(SMOKE.glob("models/SYA/*seed0_ckpt2000.zip"), None)
    if model_path is None:
        raise FileNotFoundError("141E1 checkpoint 2000 model")

    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    old = {
        key: getattr(forecast.engine, key)
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    try:
        forecast.engine.TASK_ID = "141E1"
        forecast.engine.TASK_NAME = "sya_actionable_weather_encoding_validation"
        forecast.engine.BASE_OUT = OUT
        forecast.engine.PROMPT = PROMPT
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.STATION = "SYA"
        forecast.engine.SITES = ["SYA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        model = MaskablePPO.load(str(model_path), device="cpu")
        states = collect_states(model, make_env, runtime_config, env_config, "SYA", int(cfg["seed"]))
        states.to_json(OUT / "141E1_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", YEARS[0], int(cfg["seed"]), "141E1_grid_probe", evaluation=True)
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

    decisions.to_csv(OUT / "141E1_counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "141E1_metric_comparison.csv", index=False, encoding="utf-8-sig")
    swap_rate = float(decisions["swap_changed"].mean()) if len(decisions) else 0.0
    shuffled_rate = float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0
    response_passed = bool(
        len(decisions)
        and decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()
        and max(swap_rate, shuffled_rate) > 0.01
    )
    e1_2k = metrics[(metrics["branch"] == "E1") & (metrics["checkpoint_step"] == 2000)].iloc[0]
    flat_2k = metrics[(metrics["branch"] == "140F_flat28") & (metrics["checkpoint_step"] == 2000)].iloc[0]
    nof_2k = metrics[(metrics["branch"] == "140N_no_forecast") & (metrics["checkpoint_step"] == 2000)].iloc[0]
    five_k_candidate = bool(
        smoke_result["smoke_gate"]["next_step_allowed"]
        and iso["passed"]
        and strict["strict_gate_passed"]
        and response_passed
        and e1_2k["mean_yield_kg_ha"] >= flat_2k["mean_yield_kg_ha"]
    )
    summary = {
        "status": "completed_E1_smoke_and_read_only_response_validation",
        "as_of": "2026-08-16 Asia/Shanghai",
        "model": rel(model_path),
        "smoke_gate": smoke_result["smoke_gate"],
        "isolation_audit": iso,
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
            "E1": e1_2k.to_dict(),
            "140F_flat28": flat_2k.to_dict(),
            "140N_no_forecast": nof_2k.to_dict(),
            "E1_minus_flat28_yield_kg_ha": float(e1_2k["mean_yield_kg_ha"] - flat_2k["mean_yield_kg_ha"]),
            "E1_minus_no_forecast_yield_kg_ha": float(e1_2k["mean_yield_kg_ha"] - nof_2k["mean_yield_kg_ha"]),
        },
        "wp_et": "unavailable_without_valid_Summary.OUT_ETCP; not inferred",
        "five_k_candidate": five_k_candidate,
        "five_k_executed": False,
        "interpretation": "E1 can be considered a 5K candidate only if isolation, chain, strict action, and >1% same-state weather-response gates pass and 2K mean yield is not below flat-28 forecast.",
    }
    (OUT / "141E1_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# 141E1 SY 天气编码 2K 验证记录",
        "",
        "## 验证范围",
        "",
        "E1 只把 140F 的 28 个逐日天气值替换为 12 个有界时序摘要。reward、安全 mask、16 动作格、PPO、seed、年份和输入根保持不变。反事实审计固定同一基础状态和 action mask，只交换其他年份同 DAP 的 E1 天气向量；不训练，也不把交换动作执行进 DSSAT。",
        "",
        "## 1K/2K 指标对照",
        "",
        metrics.to_markdown(index=False),
        "",
        "## 严格动作审计",
        "",
        "```json",
        json.dumps(strict, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 同状态天气响应",
        "",
        f"- 决策状态：{len(decisions)}；swap 动作变化率：{swap_rate:.4%}；shuffled 动作变化率：{shuffled_rate:.4%}。",
        f"- 所有变体动作合法：{bool(decisions[['real_legal', 'swap_legal', 'shuffled_legal']].all().all())}；反事实动作未执行进 DSSAT。",
        "",
        "## 5K 判定",
        "",
        f"- five_k_candidate：`{five_k_candidate}`；5K 未执行。",
        "- WP_ET 因缺少有效 Summary.OUT/ETCP 保持 unavailable，未从逐日 CSV 推断。",
        "- 2K 仅用于链路、早期动作与天气输入敏感性诊断，不能表述为稳定性能优势。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

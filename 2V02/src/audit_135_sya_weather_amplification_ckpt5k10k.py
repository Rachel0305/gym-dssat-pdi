"""Read-only weather amplification audit for existing SY 5K and 10K checkpoints."""

from __future__ import annotations

import copy
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

import audit_134_sya_within_site_weather_amplification as base_audit

CONFIG = ROOT / "configs/135_00_sya_weather_amplification_ckpt5k10k_readonly.json"
BASE_CFG = ROOT / "configs/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_weather_amplification_ckpt5k10k_readonly.md"
MODEL_ROOT = ROOT / "benchmark_results/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k/models/SYA"
OUT = ROOT / "benchmark_results/135_00_sya_weather_amplification_ckpt5k10k_readonly"
MODELS = {
    "ckpt5000": MODEL_ROOT / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt5000.zip",
    "ckpt10000": MODEL_ROOT / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt10000.zip",
}
SCENARIOS = base_audit.SCENARIOS
YEARS = base_audit.YEARS
SEED = 0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    base_audit.rolling.patch_contract()
    base_cfg = base_audit.forecast.read_json(BASE_CFG)
    preflight = base_audit.forecast.preflight(base_cfg, ROOT / base_cfg["prompt"])
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    for path in MODELS.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    shutil.copy2(BASE_CFG, OUT / BASE_CFG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    forecast = base_audit.forecast
    old_loader = forecast.engine.load_config
    old_make_env = forecast.engine.base03222.base.make_env
    old_class = forecast.engine.base03222.base.StressAwareDiscreteWrapper
    old_root = forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_i = list(forecast.engine.BINARY_IRRIGATION_LEVELS)
    old_n = list(forecast.engine.BINARY_NITROGEN_LEVELS)
    old_station, old_sites, old_input = forecast.engine.STATION, list(forecast.engine.SITES), forecast.engine.LOWIC_INPUT_ROOT
    summaries: list[dict] = []
    try:
        forecast.engine.load_config = lambda: base_audit.tuned_load_config(old_loader)
        forecast.engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        forecast.engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        forecast.engine.STATION, forecast.engine.SITES = "SYA", ["SYA"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = forecast.engine.base04036.LateIrrigationReserveMaskWrapper
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, base_cfg)
        for label, model_path in MODELS.items():
            model = MaskablePPO.load(str(model_path), device="cpu")
            states = base_audit.collect_states(model, make_env, runtime_config, env_config)
            states.to_json(OUT / f"135_00_{label}_state_trace.json", orient="records", force_ascii=False)
            probe = make_env(runtime_config, env_config, "SYA", YEARS[0], SEED, f"135_00_{label}_grid_probe", evaluation=True)
            try:
                grid = base_audit.action_map(probe)
            finally:
                probe.close()
            rows: list[dict] = []
            for _, state in states.iterrows():
                base = np.asarray(state["base_obs"], dtype=np.float32)
                real_fc = np.asarray(state["real_forecast"], dtype=np.float32)
                mask = np.asarray(state["mask"], dtype=bool)
                real_action, _ = model.predict(np.concatenate([base, real_fc]), action_masks=mask, deterministic=True)
                real_action = int(np.asarray(real_action).reshape(-1)[0])
                row = {
                    "year": int(state["year"]),
                    "dap": int(state["dap"]),
                    "real_action": real_action,
                    "real_i": grid[real_action][0],
                    "real_n": grid[real_action][1],
                    "real_legal": bool(mask[real_action]),
                }
                for scenario in SCENARIOS:
                    fc, clipped, max_delta = base_audit.perturb(real_fc, scenario)
                    action, _ = model.predict(np.concatenate([base, fc]), action_masks=mask, deterministic=True)
                    action = int(np.asarray(action).reshape(-1)[0])
                    row.update(
                        {
                            f"{scenario}_action": action,
                            f"{scenario}_i": grid[action][0],
                            f"{scenario}_n": grid[action][1],
                            f"{scenario}_changed": bool(action != real_action),
                            f"{scenario}_legal": bool(mask[action]),
                            f"{scenario}_clipped": clipped,
                            f"{scenario}_max_feature_delta": max_delta,
                        }
                    )
                rows.append(row)
            decisions = pd.DataFrame(rows)
            decisions.to_csv(OUT / f"135_00_{label}_weather_amplification_decisions.csv", index=False, encoding="utf-8-sig")
            for scenario in SCENARIOS:
                changed = decisions[f"{scenario}_changed"]
                after = decisions["dap"] > 1
                summaries.append(
                    {
                        "checkpoint": label,
                        "scenario": scenario,
                        "decision_states": int(len(decisions)),
                        "action_change_rate": float(changed.mean()),
                        "action_change_rate_after_dap1": float(changed.loc[after].mean()),
                        "changed_rows": int(changed.sum()),
                        "all_actions_legal": bool(decisions[f"{scenario}_legal"].all()),
                        "clipped_feature_row_rate": float(decisions[f"{scenario}_clipped"].mean()),
                        "mean_max_feature_delta": float(decisions[f"{scenario}_max_feature_delta"].mean()),
                        "max_feature_delta": float(decisions[f"{scenario}_max_feature_delta"].max()),
                    }
                )
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(OUT / "135_00_checkpoint_scenario_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "status": "completed_read_only_weather_amplification_ckpt5k10k",
            "models": {label: rel(path) for label, path in MODELS.items()},
            "config": rel(CONFIG),
            "preflight": preflight,
            "years": YEARS,
            "scenario_summary": summaries,
            "counterfactual_actions_executed_in_dssat": False,
            "training_executed": False,
            "diagnostic_only": True,
            "next_step_long_training_allowed": False,
        }
        (OUT / "135_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        doc = ROOT / "docs/2026-08-16_sya_weather_amplification_ckpt5k10k_execution.md"
        doc.write_text(
            "\n".join(
                [
                    "# 135_00 SY 5K/10K checkpoint 天气敏感性只读审计",
                    "",
                    "- 使用 126_00 已保存的 seed=0、5K 和 10K 模型；未重新训练。",
                    f"- 每个 checkpoint 审计 10 个验证年份，共 `{len(YEARS)}` 个决策状态；固定作物状态、DAP 和 action mask。",
                    "- 只修改未来 7 天 28 个天气特征；没有执行 DSSAT 反事实动作。",
                    "",
                    "## 场景结果",
                    "",
                    summary_df.to_markdown(index=False),
                    "",
                    "## 输出",
                    "",
                    f"- 汇总：`{rel(OUT / '135_00_checkpoint_scenario_summary.csv')}`",
                    f"- JSON：`{rel(OUT / '135_00_summary.json')}`",
                    "",
                    "本实验仅用于比较天气输入敏感性，不新增产量、WP_ET 或 PFP_N 结论；不批准长训练。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        forecast.engine.load_config = old_loader
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = old_class
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.BINARY_IRRIGATION_LEVELS = old_i
        forecast.engine.BINARY_NITROGEN_LEVELS = old_n
        forecast.engine.STATION, forecast.engine.SITES, forecast.engine.LOWIC_INPUT_ROOT = old_station, old_sites, old_input
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

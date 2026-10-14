"""Read-only within-SY weather amplification audit for the 111 checkpoint."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_111_112_sy_lc_rolling7d_perfect_forecast_smoke2k as rolling

CONFIG = ROOT / "configs/134_00_sya_within_site_weather_amplification_readonly.json"
BASE_CFG = ROOT / "configs/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_within_site_weather_amplification_readonly.md"
MODEL = ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
OUT = ROOT / "benchmark_results/134_00_sya_within_site_weather_amplification_readonly"
YEARS = list(range(2014, 2024))
SEED = 0
SCENARIOS = (
    "real",
    "rain_half",
    "rain_double",
    "heat_minus3c",
    "heat_plus3c",
    "srad_half",
    "srad_double",
    "dry_hot_bright",
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_map(env) -> dict[int, tuple[float, float]]:
    return {
        int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0)))
        for idx, row in enumerate(getattr(env, "grid"))
    }


def tuned_load_config(original_loader):
    cfg = copy.deepcopy(original_loader())
    cfg["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    cfg["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    return cfg


def collect_states(model, make_env, runtime_config, env_config) -> pd.DataFrame:
    rows: list[dict] = []
    fc_len = len(forecast.FORECAST_FEATURE_NAMES)
    for year in YEARS:
        env = make_env(runtime_config, env_config, "SYA", year, SEED, f"134_00_real_{year}", evaluation=True)
        try:
            obs, info = env.reset()
            done, step = False, 0
            while not done and step < int(runtime_config["runtime"]["max_steps"]):
                latest = forecast.direct_ppo.latest_observation_dict(env, obs, info)
                dap_raw = forecast.direct_ppo.scalar(latest.get("dap", step + 1), step + 1)
                dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step + 1
                mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                vector = np.asarray(obs, dtype=np.float32).reshape(-1)
                obs_next, _reward, terminated, truncated, info = env.step(action)
                rows.append(
                    {
                        "year": year,
                        "dap": dap,
                        "step": step,
                        "base_obs": vector[:-fc_len].copy().tolist(),
                        "real_forecast": vector[-fc_len:].copy().tolist(),
                        "mask": mask.astype(int).tolist(),
                    }
                )
                obs, done = obs_next, bool(terminated or truncated)
                step += 1
        finally:
            env.close()
    return pd.DataFrame(rows)


def perturb(vector: np.ndarray, scenario: str) -> tuple[np.ndarray, bool, float]:
    if vector.size != 28:
        raise RuntimeError(f"expected 28 forecast features, got {vector.size}")
    out = vector.astype(np.float32, copy=True).reshape(7, 4)
    if scenario == "rain_half":
        out[:, 0] *= 0.5
    elif scenario == "rain_double":
        out[:, 0] *= 2.0
    elif scenario == "heat_minus3c":
        out[:, 2:4] -= 0.3
    elif scenario == "heat_plus3c":
        out[:, 2:4] += 0.3
    elif scenario == "srad_half":
        out[:, 1] *= 0.5
    elif scenario == "srad_double":
        out[:, 1] *= 2.0
    elif scenario == "dry_hot_bright":
        out[:, 0] *= 0.25
        out[:, 1] *= 1.25
        out[:, 2:4] += 0.3
    elif scenario != "real":
        raise ValueError(f"unknown scenario: {scenario}")
    raw = out.reshape(-1)
    clipped = bool(np.any(raw < -5.0) or np.any(raw > 5.0))
    clipped_out = np.clip(raw, -5.0, 5.0).astype(np.float32)
    return clipped_out, clipped, float(np.max(np.abs(clipped_out - vector)))


def main() -> int:
    rolling.patch_contract()
    cfg = forecast.read_json(CONFIG)
    base_cfg = forecast.read_json(BASE_CFG)
    preflight = forecast.preflight(base_cfg, ROOT / base_cfg["prompt"])
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    shutil.copy2(BASE_CFG, OUT / BASE_CFG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    old_loader = forecast.engine.load_config
    old_make_env = forecast.engine.base03222.base.make_env
    old_class = forecast.engine.base03222.base.StressAwareDiscreteWrapper
    old_root = forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_i = list(forecast.engine.BINARY_IRRIGATION_LEVELS)
    old_n = list(forecast.engine.BINARY_NITROGEN_LEVELS)
    old_station, old_sites, old_input = forecast.engine.STATION, list(forecast.engine.SITES), forecast.engine.LOWIC_INPUT_ROOT
    try:
        forecast.engine.load_config = lambda: tuned_load_config(old_loader)
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
        model = MaskablePPO.load(str(MODEL), device="cpu")
        states = collect_states(model, make_env, runtime_config, env_config)
        states.to_json(OUT / "134_00_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", YEARS[0], SEED, "134_00_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()

        rows: list[dict] = []
        fc_len = len(forecast.FORECAST_FEATURE_NAMES)
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
                fc, clipped, max_delta = perturb(real_fc, scenario)
                if len(fc) != fc_len:
                    raise RuntimeError("forecast vector length mismatch")
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
        decisions.to_csv(OUT / "134_00_weather_amplification_decisions.csv", index=False, encoding="utf-8-sig")

        summaries: list[dict] = []
        for scenario in SCENARIOS:
            changed = decisions[f"{scenario}_changed"]
            after = decisions["dap"] > 1
            summaries.append(
                {
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
        summary_df.to_csv(OUT / "134_00_scenario_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "status": "completed_read_only_within_site_weather_amplification",
            "model": rel(MODEL),
            "config": rel(CONFIG),
            "preflight": preflight,
            "n_decision_states": int(len(decisions)),
            "scenarios": list(SCENARIOS),
            "counterfactual_actions_executed_in_dssat": False,
            "training_executed": False,
            "scenario_summary": summaries,
            "diagnostic_only": True,
            "next_step_long_training_allowed": False,
        }
        (OUT / "134_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        doc = ROOT / "docs/2026-08-16_sya_within_site_weather_amplification_execution.md"
        lines = [
            "# 134_00 SY 站内天气放大/缩小只读敏感性审计",
            "",
            "## 结论边界",
            "",
            "- 使用 SY 111_00 seed=0 ckpt2000；固定 SY 作物状态、DAP 和 action mask，只改变观测中的未来 7 天 28 个天气特征。",
            f"- 共审计 `{summary['n_decision_states']}` 个决策状态；没有训练，也没有把反事实动作执行进 DSSAT。",
            "- 本实验只能判断策略输入敏感性，不能产生产量、WP_ET 或 PFP_N 证据。",
            "",
            "## 场景结果",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## 输出",
            "",
            f"- 逐状态结果：`{rel(OUT / '134_00_weather_amplification_decisions.csv')}`",
            f"- 场景汇总：`{rel(OUT / '134_00_scenario_summary.csv')}`",
            f"- JSON：`{rel(OUT / '134_00_summary.json')}`",
            "",
            "## 判定规则",
            "",
            "- 若大幅天气扰动下动作仍接近 0% 变化，说明当前策略对天气分支确实不敏感，不能只归因于自然年份差异太小。",
            "- 若动作变化只出现在大量特征裁剪的场景，不能称为稳定天气响应。",
            "- 本轮不批准 25K/50K/75K/100K 长训练。",
            "",
        ]
        doc.write_text("\n".join(lines), encoding="utf-8")
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

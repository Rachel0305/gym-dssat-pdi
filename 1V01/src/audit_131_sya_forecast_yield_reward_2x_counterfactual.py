"""Read-only same-DAP forecast swap audit for the 130 SY checkpoint."""

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

CONFIG = ROOT / "configs/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_forecast_yield_reward_2x_counterfactual_audit.md"
MODEL_ROOT = ROOT / "benchmark_results/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x_smoke2k_smoke2k"
OUT = ROOT / "benchmark_results/131_00_sya_forecast_yield_reward_2x_counterfactual_audit_rerun1"
YEARS = list(range(2014, 2024))
SEED = 0
YIELD_SCALE = 2.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_map(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def scaled_load_config(original_load_config):
    baseline = original_load_config()
    cfg = copy.deepcopy(baseline)
    reward = cfg.setdefault("reward", {})
    reward["yield_coef"] = float(reward["yield_coef"]) * YIELD_SCALE
    return cfg


def collect_states(model, make_env, runtime_config, env_config, station: str) -> pd.DataFrame:
    rows: list[dict] = []
    for year in YEARS:
        env = make_env(runtime_config, env_config, station, year, SEED, f"131_00_cf_{year}", evaluation=True)
        try:
            grid = action_map(env)
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
                fc_len = len(forecast.FORECAST_FEATURE_NAMES)
                base_obs = vector[:-fc_len].copy()
                real_fc = vector[-fc_len:].copy()
                obs_next, _reward, terminated, truncated, info = env.step(action)
                action_info = dict(getattr(env, "last_action_info", {}) or {})
                rows.append(
                    {
                        "year": year,
                        "dap": dap,
                        "step": step,
                        "actual_action": action,
                        "mask": mask.astype(int).tolist(),
                        "base_obs": base_obs.tolist(),
                        "real_forecast": real_fc.tolist(),
                        "forecast_observation_date": str(action_info.get("forecast_observation_date", "")),
                        "real_i": grid[action][0],
                        "real_n": grid[action][1],
                    }
                )
                obs, done = obs_next, bool(terminated or truncated)
                step += 1
        finally:
            env.close()
    return pd.DataFrame(rows)


def counterfactual_decisions(states: pd.DataFrame, model, grid: dict[int, tuple[float, float]], seed: int) -> pd.DataFrame:
    work = states[states["dap"].ge(1)].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sources: dict[int, tuple[int, int]] = {}
    for _dap, idxs in work.groupby("dap").groups.items():
        ids = list(idxs)
        if len(ids) < 2:
            for i in ids:
                sources[i] = (i, i)
            continue
        rotated = ids[1:] + ids[:1]
        shuffled = list(rng.permutation(ids))
        if all(a == b for a, b in zip(ids, shuffled)):
            shuffled = rotated
        sources.update({i: (rotated[pos], shuffled[pos]) for pos, i in enumerate(ids)})

    rows: list[dict] = []
    fc_len = len(forecast.FORECAST_FEATURE_NAMES)
    for idx, row in work.iterrows():
        swap_idx, shuffled_idx = sources[idx]
        mask = np.asarray(row["mask"], dtype=bool)
        base = np.asarray(row["base_obs"], dtype=np.float32)
        preds: dict[str, int] = {}
        for name, source in [("real", idx), ("swap", swap_idx), ("shuffled", shuffled_idx)]:
            fc = np.asarray(work.loc[source, "real_forecast"], dtype=np.float32)
            if len(fc) != fc_len:
                raise RuntimeError("forecast vector length mismatch")
            action, _ = model.predict(np.concatenate([base, fc]), action_masks=mask, deterministic=True)
            preds[name] = int(np.asarray(action).reshape(-1)[0])
        rows.append(
            {
                "year": int(row["year"]),
                "dap": int(row["dap"]),
                "swap_source_year": int(work.loc[swap_idx, "year"]),
                "shuffled_source_year": int(work.loc[shuffled_idx, "year"]),
                "real_action": preds["real"],
                "swap_action": preds["swap"],
                "shuffled_action": preds["shuffled"],
                "swap_changed": preds["real"] != preds["swap"],
                "shuffled_changed": preds["real"] != preds["shuffled"],
                "real_legal": bool(mask[preds["real"]]),
                "swap_legal": bool(mask[preds["swap"]]),
                "shuffled_legal": bool(mask[preds["shuffled"]]),
                "real_i": grid[preds["real"]][0],
                "real_n": grid[preds["real"]][1],
                "swap_i": grid[preds["swap"]][0],
                "swap_n": grid[preds["swap"]][1],
                "shuffled_i": grid[preds["shuffled"]][0],
                "shuffled_n": grid[preds["shuffled"]][1],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    rolling.patch_contract()
    cfg = forecast.read_json(CONFIG)
    preflight = forecast.preflight(cfg, PROMPT)
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    model_path = next(MODEL_ROOT.glob("models/SYA/*seed0_ckpt2000.zip"), None)
    if model_path is None:
        raise FileNotFoundError(f"missing 130 model under {MODEL_ROOT}")
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    old_root = forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    old_class = forecast.engine.base03222.base.StressAwareDiscreteWrapper
    old_loader = forecast.engine.load_config
    old_binary_i = list(forecast.engine.BINARY_IRRIGATION_LEVELS)
    old_binary_n = list(forecast.engine.BINARY_NITROGEN_LEVELS)
    old_station = forecast.engine.STATION
    old_sites = list(forecast.engine.SITES)
    old_input_root = forecast.engine.LOWIC_INPUT_ROOT
    try:
        forecast.engine.load_config = lambda: scaled_load_config(old_loader)
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.STATION = "SYA"
        forecast.engine.SITES = ["SYA"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = forecast.engine.base04036.LateIrrigationReserveMaskWrapper
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        model = MaskablePPO.load(str(model_path), device="cpu")
        states = collect_states(model, make_env, runtime_config, env_config, "SYA")
        states.to_json(OUT / "131_00_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", YEARS[0], SEED, "131_00_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        decisions = counterfactual_decisions(states, model, grid, SEED)
        decisions.to_csv(OUT / "131_00_counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
        by_year = decisions.groupby("year", as_index=False).agg(
            decision_states=("dap", "size"),
            swap_action_change_rate=("swap_changed", "mean"),
            shuffled_action_change_rate=("shuffled_changed", "mean"),
            swap_changes_after_dap1=("swap_changed", lambda x: int(x.sum())),
            shuffled_changes_after_dap1=("shuffled_changed", lambda x: int(x.sum())),
        )
        by_year.to_csv(OUT / "131_00_by_year_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "status": "completed_read_only_counterfactual_action_audit",
            "model": rel(model_path),
            "preflight": preflight,
            "years": YEARS,
            "seed": SEED,
            "yield_coef_effective": float(runtime_config["reward"]["yield_coef"]),
            "counterfactual_actions_executed_in_dssat": False,
            "n_decision_states": int(len(decisions)),
            "swap_action_change_rate": float(decisions["swap_changed"].mean()) if len(decisions) else 0.0,
            "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0,
            "swap_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["swap_changed"]].shape[0]),
            "shuffled_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["shuffled_changed"]].shape[0]),
            "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()) if len(decisions) else False,
            "weather_response_detected": bool(len(decisions) and decisions["swap_changed"].mean() > 0.01),
            "next_step_long_training_allowed": False,
        }
        (OUT / "131_00_counterfactual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        doc = ROOT / "docs/2026-08-16_sya_forecast_yield_reward_2x_counterfactual_audit.md"
        doc.write_text(
            "\n".join(
                [
                    "# 131_00 SY 130 模型天气敏感性只读审计",
                    "",
                    "## 结论先说",
                    "",
                    f"- 130 的有效终点产量系数为 `{summary['yield_coef_effective']}`；本轮没有训练，也没有把反事实动作写回 DSSAT。",
                    f"- 共审计 `{summary['n_decision_states']}` 个验证决策状态；所有 real/swap/shuffled 动作均合法：`{summary['all_variant_actions_legal']}`。",
                    f"- same-DAP 天气 swap 动作变化率：`{summary['swap_action_change_rate']:.4%}`；shuffled 变化率：`{summary['shuffled_action_change_rate']:.4%}`。",
                    f"- `weather_response_detected`（>1% 的诊断阈值）：`{summary['weather_response_detected']}`。",
                    "- 该结果只判断“策略是否改变动作”，不证明天气响应会提高产量；本轮仍禁止长训练。",
                    "",
                    "## 输出",
                    "",
                    f"- 状态轨迹：`{rel(OUT / '131_00_state_trace.json')}`",
                    f"- 逐状态预测：`{rel(OUT / '131_00_counterfactual_decisions.csv')}`",
                    f"- 按年汇总：`{rel(OUT / '131_00_by_year_summary.csv')}`",
                    f"- JSON：`{rel(OUT / '131_00_counterfactual_summary.json')}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        forecast.engine.load_config = old_loader
        forecast.engine.BINARY_IRRIGATION_LEVELS = old_binary_i
        forecast.engine.BINARY_NITROGEN_LEVELS = old_binary_n
        forecast.engine.STATION = old_station
        forecast.engine.SITES = old_sites
        forecast.engine.LOWIC_INPUT_ROOT = old_input_root
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = old_class
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

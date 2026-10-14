"""Read-only cross-site forecast substitution audit for the SY 111 checkpoint."""

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

CONFIG = ROOT / "configs/133_00_sya_cross_site_weather_counterfactual.json"
BASE_CFG = ROOT / "configs/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_cross_site_weather_counterfactual_readonly.md"
MODEL = ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
OUT = ROOT / "benchmark_results/133_00_sya_cross_site_weather_counterfactual_readonly"
YEARS = list(range(2014, 2024))
SOURCES = {"SY_real": "SYA", "FQ_weather": "FQA", "HL_weather": "HLA", "LC_weather": "LCA", "YC_weather": "YCA"}
SEED = 0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_map(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def tuned_load_config(original_loader):
    cfg = copy.deepcopy(original_loader())
    cfg["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    cfg["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    return cfg


def collect_states(model, make_env, runtime_config, env_config) -> pd.DataFrame:
    rows: list[dict] = []
    for year in YEARS:
        env = make_env(runtime_config, env_config, "SYA", year, SEED, f"133_00_real_{year}", evaluation=True)
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
                fc_len = len(forecast.FORECAST_FEATURE_NAMES)
                obs_next, _reward, terminated, truncated, info = env.step(action)
                rows.append({
                    "year": year,
                    "dap": dap,
                    "step": step,
                    "base_obs": vector[:-fc_len].copy().tolist(),
                    "real_forecast": vector[-fc_len:].copy().tolist(),
                    "mask": mask.astype(int).tolist(),
                })
                obs, done = obs_next, bool(terminated or truncated)
                step += 1
        finally:
            env.close()
    return pd.DataFrame(rows)


def source_vectors(runtime_config, env_config, scales: dict[str, float], source_station: str, year: int, dap: int, weather_cache: dict[str, pd.DataFrame]) -> tuple[np.ndarray, bool]:
    planting = pd.Timestamp(forecast.direct_ppo.find_year(env_config, source_station, year)["planting_date"])
    date = planting + pd.Timedelta(days=int(dap) - 1)
    raw = rolling.raw_rolling_7d(weather_cache[source_station], date, scales)
    norm = rolling.normalize_rolling_7d(raw, scales)
    vector = np.asarray([norm[name] for name in forecast.FORECAST_FEATURE_NAMES], dtype=np.float32)
    return vector, bool(np.any(vector < -5.0) or np.any(vector > 5.0))


def main() -> int:
    rolling.patch_contract()
    cfg = forecast.read_json(CONFIG)
    base_cfg = forecast.read_json(BASE_CFG)
    pf = forecast.preflight(base_cfg, ROOT / base_cfg["prompt"])
    if not pf["next_step_allowed"]:
        raise RuntimeError(pf["issues"])
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
        states.to_json(OUT / "133_00_state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, "SYA", YEARS[0], SEED, "133_00_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        weather_cache = {station: forecast._weather_table(runtime_config, station) for station in set(SOURCES.values())}
        scales = dict(base_cfg["forecast_features"]["scales"])
        rows: list[dict] = []
        for idx, state in states.iterrows():
            base = np.asarray(state["base_obs"], dtype=np.float32)
            mask = np.asarray(state["mask"], dtype=bool)
            real_action, _ = model.predict(np.concatenate([base, np.asarray(state["real_forecast"], dtype=np.float32)]), action_masks=mask, deterministic=True)
            real_action = int(np.asarray(real_action).reshape(-1)[0])
            row = {"year": int(state["year"]), "dap": int(state["dap"]), "real_action": real_action, "real_i": grid[real_action][0], "real_n": grid[real_action][1], "real_legal": bool(mask[real_action])}
            for label, station in SOURCES.items():
                fc, ood = source_vectors(runtime_config, env_config, scales, station, int(state["year"]), int(state["dap"]), weather_cache)
                action, _ = model.predict(np.concatenate([base, fc]), action_masks=mask, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                row[f"{label}_action"] = action
                row[f"{label}_i"] = grid[action][0]
                row[f"{label}_n"] = grid[action][1]
                row[f"{label}_changed"] = bool(action != real_action)
                row[f"{label}_legal"] = bool(mask[action])
                row[f"{label}_ood"] = bool(ood)
            rows.append(row)
        decisions = pd.DataFrame(rows)
        decisions.to_csv(OUT / "133_00_cross_site_decisions.csv", index=False, encoding="utf-8-sig")
        summaries = []
        for label in SOURCES:
            summaries.append({
                "weather_source": label,
                "source_station": SOURCES[label],
                "decision_states": int(len(decisions)),
                "action_change_rate": float(decisions[f"{label}_changed"].mean()),
                "action_change_rate_after_dap1": float(decisions.loc[decisions["dap"] > 1, f"{label}_changed"].mean()),
                "changed_rows": int(decisions[f"{label}_changed"].sum()),
                "all_actions_legal": bool(decisions[f"{label}_legal"].all()),
                "ood_feature_row_rate": float(decisions[f"{label}_ood"].mean()),
            })
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(OUT / "133_00_source_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "status": "completed_read_only_cross_site_weather_counterfactual",
            "model": rel(MODEL),
            "preflight": pf,
            "n_decision_states": int(len(decisions)),
            "counterfactual_actions_executed_in_dssat": False,
            "source_summary": summaries,
            "diagnostic_only": True,
            "next_step_long_training_allowed": False,
        }
        (OUT / "133_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        doc = ROOT / "docs/2026-08-16_sya_cross_site_weather_counterfactual_execution.md"
        lines = [
            "# 133_00 SY 跨站点天气替换只读敏感性审计",
            "",
            "## 结论先说",
            "",
            "- 使用 SY 111_00 seed=0 ckpt2000；固定 SY 的 base observation、DAP 和 action mask，只替换未来 7 天天气向量。",
            f"- 共审计 `{summary['n_decision_states']}` 个决策状态；没有训练，也没有把反事实动作执行进 DSSAT。",
            "- 跨站点天气与 SY 作物状态不物理一致，因此本轮只判断策略是否读取天气，不能作为产量、WP_ET 或 PFP_N 证据。",
            "",
            "## 来源站点结果",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## 输出",
            "",
            f"- 逐状态决策：`{rel(OUT / '133_00_cross_site_decisions.csv')}`",
            f"- 来源汇总：`{rel(OUT / '133_00_source_summary.csv')}`",
            f"- JSON：`{rel(OUT / '133_00_summary.json')}`",
            "",
            "## 边界",
            "",
            "- OOD 比例较高时，动作变化只能说明模型对分布外输入敏感，不能称为稳定天气响应。",
            "- 下一步若要判断 originIC 是否缓冲天气，需要在 SY originIC 内做 oracle DSSAT 天气/动作价值测试。",
            "- 本轮不批准任何长训练。",
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

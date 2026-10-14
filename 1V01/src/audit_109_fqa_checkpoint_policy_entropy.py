"""Read-only comparison of FQ 5K/10K policy concentration by DAP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as reserve
import run_107_fqa_current_weather_observation_smoke2k as weather_base

ROOT_OUT = ROOT / "benchmark_results/109_00_fqa_checkpoint_policy_entropy_audit"
DOC = ROOT / "docs/109_00_fqa_checkpoint_policy_entropy_audit_record.md"
MODELS = {
    "ckpt5000": ROOT / "benchmark_results/108_00_fqa_current_weather_observation_5k10k_corrected2018/models/FQA/FQA_half_split_stress_aware_maskableppo_seed1_ckpt5000.zip",
    "ckpt10000": ROOT / "benchmark_results/108_00_fqa_current_weather_observation_5k10k_corrected2018/models/FQA/FQA_half_split_stress_aware_maskableppo_seed1_ckpt10000.zip",
}
YEARS = list(range(2014, 2024))
DAPS = [1, 8, 15, 32, 61]


def entropy(values: np.ndarray) -> float:
    p = values[np.isfinite(values) & (values > 0)]
    return float(-(p * np.log(p)).sum()) if len(p) else np.nan


def main() -> int:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    (ROOT_OUT / "tables").mkdir(exist_ok=True)
    cfg = direct_ppo.load_yaml(ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml")
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = 1
    cfg["paths"]["output_root"] = str(ROOT_OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["ppo"]["ent_coef"] = 0.02
    cfg["reward"]["nitrogen_cost"] = 2.5
    cfg["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    cfg["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    split = batch.load_split(); split = split[split["station_code"].eq("FQA")]
    selection = batch.build_selection(split)
    env_config = direct_ppo.build_env_config(cfg, selection)
    original_wrapper = batch.base.StressAwareDiscreteWrapper
    original_make_env = batch.base.make_env
    original_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
    base_make_env = original_make_env

    def make_env_current(config, env_config, station, year, seed, run_tag, evaluation=False):
        return weather_base.CurrentWeatherObservationWrapper(base_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation), config, env_config, station, year)

    batch.base.make_env = make_env_current
    rows: list[dict[str, Any]] = []
    try:
        for label, model_path in MODELS.items():
            model = MaskablePPO.load(str(model_path), device="cpu")
            for year in YEARS:
                env = batch.base.make_env(cfg, env_config, "FQA", year, 1, f"FQA_{year}_109_entropy", evaluation=True)
                try:
                    obs, info = env.reset(); done = False; steps = 0; seen: set[int] = set()
                    while not done and steps < 260:
                        latest = batch.base.latest_observation_dict(env, obs, info)
                        dap_raw = direct_ppo.scalar(latest.get("dap", steps + 1), steps + 1)
                        dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else steps + 1
                        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                        obs_tensor, _ = model.policy.obs_to_tensor(obs)
                        dist = model.policy.get_distribution(obs_tensor, action_masks=mask.reshape(1, -1))
                        probs = getattr(getattr(dist, "distribution", dist), "probs", None)
                        p = probs.detach().cpu().numpy().reshape(-1).astype(float) if probs is not None else np.full(len(mask), np.nan)
                        action_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        action = int(np.asarray(action_raw).reshape(-1)[0])
                        if dap in DAPS and dap not in seen:
                            seen.add(dap)
                            current = getattr(env, "last_weather_features", {})
                            rows.append({"checkpoint": label, "year": year, "dap": dap, "chosen_action": action, "legal_actions": int(mask.sum()), "top_probability": float(np.nanmax(p)) if np.isfinite(p).any() else np.nan, "policy_entropy_legal": entropy(p[mask]), "weather_rain": current.get("rain_today_mm", np.nan), "weather_srad": current.get("srad_today_mj_m2", np.nan), "weather_tmax": current.get("tmax_today_c", np.nan), "weather_tmin": current.get("tmin_today_c", np.nan)})
                        obs, _reward, terminated, truncated, info = env.step(action); done = bool(terminated or truncated); steps += 1
                finally:
                    env.close()
    finally:
        batch.base.make_env = original_make_env; batch.base.StressAwareDiscreteWrapper = original_wrapper; ppo_safe_rendering.MULTISITE_INPUT_ROOT = original_root
    df = pd.DataFrame(rows)
    df.to_csv(ROOT_OUT / "tables/109_00_policy_entropy_by_year_dap.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["checkpoint", "dap"], as_index=False).agg(years=("year", "nunique"), mean_entropy=("policy_entropy_legal", "mean"), mean_top_probability=("top_probability", "mean"), min_entropy=("policy_entropy_legal", "min"), chosen_action_count=("chosen_action", "nunique"), mean_legal_actions=("legal_actions", "mean"), weather_tmax_range=("weather_tmax", lambda x: float(x.max() - x.min())), weather_rain_range=("weather_rain", lambda x: float(x.max() - x.min())))
    summary.to_csv(ROOT_OUT / "tables/109_00_policy_entropy_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# 109_00 FQ checkpoint policy concentration audit", "", "- 只读加载 108 的 5K/10K 模型；无训练、无五情景 replay。", "- 在相同 FQ 验证环境中记录合法动作上的 policy entropy、top probability、动作选择和当前天气。", "", "## Summary", "", summary.to_markdown(index=False), "", "## Interpretation boundary", "", "- entropy/top-probability 是确定性策略集中度证据，不是性能指标。", "- 需要与 108 的 reward/action gate 和五情景表联合解释；不单独用于冻结。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

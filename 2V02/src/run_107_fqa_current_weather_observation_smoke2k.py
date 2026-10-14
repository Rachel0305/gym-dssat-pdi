"""107_00 isolated FQ current-weather observation append 2K smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
import yaml
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as reserve
import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as audit_base

BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
CONFIG = ROOT / "configs/107_00_fqa_current_weather_observation_smoke2k.json"
PROMPT = ROOT / "prompts/107_00_fq_current_weather_observation_smoke2k.md"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
WEATHER_NAMES = ["rain_today_mm", "srad_today_mj_m2", "tmax_today_c", "tmin_today_c"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = audit_base.direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(str(station))].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "srad", "tmax", "tmin"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


class CurrentWeatherObservationWrapper(gym.Env):
    """Append only today's four weather values; no forecast or future window."""

    def __init__(self, env: gym.Env, config: dict[str, Any], env_config: dict[str, Any], station: str, year: int):
        super().__init__()
        self.env = env
        self.config = config
        self.env_config = env_config
        self.station = str(station)
        self.year = int(year)
        self.weather = weather_table(config, self.station)
        self.planting = pd.Timestamp(audit_base.direct_ppo.find_year(env_config, self.station, self.year)["planting_date"])
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        weather_low = np.array([0.0, 0.0, -50.0, -50.0], dtype=np.float32)
        weather_high = np.array([400.0, 80.0, 70.0, 70.0], dtype=np.float32)
        self.base_observation_dim = int(len(low))
        self.observation_space = spaces.Box(low=np.concatenate([low, weather_low]), high=np.concatenate([high, weather_high]), dtype=np.float32)
        self.last_weather_features: dict[str, float] = {}

    def _date(self, obs: Any, info: dict[str, Any] | None = None) -> pd.Timestamp:
        latest = audit_base.direct_ppo.latest_observation_dict(self.env, obs, info or {})
        dap = audit_base.direct_ppo.scalar(latest.get("dap", np.nan), np.nan)
        return self.planting + pd.Timedelta(days=int(round(float(dap))) - 1) if np.isfinite(dap) and dap > 0 else self.planting

    def _features(self, date: pd.Timestamp) -> np.ndarray:
        row = self.weather[self.weather["date"].eq(date)]
        values = [float(row[col].iloc[0]) if not row.empty and pd.notna(row[col].iloc[0]) else 0.0 for col in ["rain", "srad", "tmax", "tmin"]]
        self.last_weather_features = {name: value for name, value in zip(WEATHER_NAMES, values)}
        return np.asarray(values, dtype=np.float32)

    def _augment(self, obs: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        base = np.asarray(obs, dtype=np.float32).reshape(-1)
        if len(base) != self.base_observation_dim:
            raise RuntimeError(f"base observation dimension changed: expected {self.base_observation_dim}, got {len(base)}")
        out = np.concatenate([base, self._features(self._date(base, info))]).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("current-weather observation contains NaN/Inf")
        return out

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs, info), dict(info or {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, dict(info or {})

    def action_masks(self):
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        data = dict(getattr(self.env, "last_action_info", {}) or {})
        data.update({f"obs_{key}": value for key, value in self.last_weather_features.items()})
        data["current_weather_observation_enabled_107"] = True
        data["current_weather_observation_dim_107"] = int(self.observation_space.shape[0])
        return data

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not (INPUT_ROOT / "FQ/CNFQ0801.MZX").exists():
        issues.append("originIC FQ input/MZX missing")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    if cfg.get("observation_contract", {}).get("weather_forecast_enabled") is not False:
        issues.append("forecast must remain disabled")
    if cfg.get("observation_contract", {}).get("current_weather_features") != WEATHER_NAMES:
        issues.append("current-weather feature contract mismatch")
    return {"task_id": cfg["task_id"], "station_code": "FQA", "site": "FQ", "seed": int(cfg["seed"]), "issues": issues, "next_step_allowed": not issues, "observation_append_dim": 4, "output_root": cfg["output_root"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    pf = preflight(cfg)
    out = ROOT / cfg["output_root"]
    if args.dry_run:
        print(json.dumps({**pf, "mode": "dry_run", "smoke_only": True, "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return 0
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml = ROOT / "configs/107_00_effective_fqa_current_weather_ppo_config.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(cfg["seed"])
    ppo = dict(cfg["fixed_ppo"]); ppo.update(cfg.get("ppo_override", {}))
    yaml_cfg["ppo"].update(ppo)
    yaml_cfg["reward"].update(cfg["reward_override"])
    yaml_cfg["discrete_actions"]["irrigation_levels"] = LEVELS_I
    yaml_cfg["discrete_actions"]["nitrogen_levels"] = LEVELS_N
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")
    old_engine = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = {key: getattr(batch, key) for key in ["CONFIG", "OUT", "DOC", "SEED"]}
    old_site_names = dict(batch.SITE_NAMES)
    old_wrapper = batch.base.StressAwareDiscreteWrapper
    old_make_env = batch.base.make_env
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    start = time.perf_counter(); rss0 = psutil.Process().memory_info().rss
    doc = ROOT / "docs/107_00_fqa_current_weather_observation_smoke2k_record.md"
    try:
        engine.TASK_ID = cfg["task_id"]; engine.TASK_NAME = cfg["task_name"]; engine.BASE_OUT = out; engine.BASE_DOC = doc; engine.PROMPT = PROMPT; engine.LOWIC_INPUT_ROOT = INPUT_ROOT; engine.STATION = "FQA"; engine.SITES = ["FQA"]; engine.BINARY_IRRIGATION_LEVELS = LEVELS_I; engine.BINARY_NITROGEN_LEVELS = LEVELS_N
        batch.SEED = int(cfg["seed"]); batch.CONFIG = effective_yaml; batch.OUT = out; batch.DOC = doc; batch.SITE_NAMES["FQA"] = "FQ"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
        base_make_env = batch.base.make_env
        def make_env_current(config, env_config, station, year, seed, run_tag, evaluation=False):
            return CurrentWeatherObservationWrapper(base_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation), config, env_config, station, year)
        batch.base.make_env = make_env_current
        engine.run_training(2000, [1000, 2000], suffix="")
    finally:
        batch.base.make_env = old_make_env; batch.base.StressAwareDiscreteWrapper = old_wrapper; ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        for key, value in old_engine.items(): setattr(engine, key, value)
        for key, value in old_batch.items(): setattr(batch, key, value)
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_site_names)
    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    inventory_path = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not summary_path.exists() or not inventory_path.exists():
        raise FileNotFoundError("training did not produce expected summary/inventory")
    summary = pd.read_csv(summary_path, keep_default_na=False); inventory = pd.read_csv(inventory_path, keep_default_na=False)
    summary.to_csv(out / "evaluation/107_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig"); inventory.to_csv(out / "evaluation/107_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    audit, gate = audit_base.audit_25k(summary, out, 2000)
    audit.to_csv(out / "audits/107_00_action_audit.csv", index=False, encoding="utf-8-sig")
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    result = {**pf, "mode": "completed_smoke", "effective_yaml": rel(effective_yaml), "observation_dim_expected": "base+4", "validation_metrics_mean": final[[c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict(), "action_gate": gate, "elapsed_s": round(time.perf_counter() - start, 3), "rss_delta_bytes": psutil.Process().memory_info().rss - rss0, "next_step_allowed": bool(gate.get("mechanism_pass")), "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred"}
    (out / "107_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "107_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 107_00 FQ current-weather observation seed=1 2K smoke", "", "- 唯一干预：在 046_02 raw observation 后追加当日 rain/srad/tmax/tmin 四字段。", "- 无 future forecast、无 external-N；保留 DSSAT 自动灌溉和原安全 mask。", "", "## Gate", "", *[f"- `{k}`: `{v}`" for k, v in gate.items()], "", "- WP_ET 未从 2K summary 推断。", "- 只有 mechanism_pass=true 才可另行申请长训；本任务不自动进入 25K/100K。", ""]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

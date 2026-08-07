"""046_07: train-year-statistics-only normalization experiment for 046_02 PPO.

No forecast, teacher, behaviour cloning, reward change, or action change is
included.  The train-only statistics are saved before PPO training and reused
unchanged by every validation episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


DEFAULT_CONFIG = ROOT / "configs" / "046_07_sya_originIC_binary_timing_train_stats_normalized_ppo.json"
PROMPT = ROOT / "prompts" / "046_07_sya_originIC_train_stats_normalized_ppo.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
CLIP = 5.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("station_code") != "SYA":
        raise ValueError("046_07 currently supports SYA only")
    if cfg.get("input_profile") not in INPUT_PROFILES:
        raise ValueError("input_profile must be originIC or lowIC")
    return cfg


class FixedTrainStatisticsObservationWrapper(gym.Env):
    """Apply pre-fitted train-only z-score normalization without touching reward."""

    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.env = env
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=np.full(len(self.mean), -CLIP, dtype=np.float32), high=np.full(len(self.mean), CLIP, dtype=np.float32), dtype=np.float32)
        self.metadata = getattr(env, "metadata", {})
        self.last_raw_obs: np.ndarray | None = None
        self.last_normalized_obs: np.ndarray | None = None

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        if raw.shape != self.mean.shape:
            raise RuntimeError(f"Observation dimension changed: raw={raw.shape}, train_stats={self.mean.shape}")
        if not np.isfinite(raw).all():
            raise RuntimeError("Raw observation contains NaN or Inf before normalization")
        normal = np.clip((raw - self.mean) / self.std, -CLIP, CLIP).astype(np.float32)
        if not np.isfinite(normal).all():
            raise RuntimeError("Normalized observation contains NaN or Inf")
        self.last_raw_obs, self.last_normalized_obs = raw, normal
        return normal

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._transform(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.env, "last_action_info", {}))

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def patch_engine(cfg: dict[str, Any], out: Path, doc: Path) -> dict[str, Any]:
    old = {name: getattr(engine, name) for name in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT, engine.BASE_DOC = str(cfg["task_id"]), str(cfg["task_name"]), out, doc
    engine.PROMPT = PROMPT
    engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
    engine.STATION, engine.SITES = "SYA", ["SYA"]
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    engine.patch_base_module(out, doc, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
    return old


def restore_engine(old: dict[str, Any]) -> None:
    for name, value in old.items():
        setattr(engine, name, value)


def collect_train_statistics(cfg: dict[str, Any], out: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Collect raw observations from train years only under transparent no-op actions."""
    config = engine.base03222.load_config()
    selection = engine.base03222.build_selection(engine.base03222.load_split())
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = rel(out)
    train_years = list(map(int, cfg["scope"]["train_years"]))
    samples: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for year in train_years:
        env = engine.base03222.base.make_env(config, env_config, "SYA", year, int(cfg.get("seed", 0)), f"SYA_{year}_046_07_train_stats_noop", evaluation=True)
        try:
            obs, _info = env.reset()
            done, step = False, 0
            while not done and step < 260:
                raw = np.asarray(obs, dtype=np.float64).reshape(-1)
                if not np.isfinite(raw).all():
                    raise RuntimeError(f"Train-stat collection received NaN/Inf: year={year}, step={step}")
                samples.append(raw)
                rows.append({"year": year, "step": step, "observation_dim": len(raw)})
                # In this action grid index 0 is explicitly forced to I=0,N=0.
                obs, _reward, terminated, truncated, _info = env.step(0)
                done = bool(terminated or truncated)
                step += 1
            if not done:
                raise RuntimeError(f"Train-stat no-op reference did not finish: SYA{year}")
        finally:
            env.close()
    values = np.vstack(samples)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    if (std < 1e-6).any():
        # Constant features remain valid: denominator is floored to one so the
        # transformed feature stays zero instead of producing an unstable value.
        std = np.where(std < 1e-6, 1.0, std)
    stats = pd.DataFrame({"feature_index": np.arange(values.shape[1]), "count": len(values), "mean_train": mean, "std_train": std, "min_train": values.min(axis=0), "max_train": values.max(axis=0)})
    return mean.astype(np.float32), std.astype(np.float32), stats


def run(cfg_path: Path, dry_run: bool, smoke: bool) -> dict[str, Any]:
    cfg = read_config(cfg_path)
    if smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["task_name"] = f"{cfg['task_name']}_smoke2k"
        cfg["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    out = ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    profile_root = INPUT_PROFILES[str(cfg["input_profile"])]
    preflight = {"task": f"{cfg['task_id']}_{cfg['task_name']}", "input_profile": cfg["input_profile"], "resolved_input_root": rel(profile_root), "input_root_exists": profile_root.exists(), "train_years": cfg["scope"]["train_years"], "validation_years": cfg["scope"]["validation_years"], "timesteps": cfg["training"]["total_timesteps"], "checkpoints": cfg["training"]["checkpoint_steps"], "next_step_allowed": bool(profile_root.exists() and PROMPT.exists())}
    if dry_run:
        print(json.dumps({"mode": "dry_run", **preflight}, ensure_ascii=False, indent=2))
        return preflight
    if not preflight["next_step_allowed"]:
        raise RuntimeError("046_07 preflight failed")
    for directory in [out / "configs", out / "audits", out / "evaluation"]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    old_engine, old_input_root, old_make_env = None, ppo_safe_rendering.MULTISITE_INPUT_ROOT, engine.base03222.base.make_env
    try:
        old_engine = patch_engine(cfg, out, doc)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = profile_root
        mean, std, stats = collect_train_statistics(cfg, out)
        stats.to_csv(out / "audits" / "046_07_train_only_observation_statistics.csv", index=False, encoding="utf-8-sig")
        (out / "configs" / "046_07_train_only_observation_statistics.json").write_text(json.dumps({"source": "2005-2013 no-op reference trajectories only", "clip": CLIP, "mean": mean.tolist(), "std": std.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")

        def normalized_make_env(config: dict, env_config: dict, station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
            return FixedTrainStatisticsObservationWrapper(old_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation), mean, std)

        engine.base03222.base.make_env = normalized_make_env
        # Smoke is intentionally a real short train/evaluate run; full 100K is
        # only run after its audit confirms the profile and statistics.
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
        # The inherited engine retains compatibility filenames (032_22/042_10).
        # Add task-named copies so a person can inspect this run without knowing
        # engine internals; shared report code can also discover either form.
        rename_map = {
            "042_10_training_checkpoint_inventory.csv": "046_07_training_checkpoint_inventory.csv",
            "042_10_checkpoint_validation_summary.csv": "046_07_checkpoint_validation_summary.csv",
            "042_10_validation_summary_by_station_checkpoint.csv": "046_07_validation_summary_by_station_checkpoint.csv",
        }
        for source_name, target_name in rename_map.items():
            source = out / "evaluation" / source_name
            if source.exists():
                shutil.copy2(source, out / "evaluation" / target_name)
    finally:
        engine.base03222.base.make_env = old_make_env
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        if old_engine is not None:
            restore_engine(old_engine)
    record = [
        f"# 046_07 SYA {cfg['input_profile']} 训练年统计量归一化 MaskablePPO",
        "",
        f"- 训练步数：`{cfg['training']['total_timesteps']}`；checkpoint：`{cfg['training']['checkpoint_steps']}`。",
        "- 仅将 25 维 observation 以训练年无操作参考轨迹统计量标准化；reward、动作和安全约束未改。",
        "- 统计量仅来自 2005–2013；验证年不参与拟合。",
        "- 统计量：`benchmark_results/.../audits/046_07_train_only_observation_statistics.csv`。",
    ]
    doc.write_text("\n".join(record) + "\n", encoding="utf-8")
    result = {"task": f"{cfg['task_id']}_{cfg['task_name']}", "algorithm": "MaskablePPO", "normalization": "train_year_noop_mean_std_only", "output_root": rel(out), "statistics_csv": rel(out / "audits" / "046_07_train_only_observation_statistics.csv"), "record_md": rel(doc), "timesteps": cfg["training"]["total_timesteps"]}
    (out / "046_07_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    run(config_path, args.dry_run, args.smoke)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results" / "032_03_lc2010_stress_aware_ppo_training_length"
DOC = ROOT / "docs" / "032_03_lc2010_stress_aware_ppo_training_length_record.md"
CHECKPOINT_STEPS = [10000, 25000, 50000, 75000, 100000]


def ensure_dirs() -> None:
    for rel in ["configs", "models/LCA", "daily_outputs/LCA", "evaluation", "tensorboard/LCA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_path(station: str, year: int, seed: int, step: int) -> Path:
    return OUT / "models" / station / f"{station}_{year}_stress_aware_maskableppo_seed{seed}_ckpt{step}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, station: str, year: int, seed: int, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step and x not in self.outer.saved_steps]
                for target in pending:
                    path = checkpoint_path(self.outer.station, self.outer.year, self.outer.seed, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(target)
                return True

        self.station = station
        self.year = int(year)
        self.seed = int(seed)
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["total_timesteps"] = 100000
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    return cfg


def train(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    seed = int(config["seed"])
    expected = [checkpoint_path(station, year, seed, step) for step in CHECKPOINT_STEPS]
    if all(path.exists() for path in expected):
        rows = [
            {
                "station_code": station,
                "year": year,
                "seed": seed,
                "checkpoint_step": step,
                "run_status": "ok_existing",
                "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "model_sha256": sha256_file(path),
            }
            for step, path in zip(CHECKPOINT_STEPS, expected)
        ]
        return pd.DataFrame(rows)

    rows = []
    env = None
    try:
        env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_032_03_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **base.ppo_kwargs(config),
        )
        cb = FixedStepCheckpointCallback(station, year, seed, CHECKPOINT_STEPS)
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False, callback=cb.callback)
        for step, path in zip(CHECKPOINT_STEPS, expected):
            rows.append(
                {
                    "station_code": station,
                    "year": year,
                    "seed": seed,
                    "checkpoint_step": int(step),
                    "run_status": "ok" if path.exists() else "missing",
                    "model_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "",
                    "model_sha256": sha256_file(path) if path.exists() else "",
                }
            )
    except Exception:
        rows.append(
            {
                "station_code": station,
                "year": year,
                "seed": seed,
                "checkpoint_step": "",
                "run_status": "failed",
                "notes": traceback.format_exc()[-4000:],
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return pd.DataFrame(rows)


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = str(row["station_code"])
    year = int(row["year"])
    seed = int(row["seed"])
    step = int(row["checkpoint_step"])
    model_path = ROOT / str(row["model_path"])
    daily_path = OUT / "daily_outputs" / station / f"{station}_{year}_seed{seed}_ckpt{step}_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        out = base.summarize_daily("MaskablePPO", daily, daily_path, model_path)
        out.update({"station_code": station, "year": year, "seed": seed, "checkpoint_step": step, "run_status": "ok_existing"})
        return out
    if not model_path.exists():
        return {"station_code": station, "year": year, "seed": seed, "checkpoint_step": step, "run_status": "missing_model"}

    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    records = []
    env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_032_03_ckpt{step}_eval", evaluation=True)
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "seed": seed,
                    "checkpoint_step": step,
                    "algorithm": "MaskablePPO",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()
    daily = pd.DataFrame(records)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    out = base.summarize_daily("MaskablePPO", daily, daily_path, model_path)
    out.update({"station_code": station, "year": year, "seed": seed, "checkpoint_step": step, "run_status": "ok"})
    return out


def write_record(config: dict[str, Any], train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    cols = [
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "reward_stress_aware_sum",
        "stress_relief_bonus_sum_unscaled",
        "early_dap1_10_irrigation",
        "early_dap1_10_n",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "action_sequence",
    ]
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy() if not eval_df.empty else pd.DataFrame()
    best_reward = ok.sort_values("reward_stress_aware_sum", ascending=False).iloc[0].to_dict() if not ok.empty else {}
    lines = [
        "# 032_03 LC2010 stress-aware PPO training-length record",
        "",
        "## Scope",
        "",
        "- LCA2010 seed0 only.",
        "- MaskablePPO only.",
        "- Same reward/action/constraint design as 032_00.",
        "- Total timesteps: 100,000.",
        "- Checkpoints: 10k, 25k, 50k, 75k, 100k.",
        "- No checkpoint cherry-picking; full trajectory reported.",
        "",
        "## Training checkpoint inventory",
        "",
        train_df.to_string(index=False),
        "",
        "## Checkpoint evaluation",
        "",
        eval_df[[c for c in cols if c in eval_df.columns]].to_string(index=False) if not eval_df.empty else "No eval rows.",
        "",
        "## First-read branch",
        "",
        f"- Highest reward checkpoint: {best_reward.get('checkpoint_step', 'NA')}.",
        "- Compare against 032_01 `stress_triggered` rule: reward 1.033552, I30/N240, yield 8273.04.",
        "- If no checkpoint approaches lower-water high-reward behavior, longer training alone is not enough.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copyfile(ROOT / "prompts" / "032_03_lc2010_stress_aware_ppo_training_length.md", OUT / "configs" / "032_03_prompt.md")
    config = load_config()
    selection = base.make_selection(config)
    selection.to_csv(OUT / "configs" / "032_03_lc2010_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_03_resolved_env_config.yaml")
    train_df = train(config, env_config)
    train_df.to_csv(OUT / "evaluation" / "032_03_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    eval_rows = []
    for _, row in train_df.iterrows():
        if str(row.get("run_status", "")).startswith("ok"):
            eval_rows.append(evaluate_checkpoint(config, env_config, row))
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT / "evaluation" / "032_03_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    write_record(config, train_df, eval_df)
    result = {
        "task": "032_03_lc2010_stress_aware_ppo_training_length",
        "training_run": True,
        "site_year": "LCA2010",
        "seed": int(config["seed"]),
        "timesteps": int(config["total_timesteps"]),
        "checkpoint_inventory": str((OUT / "evaluation" / "032_03_checkpoint_inventory.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "032_03_checkpoint_eval_summary.csv").relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "032_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        print(eval_df[["checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "reward_stress_aware_sum", "irrigation_event_count", "n_event_count", "action_sequence"]].to_string(index=False))


if __name__ == "__main__":
    main()


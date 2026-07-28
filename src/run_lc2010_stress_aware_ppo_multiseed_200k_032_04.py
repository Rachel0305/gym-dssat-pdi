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
PROMPT = ROOT / "prompts" / "032_04_lc2010_stress_aware_ppo_multiseed_200k.md"
OUT = ROOT / "benchmark_results" / "032_04_lc2010_stress_aware_ppo_multiseed_200k"
DOC = ROOT / "docs" / "032_04_lc2010_stress_aware_ppo_multiseed_200k_record.md"
CHECKPOINT_STEPS = [10000, 20000, 25000, 50000, 75000, 100000, 150000, 200000]
SEEDS = [0, 1, 2]
STATION = "LCA"
YEAR = 2010


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


def load_config(seed: int) -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = 200000
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = YEAR
    return cfg


def train_one_seed(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    seed = int(config["seed"])
    expected = [checkpoint_path(station, year, seed, step) for step in CHECKPOINT_STEPS]
    if all(path.exists() for path in expected):
        return pd.DataFrame(
            [
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
        )

    rows: list[dict[str, Any]] = []
    env = None
    try:
        env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_032_04_seed{seed}_train", evaluation=False)
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
                "model_path": "",
                "model_sha256": "",
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
        out.update(stress_summary(daily))
        return out
    if not model_path.exists():
        return {"station_code": station, "year": year, "seed": seed, "checkpoint_step": step, "run_status": "missing_model"}

    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    records = []
    env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_032_04_seed{seed}_ckpt{step}_eval", evaluation=True)
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
    out.update(stress_summary(daily))
    return out


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def selected_by_seed(eval_df: pd.DataFrame) -> pd.DataFrame:
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    selected = []
    for seed, g in ok.groupby("seed"):
        selected.append(g.sort_values(["reward_stress_aware_sum", "checkpoint_step"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(selected)


def write_record(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    cols = [
        "seed",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_swfac",
        "swfac_days_gt_0p001",
        "max_nstres",
        "nstres_days_gt_0p001",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "action_sequence",
    ]
    selected = selected_by_seed(eval_df)
    final_200k = eval_df[eval_df.get("checkpoint_step", pd.Series(dtype=int)).eq(200000)].copy() if not eval_df.empty else pd.DataFrame()
    lines = [
        "# 032_04 LC2010 stress-aware PPO multiseed 200k record",
        "",
        "## Scope",
        "",
        "- LCA2010 only.",
        "- Seeds: 0, 1, 2.",
        "- MaskablePPO only.",
        "- Total timesteps per seed: 200,000.",
        "- Checkpoints: 10k, 20k, 25k, 50k, 75k, 100k, 150k, 200k.",
        "- Same action constraints and stress-aware reward as 032_03.",
        "- Selection rule: highest `reward_stress_aware_sum` per seed; final 200k is reported separately.",
        "",
        "## Training checkpoint inventory",
        "",
        train_df.to_string(index=False) if not train_df.empty else "No train rows.",
        "",
        "## All checkpoint evaluation",
        "",
        eval_df[[c for c in cols if c in eval_df.columns]].to_string(index=False) if not eval_df.empty else "No eval rows.",
        "",
        "## Selected checkpoint by pre-specified reward rule",
        "",
        selected[[c for c in cols if c in selected.columns]].to_string(index=False) if not selected.empty else "No selected rows.",
        "",
        "## Final 200k checkpoint",
        "",
        final_200k[[c for c in cols if c in final_200k.columns]].to_string(index=False) if not final_200k.empty else "No final rows.",
        "",
        "## First-read interpretation",
        "",
        "- This record should be interpreted only after all three seeds complete.",
        "- If 150k/200k consistently improves over 50k, long training may be useful.",
        "- If selected checkpoints cluster before 100k and final 200k degrades, future work should use validation/early stopping instead of longer training.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    direct_ppo.OUTPUT_ROOT = OUT

    train_parts = []
    eval_rows = []
    for seed in SEEDS:
        config = load_config(seed)
        selection = base.make_selection(config)
        selection.to_csv(OUT / "configs" / f"032_04_lc2010_selection_seed{seed}.csv", index=False, encoding="utf-8-sig")
        env_config = direct_ppo.build_env_config(config, selection)
        direct_ppo.write_yaml(env_config, OUT / "configs" / f"032_04_resolved_env_config_seed{seed}.yaml")
        train_df = train_one_seed(config, env_config)
        train_parts.append(train_df)
        for _, row in train_df.iterrows():
            if str(row.get("run_status", "")).startswith("ok"):
                eval_rows.append(evaluate_checkpoint(config, env_config, row))

    train_all = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    eval_df = pd.DataFrame(eval_rows)
    train_all.to_csv(OUT / "evaluation" / "032_04_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT / "evaluation" / "032_04_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    selected = selected_by_seed(eval_df)
    selected.to_csv(OUT / "evaluation" / "032_04_selected_by_reward.csv", index=False, encoding="utf-8-sig")
    write_record(train_all, eval_df)

    result = {
        "task": "032_04_lc2010_stress_aware_ppo_multiseed_200k",
        "training_run": True,
        "site_year": "LCA2010",
        "seeds": SEEDS,
        "timesteps_per_seed": 200000,
        "checkpoints": CHECKPOINT_STEPS,
        "checkpoint_inventory": str((OUT / "evaluation" / "032_04_checkpoint_inventory.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "032_04_checkpoint_eval_summary.csv").relative_to(ROOT)),
        "selected_summary": str((OUT / "evaluation" / "032_04_selected_by_reward.csv").relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "032_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not selected.empty:
        print("\nSelected checkpoints:")
        print(selected[["seed", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "reward_stress_aware_sum", "action_sequence"]].to_string(index=False))


if __name__ == "__main__":
    main()

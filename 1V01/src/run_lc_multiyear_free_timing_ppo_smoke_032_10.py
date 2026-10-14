from __future__ import annotations

import hashlib
import json
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_10_lc_multiyear_free_timing_ppo_smoke.md"
OUT = ROOT / "benchmark_results" / "032_10_lc_multiyear_free_timing_ppo_smoke"
DOC = ROOT / "docs" / "032_10_lc_multiyear_free_timing_ppo_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"

STATION = "LCA"
TRAIN_YEARS = [2005, 2006, 2007, 2008, 2009, 2010]
SEED = 0
TOTAL_TIMESTEPS = 12000
CHECKPOINT_STEPS = [2000, 5000, 10000, 12000]


def ensure_dirs() -> None:
    for rel in [
        "configs",
        "models/LCA",
        "daily_outputs/LCA",
        "evaluation",
        "logs",
        "tensorboard/LCA",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_config() -> dict[str, Any]:
    config = direct_ppo.load_yaml(CONFIG)
    config = json.loads(json.dumps(config))
    config["seed"] = SEED
    config["total_timesteps"] = TOTAL_TIMESTEPS
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = TRAIN_YEARS[0]
    return config


def make_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    rows = pool[(pool["station_code"].eq(STATION)) & (pool["year"].astype(int).isin(TRAIN_YEARS))].copy()
    found = sorted(rows["year"].astype(int).unique().tolist())
    if found != TRAIN_YEARS:
        raise RuntimeError(f"Expected LC train years {TRAIN_YEARS}, found {found}")
    rows["selected_for_train"] = True
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "032_10_lc_multiyear_free_timing_ppo_smoke"
    return rows.sort_values(["station_code", "year"]).reset_index(drop=True)


def checkpoint_path(step: int) -> Path:
    return OUT / "models" / STATION / f"{STATION}_multiyear_2005_2010_stress_aware_maskableppo_seed{SEED}_ckpt{int(step)}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step_now = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step_now and x not in self.outer.saved_steps]
                for target in pending:
                    path = checkpoint_path(target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(int(target))
                return True

        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


class RandomYearEnv(gym.Env):
    """One SB3 env whose reset samples one fixed DSSAT year from TRAIN_YEARS."""

    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], station: str, years: list[int], seed: int) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.station = station
        self.years = [int(y) for y in years]
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        first_env = self._env_for(self.years[0])
        self.action_space = first_env.action_space
        self.observation_space = first_env.observation_space
        self.metadata = getattr(first_env, "metadata", {})

    def _env_for(self, year: int):
        year = int(year)
        if year not in self.envs:
            self.envs[year] = base.make_env(
                self.config,
                self.env_config,
                self.station,
                year,
                self.seed,
                f"{self.station}_{year}_032_10_multiyear_train",
                evaluation=False,
            )
        return self.envs[year]

    @property
    def current_env(self):
        if self.current_year is None:
            self.current_year = self.years[0]
        return self._env_for(self.current_year)

    def reset(self, *args, **kwargs):
        year = int(self.rng.choice(self.years))
        self.current_year = year
        self.reset_counts[year] += 1
        self.switch_log.append({"episode_index": int(sum(self.reset_counts.values())), "year": year})
        obs, info = self.current_env.reset(*args, **kwargs)
        info = dict(info or {})
        info["active_year"] = year
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info or {})
        info["active_year"] = int(self.current_year)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.current_env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.current_env, "last_action_info", {}))

    def close(self):
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.current_env, name)


def train(config: dict[str, Any], env_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sb3_contrib import MaskablePPO

    expected = [checkpoint_path(step) for step in CHECKPOINT_STEPS]
    if all(path.exists() for path in expected):
        rows = [
            {
                "station_code": STATION,
                "train_years": ",".join(map(str, TRAIN_YEARS)),
                "seed": SEED,
                "checkpoint_step": step,
                "run_status": "ok_existing",
                "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "model_sha256": sha256_file(path),
            }
            for step, path in zip(CHECKPOINT_STEPS, expected)
        ]
        sample = pd.read_csv(OUT / "logs" / "032_10_training_year_reset_counts.csv") if (OUT / "logs" / "032_10_training_year_reset_counts.csv").exists() else pd.DataFrame()
        return pd.DataFrame(rows), sample

    env: RandomYearEnv | None = None
    rows: list[dict[str, Any]] = []
    status = "ok"
    notes = ""
    try:
        env = RandomYearEnv(config, env_config, STATION, TRAIN_YEARS, SEED)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED,
            tensorboard_log=str(OUT / "tensorboard" / STATION),
            **base.ppo_kwargs(config),
        )
        callback = FixedStepCheckpointCallback(CHECKPOINT_STEPS)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=True, progress_bar=False, callback=callback.callback)
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        reset_df = pd.DataFrame(
            [{"year": int(year), "episode_count": int((env.reset_counts if env is not None else Counter()).get(year, 0))} for year in TRAIN_YEARS]
        )
        reset_df.to_csv(OUT / "logs" / "032_10_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        if env is not None:
            pd.DataFrame(env.switch_log).to_csv(OUT / "logs" / "032_10_training_year_switch_log.csv", index=False, encoding="utf-8-sig")
            env.close()

    for step, path in zip(CHECKPOINT_STEPS, expected):
        rows.append(
            {
                "station_code": STATION,
                "train_years": ",".join(map(str, TRAIN_YEARS)),
                "seed": SEED,
                "checkpoint_step": int(step),
                "run_status": status if status != "ok" else ("ok" if path.exists() else "missing"),
                "model_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "",
                "model_sha256": sha256_file(path) if path.exists() else "",
                "notes": notes[-4000:] if notes else "",
            }
        )
    return pd.DataFrame(rows), reset_df


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series, year: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    step = int(row["checkpoint_step"])
    model_path = ROOT / str(row["model_path"])
    daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_seed{SEED}_ckpt{step}_daily.csv"
    if not model_path.exists():
        return {"station_code": STATION, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "missing_model"}
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        out = base.summarize_daily("MaskablePPO", daily, daily_path, model_path)
        out.update({"station_code": STATION, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "ok_existing"})
        out.update(stress_summary(daily))
        return out

    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    env = base.make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_032_10_ckpt{step}_eval", evaluation=True)
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
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
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "year": int(year),
                    "seed": SEED,
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
    out.update({"station_code": STATION, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "ok"})
    out.update(stress_summary(daily))
    return out


def write_record(config: dict[str, Any], train_df: pd.DataFrame, reset_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    eval_cols = [
        "year",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_swfac",
        "max_nstres",
        "swfac_days_gt_0p05",
        "nstres_days_gt_0p05",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "action_sequence",
    ]
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy() if not eval_df.empty else pd.DataFrame()
    by_ckpt = (
        ok.groupby("checkpoint_step", as_index=False)
        .agg(
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            mean_reward=("reward_stress_aware_sum", "mean"),
            max_nstres=("max_nstres", "max"),
            max_swfac=("max_swfac", "max"),
        )
        if not ok.empty
        else pd.DataFrame()
    )
    sampled_all_years = bool((pd.to_numeric(reset_df.get("episode_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).all())
    all_ckpts_saved = bool(train_df["run_status"].astype(str).isin(["ok", "ok_existing"]).all())
    all_evals_complete = bool(len(ok) == len(CHECKPOINT_STEPS) * len(TRAIN_YEARS))
    lines = [
        "# 032_10 LC multi-year free-timing MaskablePPO smoke record",
        "",
        "## Status",
        "",
        f"- Smoke pass: `{sampled_all_years and all_ckpts_saved and all_evals_complete}`.",
        f"- All train years sampled: `{sampled_all_years}`.",
        f"- All checkpoints saved: `{all_ckpts_saved}`.",
        f"- All checkpoint-year evaluations complete: `{all_evals_complete}`.",
        "",
        "## Scope",
        "",
        "- Station: LCA / LC.",
        f"- Training years: {', '.join(map(str, TRAIN_YEARS))}.",
        "- Algorithm: MaskablePPO only.",
        "- Weather forecast: not included.",
        f"- Seed: {SEED}.",
        f"- Total timesteps: {TOTAL_TIMESTEPS}.",
        f"- Checkpoints: {', '.join(map(str, CHECKPOINT_STEPS))}.",
        "- Reward/actions/constraints inherited unchanged from 032_00.",
        "",
        "## Training checkpoint inventory",
        "",
        train_df.to_string(index=False),
        "",
        "## Training year sampling",
        "",
        reset_df.to_string(index=False),
        "",
        "## Mean performance by checkpoint across LC2005-2010",
        "",
        by_ckpt.to_string(index=False) if not by_ckpt.empty else "No successful evaluations.",
        "",
        "## Per-year checkpoint evaluation",
        "",
        eval_df[[c for c in eval_cols if c in eval_df.columns]].to_string(index=False) if not eval_df.empty else "No eval rows.",
        "",
        "## Interpretation boundary",
        "",
        "- This is a smoke test of the no-forecast multi-year training wrapper.",
        "- It is not a final model-selection result.",
        "- It does not evaluate LC2011-2020 or LC2021-2023.",
        "- It does not prove cross-year transfer performance.",
        "- If a follow-up changes reward, forecast inputs, training length, or checkpoint selection, it must be separately pre-registered.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    config = load_config()
    selection = make_selection()
    selection.to_csv(OUT / "configs" / "032_10_lc_train_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_10_resolved_env_config.yaml")

    train_df, reset_df = train(config, env_config)
    train_df.to_csv(OUT / "evaluation" / "032_10_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")

    eval_rows = []
    for _, row in train_df.iterrows():
        if str(row["run_status"]) not in {"ok", "ok_existing"}:
            continue
        for year in TRAIN_YEARS:
            eval_rows.append(evaluate_checkpoint(config, env_config, row, year))
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT / "evaluation" / "032_10_train_year_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")

    write_record(config, train_df, reset_df, eval_df)
    result = {
        "task": "032_10_lc_multiyear_free_timing_ppo_smoke",
        "training_run": True,
        "station": STATION,
        "train_years": TRAIN_YEARS,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoints": CHECKPOINT_STEPS,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "eval_summary": str((OUT / "evaluation" / "032_10_train_year_checkpoint_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_10_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        print(eval_df[["year", "checkpoint_step", "run_status", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "max_nstres", "action_sequence"]].to_string(index=False))


if __name__ == "__main__":
    main()

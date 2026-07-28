from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_linked_free_timing_ppo_dqn_train_smoke_034_04 as linked03404
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


TASK_ID = "035_04"
STATION = "FQA"
YEAR = 2014
SEED = 0
CHECKPOINTS = [10000, 20000, 30000, 40000, 50000]
OUT = ROOT / "benchmark_results" / "035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail"
DOC = ROOT / "docs" / "035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_record.md"
PROMPT = ROOT / "prompts" / "035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail.md"
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "034_00_multisite_input_ic1_four_baseline_rebuild"
    / "evaluation"
    / "034_00_full_baseline_summary.csv"
)


def ensure_dirs() -> None:
    for rel in ["configs", "models/FQA", "daily_outputs/FQA", "snapshots/FQA/2014", "tensorboard/FQA", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env: Any, obs: Any | None = None, info: dict | None = None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


class ProjectProfitDiscreteWrapper(gym.Env):
    """Free-timing discrete wrapper with the 035_04 project simple-profit reward."""

    def __init__(self, env: Any, config: dict[str, Any]):
        super().__init__()
        from gymnasium import spaces

        self.env = env
        self.config = config
        self.grid = base032.action_grid(config)
        self.action_space = spaces.Discrete(len(self.grid))
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.safety_state = ActionSafetyState()
        self.last_action_info: dict[str, Any] = {}
        self.last_obs_dict: dict[str, Any] = {}

    def reset(self, *args: Any, **kwargs: Any):
        self.safety_state = ActionSafetyState()
        self.last_action_info = {}
        result = self.env.reset(*args, **kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self.last_obs_dict = latest_observation_dict(self.env, obs, info)
        return obs, info

    def _dap(self) -> int:
        dap_raw = scalar(self.last_obs_dict.get("dap", 1))
        return int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else 1

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        i = float(raw.get("amir", 0.0))
        n = float(raw.get("anfer", 0.0))
        if i == 0.0 and n == 0.0:
            return True
        cfg = self.config["action_safety"]
        remaining_i = float(cfg["season_irrigation_soft_limit"]) - float(self.safety_state.cumulative_irrigation)
        remaining_n = float(cfg["season_n_soft_limit"]) - float(self.safety_state.cumulative_n)
        i_min, i_max = [int(x) for x in cfg["irrigation_allowed_dap_range"]]
        n_min, n_max = [int(x) for x in cfg["fertilization_allowed_dap_range"]]
        if i > 0:
            if dap < i_min or dap > i_max:
                return False
            if i - remaining_i > 1e-9:
                return False
            if self.safety_state.last_irrigation_dap is not None and dap - int(self.safety_state.last_irrigation_dap) < int(cfg["min_days_between_irrigation"]):
                return False
        if n > 0:
            if dap < n_min or dap > n_max:
                return False
            if n - remaining_n > 1e-9:
                return False
            if self.safety_state.last_fertilization_dap is not None and dap - int(self.safety_state.last_fertilization_dap) < int(cfg["min_days_between_fertilization"]):
                return False
        return True

    def action_masks(self) -> np.ndarray:
        dap = self._dap()
        mask = np.asarray([self._action_is_legal_without_clipping(raw, dap) for raw in self.grid], dtype=bool)
        mask[0] = True
        if not bool(mask.any()):
            mask[0] = True
        return mask

    def step(self, action: Any):
        prev = dict(self.last_obs_dict)
        dap = self._dap()
        requested_action_index = int(np.asarray(action).item())
        mask = self.action_masks()
        action_index = requested_action_index
        mask_forced_noop = False
        if not bool(mask[action_index]):
            action_index = 0
            mask_forced_noop = True
        raw_real = dict(self.grid[action_index])
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, {**self.config["action_safety"], "enabled": True})
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest

        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward_cfg = self.config["reward"]
        water_cost = float(reward_cfg["water_cost"])
        nitrogen_cost = float(reward_cfg["nitrogen_cost"])
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        final_yield_component = 0.0
        if bool(terminated or truncated):
            final_yield_component = scalar(latest.get("grnwt"), scalar(prev.get("grnwt"), 0.0))
        resource_cost = water_cost * safe_i + nitrogen_cost * safe_n
        unscaled_reward = final_yield_component - resource_cost
        reward = unscaled_reward * reward_scale
        self.last_action_info = {
            "discrete_action_index": action_index,
            "requested_discrete_action_index": requested_action_index,
            "mask_forced_noop": bool(mask_forced_noop),
            "raw_action_amir": raw_real.get("amir", np.nan),
            "raw_action_anfer": raw_real.get("anfer", np.nan),
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": safety_result.safety_rule_triggered,
            "mask_valid_action_count": int(self.action_masks().sum()),
            "yield_component": final_yield_component,
            "resource_cost": resource_cost,
            "water_stress_relief": 0.0,
            "nitrogen_stress_relief": 0.0,
            "stress_relief_bonus": 0.0,
            "reward_unscaled": unscaled_reward,
            "reward_scale": reward_scale,
            "reward_stress_aware": reward,
            "previous_cumulative_irrigation": float(before.cumulative_irrigation),
            "previous_cumulative_n": float(before.cumulative_n),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def configure_modules() -> None:
    base032.OUT = OUT
    linked03404.OUT = OUT
    linked03404.TASK_ID = TASK_ID
    linked03404.DOC = DOC
    linked03404.PROMPT = PROMPT
    direct_ppo.OUTPUT_ROOT = OUT


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    base_env = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return ProjectProfitDiscreteWrapper(base_env, config)


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    configure_modules()
    config, env_config = linked03404.load_config_and_env()
    config["seed"] = SEED
    config["total_timesteps"] = max(CHECKPOINTS)
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = YEAR
    config["reward"]["reward_type"] = "harvest_project_simple_profit_scaled_0p001"
    config["reward"]["yield_coef"] = 1.0
    config["reward"]["water_cost"] = 1.1
    config["reward"]["nitrogen_cost"] = 1.58
    config["reward"]["water_stress_relief_coef"] = 0.0
    config["reward"]["nitrogen_stress_relief_coef"] = 0.0
    config["reward"]["reward_scale"] = 0.001
    direct_ppo.write_yaml(config, OUT / "configs" / "035_04_train_config.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "035_04_resolved_env_config.yaml")
    return config, env_config


def checkpoint_model_path(step: int) -> Path:
    return OUT / "models" / STATION / f"maskableppo_project_profit_seed{SEED}_step{step}.zip"


def train_checkpoints(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    rows: list[dict[str, Any]] = []
    env = None
    try:
        env = make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED,
            tensorboard_log=str(OUT / "tensorboard" / STATION),
            **base032.ppo_kwargs(config),
        )
        prev = 0
        for step in CHECKPOINTS:
            chunk = int(step - prev)
            model.learn(total_timesteps=chunk, reset_num_timesteps=(prev == 0), progress_bar=False)
            path = checkpoint_model_path(step)
            model.save(str(path.with_suffix("")))
            rows.append(
                {
                    "algorithm": "MaskablePPO",
                    "station_code": STATION,
                    "year": YEAR,
                    "seed": SEED,
                    "checkpoint_step": step,
                    "chunk_timesteps": chunk,
                    "run_status": "ok",
                    "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "task": TASK_ID,
                    "reward_type": config["reward"]["reward_type"],
                    "linked_management_expected": True,
                }
            )
            prev = step
    except Exception:
        rows.append(
            {
                "algorithm": "MaskablePPO",
                "station_code": STATION,
                "year": YEAR,
                "seed": SEED,
                "checkpoint_step": -1,
                "run_status": "failed",
                "model_path": "",
                "task": TASK_ID,
                "notes": traceback.format_exc()[-5000:],
            }
        )
    finally:
        if env is not None:
            env.close()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "evaluation" / "035_04_training_checkpoints.csv", index=False, encoding="utf-8-sig")
    return out


def overview_modes(snapshot: Path) -> str:
    overview = snapshot / "OVERVIEW.OUT"
    if not overview.exists():
        return ""
    lines = [line.strip() for line in overview.read_text(encoding="utf-8", errors="ignore").splitlines() if "MANAGEMENT OPT" in line]
    return " | ".join(lines[:3])


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], train_row: pd.Series) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    if str(train_row.get("run_status")) != "ok":
        return pd.DataFrame([{"run_status": "failed", "notes": train_row.get("notes", "")}])
    model_path = ROOT / str(train_row["model_path"])
    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    env = make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_step{int(train_row['checkpoint_step'])}_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, YEAR)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "year": YEAR,
                    "seed": SEED,
                    "algorithm": "MaskablePPO",
                    "checkpoint_step": int(train_row["checkpoint_step"]),
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
        daily = pd.DataFrame(records)
        step = int(train_row["checkpoint_step"])
        daily_path = OUT / "daily_outputs" / STATION / f"{YEAR}_maskableppo_project_profit_step{step}_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]) if len(daily) else np.nan
        snapshot = OUT / "snapshots" / STATION / str(YEAR) / f"maskableppo_project_profit_step{step}"
        tmp = siteppo.snapshot_from_env(env)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(tmp, snapshot)
        metrics = baseline03400.metrics_from_snapshot(snapshot, final_y)
        summary = base032.summarize_daily("MaskablePPO", daily, daily_path, model_path)
        overview = overview_modes(snapshot)
        summary.update(
            {
                "task": TASK_ID,
                "station_code": STATION,
                "year": YEAR,
                "seed": SEED,
                "checkpoint_step": step,
                "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
                "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
                "overview_management_opt": overview,
                "overview_is_linked": "IRRIG   :L" in overview and "FERT :L" in overview,
                "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
                "safe_summary_i_match": abs(float(summary["total_irrigation"]) - float(metrics["actual_irrigation_mm"])) < 1e-6,
                "safe_summary_n_match": abs(float(summary["total_n"]) - float(metrics["actual_nitrogen_kg_ha"])) < 1e-6,
                **metrics,
            }
        )
        summary["interface_pass"] = (
            bool(summary["safe_summary_i_match"])
            and bool(summary["safe_summary_n_match"])
            and bool(summary["overview_is_linked"])
        )
        summary["grain_yield_kg_ha"] = float(summary["final_grnwt"])
        summary["actual_irrigation_mm"] = float(summary["summary_irrigation_mm"])
        summary["actual_nitrogen_kg_ha"] = float(summary["summary_n_kg_ha"])
        summary["project_simple_profit"] = (
            float(summary["grain_yield_kg_ha"])
            - 1.1 * float(summary["actual_irrigation_mm"])
            - 1.58 * float(summary["actual_nitrogen_kg_ha"])
        )
        return pd.DataFrame([summary])
    except Exception:
        return pd.DataFrame(
            [
                {
                    "algorithm": "MaskablePPO",
                    "checkpoint_step": int(train_row.get("checkpoint_step", -1)),
                    "run_status": "eval_failed",
                    "notes": traceback.format_exc()[-5000:],
                }
            ]
        )
    finally:
        env.close()


def evaluate_checkpoints(config: dict[str, Any], env_config: dict[str, Any], train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in train.iterrows():
        rows.append(evaluate_checkpoint(config, env_config, row))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out.to_csv(OUT / "evaluation" / "035_04_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    return out


def expert_row() -> pd.Series:
    baseline = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    baseline["year"] = pd.to_numeric(baseline["year"], errors="coerce").astype(int)
    row = baseline[
        baseline["station_code"].astype(str).eq(STATION)
        & baseline["year"].eq(YEAR)
        & baseline["scenario"].astype(str).eq("official_extension_expert")
    ]
    if len(row) != 1:
        raise ValueError(f"official_extension_expert not found for {STATION}{YEAR}")
    out = row.iloc[0].copy()
    for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["project_simple_profit"] = (
        float(out["grain_yield_kg_ha"]) - 1.1 * float(out["actual_irrigation_mm"]) - 1.58 * float(out["actual_nitrogen_kg_ha"])
    )
    return out


def guardrail(eval_df: pd.DataFrame) -> pd.DataFrame:
    expert = expert_row()
    out = eval_df.copy()
    out["grain_yield_kg_ha"] = pd.to_numeric(out["grain_yield_kg_ha"], errors="coerce")
    out["actual_irrigation_mm"] = pd.to_numeric(out["actual_irrigation_mm"], errors="coerce")
    out["actual_nitrogen_kg_ha"] = pd.to_numeric(out["actual_nitrogen_kg_ha"], errors="coerce")
    out["WP_ET_kg_m3"] = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce")
    out["PFP_N_kg_kg"] = pd.to_numeric(out["PFP_N_kg_kg"], errors="coerce")
    out["project_simple_profit"] = pd.to_numeric(out["project_simple_profit"], errors="coerce")
    out["delta_yield_vs_expert"] = out["grain_yield_kg_ha"] - float(expert["grain_yield_kg_ha"])
    out["delta_i_vs_expert"] = out["actual_irrigation_mm"] - float(expert["actual_irrigation_mm"])
    out["delta_n_vs_expert"] = out["actual_nitrogen_kg_ha"] - float(expert["actual_nitrogen_kg_ha"])
    out["delta_wp_vs_expert"] = out["WP_ET_kg_m3"] - float(expert["WP_ET_kg_m3"])
    out["delta_pfp_vs_expert"] = out["PFP_N_kg_kg"] - float(expert["PFP_N_kg_kg"])
    out["delta_profit_vs_expert"] = out["project_simple_profit"] - float(expert["project_simple_profit"])
    out["beats_expert_yield"] = out["delta_yield_vs_expert"] >= 0
    out["beats_expert_wp"] = out["delta_wp_vs_expert"] > 0
    out["beats_expert_pfp"] = out["delta_pfp_vs_expert"] > 0
    out["guardrail_pass"] = (
        out["interface_pass"].astype(bool)
        & out["beats_expert_yield"].astype(bool)
        & out[["beats_expert_yield", "beats_expert_wp", "beats_expert_pfp"]].any(axis=1)
    )
    out = out.sort_values(
        ["guardrail_pass", "project_simple_profit", "actual_nitrogen_kg_ha", "actual_irrigation_mm"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    out.insert(0, "guardrail_rank", range(1, len(out) + 1))
    out.to_csv(OUT / "evaluation" / "035_04_guardrail_ranking.csv", index=False, encoding="utf-8-sig")
    return out


def write_record(train: pd.DataFrame, eval_df: pd.DataFrame, ranked: pd.DataFrame, elapsed: float) -> None:
    expert = expert_row()
    pass_count = int(ranked["guardrail_pass"].astype(bool).sum()) if "guardrail_pass" in ranked else 0
    lines = [
        "# 035_04 FQA2014 linked 自由时序 PPO 项目 simple_profit reward 记录",
        "",
        "## 结论先说",
        "",
        f"- 训练 checkpoint 数：{len(train)}；评估 checkpoint 数：{len(eval_df)}。",
        f"- guardrail 通过数：{pass_count}/{len(ranked)}。",
        "- 本任务只改 reward 口径：训练 reward 等价于 `final_GRNWT - 1.1I - 1.58N` 后乘 0.001。",
        "- 本任务没有调 PPO 超参数、没有增加 seed、没有扩展站点年份。",
        "",
        "## expert 参照",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "expert_yield": expert["grain_yield_kg_ha"],
                        "expert_irrigation": expert["actual_irrigation_mm"],
                        "expert_nitrogen": expert["actual_nitrogen_kg_ha"],
                        "expert_WP_ET": expert["WP_ET_kg_m3"],
                        "expert_PFP_N": expert["PFP_N_kg_kg"],
                        "expert_project_simple_profit": expert["project_simple_profit"],
                    }
                ]
            )
        ),
        "",
        "## 训练 checkpoints",
        "",
        md_table(train, 20),
        "",
        "## checkpoint guardrail 排名",
        "",
        md_table(
            ranked[
                [
                    "guardrail_rank",
                    "checkpoint_step",
                    "guardrail_pass",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "project_simple_profit",
                    "delta_yield_vs_expert",
                    "delta_i_vs_expert",
                    "delta_n_vs_expert",
                    "delta_wp_vs_expert",
                    "delta_pfp_vs_expert",
                    "interface_pass",
                    "action_sequence",
                ]
            ],
            20,
        ),
        "",
        "## 边界",
        "",
        "- 若本任务失败，只能说明该 reward 变体在 FQA2014 seed0 50K 内没有解决问题，不能推出 PPO/DQN 或自由时序整体不可行。",
        "- 若本任务通过，也只能说明单站点单年单 seed 有正向信号，不能直接扩展到全站点。",
        "",
        f"耗时：{elapsed:.1f} 秒",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    train = train_checkpoints(config, env_config)
    eval_df = evaluate_checkpoints(config, env_config, train)
    ranked = guardrail(eval_df) if len(eval_df) else pd.DataFrame()
    elapsed = time.time() - start
    write_record(train, eval_df, ranked, elapsed)
    print(
        {
            "task": TASK_ID,
            "trained_checkpoints": int(len(train)),
            "evaluated_checkpoints": int(len(eval_df)),
            "guardrail_pass": int(ranked["guardrail_pass"].astype(bool).sum()) if len(ranked) else 0,
            "record_md": str(DOC.relative_to(ROOT)),
            "guardrail_csv": str((OUT / "evaluation" / "035_04_guardrail_ranking.csv").relative_to(ROOT)),
        }
    )
    if len(ranked):
        print(
            ranked[
                [
                    "guardrail_rank",
                    "checkpoint_step",
                    "guardrail_pass",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "project_simple_profit",
                    "action_sequence",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

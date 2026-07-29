from __future__ import annotations

import argparse
import json
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as split_base
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as rl_base
from strict_maskable_dqn_040 import StrictMaskableDQN


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_02"
TASK_NAME = "sya_lowIC_strict_maskable_dqn"

CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DEFAULT_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DEFAULT_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
LOWIC_CLASSIFICATION = (
    ROOT
    / "benchmark_results"
    / "039_02_original_vs_lowIC_three_baseline_audit"
    / "tables"
    / "039_02_lowIC_usability_classification.csv"
)

STATION = "SYA"
SITE = "SY"
SEED = 0
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

OUT = DEFAULT_OUT
DOC = DEFAULT_DOC
TOTAL_TIMESTEPS = DEFAULT_TOTAL_TIMESTEPS
CHECKPOINT_STEPS = list(DEFAULT_CHECKPOINT_STEPS)


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "logs", "models/SYA", "daily_outputs/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 160) -> str:
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


def load_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED
    cfg["total_timesteps"] = int(TOTAL_TIMESTEPS)
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    return cfg


def load_split() -> pd.DataFrame:
    old_sites = split_base.SITES
    split_base.SITES = [STATION]
    try:
        split = split_base.load_split().copy()
    finally:
        split_base.SITES = old_sites
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split.sort_values(["station_code", "year"]).reset_index(drop=True)


def build_selection(split: pd.DataFrame) -> pd.DataFrame:
    old_sites = split_base.SITES
    split_base.SITES = [STATION]
    try:
        selected = split_base.build_selection(split).copy()
    finally:
        split_base.SITES = old_sites
    selected["selection_reason"] = f"{TASK_ID}_{TASK_NAME}"
    return selected.sort_values(["station_code", "year"]).reset_index(drop=True)


def load_lowic_classification() -> pd.DataFrame:
    if not LOWIC_CLASSIFICATION.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOWIC_CLASSIFICATION, keep_default_na=False)
    df = df[df["station_code"].eq(STATION)].copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.sort_values(["station_code", "year"]).reset_index(drop=True)


def model_path(step: int) -> Path:
    return OUT / "models" / STATION / f"{STATION}_lowIC_strict_maskable_dqn_seed{SEED}_ckpt{int(step)}.pt"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RandomYearEnv(gym.Env):
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], years: list[int]) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.years = [int(y) for y in years]
        self.rng = np.random.default_rng(SEED)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        first = self._env_for(self.years[0])
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.metadata = getattr(first, "metadata", {})

    def _env_for(self, year: int):
        year = int(year)
        if year not in self.envs:
            self.envs[year] = rl_base.make_env(
                self.config,
                self.env_config,
                STATION,
                year,
                SEED,
                f"{STATION}_{year}_{TASK_ID}_strict_maskable_dqn_train",
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
        self.switch_log.append({"episode_index": int(sum(self.reset_counts.values())), "station_code": STATION, "year": year})
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


def flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def make_model(config: dict[str, Any], observation_dim: int, action_dim: int) -> StrictMaskableDQN:
    dqn_cfg = config["dqn"]
    return StrictMaskableDQN(
        observation_dim=observation_dim,
        action_dim=action_dim,
        learning_rate=float(dqn_cfg["learning_rate"]),
        gamma=float(dqn_cfg["gamma"]),
        buffer_size=int(dqn_cfg["buffer_size"]),
        batch_size=int(dqn_cfg["batch_size"]),
        net_arch=[int(x) for x in dqn_cfg["net_arch"]],
        weight_decay=float(dqn_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(dqn_cfg["max_grad_norm"]),
        exploration_initial_eps=float(dqn_cfg["exploration_initial_eps"]),
        exploration_final_eps=float(dqn_cfg["exploration_final_eps"]),
        exploration_fraction=float(dqn_cfg["exploration_fraction"]),
        total_timesteps=int(config["total_timesteps"]),
        seed=SEED,
        device="cpu",
    )


def train_strict_maskable_dqn(config: dict[str, Any], env_config: dict[str, Any], train_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = [model_path(step) for step in CHECKPOINT_STEPS]
    rows: list[dict[str, Any]] = []
    if all(path.exists() for path in expected):
        for step, path in zip(CHECKPOINT_STEPS, expected):
            restored, saved_step = StrictMaskableDQN.load(path)
            rows.append(
                {
                    "algorithm": "StrictMaskableDQN",
                    "station_code": STATION,
                    "site": SITE,
                    "train_years": ",".join(map(str, train_years)),
                    "seed": SEED,
                    "checkpoint_step": step,
                    "saved_step": saved_step,
                    "run_status": "ok_existing",
                    "model_path": path.relative_to(ROOT).as_posix(),
                    "model_sha256": sha256_file(path),
                    "optimizer_updates": restored.optimizer_updates,
                    "target_updates": restored.target_updates,
                    "notes": "Existing checkpoint reused.",
                }
            )
        return pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()

    env: RandomYearEnv | None = None
    status = "ok"
    notes = ""
    update_rows: list[dict[str, Any]] = []
    reset_df = pd.DataFrame()
    try:
        env = RandomYearEnv(config, env_config, train_years)
        obs, info = env.reset()
        obs_arr = flatten_obs(obs)
        model = make_model(config, obs_arr.shape[0], int(env.action_space.n))
        learning_starts = int(config["dqn"]["learning_starts"])
        train_freq = int(config["dqn"]["train_freq"])
        gradient_steps = int(config["dqn"]["gradient_steps"])
        target_update_interval = int(config["dqn"]["target_update_interval"])
        checkpoint_set = set(int(x) for x in CHECKPOINT_STEPS)
        for global_step in range(1, int(TOTAL_TIMESTEPS) + 1):
            mask = np.asarray(env.action_masks(), dtype=bool)
            action = model.select_action(obs_arr, mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            next_obs_arr = flatten_obs(next_obs)
            next_mask = np.zeros(model.action_dim, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool)
            model.add_transition(
                observation=obs_arr,
                action=action,
                reward=float(reward),
                next_observation=next_obs_arr,
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            if global_step > learning_starts and global_step % train_freq == 0:
                for _ in range(gradient_steps):
                    update_rows.append({"global_step": global_step, **model.train_step()})
            if global_step % target_update_interval == 0:
                model.sync_target()
            if global_step in checkpoint_set:
                model.save(model_path(global_step), global_step=global_step)
            if done and global_step < int(TOTAL_TIMESTEPS):
                obs, info = env.reset()
                obs_arr = flatten_obs(obs)
            else:
                obs_arr = next_obs_arr
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            reset_df = pd.DataFrame(
                [{"station_code": STATION, "year": int(year), "episode_count": int(env.reset_counts.get(year, 0))} for year in train_years]
            )
            pd.DataFrame(env.switch_log).to_csv(OUT / "logs" / "040_02_training_year_switch_log.csv", index=False, encoding="utf-8-sig")
            env.close()

    for step, path in zip(CHECKPOINT_STEPS, expected):
        saved_step = ""
        optimizer_updates = ""
        target_updates = ""
        if path.exists():
            restored, saved_step_int = StrictMaskableDQN.load(path)
            saved_step = int(saved_step_int)
            optimizer_updates = int(restored.optimizer_updates)
            target_updates = int(restored.target_updates)
        rows.append(
            {
                "algorithm": "StrictMaskableDQN",
                "station_code": STATION,
                "site": SITE,
                "train_years": ",".join(map(str, train_years)),
                "seed": SEED,
                "checkpoint_step": step,
                "saved_step": saved_step,
                "run_status": status if status != "ok" else ("ok" if path.exists() else "missing"),
                "model_path": path.relative_to(ROOT).as_posix() if path.exists() else "",
                "model_sha256": sha256_file(path) if path.exists() else "",
                "optimizer_updates": optimizer_updates,
                "target_updates": target_updates,
                "notes": notes[-4000:] if notes else "",
            }
        )
    return pd.DataFrame(rows), reset_df, pd.DataFrame(update_rows)


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series, year: int) -> dict[str, Any]:
    model, saved_step = StrictMaskableDQN.load(ROOT / str(row["model_path"]))
    weather = direct_ppo.weather_for_daily(config)
    env = rl_base.make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_{TASK_ID}_eval_{int(row['checkpoint_step'])}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        obs_arr = flatten_obs(obs)
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = rl_base.latest_observation_dict(env, obs, info)
            dap_raw = rl_base.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = np.asarray(env.action_masks(), dtype=bool)
            action = model.select_action(obs_arr, mask, step_count, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_arr = flatten_obs(obs)
            done = bool(terminated or truncated)
            latest = rl_base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "seed": SEED,
                    "algorithm": "StrictMaskableDQN",
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "saved_step": int(saved_step),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": rl_base.scalar(wrow.get("rain"), np.nan),
                    "srad": rl_base.scalar(wrow.get("srad"), np.nan),
                    "tmax": rl_base.scalar(wrow.get("tmax"), np.nan),
                    "tmin": rl_base.scalar(wrow.get("tmin"), np.nan),
                    "swfac": rl_base.scalar(latest.get("swfac")),
                    "nstres": rl_base.scalar(latest.get("nstres")),
                    "topwt": rl_base.scalar(latest.get("topwt")),
                    "grnwt": rl_base.scalar(latest.get("grnwt")),
                    "xlai": rl_base.scalar(latest.get("xlai")),
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
    daily_path = OUT / "daily_outputs" / STATION / f"{int(year)}_ckpt{int(row['checkpoint_step'])}_strict_maskable_dqn_eval_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    out = rl_base.summarize_daily("StrictMaskableDQN", daily, daily_path, ROOT / str(row["model_path"]))
    out.update(
        {
            "station_code": STATION,
            "site": SITE,
            "year": int(year),
            "seed": SEED,
            "checkpoint_step": int(row["checkpoint_step"]),
            "saved_step": int(saved_step),
            "run_status": out.get("run_status", "ok"),
        }
    )
    return out


def summarize_by_checkpoint(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    for col in ["final_grnwt", "total_irrigation", "total_n", "PFP_N", "swfac_stress_days_gt_0p05", "nstres_days_gt_0p05"]:
        if col not in ok.columns:
            ok[col] = pd.NA
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    return (
        ok.groupby(["station_code", "site", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            mean_swfac_stress_days_gt_0p05=("swfac_stress_days_gt_0p05", "mean"),
            mean_nstres_days_gt_0p05=("nstres_days_gt_0p05", "mean"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


def write_record(split: pd.DataFrame, train_df: pd.DataFrame, reset_df: pd.DataFrame, update_df: pd.DataFrame, eval_df: pd.DataFrame, by_ckpt: pd.DataFrame) -> None:
    cls = load_lowic_classification()
    lines = [
        "# 040_02 SYA lowIC 严格 MaskableDQN 记录",
        "",
        "## 结论先说",
        "",
        "- 本任务使用项目本地 StrictMaskableDQN；mask 进入探索、贪心、replay 和 Bellman target。",
        "- 除 DQN 框架外，输入、年份、动作空间、约束和奖励均沿用 040_00/040_01。",
        "",
        "## 固定配置",
        "",
        f"- 输入根目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 站点：`{STATION}`",
        f"- seed：`{SEED}`",
        f"- 训练步数：`{TOTAL_TIMESTEPS}`",
        f"- checkpoint：`{', '.join(map(str, CHECKPOINT_STEPS))}`",
        "",
        "## SYA lowIC 可用性分类",
        "",
        md_table(cls[["station_code", "year", "lowIC_usability_class", "lowIC_usability_reason"]] if not cls.empty else cls),
        "",
        "## 年份划分",
        "",
        md_table(split[[c for c in ["station_code", "site", "year", "split", "selected_for_train", "selected_for_eval"] if c in split.columns]]),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train_df),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset_df),
        "",
        "## 更新日志节选",
        "",
        md_table(update_df.tail(40), max_rows=40),
        "",
        "## 验证集 checkpoint 汇总",
        "",
        md_table(by_ckpt),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, max_rows=240),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run() -> dict[str, Any]:
    split = load_split()
    cls = load_lowic_classification()
    config = load_config()
    return {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "station": STATION,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "lowIC_class_counts": cls["lowIC_usability_class"].value_counts().to_dict() if "lowIC_usability_class" in cls.columns else {},
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "config_dqn": config.get("dqn", {}),
        "config_action_safety": config.get("action_safety", {}),
        "config_discrete_actions": config.get("discrete_actions", {}),
        "config_reward": config.get("reward", {}),
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and not split.empty),
    }


def run_training() -> None:
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)
    ensure_dirs()
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)

    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        config = load_config()
        split = load_split()
        selection = build_selection(split)
        selection.to_csv(OUT / "configs" / "040_02_half_split_selection.csv", index=False, encoding="utf-8-sig")
        direct_ppo.OUTPUT_ROOT = OUT
        env_config = direct_ppo.build_env_config(config, selection)
        direct_ppo.write_yaml(env_config, OUT / "configs" / "040_02_resolved_env_config.yaml")
        train_years = split[split["split"].eq("train")]["year"].astype(int).tolist()
        validation_years = split[split["split"].eq("validation")]["year"].astype(int).tolist()

        train_df, reset_df, update_df = train_strict_maskable_dqn(config, env_config, train_years)
        train_df.to_csv(OUT / "evaluation" / "040_02_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        reset_df.to_csv(OUT / "logs" / "040_02_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        update_df.to_csv(OUT / "logs" / "040_02_update_log.csv", index=False, encoding="utf-8-sig")

        eval_rows: list[dict[str, Any]] = []
        train_ok = train_df[train_df["run_status"].astype(str).str.startswith("ok")].copy()
        for _, row in train_ok.iterrows():
            for year in validation_years:
                try:
                    eval_rows.append(evaluate_checkpoint(config, env_config, row, int(year)))
                except Exception:
                    eval_rows.append(
                        {
                            "algorithm": "StrictMaskableDQN",
                            "station_code": STATION,
                            "site": SITE,
                            "year": int(year),
                            "seed": SEED,
                            "checkpoint_step": int(row["checkpoint_step"]),
                            "run_status": "failed",
                            "notes": traceback.format_exc()[-4000:],
                        }
                    )
        eval_df = pd.DataFrame(eval_rows)
        by_ckpt = summarize_by_checkpoint(eval_df)
        eval_df.to_csv(OUT / "evaluation" / "040_02_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        by_ckpt.to_csv(OUT / "evaluation" / "040_02_validation_summary_by_checkpoint.csv", index=False, encoding="utf-8-sig")
        write_record(selection, train_df, reset_df, update_df, eval_df, by_ckpt)
        result = {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "algorithm": "StrictMaskableDQN",
            "record_md": DOC.relative_to(ROOT).as_posix(),
            "train_inventory": (OUT / "evaluation" / "040_02_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
            "validation_summary": (OUT / "evaluation" / "040_02_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
            "by_checkpoint": (OUT / "evaluation" / "040_02_validation_summary_by_checkpoint.csv").relative_to(ROOT).as_posix(),
            "stations": [STATION],
            "total_timesteps_per_station": TOTAL_TIMESTEPS,
            "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        }
        (OUT / "040_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root


def parse_checkpoint_steps(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("checkpoint-steps cannot be empty")
    return sorted(values)


def configure_run(suffix: str | None, timesteps: int | None, checkpoint_steps: list[int] | None) -> None:
    global OUT, DOC, TOTAL_TIMESTEPS, CHECKPOINT_STEPS
    if suffix:
        OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}"
        DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md"
    if timesteps is not None:
        TOTAL_TIMESTEPS = int(timesteps)
    if checkpoint_steps is not None:
        CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]


def main() -> None:
    parser = argparse.ArgumentParser(description="040_02 SYA lowIC strict MaskableDQN")
    parser.add_argument("--dry-run", action="store_true", help="Only verify routing/config/year split; do not train.")
    parser.add_argument("--timesteps", type=int, default=None, help="Optional engineering-smoke override; default is 100000.")
    parser.add_argument("--checkpoint-steps", type=str, default=None, help="Comma-separated checkpoint override for smoke runs.")
    parser.add_argument("--suffix", type=str, default=None, help="Optional output suffix, useful for smoke runs.")
    args = parser.parse_args()
    configure_run(args.suffix, args.timesteps, parse_checkpoint_steps(args.checkpoint_steps) if args.checkpoint_steps else None)
    if args.dry_run:
        summary = dry_run()
        ensure_dirs()
        (OUT / "040_02_dry_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(summary["split_years"]).to_csv(OUT / "configs" / "040_02_half_split_selection_dry_run.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    run_training()


if __name__ == "__main__":
    main()


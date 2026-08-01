from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out, select_matching_row
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table

import build_sya_lowIC_teacher_candidate_search_041_00 as base04100
import ppo_safe_rendering
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as ppo04040


TASK_ID = "041_03"
TASK_NAME = "sya_lowIC_teacher_warmstart_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

TEACHER_SELECTED = (
    ROOT
    / "benchmark_results"
    / "041_02_sya_lowIC_layered_teacher_imitation_dataset"
    / "tables"
    / "041_02_selected_teacher_trajectories.csv"
)
YEARS = list(range(2014, 2024))
STATION = "SYA"
SITE = "SY"
SEED = 0
DEFAULT_TIMESTEPS = 100_000
DEFAULT_CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
BC_EPOCHS = 20
BC_BATCH_SIZE = 256
BC_LR = 1e-4


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def ensure_dirs(out: Path) -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "bc_dataset"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    BASE_DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def parse_checkpoint_steps(raw: str | None, timesteps: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [x for x in DEFAULT_CHECKPOINTS if x <= int(timesteps)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selected_teachers() -> pd.DataFrame:
    if not TEACHER_SELECTED.exists():
        raise FileNotFoundError(TEACHER_SELECTED)
    df = pd.read_csv(TEACHER_SELECTED, keep_default_na=False)
    if sorted(df["year"].astype(int).tolist()) != YEARS:
        raise RuntimeError(f"041_02 selected teacher years mismatch: {df['year'].tolist()}")
    return df.sort_values("year").reset_index(drop=True)


def build_config_and_env_config(years: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = ppo04040.load_config()
    selection = base04100.build_selection_for_years(years)
    env_config = ppo04040.base03222.direct_ppo.build_env_config(config, selection)
    return config, env_config


def action_index_for(env: Any, amir: float, anfer: float) -> int:
    matches = [
        idx
        for idx, raw in enumerate(env.grid)
        if abs(float(raw.get("amir", 0.0)) - float(amir)) < 1e-9
        and abs(float(raw.get("anfer", 0.0)) - float(anfer)) < 1e-9
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot map action amir={amir}, anfer={anfer}; matches={matches}")
    return int(matches[0])


def combined_schedule(row: Any) -> dict[int, dict[str, float]]:
    water = {int(k): float(v) for k, v in json.loads(str(row.water_schedule_json)).items()}
    nitrogen = {int(k): float(v) for k, v in json.loads(str(row.n_schedule_json)).items()}
    schedule: dict[int, dict[str, float]] = {}
    for dap, amount in water.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["amir"] += float(amount)
    for dap, amount in nitrogen.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["anfer"] += float(amount)
    return dict(sorted(schedule.items()))


def collect_bc_dataset(out: Path, selected: pd.DataFrame, config: dict[str, Any], env_config: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    weight_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04100.LOWIC_INPUT_ROOT
    try:
        for teacher in selected.itertuples(index=False):
            year = int(teacher.year)
            schedule = combined_schedule(teacher)
            env = ppo04040.make_env_with_yield_guardrail(
                config,
                env_config,
                STATION,
                year,
                SEED,
                f"{STATION}_{year}_041_03_bc_collect",
                evaluation=True,
            )
            fired: set[int] = set()
            try:
                obs, info = env.reset()
                for step in range(420):
                    dap = int(round(scalar(env.last_obs_dict.get("dap", step + 1), step + 1)))
                    requested = schedule[dap] if dap in schedule and dap not in fired else {"amir": 0.0, "anfer": 0.0}
                    if dap in schedule:
                        fired.add(dap)
                    action_idx = action_index_for(env, requested["amir"], requested["anfer"])
                    mask = np.asarray(env.action_masks(), dtype=bool)
                    if not bool(mask[action_idx]):
                        raise RuntimeError(f"Teacher action masked out: year={year}, step={step}, dap={dap}, action={action_idx}")
                    nonzero = bool(float(requested["amir"]) > 0 or float(requested["anfer"]) > 0)
                    tier = str(teacher.teacher_tier)
                    sample_weight = 1.0 if tier == "strong_all3" and nonzero else 0.2 if tier == "strong_all3" else 0.5 if nonzero else 0.1
                    obs_rows.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
                    mask_rows.append(mask.copy())
                    action_rows.append(int(action_idx))
                    weight_rows.append(float(sample_weight))
                    meta_rows.append(
                        {
                            "station_code": STATION,
                            "site": SITE,
                            "year": year,
                            "step": int(step),
                            "dap": int(dap),
                            "teacher_tier": tier,
                            "candidate_id": str(teacher.candidate_id),
                            "teacher_action_index": int(action_idx),
                            "teacher_irrigation_mm": float(requested["amir"]),
                            "teacher_nitrogen_kg_ha": float(requested["anfer"]),
                            "is_nonzero_action_day": nonzero,
                            "sample_weight": float(sample_weight),
                        }
                    )
                    obs, reward, terminated, truncated, info = env.step(action_idx)
                    if terminated or truncated:
                        break
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    dataset = {
        "obs": np.stack(obs_rows).astype(np.float32),
        "masks": np.stack(mask_rows).astype(bool),
        "actions": np.asarray(action_rows, dtype=np.int64),
        "weights": np.asarray(weight_rows, dtype=np.float32),
    }
    meta = pd.DataFrame(meta_rows)
    np.savez_compressed(out / "bc_dataset" / "041_03_bc_dataset.npz", **dataset)
    meta.to_csv(out / "bc_dataset" / "041_03_bc_dataset_meta.csv", index=False)
    return dataset, meta


class RandomYearEnv(gym.Env):
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], years: list[int], seed: int) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.years = [int(y) for y in years]
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
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
            self.envs[year] = ppo04040.make_env_with_yield_guardrail(
                self.config,
                self.env_config,
                STATION,
                year,
                self.seed,
                f"{STATION}_{year}_041_03_train",
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


class FixedStepCheckpointCallback:
    def __init__(self, out: Path, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                step_now = int(self.model.num_timesteps)
                pending = [x for x in outer.checkpoint_steps if x <= step_now and x not in outer.saved_steps]
                for target in pending:
                    path = outer.model_path(target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path.with_suffix("")))
                    outer.saved_steps.append(int(target))
                return True

        self.out = out
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback()

    def model_path(self, step: int) -> Path:
        return self.out / "models" / STATION / f"{STATION}_teacher_warmstart_maskableppo_seed{SEED}_ckpt{int(step)}.zip"


def pretrain_policy_bc(model: Any, dataset: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    device = model.device
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    masks = torch.as_tensor(dataset["masks"], dtype=torch.bool, device=device)
    weights = torch.as_tensor(dataset["weights"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=BC_LR)
    rng = np.random.default_rng(SEED)
    rows = []
    n = int(obs.shape[0])
    for epoch in range(1, BC_EPOCHS + 1):
        order = rng.permutation(n)
        losses = []
        accuracies = []
        for start in range(0, n, BC_BATCH_SIZE):
            idx_np = order[start : start + BC_BATCH_SIZE]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_masks = masks[idx]
            batch_weights = weights[idx]
            try:
                distribution = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            except TypeError:
                distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_actions)
            loss = -((log_prob * batch_weights).sum() / batch_weights.sum().clamp_min(1e-9))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 10.0)
            optimizer.step()
            with torch.no_grad():
                probs = distribution.distribution.probs
                masked_probs = probs.masked_fill(~batch_masks, -1.0)
                pred = torch.argmax(masked_probs, dim=1)
                acc = (pred == batch_actions).float().mean()
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float(acc.detach().cpu()))
        rows.append({"bc_epoch": epoch, "bc_loss": float(np.mean(losses)), "bc_action_accuracy": float(np.mean(accuracies))})
        if not np.isfinite(rows[-1]["bc_loss"]):
            raise RuntimeError(f"BC loss is not finite at epoch {epoch}")
    bc_log = pd.DataFrame(rows)
    bc_log.to_csv(out / "logs" / "041_03_bc_pretrain_log.csv", index=False)
    return bc_log


def evaluate_checkpoint(out: Path, model_path: Path, checkpoint_step: int, config: dict[str, Any], env_config: dict[str, Any], thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="auto")
    thresholds_by_year = {int(r.year): r._asdict() for r in thresholds.itertuples(index=False)}
    rows = []
    daily_frames = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04100.LOWIC_INPUT_ROOT
    try:
        for year in YEARS:
            env = ppo04040.make_env_with_yield_guardrail(config, env_config, STATION, year, SEED, f"{STATION}_{year}_041_03_ckpt{checkpoint_step}_eval", evaluation=True)
            daily_rows = []
            try:
                obs, info = env.reset()
                for step in range(420):
                    mask = env.action_masks()
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                    obs, reward, terminated, truncated, info = env.step(action)
                    latest = latest_observation_dict(env, obs, info)
                    last = dict(getattr(env, "last_action_info", {}))
                    daily_rows.append(
                        {
                            "station_code": STATION,
                            "site": SITE,
                            "year": int(year),
                            "checkpoint_step": int(checkpoint_step),
                            "step": int(step),
                            "dap": scalar(latest.get("dap")),
                            "yrdoy": scalar(latest.get("yrdoy")),
                            "rain": scalar(latest.get("rain")),
                            "grnwt": scalar(latest.get("grnwt")),
                            "topwt": scalar(latest.get("topwt")),
                            "swfac": scalar(latest.get("swfac")),
                            "nstres": scalar(latest.get("nstres")),
                            "action_index": int(np.asarray(action).item()),
                            "irrigation_mm_action": float(last.get("safe_action_amir", np.nan)),
                            "nitrogen_kg_ha_action": float(last.get("safe_action_anfer", np.nan)),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    if terminated or truncated:
                        break
                daily = pd.DataFrame(daily_rows)
                daily_path = out / "daily_outputs" / STATION / f"041_03_{STATION}_{year}_ckpt{checkpoint_step}_daily.csv"
                daily.to_csv(daily_path, index=False)
                daily_frames.append(daily)
                snapshot = Path(getattr(env.unwrapped, "_tmp_folder"))
                plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
                gwad = pd.to_numeric(plantgro["GWAD"], errors="coerce").dropna()
                cwad = pd.to_numeric(plantgro["CWAD"], errors="coerce").dropna()
                final_gwad = float(gwad.iloc[-1]) if not gwad.empty else np.nan
                final_cwad = float(cwad.iloc[-1]) if not cwad.empty else np.nan
                i_total = float(pd.to_numeric(daily["irrigation_mm_action"], errors="coerce").fillna(0).sum())
                n_total = float(pd.to_numeric(daily["nitrogen_kg_ha_action"], errors="coerce").fillna(0).sum())
                summary_rows = parse_summary_out(snapshot / "Summary.OUT")
                srow, match_score, row_index = select_matching_row(summary_rows, final_gwad, i_total, n_total)
                ircm, nicm, etcp = num(srow, "IRCM"), num(srow, "NICM"), num(srow, "ETCP")
                ypem, ypnam = num(srow, "YPEM"), num(srow, "YPNAM")
                thr = thresholds_by_year[int(year)]
                row = {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "checkpoint_step": int(checkpoint_step),
                    "grain_yield_kg_ha": final_gwad,
                    "biomass_kg_ha": final_cwad,
                    "summary_irrigation_total": ircm,
                    "summary_nitrogen_total": nicm,
                    "etcp_mm": etcp,
                    "WP_ET_kg_m3": ypem * 0.1 if ypem is not None and ypem >= 0 else np.nan,
                    "PFP_N_kg_kg": ypnam if nicm and nicm > 0 and ypnam is not None and ypnam >= 0 else np.nan,
                    "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
                    "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
                    "summary_match_score": match_score,
                    "summary_row_index": row_index,
                    "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
                }
                row["gap_yield_vs_four_max"] = row["grain_yield_kg_ha"] - float(thr["four_max_yield_kg_ha"])
                row["gap_wp_et_vs_four_max"] = row["WP_ET_kg_m3"] - float(thr["four_max_wp_et_kg_m3"])
                row["gap_pfp_n_vs_four_max"] = row["PFP_N_kg_kg"] - float(thr["four_max_pfp_n_kg_kg"])
                row["yield_win_vs_four_max"] = bool(row["gap_yield_vs_four_max"] > 0)
                row["wp_et_win_vs_four_max"] = bool(row["gap_wp_et_vs_four_max"] > 0)
                row["pfp_n_win_vs_four_max"] = bool(row["gap_pfp_n_vs_four_max"] > 0)
                row["any_metric_win_vs_four_max"] = bool(row["yield_win_vs_four_max"] or row["wp_et_win_vs_four_max"] or row["pfp_n_win_vs_four_max"])
                row["all3_win_vs_four_max"] = bool(row["yield_win_vs_four_max"] and row["wp_et_win_vs_four_max"] and row["pfp_n_win_vs_four_max"])
                rows.append(row)
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    return pd.DataFrame(rows), pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()


def write_record(out: Path, doc: Path, result: dict[str, Any], bc_log: pd.DataFrame, train_inventory: pd.DataFrame, eval_summary: pd.DataFrame, by_ckpt: pd.DataFrame) -> None:
    lines = [
        f"# {TASK_ID} SYA lowIC teacher warm-start MaskablePPO 记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- BC epoch：{result['bc_epochs']}",
        f"- PPO fine-tune steps：{result['total_timesteps']}",
        f"- checkpoint：{', '.join(map(str, result['checkpoint_steps']))}",
        f"- 训练性质：teacher-assisted same-year feasibility，不是严格跨年泛化。",
        "",
        "## BC 训练日志",
        "",
        md_table(bc_log, max_rows=30),
        "",
        "## checkpoint 汇总",
        "",
        md_table(by_ckpt, max_rows=20),
        "",
        "## 训练模型清单",
        "",
        md_table(train_inventory, max_rows=20),
        "",
        "## 输出文件",
        "",
        f"- 训练清单：`{result['outputs']['training_inventory']}`",
        f"- 评估总表：`{result['outputs']['validation_summary']}`",
        f"- checkpoint 汇总：`{result['outputs']['by_checkpoint']}`",
        f"- BC 数据：`{result['outputs']['bc_dataset_npz']}`",
        f"- JSON：`{result['outputs']['result_json']}`",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    ensure_dirs(out)
    selected = load_selected_teachers()
    config, env_config = build_config_and_env_config(YEARS)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "teacher_selected_exists": TEACHER_SELECTED.exists(),
        "selected_years": selected["year"].astype(int).tolist(),
        "teacher_tier_counts": selected["teacher_tier"].value_counts().to_dict(),
        "lowIC_input_root": base04100.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": base04100.LOWIC_INPUT_ROOT.exists(),
        "total_timesteps": int(timesteps),
        "checkpoint_steps": checkpoint_steps,
        "bc_epochs": BC_EPOCHS,
        "bc_batch_size": BC_BATCH_SIZE,
        "bc_lr": BC_LR,
        "action_safety": config["action_safety"],
        "discrete_actions": config["discrete_actions"],
        "reward": config["reward"],
        "next_step_allowed": bool(TEACHER_SELECTED.exists() and base04100.LOWIC_INPUT_ROOT.exists() and PROMPT.exists()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    from sb3_contrib import MaskablePPO

    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    ensure_dirs(out)
    selected = load_selected_teachers()
    config, env_config = build_config_and_env_config(YEARS)
    (out / "configs" / "041_03_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "configs" / "041_03_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset, bc_meta = collect_bc_dataset(out, selected, config, env_config)
    train_env = RandomYearEnv(config, env_config, YEARS, SEED)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        tensorboard_log=str(out / "tensorboard" / STATION),
        **ppo04040.base03222.base.ppo_kwargs(config),
    )
    bc_log = pretrain_policy_bc(model, dataset, out)
    bc_model = out / "models" / STATION / f"{STATION}_teacher_warmstart_maskableppo_seed{SEED}_bc_init.zip"
    model.save(str(bc_model.with_suffix("")))

    callback = FixedStepCheckpointCallback(out, checkpoint_steps)
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04100.LOWIC_INPUT_ROOT
    try:
        model.learn(total_timesteps=int(timesteps), reset_num_timesteps=True, callback=callback.callback, progress_bar=False)
    finally:
        train_env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    rows = [
        {
            "station_code": STATION,
            "site": SITE,
            "seed": SEED,
            "checkpoint_step": 0,
            "stage": "bc_init",
            "model_path": bc_model.relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(bc_model),
        }
    ]
    for step in checkpoint_steps:
        path = callback.model_path(step)
        if path.exists():
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "seed": SEED,
                    "checkpoint_step": int(step),
                    "stage": "ppo_finetune",
                    "model_path": path.relative_to(ROOT).as_posix(),
                    "model_sha256": sha256_file(path),
                }
            )
    train_inventory = pd.DataFrame(rows)
    train_inventory_path = out / "evaluation" / "041_03_training_checkpoint_inventory.csv"
    train_inventory.to_csv(train_inventory_path, index=False)

    thresholds = base04100.load_baseline_thresholds()
    thresholds = thresholds[thresholds["year"].isin(YEARS)].copy()
    eval_frames = []
    for row in train_inventory.itertuples(index=False):
        eval_df, _daily = evaluate_checkpoint(out, ROOT / row.model_path, int(row.checkpoint_step), config, env_config, thresholds)
        eval_df["stage"] = str(row.stage)
        eval_frames.append(eval_df)
    eval_summary = pd.concat(eval_frames, ignore_index=True)
    eval_summary_path = out / "evaluation" / "041_03_checkpoint_validation_summary.csv"
    eval_summary.to_csv(eval_summary_path, index=False)
    by_ckpt = (
        eval_summary.groupby(["stage", "checkpoint_step"], as_index=False)
        .agg(
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_wp_et=("WP_ET_kg_m3", "mean"),
            mean_pfp_n=("PFP_N_kg_kg", "mean"),
            mean_irrigation=("summary_irrigation_total", "mean"),
            mean_nitrogen=("summary_nitrogen_total", "mean"),
            any_metric_win_years=("any_metric_win_vs_four_max", "sum"),
            all3_win_years=("all3_win_vs_four_max", "sum"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        .sort_values("checkpoint_step")
    )
    by_ckpt_path = out / "evaluation" / "041_03_validation_summary_by_checkpoint.csv"
    by_ckpt.to_csv(by_ckpt_path, index=False)

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "branch": "A_teacher_warmstart_training_completed",
        "algorithm": "MaskablePPO",
        "training_run": True,
        "same_year_teacher_feasibility": True,
        "strict_cross_year_generalization": False,
        "total_timesteps": int(timesteps),
        "checkpoint_steps": checkpoint_steps,
        "bc_epochs": BC_EPOCHS,
        "bc_final_loss": float(bc_log["bc_loss"].iloc[-1]),
        "bc_final_action_accuracy": float(bc_log["bc_action_accuracy"].iloc[-1]),
        "outputs": {
            "record_md": doc.relative_to(ROOT).as_posix(),
            "training_inventory": train_inventory_path.relative_to(ROOT).as_posix(),
            "validation_summary": eval_summary_path.relative_to(ROOT).as_posix(),
            "by_checkpoint": by_ckpt_path.relative_to(ROOT).as_posix(),
            "bc_dataset_npz": (out / "bc_dataset" / "041_03_bc_dataset.npz").relative_to(ROOT).as_posix(),
            "bc_dataset_meta": (out / "bc_dataset" / "041_03_bc_dataset_meta.csv").relative_to(ROOT).as_posix(),
            "bc_log": (out / "logs" / "041_03_bc_pretrain_log.csv").relative_to(ROOT).as_posix(),
            "result_json": (out / "041_03_result.json").relative_to(ROOT).as_posix(),
        },
    }
    (out / "041_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(out, doc, result, bc_log, train_inventory, eval_summary, by_ckpt)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        dry_run(args.timesteps, checkpoint_steps, args.suffix)
    else:
        run_training(args.timesteps, checkpoint_steps, args.suffix)


if __name__ == "__main__":
    main()

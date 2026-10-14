from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering  # noqa: E402
import run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05 as base04305  # noqa: E402
from strict_maskable_dqn_040 import MaskReplayBuffer, ReplayBatch, StrictMaskableDQN  # noqa: E402


TASK_ID = "044_00"
TASK_NAME = "sya_lowIC_binary_forecast_demo_dqn_smoke"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
SITE = "SY"
SEED = 0
DEFAULT_TIMESTEPS = 2_000
DEFAULT_CHECKPOINTS = [1_000, 2_000]
DEMO_BATCH_SIZE = 256
DEMO_TD_COEF = 1.0
DEMO_MARGIN_COEF = 1.0
DEMO_MARGIN = 0.8


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else DOC


def ensure_dirs(out: Path) -> None:
    for rel in ["configs", "models/SYA", "logs", "evaluation", "daily_outputs/SYA", "demo_buffer"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    doc_for_suffix("").parent.mkdir(parents=True, exist_ok=True)


def flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def make_model(config: dict[str, Any], observation_dim: int, action_dim: int, timesteps: int) -> StrictMaskableDQN:
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
        total_timesteps=int(timesteps),
        seed=SEED,
        device="cpu",
    )


def model_path(out: Path, step: int) -> Path:
    return out / "models" / STATION / f"{STATION}_binary_forecast_demo_dqn_seed{SEED}_ckpt{int(step)}.pt"


def collect_demo_buffer(out: Path, config: dict[str, Any], env_config: dict[str, Any]) -> tuple[MaskReplayBuffer, pd.DataFrame, pd.DataFrame]:
    selected = base04305.base04103.load_selected_teachers()
    # Discover shapes.
    probe_env = base04305.make_env(config, env_config, int(base04305.YEARS[0]), f"{STATION}_04400_probe", evaluation=True)
    obs, _info = probe_env.reset()
    observation_dim = int(flatten_obs(obs).shape[0])
    action_dim = int(probe_env.action_space.n)
    probe_env.close()
    demo = MaskReplayBuffer(20_000, observation_dim, action_dim)
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04305.base04302.LOWIC_INPUT_ROOT
    try:
        for teacher in selected.itertuples(index=False):
            year = int(teacher.year)
            schedule = base04305.combined_schedule(teacher)
            env = base04305.make_env(config, env_config, year, f"{STATION}_{year}_04400_demo_collect", evaluation=True)
            fired: set[int] = set()
            try:
                obs, _info = env.reset()
                obs_arr = flatten_obs(obs)
                for step in range(420):
                    dap = int(round(base04305.scalar(env.last_obs_dict.get("dap", step + 1), step + 1)))
                    requested_raw = schedule[dap] if dap in schedule and dap not in fired else {"amir": 0.0, "anfer": 0.0}
                    if dap in schedule:
                        fired.add(dap)
                    requested = base04305.binary_teacher_action(requested_raw)
                    action_idx = base04305.action_index_for(env, requested["amir"], requested["anfer"])
                    mask = np.asarray(env.action_masks(), dtype=bool)
                    skipped = False
                    if not bool(mask[action_idx]):
                        skipped = True
                        skipped_rows.append(
                            {
                                "year": year,
                                "step": int(step),
                                "dap": int(dap),
                                "raw_teacher_irrigation_mm": float(requested_raw["amir"]),
                                "raw_teacher_nitrogen_kg_ha": float(requested_raw["anfer"]),
                                "binary_teacher_irrigation_mm": float(requested["amir"]),
                                "binary_teacher_nitrogen_kg_ha": float(requested["anfer"]),
                                "reason": "mapped_binary_teacher_action_masked_by_current_action_safety",
                            }
                        )
                        requested = {"amir": 0.0, "anfer": 0.0}
                        action_idx = base04305.action_index_for(env, 0.0, 0.0)
                    next_obs, reward, terminated, truncated, _info = env.step(action_idx)
                    done = bool(terminated or truncated)
                    next_obs_arr = flatten_obs(next_obs)
                    next_mask = np.zeros(action_dim, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool)
                    demo.add(
                        observation=obs_arr,
                        action=int(action_idx),
                        reward=float(reward),
                        next_observation=next_obs_arr,
                        done=done,
                        mask=mask,
                        next_mask=next_mask,
                    )
                    rows.append(
                        {
                            "year": year,
                            "step": int(step),
                            "dap": int(dap),
                            "action_index": int(action_idx),
                            "is_nonzero_action": bool(action_idx != 0),
                            "reward": float(reward),
                            "skipped_original_teacher_action": bool(skipped),
                        }
                    )
                    if done:
                        break
                    obs_arr = next_obs_arr
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    meta = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped_rows)
    meta.to_csv(out / "demo_buffer" / "044_00_demo_transition_meta.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out / "demo_buffer" / "044_00_skipped_masked_teacher_actions.csv", index=False, encoding="utf-8-sig")
    return demo, meta, skipped


class RandomYearEnv:
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], years: list[int]) -> None:
        self.config = config
        self.env_config = env_config
        self.years = [int(y) for y in years]
        self.rng = np.random.default_rng(SEED)
        self.envs: dict[int, Any] = {}
        self.current_env: Any | None = None
        self.current_year: int | None = None
        self.reset_counts: dict[int, int] = {int(y): 0 for y in years}

    def reset(self) -> tuple[Any, dict[str, Any]]:
        year = int(self.rng.choice(self.years))
        if year not in self.envs:
            self.envs[year] = base04305.make_env(self.config, self.env_config, year, f"{STATION}_{year}_04400_train", evaluation=False)
        self.current_env = self.envs[year]
        self.current_year = year
        self.reset_counts[year] += 1
        return self.current_env.reset()

    def step(self, action: int):
        return self.current_env.step(action)

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.current_env.action_masks(), dtype=bool)

    @property
    def action_space(self):
        return self.current_env.action_space

    def close(self) -> None:
        for env in self.envs.values():
            env.close()


def td_loss(model: StrictMaskableDQN, batch: ReplayBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_q = model.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_all = model.target(batch.next_observations)
        safe_next_masks = batch.next_masks.clone()
        safe_next_masks[batch.dones.bool(), 0] = True
        next_q = next_all.masked_fill(~safe_next_masks, -torch.inf).max(dim=1).values
        next_q = torch.where(batch.dones.bool(), torch.zeros_like(next_q), next_q)
        targets = batch.rewards + model.gamma * next_q
    return F.smooth_l1_loss(chosen_q, targets), chosen_q, targets


def demo_margin_loss(model: StrictMaskableDQN, batch: ReplayBatch) -> torch.Tensor:
    q = model.online(batch.observations)
    chosen = q.gather(1, batch.actions[:, None]).squeeze(1)
    valid_other = batch.masks.clone()
    valid_other.scatter_(1, batch.actions[:, None], False)
    if not bool(valid_other.any()):
        return torch.zeros((), device=model.device)
    competitor = (q + DEMO_MARGIN).masked_fill(~valid_other, -torch.inf).max(dim=1).values
    raw = competitor - chosen
    raw = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw))
    return F.relu(raw).mean()


def train_step_demo(model: StrictMaskableDQN, demo: MaskReplayBuffer) -> dict[str, float]:
    online_batch = model.replay.sample(model.batch_size, model.rng, model.device)
    demo_batch = demo.sample(DEMO_BATCH_SIZE, model.rng, model.device)
    online_loss, online_q, online_target = td_loss(model, online_batch)
    demo_td, demo_q, demo_target = td_loss(model, demo_batch)
    margin = demo_margin_loss(model, demo_batch)
    loss = online_loss + DEMO_TD_COEF * demo_td + DEMO_MARGIN_COEF * margin
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    preclip_norm = torch.nn.utils.clip_grad_norm_(model.online.parameters(), model.max_grad_norm)
    model.optimizer.step()
    model.optimizer_updates += 1
    return {
        "loss_total": float(loss.detach().cpu()),
        "loss_online_td": float(online_loss.detach().cpu()),
        "loss_demo_td": float(demo_td.detach().cpu()),
        "loss_demo_margin": float(margin.detach().cpu()),
        "mean_online_q": float(online_q.detach().mean().cpu()),
        "mean_online_target": float(online_target.detach().mean().cpu()),
        "mean_demo_q": float(demo_q.detach().mean().cpu()),
        "mean_demo_target": float(demo_target.detach().mean().cpu()),
        "preclip_grad_norm": float(preclip_norm.detach().cpu()),
    }


def evaluate_model(out: Path, config: dict[str, Any], env_config: dict[str, Any], model: StrictMaskableDQN, saved_step: int) -> pd.DataFrame:
    rows = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04305.base04302.LOWIC_INPUT_ROOT
    try:
        for year in base04305.YEARS:
            env = base04305.make_env(config, env_config, int(year), f"{STATION}_{year}_04400_eval_{saved_step}", evaluation=True)
            daily_rows = []
            try:
                obs, info = env.reset()
                obs_arr = flatten_obs(obs)
                for step in range(420):
                    mask = np.asarray(env.action_masks(), dtype=bool)
                    action = model.select_action(obs_arr, mask, step, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    latest = base04305.latest_observation_dict(env, obs, info)
                    last = dict(getattr(env, "last_action_info", {}))
                    daily_rows.append(
                        {
                            "year": int(year),
                            "step": int(step),
                            "dap": base04305.scalar(latest.get("dap")),
                            "action_index": int(action),
                            "irrigation_mm_action": float(last.get("safe_action_amir", np.nan)),
                            "nitrogen_kg_ha_action": float(last.get("safe_action_anfer", np.nan)),
                            "grnwt": base04305.scalar(latest.get("grnwt")),
                            "swfac": base04305.scalar(latest.get("swfac")),
                            "nstres": base04305.scalar(latest.get("nstres")),
                            "reward": float(reward),
                        }
                    )
                    if terminated or truncated:
                        break
                    obs_arr = flatten_obs(obs)
                daily = pd.DataFrame(daily_rows)
                daily_path = out / "daily_outputs" / STATION / f"044_00_{STATION}_{year}_ckpt{saved_step}_daily.csv"
                daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
                snapshot = Path(getattr(env.unwrapped, "_tmp_folder"))
                plantgro = base04305.parse_dssat_table(snapshot / "PlantGro.OUT")
                gwad = pd.to_numeric(plantgro["GWAD"], errors="coerce").dropna()
                final_gwad = float(gwad.iloc[-1]) if not gwad.empty else np.nan
                rows.append(
                    {
                        "year": int(year),
                        "checkpoint_step": int(saved_step),
                        "grain_yield_kg_ha": final_gwad,
                        "total_irrigation": float(pd.to_numeric(daily["irrigation_mm_action"], errors="coerce").fillna(0).sum()),
                        "total_nitrogen": float(pd.to_numeric(daily["nitrogen_kg_ha_action"], errors="coerce").fillna(0).sum()),
                        "irrigation_event_count": int((pd.to_numeric(daily["irrigation_mm_action"], errors="coerce").fillna(0) > 0).sum()),
                        "n_event_count": int((pd.to_numeric(daily["nitrogen_kg_ha_action"], errors="coerce").fillna(0) > 0).sum()),
                        "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
                        "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
                        "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
                    }
                )
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    return pd.DataFrame(rows)


def run_smoke(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    ensure_dirs(out)
    config, env_config = base04305.build_config_and_env_config(base04305.YEARS)
    (out / "configs" / "044_00_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "configs" / "044_00_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")
    demo, demo_meta, skipped = collect_demo_buffer(out, config, env_config)
    env = RandomYearEnv(config, env_config, base04305.YEARS)
    update_rows = []
    eval_frames = []
    reset_df = pd.DataFrame()
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04305.base04302.LOWIC_INPUT_ROOT
    try:
        obs, _info = env.reset()
        obs_arr = flatten_obs(obs)
        model = make_model(config, obs_arr.shape[0], int(env.action_space.n), timesteps)
        learning_starts = min(int(config["dqn"]["learning_starts"]), 256)
        train_freq = int(config["dqn"]["train_freq"])
        gradient_steps = int(config["dqn"]["gradient_steps"])
        target_update_interval = int(config["dqn"]["target_update_interval"])
        checkpoint_set = set(int(x) for x in checkpoint_steps)
        for global_step in range(1, int(timesteps) + 1):
            mask = np.asarray(env.action_masks(), dtype=bool)
            action = model.select_action(obs_arr, mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            next_obs_arr = flatten_obs(next_obs)
            next_mask = np.zeros(model.action_dim, dtype=bool) if done else np.asarray(env.action_masks(), dtype=bool)
            model.add_transition(
                observation=obs_arr,
                action=int(action),
                reward=float(reward),
                next_observation=next_obs_arr,
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            if global_step > learning_starts and global_step % train_freq == 0:
                for _ in range(gradient_steps):
                    update_rows.append({"global_step": global_step, **train_step_demo(model, demo)})
            if global_step % target_update_interval == 0:
                model.sync_target()
            if global_step in checkpoint_set:
                path = model_path(out, global_step)
                model.save(path, global_step=global_step)
                eval_frames.append(evaluate_model(out, config, env_config, model, global_step))
            if done and global_step < int(timesteps):
                obs, _info = env.reset()
                obs_arr = flatten_obs(obs)
            else:
                obs_arr = next_obs_arr
    finally:
        reset_df = pd.DataFrame([{"year": int(y), "episode_count": int(env.reset_counts.get(y, 0))} for y in base04305.YEARS])
        env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    updates = pd.DataFrame(update_rows)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    by_ckpt = (
        eval_df.groupby("checkpoint_step", as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_nitrogen=("total_nitrogen", "mean"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        if not eval_df.empty
        else pd.DataFrame()
    )
    demo_meta.to_csv(out / "demo_buffer" / "044_00_demo_transition_meta.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out / "demo_buffer" / "044_00_skipped_masked_teacher_actions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(out / "logs" / "044_00_update_log.csv", index=False, encoding="utf-8-sig")
    reset_df.to_csv(out / "logs" / "044_00_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(out / "evaluation" / "044_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    by_ckpt.to_csv(out / "evaluation" / "044_00_validation_summary_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "branch": "A_demo_dqn_smoke_completed",
        "algorithm": "StrictMaskableDQN_with_demo_replay_margin_loss",
        "timesteps": int(timesteps),
        "checkpoint_steps": checkpoint_steps,
        "demo_transition_count": int(demo.size),
        "demo_nonzero_count": int(demo_meta["is_nonzero_action"].sum()),
        "skipped_masked_teacher_action_count": int(len(skipped)),
        "update_count": int(len(updates)),
        "outputs": {
            "record_md": doc.relative_to(ROOT).as_posix(),
            "update_log": (out / "logs" / "044_00_update_log.csv").relative_to(ROOT).as_posix(),
            "validation_summary": (out / "evaluation" / "044_00_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
            "by_checkpoint": (out / "evaluation" / "044_00_validation_summary_by_checkpoint.csv").relative_to(ROOT).as_posix(),
            "result_json": (out / "044_00_result.json").relative_to(ROOT).as_posix(),
        },
    }
    (out / "044_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(doc, result, demo_meta, skipped, updates, by_ckpt)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_record(doc: Path, result: dict[str, Any], demo_meta: pd.DataFrame, skipped: pd.DataFrame, updates: pd.DataFrame, by_ckpt: pd.DataFrame) -> None:
    lines = [
        "# 044_00 SYA lowIC binary-forecast Demo-DQN smoke 记录",
        "",
        "## 一句话结论",
        "",
        f"分支：`{result['branch']}`。本任务是 2K 机制 smoke，用 demo replay + margin loss 测试 DQfD 思路是否接入训练。",
        "",
        "## Demo buffer",
        "",
        f"- demo transition 数：{len(demo_meta)}",
        f"- demo 非零动作数：{int(demo_meta['is_nonzero_action'].sum()) if not demo_meta.empty else 0}",
        f"- 被 mask 跳过的 teacher 正动作数：{len(skipped)}",
        "",
        "## 更新日志尾部",
        "",
        base04305.md_table(updates.tail(20), 20) if not updates.empty else "无更新。",
        "",
        "## Checkpoint 汇总",
        "",
        base04305.md_table(by_ckpt, 20),
        "",
        "## 输出文件",
        "",
        "```json",
        json.dumps(result["outputs"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")


def dry_run(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    ensure_dirs(out)
    config, _env_config = base04305.build_config_and_env_config(base04305.YEARS)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "prompt_exists": PROMPT.exists(),
        "input_root": base04305.base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "input_root_exists": base04305.base04302.LOWIC_INPUT_ROOT.exists(),
        "timesteps": int(timesteps),
        "checkpoint_steps": checkpoint_steps,
        "demo_batch_size": DEMO_BATCH_SIZE,
        "demo_margin": DEMO_MARGIN,
        "demo_margin_coef": DEMO_MARGIN_COEF,
        "dqn_config": config.get("dqn", {}),
        "discrete_actions": config.get("discrete_actions", {}),
        "next_step_allowed": bool(PROMPT.exists() and base04305.base04302.LOWIC_INPUT_ROOT.exists()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_steps(raw: str | None, timesteps: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [x for x in DEFAULT_CHECKPOINTS if x <= int(timesteps)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="smoke2k")
    args = parser.parse_args()
    steps = parse_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        dry_run(args.timesteps, steps, args.suffix)
    else:
        run_smoke(args.timesteps, steps, args.suffix)


if __name__ == "__main__":
    main()

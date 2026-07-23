from __future__ import annotations

import hashlib
import json
import math
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19 as dqn19


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke"
DOC = ROOT / "docs" / "031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_22_literature_aligned_ddqn_smoke"
    return row


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(observation_dim)
        for width in hidden:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.ReLU())
            last = int(width)
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(last, 1)
        self.advantage = nn.Linear(last, int(action_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.trunk(observation)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass(frozen=True)
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor
    next_masks: torch.Tensor


class MaskReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.masks = np.zeros((self.capacity, action_dim), dtype=bool)
        self.next_masks = np.zeros((self.capacity, action_dim), dtype=bool)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        mask = np.asarray(mask, dtype=bool)
        next_mask = np.asarray(next_mask, dtype=bool)
        if not bool(mask[int(action)]):
            raise ValueError("Cannot store masked action")
        i = self.position
        self.observations[i] = np.asarray(observation, dtype=np.float32)
        self.next_observations[i] = np.asarray(next_observation, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.masks[i] = mask
        self.next_masks[i] = next_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> ReplayBatch:
        indices = rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device).long(),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            masks=torch.as_tensor(self.masks[indices], device=device),
            next_masks=torch.as_tensor(self.next_masks[indices], device=device),
        )


class MaskAwareDoubleDuelingDQN:
    def __init__(self, observation_dim: int, action_dim: int, config: dict, seed: int = 0, device: str = "cpu") -> None:
        cfg = config["ddqn"]
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(cfg["gamma"])
        self.batch_size = int(cfg["batch_size"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.exploration_fraction = float(cfg["exploration_fraction"])
        self.exploration_initial_eps = float(cfg["exploration_initial_eps"])
        self.exploration_final_eps = float(cfg["exploration_final_eps"])
        self.total_timesteps = int(config["total_timesteps"])
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        hidden = [int(x) for x in cfg["net_arch"]]
        self.online = DuelingQNetwork(observation_dim, action_dim, hidden).to(self.device)
        self.target = DuelingQNetwork(observation_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(cfg["learning_rate"]))
        self.replay = MaskReplayBuffer(int(cfg["replay_capacity"]), observation_dim, action_dim)
        self.optimizer_updates = 0
        self.target_updates = 0

    def epsilon(self, step: int) -> float:
        decay_steps = max(1.0, float(self.total_timesteps) * self.exploration_fraction)
        progress = min(max(float(step), 0.0) / decay_steps, 1.0)
        return float(self.exploration_initial_eps + progress * (self.exploration_final_eps - self.exploration_initial_eps))

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
            return self.online(x).detach().cpu().numpy()[0]

    def select_action(self, observation: np.ndarray, mask: np.ndarray, step: int, deterministic: bool = False) -> int:
        mask = np.asarray(mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("No valid actions")
        if not deterministic and self.rng.random() < self.epsilon(step):
            return int(self.rng.choice(valid))
        q = self.q_values(observation)
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def train_step(self) -> dict[str, float]:
        batch = self.replay.sample(self.batch_size, self.rng, self.device)
        q_all = self.online(batch.observations)
        chosen_q = q_all.gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.online(batch.next_observations)
            safe_next_masks = batch.next_masks.clone()
            safe_next_masks[batch.dones.bool(), 0] = True
            online_next_masked = online_next.masked_fill(~safe_next_masks, -torch.inf)
            next_actions = online_next_masked.argmax(dim=1)
            target_next = self.target(batch.next_observations).gather(1, next_actions[:, None]).squeeze(1)
            target_next = torch.where(batch.dones.bool(), torch.zeros_like(target_next), target_next)
            targets = batch.rewards + self.gamma * target_next
        loss = F.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer_updates += 1
        td_error = (targets - chosen_q).detach()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_chosen_q": float(chosen_q.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
            "mean_abs_td_error": float(td_error.abs().mean().cpu()),
            "preclip_grad_norm": float(grad_norm.detach().cpu()),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target_updates += 1

    @staticmethod
    def module_hash(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: Path, global_step: int) -> None:
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_updates": self.optimizer_updates,
                "target_updates": self.target_updates,
                "global_step": int(global_step),
            },
            path,
        )


def summarize_daily(daily: pd.DataFrame, daily_path: Path, model_path: Path) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "algorithm": "mask_aware_double_dueling_DQN",
        "policy_name": "free_discrete_mask_aware_double_dueling_dqn_ncost2x_scaled_reward_20k_seed0",
        "split": "eval",
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "literature_reward_sum": float(pd.to_numeric(daily.get("literature_reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": float(irr[dap <= 10].sum()) if len(daily) else np.nan,
        "early_dap1_10_n": float(n[dap <= 10].sum()) if len(daily) else np.nan,
        "reached_both_caps_by_dap10": bool(float(irr[dap <= 10].sum()) >= 159.999 and float(n[dap <= 10].sum()) >= 249.999) if len(daily) else False,
        "mask_forced_noop_count": int(pd.to_numeric(daily.get("mask_forced_noop", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(daily) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "action_sequence": action_sequence,
        "model_path": str(model_path.relative_to(ROOT)),
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
        "station_code": "SYA",
        "year": 2014,
        "seed": 0,
    }


def evaluate_model(config: dict, env_config: dict, model: MaskAwareDoubleDuelingDQN, model_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    weather = direct_ppo.weather_for_daily(config)
    env = dqn19.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_22_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = dqn19.latest_observation_dict(env, obs, info)
            dap_raw = dqn19.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = env.action_masks().copy()
            action = model.select_action(np.asarray(obs, dtype=np.float32), mask, step_count, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = dqn19.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "seed": seed,
                    "split": "eval",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": dqn19.scalar(wrow.get("rain"), np.nan),
                    "srad": dqn19.scalar(wrow.get("srad"), np.nan),
                    "tmax": dqn19.scalar(wrow.get("tmax"), np.nan),
                    "tmin": dqn19.scalar(wrow.get("tmin"), np.nan),
                    "swfac": dqn19.scalar(latest.get("swfac")),
                    "nstres": dqn19.scalar(latest.get("nstres")),
                    "topwt": dqn19.scalar(latest.get("topwt")),
                    "grnwt": dqn19.scalar(latest.get("grnwt")),
                    "xlai": dqn19.scalar(latest.get("xlai")),
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
    daily_path = OUT / "daily_outputs" / station / "2014_eval_free_discrete_mask_aware_double_dueling_dqn_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    return daily, summarize_daily(daily, daily_path, model_path)


def train_ddqn(config: dict, env_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    cfg = config["ddqn"]
    env = dqn19.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_22_train", evaluation=False)
    transitions: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    status = "ok"
    notes = ""
    model_path = OUT / "models" / station / "free_timing_mask_aware_double_dueling_dqn_ncost2x_scaled_reward_seed0.pt"
    try:
        obs, info = env.reset()
        obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
        action_dim = int(env.action_space.n)
        model = MaskAwareDoubleDuelingDQN(obs_dim, action_dim, config, seed=seed)
        done = False
        for global_step in range(1, int(config["total_timesteps"]) + 1):
            mask = env.action_masks().copy()
            action = model.select_action(np.asarray(obs, dtype=np.float32), mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_mask = np.zeros(action_dim, dtype=bool) if done else env.action_masks().copy()
            model.replay.add(
                observation=np.asarray(obs, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_observation=np.asarray(next_obs, dtype=np.float32),
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            if global_step % 10 == 0 or done:
                transitions.append(
                    {
                        "global_step": global_step,
                        "action": action,
                        "reward": float(reward),
                        "done": done,
                        "epsilon": model.epsilon(global_step - 1),
                        "replay_size": model.replay.size,
                        "season_i": float(env.safety_state.cumulative_irrigation),
                        "season_n": float(env.safety_state.cumulative_n),
                        **dict(env.last_action_info),
                    }
                )
            if global_step >= int(cfg["learning_starts"]) and global_step % int(cfg["train_freq"]) == 0:
                metrics = model.train_step()
                if global_step % 100 == 0:
                    updates.append({"global_step": global_step, **metrics})
            if global_step % int(cfg["target_update_interval"]) == 0:
                model.sync_target()
            if done and global_step < int(config["total_timesteps"]):
                obs, info = env.reset()
            else:
                obs = next_obs
        model.save(model_path, global_step=int(config["total_timesteps"]))
        daily, eval_row = evaluate_model(config, env_config, model, model_path)
        eval_row.update(
            {
                "run_status": "ok",
                "optimizer_updates": model.optimizer_updates,
                "target_updates": model.target_updates,
                "online_hash": model.module_hash(model.online),
                "target_hash": model.module_hash(model.target),
                "model_sha256": sha256_file(model_path),
            }
        )
        train_summary = {
            "algorithm": "mask_aware_double_dueling_DQN",
            "station_code": station,
            "train_years": str(year),
            "seed": seed,
            "total_timesteps": int(config["total_timesteps"]),
            "run_status": status,
            "model_path": str(model_path.relative_to(ROOT)),
            "optimizer_updates": model.optimizer_updates,
            "target_updates": model.target_updates,
            "notes": notes,
        }
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
        train_summary = {
            "algorithm": "mask_aware_double_dueling_DQN",
            "station_code": station,
            "train_years": str(year),
            "seed": seed,
            "total_timesteps": int(config["total_timesteps"]),
            "run_status": status,
            "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
            "notes": notes[-2500:],
        }
        eval_row = {"algorithm": "mask_aware_double_dueling_DQN", "run_status": "failed", "notes": notes[-2500:]}
    finally:
        env.close()
    pd.DataFrame(transitions).to_csv(OUT / "logs" / "031_22_ddqn_transition_sample_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(updates).to_csv(OUT / "logs" / "031_22_ddqn_update_log.csv", index=False, encoding="utf-8-sig")
    train_df = pd.DataFrame([train_summary])
    eval_df = pd.DataFrame([eval_row])
    train_df.to_csv(OUT / "evaluation" / "031_22_ddqn_training_summary.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT / "evaluation" / "031_22_ddqn_eval_summary.csv", index=False, encoding="utf-8-sig")
    return train_df, eval_df, eval_row


def load_existing_eval(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = df[df["split"].eq("eval")].copy()
    rows["comparison_label"] = label
    return rows


def write_record(train_df: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# 031_22 Literature-aligned PPO / DQN / Double-Dueling DQN comparison smoke",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- PPO and vanilla DQN rows are existing 20k outputs from 031_17 and 031_19; they are not retrained.",
        "- New training in this task: project-local mask-aware Double-Dueling DQN, 20k steps.",
        "- Same free-timing discrete action grid, reward, caps, 7-day intervals, irrigation DAP 1-120, fertilization DAP 1-90.",
        "- Final model only; no post-hoc checkpoint selection.",
        "",
        "## DDQN training summary",
        "",
        train_df.to_string(index=False),
        "",
        "## Algorithm comparison",
        "",
        comparison[
            [
                "comparison_label",
                "final_grnwt",
                "total_irrigation",
                "total_n",
                "profit_simple",
                "PFP_N",
                "early_dap1_10_irrigation",
                "early_dap1_10_n",
                "swfac_stress_days_gt_0p05",
                "nstres_days_gt_0p05",
                "action_sequence",
            ]
        ].to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- This is an algorithm-family smoke, not a full site/year result.",
        "- It can justify whether Double-Dueling DQN deserves more runs under the current free-timing setup.",
        "- It cannot by itself prove cross-year or cross-site robustness.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_22_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_22_resolved_env_config.yaml")

    train_df, ddqn_eval_df, _ = train_ddqn(config, env_config)
    ppo_df = load_existing_eval(ROOT / config["paths"]["existing_ppo_eval_summary"], "MaskablePPO_031_17_20k_seed0")
    vanilla_df = load_existing_eval(ROOT / config["paths"]["existing_vanilla_dqn_eval_summary"], "SB3_vanilla_DQN_031_19_20k_seed0")
    ddqn_eval_df = ddqn_eval_df.copy()
    ddqn_eval_df["comparison_label"] = "mask_aware_DoubleDuelingDQN_031_22_20k_seed0"
    comparison = pd.concat([ppo_df, vanilla_df, ddqn_eval_df], ignore_index=True, sort=False)
    comparison.to_csv(OUT / "evaluation" / "031_22_algorithm_comparison_summary.csv", index=False, encoding="utf-8-sig")
    write_record(train_df, comparison)

    result = {
        "task": "031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke",
        "record_md": str(DOC.relative_to(ROOT)),
        "comparison_summary": str((OUT / "evaluation" / "031_22_algorithm_comparison_summary.csv").relative_to(ROOT)),
        "ddqn_eval_summary": str((OUT / "evaluation" / "031_22_ddqn_eval_summary.csv").relative_to(ROOT)),
    }
    (OUT / "031_22_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        comparison[
            [
                "comparison_label",
                "final_grnwt",
                "total_irrigation",
                "total_n",
                "profit_simple",
                "PFP_N",
                "early_dap1_10_n",
                "nstres_days_gt_0p05",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()


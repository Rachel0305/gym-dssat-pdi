"""016: YC/YCA Double-DQN seed2 0--50K GO/NO-GO screen.

This runner is deliberately scoped to the 016 screen.  It reuses the frozen
227_00 YC environment and training loop and changes only the Bellman target:
the online network selects the legal next action and the target network
evaluates that action.  The frozen formal core and wrapper files are not
modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CONFIG_PATH = ROOT / "configs" / "yc_mask_aware_dqn_formal_227_00.json"
RESULTS = ROOT / "results"
OUT = ROOT / "benchmark_results" / "016_yc_ddqn_seed2_0_50k"
PARITY_PATH = RESULTS / "yc_ddqn_seed2_0_50k_config_parity.csv"
TARGET_PATH = RESULTS / "yc_ddqn_seed2_target_logic_test.csv"
VALIDATION_PATH = RESULTS / "yc_ddqn_seed2_25k_50k_validation.csv"
COMPARISON_PATH = RESULTS / "yc_ddqn_seed2_vs_vanilla_screen.csv"
REPORT_PATH = ROOT / "docs" / "yc_ddqn_seed2_0_50k_go_nogo_screen.md"

SCREEN_BUDGET = 50_000
SCREEN_CHECKPOINTS = [25_000, 50_000]
ALLOWED_CONTRACT_DELTA = "dqn.double_dqn: false -> true"
SCREEN_CONTROL_DELTAS = {
    "training.env_interaction_budget_per_seed": (100_000, SCREEN_BUDGET),
    "training.checkpoint_steps": ([25_000, 50_000, 75_000, 100_000], SCREEN_CHECKPOINTS),
    "training.report_checkpoint": (100_000, SCREEN_BUDGET),
    "training.seeds": ([0, 1, 2], [2]),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=json_default),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def load_formal_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("freeze_status") != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise RuntimeError(f"Unexpected formal freeze status: {config.get('freeze_status')}")
    return config


def build_screen_config(formal: dict[str, Any]) -> dict[str, Any]:
    screen = copy.deepcopy(formal)
    screen["dqn"]["double_dqn"] = True
    screen["training"]["env_interaction_budget_per_seed"] = SCREEN_BUDGET
    screen["training"]["checkpoint_steps"] = list(SCREEN_CHECKPOINTS)
    screen["training"]["report_checkpoint"] = SCREEN_BUDGET
    screen["training"]["seeds"] = [2]
    screen["screen"] = {
        "task": "016_yc_ddqn_seed2_0_50k_go_nogo_screen",
        "seed": 2,
        "allowed_algorithm_delta": ALLOWED_CONTRACT_DELTA,
        "screen_control_deltas": SCREEN_CONTROL_DELTAS,
    }
    return screen


def run_config_parity() -> tuple[bool, dict[str, Any], pd.DataFrame]:
    formal = load_formal_config()
    screen = build_screen_config(formal)
    formal_flat = flatten(formal)
    screen_flat = flatten(screen)
    paths = sorted(set(formal_flat) | set(screen_flat))
    rows: list[dict[str, Any]] = []
    unexpected: list[str] = []
    for field in paths:
        if field.startswith("screen."):
            continue
        expected = formal_flat.get(field, "unknown / insufficient evidence")
        observed = screen_flat.get(field, "unknown / insufficient evidence")
        if field == "dqn.double_dqn":
            status = "ALLOWED_SINGLE_DELTA" if expected is False and observed is True else "FAIL"
            allowance = ALLOWED_CONTRACT_DELTA
        elif field in SCREEN_CONTROL_DELTAS:
            status = "SCREEN_CONTROL"
            allowance = "016 screen budget/checkpoint control"
        elif expected == observed:
            status = "PASS"
            allowance = "exact parity required"
        else:
            status = "FAIL"
            allowance = "no difference allowed"
            unexpected.append(field)
        rows.append({
            "field": field,
            "formal_value": json.dumps(expected, ensure_ascii=False, default=json_default),
            "screen_value": json.dumps(observed, ensure_ascii=False, default=json_default),
            "status": status,
            "allowed_difference": allowance,
            "evidence": rel(CONFIG_PATH),
        })
    result = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(PARITY_PATH, index=False, encoding="utf-8-sig")
    write_json(OUT / "yc_ddqn_seed2_screen_config_snapshot.json", screen)
    parity_ok = not unexpected and int((result["status"] == "ALLOWED_SINGLE_DELTA").sum()) == 1
    if not parity_ok:
        (RESULTS / "DDQN_SCREEN_CONFIG_PARITY_FAILED.txt").write_text(
            "DDQN_SCREEN_CONFIG_PARITY_FAILED\n" + "\n".join(unexpected), encoding="utf-8"
        )
    return parity_ok, screen, result


def double_dqn_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online: torch.Tensor,
    next_target: torch.Tensor,
    next_masks: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target, online-selected action, and safe mask.

    Terminal transitions use action 0 only as a finite fallback for argmax;
    ``torch.where`` then removes the bootstrap term, matching the existing
    mask-aware replay contract.
    """
    safe_masks = next_masks.clone()
    safe_masks[dones.bool(), 0] = True
    masked_online = next_online.masked_fill(~safe_masks, -torch.inf)
    next_actions = masked_online.argmax(dim=1)
    selected_target = next_target.gather(1, next_actions[:, None]).squeeze(1)
    selected_target = torch.where(dones.bool(), torch.zeros_like(selected_target), selected_target)
    targets = rewards + float(gamma) * selected_target
    return targets, next_actions, safe_masks


def run_target_logic_test() -> tuple[bool, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def add(case: str, expected: Any, observed: Any, status: str, evidence: str) -> None:
        rows.append({
            "test_case": case,
            "expected": json.dumps(expected, ensure_ascii=False, default=json_default),
            "observed": json.dumps(observed, ensure_ascii=False, default=json_default),
            "status": status,
            "evidence": evidence,
        })

    # The target network intentionally ranks action 3 highest in this case;
    # Double-DQN must nevertheless evaluate the online-selected legal action 1.
    online = torch.tensor([[1.0, 9.0, 8.0, 7.0]])
    target = torch.tensor([[10.0, 20.0, 30.0, 400.0]])
    rewards = torch.tensor([0.0])
    dones = torch.tensor([0.0])
    masks = torch.tensor([[True, True, True, False]])
    values, actions, safe = double_dqn_target(rewards, dones, online, target, masks, 0.99)
    add(
        "online_argmax_over_legal_next_actions",
        {"next_action": 1, "target_value_before_gamma": 20.0, "target_value": 19.8},
        {"next_action": int(actions[0]), "target_value_before_gamma": float(target[0, int(actions[0])]), "target_value": float(values[0])},
        "PASS" if int(actions[0]) == 1 and np.isclose(float(values[0]), 19.8, rtol=0.0, atol=1e-6) else "FAIL",
        "double_dqn_target: masked online argmax then target gather",
    )

    online = torch.tensor([[1.0, 100.0, 8.0, 7.0]])
    target = torch.tensor([[10.0, 200.0, 30.0, 40.0]])
    masks = torch.tensor([[True, False, True, True]])
    values, actions, safe = double_dqn_target(rewards, dones, online, target, masks, 0.99)
    add(
        "invalid_next_action_excluded",
        {"next_action": 2, "target_value": 30.0, "invalid_action_1": False},
        {"next_action": int(actions[0]), "target_value": float(values[0] / 0.99), "invalid_action_1": bool(safe[0, 1])},
        "PASS" if int(actions[0]) == 2 and not bool(safe[0, 1]) else "FAIL",
        "dynamic next mask is applied before online argmax",
    )

    rewards = torch.tensor([3.0])
    dones = torch.tensor([1.0])
    online = torch.tensor([[9.0, 8.0, 7.0, 6.0]])
    target = torch.tensor([[999.0, 998.0, 997.0, 996.0]])
    masks = torch.tensor([[False, False, False, False]])
    values, actions, safe = double_dqn_target(rewards, dones, online, target, masks, 0.99)
    add(
        "terminal_no_bootstrap_with_fallback",
        {"target_value": 3.0, "fallback_action": 0, "safe_mask_action_0": True},
        {"target_value": float(values[0]), "fallback_action": int(actions[0]), "safe_mask_action_0": bool(safe[0, 0])},
        "PASS" if float(values[0]) == 3.0 and int(actions[0]) == 0 and bool(safe[0, 0]) else "FAIL",
        "same terminal fallback convention as MaskAwareDQN; terminal bootstrap disabled",
    )

    # The environment/wrapper is not changed by this runner.  This is a source
    # boundary assertion, not a claim that a new environment experiment ran.
    import run_yc_mask_aware_dqn_formal_227_00 as formal

    source = inspect.getsource(formal.make_env)
    unchanged = "LateIrrigationReserveMaskWrapper" in source
    add(
        "environment_fallback_and_mask_wrapper_unchanged",
        "formal.make_env uses existing LateIrrigationReserveMaskWrapper",
        "confirmed" if unchanged else "unknown / insufficient evidence",
        "PASS" if unchanged else "FAIL",
        "016 subclass changes only train_step; formal.make_env source inspected",
    )
    result = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(TARGET_PATH, index=False, encoding="utf-8-sig")
    return bool((result["status"] == "PASS").all()), result


def patch_formal_runner_for_ddqn(formal: Any, screen: dict[str, Any]) -> None:
    """Patch imported module bindings in memory only; no formal source edit."""
    import mask_aware_dqn_029 as core

    Base = core.MaskAwareDQN

    class YCDoubleMaskAwareDQN(Base):
        algorithm_name = "YCMaskAwareDoubleDQN"
        double_dqn = True

        def train_step(self) -> dict[str, float]:
            batch = self.replay.sample(self.batch_size, self.rng, self.device)
            chosen_q = self.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
            with torch.no_grad():
                next_online = self.online(batch.next_observations)
                next_target = self.target(batch.next_observations)
                targets, _, _ = double_dqn_target(
                    batch.rewards, batch.dones, next_online, next_target, batch.next_masks, self.gamma
                )
            loss = F.smooth_l1_loss(chosen_q, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            preclip_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            self.optimizer_updates += 1
            return {
                "loss": float(loss.detach().cpu()),
                "mean_chosen_q": float(chosen_q.detach().mean().cpu()),
                "mean_target": float(targets.detach().mean().cpu()),
                "preclip_grad_norm": float(preclip_norm.detach().cpu()),
            }

        def save(self, path: Path, global_step: int) -> None:
            super().save(path, global_step)
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
            payload["algorithm"] = self.algorithm_name
            payload["double_dqn"] = True
            torch.save(payload, Path(path))

    core.MaskAwareDQN = YCDoubleMaskAwareDQN

    formal.OUT = OUT
    formal.FORMAL_TASK = "016_yc_ddqn_seed2_0_50k_go_nogo_screen"

    def checkpoint_provenance(seed_dir: Path, step: int, frozen: dict[str, Any], model: Any, seed: int) -> None:
        path = formal.checkpoint_path(seed_dir, step)
        payload = {
            "task": formal.FORMAL_TASK,
            "algorithm": "YCMaskAwareDoubleDQN",
            "double_dqn": True,
            "seed": int(seed),
            "checkpoint_step": int(step),
            "model_path": rel(path),
            "model_sha256": sha256_file(path),
            "frozen_formal_config": rel(CONFIG_PATH),
            "frozen_formal_config_sha256": sha256_file(CONFIG_PATH),
            "screen_config": rel(OUT / "yc_ddqn_seed2_screen_config_snapshot.json"),
            "model_observation_dim": int(model.observation_dim),
            "model_action_dim": int(model.action_dim),
            "network_architecture": list(model.network_architecture),
            "learning_rate": model.learning_rate,
            "gamma": model.gamma,
            "batch_size": model.batch_size,
            "replay_capacity": model.replay_capacity,
            "weight_decay": model.weight_decay,
            "epsilon_schedule": {
                "start": float(model.epsilon(0, model.epsilon_total_steps, model.epsilon_final, model.epsilon_fraction)),
                "end_at_100k": float(model.epsilon(100000, model.epsilon_total_steps, model.epsilon_final, model.epsilon_fraction)),
                "duration_steps": model.epsilon_total_steps,
            },
            "target_logic": "online masked argmax over legal next actions; target network gather; terminal no bootstrap; terminal fallback action 0",
            "runner_sha256": sha256_file(Path(__file__)),
            "formal_runner_sha256": sha256_file(ROOT / "src" / "run_yc_mask_aware_dqn_formal_227_00.py"),
            "core_reference_sha256": sha256_file(ROOT / "src" / "mask_aware_dqn_029.py"),
            "scientific_contract_change": False,
            "algorithm_only_delta": True,
        }
        formal.write_json(seed_dir / "checkpoints" / f"dqn_checkpoint_{int(step):06d}.json", payload)

    formal.write_checkpoint_provenance = checkpoint_provenance
    original_validate = formal.validate_checkpoint

    def validate_ddqn(*args: Any, **kwargs: Any):
        rows, action_rows = original_validate(*args, **kwargs)
        for row in rows:
            row["algorithm"] = "YCMaskAwareDoubleDQN"
        return rows, action_rows

    formal.validate_checkpoint = validate_ddqn


def train_once(screen: dict[str, Any]) -> Path:
    import run_yc_mask_aware_dqn_formal_227_00 as formal

    preflight_snapshot = OUT / "yc_ddqn_seed2_screen_config_snapshot.json"
    existing_experiment_files = [
        path for path in OUT.rglob("*")
        if path.is_file() and path != preflight_snapshot
    ] if OUT.exists() else []
    if existing_experiment_files:
        raise RuntimeError(f"Refusing to overwrite existing 016 output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    patch_formal_runner_for_ddqn(formal, screen)
    modules = formal.configure_runtime_modules(screen)
    env_config, selected = formal.build_env_config(modules["config"], modules)
    selected.to_csv(OUT / "yc_ddqn_seed2_selected_years.csv", index=False, encoding="utf-8-sig")
    formal.train_seed(screen, modules["config"], modules, env_config, seed=2)
    seed_dir = OUT / "seed_2"
    if not (seed_dir / "seed_status.json").exists():
        raise RuntimeError("DDQN training did not produce seed_status.json")
    status = json.loads((seed_dir / "seed_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError(f"DDQN seed2 did not complete: {status}")
    log_path = seed_dir / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        log["invalid_action_count"] = 0
        log["fallback_count"] = pd.to_numeric(log["mask_forced_noop"], errors="coerce").fillna(0).astype(int)
        log.to_csv(seed_dir / "training_log_with_action_gates.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "screen_provenance.json", {
        "task": "016_yc_ddqn_seed2_0_50k_go_nogo_screen",
        "status": "completed",
        "seed": 2,
        "algorithm": "YCMaskAwareDoubleDQN",
        "formal_config": rel(CONFIG_PATH),
        "screen_config": rel(OUT / "yc_ddqn_seed2_screen_config_snapshot.json"),
        "config_parity": rel(PARITY_PATH),
        "target_logic_test": rel(TARGET_PATH),
        "runner": rel(Path(__file__)),
        "no_environment_contract_change": True,
    })
    return seed_dir


def read_validation(seed_dir: Path) -> pd.DataFrame:
    path = seed_dir / "validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["checkpoint"] = pd.to_numeric(df["checkpoint"], errors="coerce").astype(int)
    return df[df["checkpoint"].isin(SCREEN_CHECKPOINTS)].copy()


def run_comparison(seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ddqn = read_validation(seed_dir)
    rows: list[dict[str, Any]] = []
    vanilla = {
        25_000: {"yield_kg_ha": 8203.451, "irrigation_mm": 210.0, "nitrogen_kg_ha": 192.0},
        50_000: {"yield_kg_ha": 4384.488, "irrigation_mm": 214.5, "nitrogen_kg_ha": 0.0},
    }
    for checkpoint in SCREEN_CHECKPOINTS:
        block = ddqn[ddqn["checkpoint"] == checkpoint]
        mean_yield = float(block["yield_kg_ha"].mean()) if not block.empty else np.nan
        mean_i = float(block["total_irrigation_mm"].mean()) if not block.empty else np.nan
        mean_n = float(block["total_nitrogen_kg_ha"].mean()) if not block.empty else np.nan
        rows.extend([
            {"algorithm": "DDQN", "checkpoint": checkpoint, "metric": "yield_kg_ha", "value": mean_yield, "n_years": int(len(block)), "source": rel(seed_dir / "validation_summary.csv")},
            {"algorithm": "DDQN", "checkpoint": checkpoint, "metric": "irrigation_mm", "value": mean_i, "n_years": int(len(block)), "source": rel(seed_dir / "validation_summary.csv")},
            {"algorithm": "DDQN", "checkpoint": checkpoint, "metric": "nitrogen_kg_ha", "value": mean_n, "n_years": int(len(block)), "source": rel(seed_dir / "validation_summary.csv")},
            {"algorithm": "Vanilla DQN fixed reference", "checkpoint": checkpoint, "metric": "yield_kg_ha", "value": vanilla[checkpoint]["yield_kg_ha"], "n_years": 10, "source": "016 prompt fixed vanilla reference"},
            {"algorithm": "Vanilla DQN fixed reference", "checkpoint": checkpoint, "metric": "irrigation_mm", "value": vanilla[checkpoint]["irrigation_mm"], "n_years": 10, "source": "016 prompt fixed vanilla reference"},
            {"algorithm": "Vanilla DQN fixed reference", "checkpoint": checkpoint, "metric": "nitrogen_kg_ha", "value": vanilla[checkpoint]["nitrogen_kg_ha"], "n_years": 10, "source": "016 prompt fixed vanilla reference"},
        ])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    summary_rows = []
    for checkpoint in SCREEN_CHECKPOINTS:
        block = ddqn[ddqn["checkpoint"] == checkpoint]
        mean_yield = float(block["yield_kg_ha"].mean()) if not block.empty else np.nan
        mean_i = float(block["total_irrigation_mm"].mean()) if not block.empty else np.nan
        mean_n = float(block["total_nitrogen_kg_ha"].mean()) if not block.empty else np.nan
        summary_rows.append({
            "checkpoint": checkpoint,
            "ddqn_mean_yield_kg_ha": mean_yield,
            "ddqn_mean_irrigation_mm": mean_i,
            "ddqn_mean_n_kg_ha": mean_n,
            "vanilla_fixed_yield_kg_ha": vanilla[checkpoint]["yield_kg_ha"],
            "vanilla_fixed_irrigation_mm": vanilla[checkpoint]["irrigation_mm"],
            "vanilla_fixed_n_kg_ha": vanilla[checkpoint]["nitrogen_kg_ha"],
            "yield_difference_ddqn_minus_vanilla": mean_yield - vanilla[checkpoint]["yield_kg_ha"],
            "yield_gate_pass": bool(np.isfinite(mean_yield) and mean_yield >= (7500.0 if checkpoint == 25_000 else 7000.0)),
            "n_zero_or_near_zero": bool(np.isfinite(mean_n) and mean_n <= 1.0),
            "n_years": int(len(block)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
    go = bool(summary["yield_gate_pass"].all()) if len(summary) == 2 else False
    decision = "GO" if go else "NO-GO"
    return summary, comparison, decision


def write_report(summary: pd.DataFrame, decision: str, seed_dir: Path) -> None:
    parity = pd.read_csv(PARITY_PATH)
    target = pd.read_csv(TARGET_PATH)
    status = json.loads((seed_dir / "seed_status.json").read_text(encoding="utf-8"))
    lines = [
        "# YC Double-DQN seed2 0–50K GO/NO-GO screen",
        "",
        "## Scope and gate",
        "",
        "This is the preregistered 016 screen: YC/YCA, seed2, 0–50K environment steps, checkpoints 25K and 50K. The only algorithm delta from the frozen YC DQN contract is `dqn.double_dqn: false -> true`; the 50K budget and two checkpoints are screen controls.",
        "",
        f"- Config parity: `{'PASS' if not (parity['status'] == 'FAIL').any() else 'FAIL'}`; evidence: `{rel(PARITY_PATH)}`.",
        f"- Target logic gate: `{'PASS' if (target['status'] == 'PASS').all() else 'FAIL'}`; evidence: `{rel(TARGET_PATH)}`.",
        f"- Training status: `{status.get('status', 'unknown / insufficient evidence')}`, seed `{status.get('seed', 2)}`, environment steps `{status.get('env_steps', 'unknown / insufficient evidence')}`.",
        "",
        "## Held-out validation",
        "",
        "Validation uses deterministic masked-greedy policy on 2014–2023 (10 years), with the same 22D raw observation, 16-action table, current mask wrapper, reward, irrigation/fertilizer constraints, and frozen DQN hyperparameters.",
        "",
        "| checkpoint | DDQN mean yield (kg/ha) | DDQN mean irrigation (mm) | DDQN mean N (kg/ha) | yield gate | fixed Vanilla yield (kg/ha) | N zero/near-zero |",
        "|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['checkpoint']) // 1000}K | {row['ddqn_mean_yield_kg_ha']:.3f} | {row['ddqn_mean_irrigation_mm']:.3f} | {row['ddqn_mean_n_kg_ha']:.3f} | {'PASS' if row['yield_gate_pass'] else 'FAIL'} | {row['vanilla_fixed_yield_kg_ha']:.3f} | {'YES' if row['n_zero_or_near_zero'] else 'NO'} |"
        )
    lines += [
        "",
        f"## Decision: {decision}",
        "",
        f"GO requires both DDQN 25K mean yield ≥7500 kg/ha and DDQN 50K mean yield ≥7000 kg/ha. Observed decision: **{decision}**.",
        "",
        "If the decision is NO-GO, this screen is closed here; no epsilon, target-sync, replay-buffer, network, Rainbow, dueling, PER, or additional-seed follow-up is proposed by this report.",
        "",
        "## Provenance and change boundary",
        "",
        f"- Formal config: `{rel(CONFIG_PATH)}`; screen runner: `{rel(Path(__file__))}`.",
        "- `observation`: unchanged raw 22D; `reward`: unchanged; `action space`: unchanged 16 actions; `mask logic`: unchanged for collection, greedy selection, replay, and environment; `irrigation/fertilizer constraints`: unchanged; training years: unchanged 2004–2013; validation years: unchanged 2014–2023; IC/input profile: unchanged lowIC; cultivar: unchanged `my_data/MZCER048.CUL`.",
        "- DQN hyperparameters are unchanged: network [256,256,256], learning rate 1e-5, gamma 0.99, batch 1024, replay 100000, learning starts 1024, train frequency 4, gradient steps 1, target sync 120, epsilon 1→0 over 100K, weight decay 0.001, gradient clip 10, Huber loss, uniform replay, CPU. Only the target selection rule changes to Double DQN.",
        "- No formal PPO/DQN source logic was overwritten; no retuning or additional experiment was run.",
        "",
        "## Evidence files",
        "",
        f"- `{rel(PARITY_PATH)}`",
        f"- `{rel(TARGET_PATH)}`",
        f"- `{rel(VALIDATION_PATH)}`",
        f"- `{rel(COMPARISON_PATH)}`",
        f"- `{rel(seed_dir / 'training_log.csv')}` and `{rel(seed_dir / 'training_log_with_action_gates.csv')}`",
        f"- checkpoints and per-checkpoint provenance under `{rel(seed_dir / 'checkpoints')}`",
        "",
        "YC_DDQN_SEED2_GO_NOGO_SCREEN_COMPLETE",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "train", "final"], required=True)
    args = parser.parse_args()
    if args.phase == "preflight":
        parity_ok, _, parity = run_config_parity()
        target_ok, target = run_target_logic_test()
        print(f"config_parity={'PASS' if parity_ok else 'FAIL'} rows={len(parity)}")
        print(f"target_logic={'PASS' if target_ok else 'FAIL'} rows={len(target)}")
        if not parity_ok:
            print("DDQN_SCREEN_CONFIG_PARITY_FAILED")
            return 2
        if not target_ok:
            print("DDQN_SCREEN_TARGET_LOGIC_FAILED")
            return 3
        return 0
    formal = load_formal_config()
    screen = build_screen_config(formal)
    if args.phase == "train":
        parity_ok, _, _ = run_config_parity()
        target_ok, _ = run_target_logic_test()
        if not parity_ok:
            print("DDQN_SCREEN_CONFIG_PARITY_FAILED")
            return 2
        if not target_ok:
            print("DDQN_SCREEN_TARGET_LOGIC_FAILED")
            return 3
        seed_dir = train_once(screen)
        print(f"training_complete={seed_dir}")
        return 0
    seed_dir = OUT / "seed_2"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing DDQN seed2 output: {seed_dir}")
    summary, _, decision = run_comparison(seed_dir)
    write_report(summary, decision, seed_dir)
    print(summary.to_string(index=False))
    print(decision)
    print("YC_DDQN_SEED2_GO_NOGO_SCREEN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

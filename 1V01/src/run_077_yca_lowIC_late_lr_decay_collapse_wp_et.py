"""077_00: isolated YC late-learning-rate-decay stabilization experiment.

This wrapper reuses the 055 YCA transfer runner and changes only the PPO
learning-rate value passed to MaskablePPO.  It intentionally exposes only a
dry-run, a 2K smoke, and an authorized 25K run; it cannot launch 50K/75K/100K.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src" / "055_yca_lowIC_site_transfer") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "055_yca_lowIC_site_transfer"))

import run_055_00_yca_lowIC_expanded_action_maskableppo as base055


CONFIG = ROOT / "configs" / "077_yca_lowIC_late_lr_decay_collapse_wp_et.json"
PROMPT = ROOT / "prompts" / "2026-08-14_yc_late_lr_decay_collapse_wp_et.md"
TASK_ID = "077_00"
SMOKE_STEPS = 2_000
FORMAL_STEPS = 25_000
DECAY_START = 10_000
START_LR = 3.0e-4
END_LR = 1.0e-4


def late_lr_schedule(progress_remaining: float) -> float:
    """Return LR as a function of SB3's remaining-progress convention."""

    progress_done = max(0.0, min(1.0, 1.0 - float(progress_remaining))) * FORMAL_STEPS
    if progress_done <= DECAY_START:
        return START_LR
    fraction = min(1.0, (progress_done - DECAY_START) / (FORMAL_STEPS - DECAY_START))
    return START_LR + fraction * (END_LR - START_LR)


def schedule_probe() -> dict[str, Any]:
    return {
        "type": "piecewise_linear",
        "total_timesteps_reference": FORMAL_STEPS,
        "decay_start_timestep": DECAY_START,
        "start_learning_rate": START_LR,
        "end_learning_rate": END_LR,
        "values": {
            str(step): late_lr_schedule(1.0 - step / FORMAL_STEPS)
            for step in [0, DECAY_START, FORMAL_STEPS]
        },
    }


def _patched_ppo_kwargs(original: Callable[[dict[str, Any]], dict[str, Any]]):
    def wrapped(config: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(original(config))
        baseline_lr = kwargs.get("learning_rate")
        kwargs["learning_rate"] = late_lr_schedule
        wrapped.last_baseline_lr = baseline_lr  # type: ignore[attr-defined]
        wrapped.last_kwargs = {  # type: ignore[attr-defined]
            key: value for key, value in kwargs.items() if key != "learning_rate"
        }
        return kwargs

    wrapped.last_baseline_lr = None  # type: ignore[attr-defined]
    wrapped.last_kwargs = {}  # type: ignore[attr-defined]
    return wrapped


def _signature_gate(audit: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    final_step = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_step)].copy()
    signatures = {
        int(row.year): tuple(
            part for part in str(row.nonzero_action_pairs).split("; ") if part
        )
        for row in final.itertuples(index=False)
    }
    unique_signatures = sorted({signature for signature in signatures.values()})
    all_pairs = sorted({pair for signature in unique_signatures for pair in signature})
    return {
        "final_checkpoint": final_step,
        "year_signature_count": len(unique_signatures),
        "year_signatures": {str(year): list(signature) for year, signature in signatures.items()},
        "distinct_nonzero_action_pair_count": len(all_pairs),
        "strict_cross_year_signature_gate": len(unique_signatures) >= 2,
        "strict_action_pair_gate": len(all_pairs) >= 3,
        "next_step_allowed": len(unique_signatures) >= 2 and len(all_pairs) >= 3,
    }


def _write_augmented_result(cfg: dict[str, Any], result: dict[str, Any], phase: str, wrapper: Callable[..., Any]) -> dict[str, Any]:
    out = base055.output_root(cfg)
    audit_path = out / "audits" / f"{TASK_ID}_{phase}_action_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    signature_gate = _signature_gate(audit, cfg) if not audit.empty else {
        "next_step_allowed": False,
        "reason": "action audit missing",
    }
    schedule_path = out / "audits" / f"{TASK_ID}_late_lr_schedule.json"
    schedule_payload = {
        "schedule": schedule_probe(),
        "baseline_learning_rate_from_config": getattr(wrapper, "last_baseline_lr", None),
        "unchanged_ppo_kwargs": getattr(wrapper, "last_kwargs", {}),
        "phase": phase,
        "seed": cfg.get("seed"),
    }
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result["late_lr_schedule"] = schedule_payload
    result["signature_gate"] = signature_gate
    if phase == "smoke":
        result["smoke_gate"] = {**result.get("smoke_gate", {}), **signature_gate}
        result["next_step_allowed"] = bool(result["smoke_gate"].get("next_step_allowed"))
        result_path = out / f"{TASK_ID}_smoke_result.json"
    else:
        result["next_step_allowed"] = bool(result.get("next_step_allowed", True)) and bool(signature_gate.get("next_step_allowed"))
        result_path = out / f"{TASK_ID}_formal_result.json"
    result["schedule_manifest"] = schedule_path.relative_to(ROOT).as_posix()
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    record = out.parent.parent / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    if record.exists():
        with record.open("a", encoding="utf-8") as handle:
            handle.write("\n## Late learning-rate intervention\n\n")
            handle.write("- Only intervention: 3e-4 through 10K, linearly to 1e-4 at 25K.\n")
            handle.write(f"- Strict signature gate: `{signature_gate}`.\n")
            handle.write(f"- Schedule audit: `{schedule_path.relative_to(ROOT).as_posix()}`.\n")
    return result


def run(config_path: Path, phase: str) -> dict[str, Any]:
    if phase not in {"dry-run", "smoke", "formal25k"}:
        raise ValueError("Allowed phases are dry-run, smoke, formal25k.")
    base_cfg = base055.read_config(config_path)
    if phase == "formal25k" and int(base_cfg["training"]["total_timesteps"]) != FORMAL_STEPS:
        raise ValueError("077 formal phase is fixed at 25,000 timesteps.")
    if phase == "dry-run":
        cfg = base055.prepare_config(base_cfg, smoke=False)
        result = base055.run(config_path, dry_run=True, smoke=False, formal=False)
        result["late_lr_schedule"] = schedule_probe()
        result["authorized_phases"] = ["dry-run", "smoke", "formal25k"]
        return result

    if phase == "smoke":
        cfg = base055.prepare_config(base_cfg, smoke=True)
        call = lambda: base055.run(config_path, dry_run=False, smoke=True, formal=False)
    else:
        cfg = base055.prepare_config(base_cfg, smoke=False)
        call = lambda: base055.run(config_path, dry_run=False, smoke=False, formal=True)

    module = base055.engine.base03222.base
    original_kwargs = module.ppo_kwargs
    original_prompt = base055.PROMPT
    wrapper = _patched_ppo_kwargs(original_kwargs)
    base055.PROMPT = PROMPT
    module.ppo_kwargs = wrapper
    try:
        result = call()
    finally:
        module.ppo_kwargs = original_kwargs
        base055.PROMPT = original_prompt
    return _write_augmented_result(cfg, result, "smoke" if phase == "smoke" else "formal25k", wrapper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--formal25k", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    phase = "dry-run" if args.dry_run else ("smoke" if args.smoke else "formal25k")
    print(json.dumps(run(config_path, phase), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

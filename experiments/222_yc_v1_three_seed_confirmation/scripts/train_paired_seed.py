"""Run one isolated YC V1 seed/method smoke or formal training job.

The caller runs jobs sequentially to keep DSSAT environments and model memory
bounded.  Seed 0 is intentionally not retrained because its frozen formal
artifacts already exist and were exact-replayed separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
YCA_SRC = SRC / "055_yca_lowIC_site_transfer"
AUGMENTED_SRC = ROOT / "experiments" / "221_yc_ppo_domain_augmentation" / "scripts"
for entry in (SRC, YCA_SRC, AUGMENTED_SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_055_00_yca_lowIC_expanded_action_maskableppo as original_runner
import run_221YCA_yc_ppo_domain_augmentation as augmented_runner


EXPERIMENT = ROOT / "experiments" / "222_yc_v1_three_seed_confirmation"
CONFIG_DIR = EXPERIMENT / "configs"
LOG_DIR = EXPERIMENT / "logs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(method: str, seed: int) -> tuple[Path, dict[str, Any]]:
    if method == "Original":
        base_path = ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json"
        cfg = read_json(base_path)
        task_id = f"222YCA_O{seed}"
        task_name = f"yc_v1_original_seed{seed}"
    else:
        base_path = EXPERIMENT.parent.parent / "experiments" / "221_yc_ppo_domain_augmentation" / "configs" / "221YCA_yc_ppo_domain_augmentation.json"
        cfg = read_json(base_path)
        task_id = f"222YCA_A{seed}"
        task_name = f"yc_v1_augmented_seed{seed}"
    cfg["task_id"] = task_id
    cfg["task_name"] = task_name
    cfg["seed"] = int(seed)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{task_id}_{task_name}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, cfg


def run_original(config_path: Path, cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    # 055_00's legacy wrapper reads the seed from base03222's module global.
    original_runner.engine.base03222.SEED = int(cfg["seed"])
    return original_runner.run(
        config_path,
        dry_run=False,
        smoke=phase == "smoke",
        formal=phase == "formal",
    )


def run_augmented(config_path: Path, cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    # 221YCA is a module-level adapter; change only its task identity so each
    # seed has an isolated benchmark_results root and no old output is touched.
    task_id = str(cfg["task_id"])
    task_name = str(cfg["task_name"])
    augmented_runner.TASK_ID = task_id
    augmented_runner.TASK_NAME = task_name
    augmented_runner.base.TASK_ID = task_id
    augmented_runner.base.TASK_NAME = task_name
    augmented_runner.base.DEFAULT_CONFIG = config_path
    augmented_runner.base.CONFIG = config_path
    return augmented_runner.base.run(
        config_path,
        dry_run=False,
        smoke=phase == "smoke",
        formal=phase == "formal",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["Original", "Augmented"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2], required=True)
    parser.add_argument("--phase", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    if args.seed not in {1, 2}:
        raise ValueError("Only seed 1 and seed 2 are trainable in this confirmation; seed 0 is frozen.")

    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path, cfg = write_config(args.method, args.seed)
    payload: dict[str, Any] = {
        "method": args.method,
        "seed": args.seed,
        "phase": args.phase,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "status": "started",
    }
    try:
        result = run_original(config_path, cfg, args.phase) if args.method == "Original" else run_augmented(config_path, cfg, args.phase)
        payload["status"] = "completed"
        payload["result"] = result
    except Exception:
        payload["status"] = "failed"
        payload["error"] = traceback.format_exc()
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        out_name = f"{args.method.lower()}_seed{args.seed}_{args.phase}_status.json"
        (LOG_DIR / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    out_name = f"{args.method.lower()}_seed{args.seed}_{args.phase}_status.json"
    (LOG_DIR / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

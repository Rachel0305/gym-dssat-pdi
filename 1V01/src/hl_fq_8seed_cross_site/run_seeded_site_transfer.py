"""Seed-aware adapter for the frozen HLA/FQA MaskablePPO site runners.

The 051/054 runners record ``config.seed`` but the shared 032_22 engine reads
its module-level ``SEED``. This adapter applies that configured seed for one
process, delegates all training/evaluation to the existing station runner,
then verifies the produced checkpoint names before writing a small audit.
No PPO, reward, action-grid, mask, or site-input settings are changed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_runner(station_code: str):
    if station_code == "HLA":
        runner_dir = SRC / "054_hla_lowIC_site_transfer"
        module_name = "run_054_00_hla_lowIC_expanded_action_maskableppo"
    elif station_code == "FQA":
        runner_dir = SRC / "051_fqa_originIC_site_transfer"
        module_name = "run_051_00_fqa_originIC_expanded_action_maskableppo"
    else:
        raise ValueError(f"Unsupported station_code for this task: {station_code}")
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    return __import__(module_name)


def run(config_path: Path, mode: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    runner = import_runner(str(config["station_code"]))
    seed_owner = runner.engine.base03222
    previous_seed = int(seed_owner.SEED)
    seed_owner.SEED = seed
    try:
        result = runner.run(
            config_path,
            dry_run=(mode == "dry-run"),
            smoke=(mode == "smoke"),
            formal=(mode == "formal"),
        )
    finally:
        seed_owner.SEED = previous_seed

    if mode == "dry-run":
        result["configured_seed"] = seed
        result["engine_seed_during_preflight"] = seed
        return result

    output_root = ROOT / result["output_root"]
    checkpoint_paths = sorted(output_root.glob(f"models/**/*seed{seed}_ckpt*.zip"))
    if not checkpoint_paths:
        raise RuntimeError(
            f"No checkpoint path confirms seed {seed} under {output_root}"
        )
    if mode == "formal" and not any("_ckpt100000.zip" in p.name for p in checkpoint_paths):
        raise RuntimeError(f"The fixed 100K checkpoint is missing for seed {seed}")

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "station_code": config["station_code"],
        "site": config["site"],
        "mode": mode,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "configured_seed": seed,
        "engine_module": "run_five_site_half_split_stress_aware_maskableppo_batch_032_22.SEED",
        "engine_seed_set_before_runner": seed,
        "checkpoint_paths": [p.relative_to(ROOT).as_posix() for p in checkpoint_paths],
        "checkpoint_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in checkpoint_paths},
        "seed_name_match": all(f"seed{seed}_" in p.name for p in checkpoint_paths),
        "formal_100k_checkpoint_present": (
            any("_ckpt100000.zip" in p.name for p in checkpoint_paths)
            if mode == "formal"
            else None
        ),
    }
    audit_path = output_root / "seed_override_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if not audit["seed_name_match"]:
        raise RuntimeError(f"Checkpoint seed verification failed; inspect {audit_path}")
    return {**result, "seed_override_audit": audit_path.relative_to(ROOT).as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "smoke", "formal"), required=True)
    args = parser.parse_args()
    result = run(args.config, args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

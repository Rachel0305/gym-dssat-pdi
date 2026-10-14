"""Replay 204HLA checkpoints to obtain exact DSSAT ETCP and WP_ET metrics.

The existing 054_03 HLA replay/evaluation code is reused without changing its
reward, actions, renderer, or metric parser.  Runs are sequential to limit CPU
and memory pressure.  Completed checkpoint replays are skipped, never
overwritten.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_03_hla_lowIC_five_scenario_figures.py"
CONFIG = ROOT / "configs" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo.json"
PPO_ROOT = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
AUDIT_ROOT = ROOT / "benchmark_results" / "205HLA_hla_weather_aug_100k_diagnostics"
LOG = AUDIT_ROOT / "logs" / "205HLA_exact_wp_et_replay_commands.jsonl"
CHECKPOINTS = [10_000, 25_000, 50_000, 75_000, 100_000]


def completed_output(checkpoint: int) -> Path | None:
    candidates = sorted(
        path
        for path in (ROOT / "benchmark_results").glob(
            f"054_03_hla_lowIC_204HLA_*ckpt{checkpoint}_exact205"
        )
        if (path / "tables" / "054_03_hla_five_scenario_season_summary.csv").exists()
    )
    return candidates[-1] if candidates else None


def append_log(payload: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    required = [
        REPLAY,
        CONFIG,
        PPO_ROOT / "204HLA_formal_result.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing completed formal-run inputs: " + "; ".join(missing))

    for checkpoint in CHECKPOINTS:
        existing = completed_output(checkpoint)
        if existing is not None:
            payload = {
                "checkpoint": checkpoint,
                "status": "skipped_existing_complete",
                "output_root": existing.relative_to(ROOT).as_posix(),
            }
            append_log(payload)
            print(json.dumps(payload, ensure_ascii=False))
            continue
        command = [
            sys.executable,
            str(REPLAY),
            "--config",
            str(CONFIG),
            "--ppo-run-dir",
            str(PPO_ROOT),
            "--checkpoint",
            str(checkpoint),
            "--label",
            "exact205",
            "--skip-daily",
            "--skip-bars",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        payload = {
            "checkpoint": checkpoint,
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        output = completed_output(checkpoint)
        if output is not None:
            payload["output_root"] = output.relative_to(ROOT).as_posix()
        append_log(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if completed.returncode != 0:
            raise RuntimeError(f"Exact replay failed at checkpoint {checkpoint}")


if __name__ == "__main__":
    main()

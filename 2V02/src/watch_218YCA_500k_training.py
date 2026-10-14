"""Read-only periodic console watchdog for the 218YCA 500K run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
LOG_JSONL = OUT / "logs" / "218YCA_training_update_metrics.jsonl"
MODEL_DIR = OUT / "models" / "YCA"
WATCHDOG_LOG = OUT / "logs" / "218YCA_watchdog.log"
WATCHDOG_STATUS = OUT / "logs" / "218YCA_watchdog_status.json"
PROCESS_PATTERN = "run_218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k.py --formal"
TOTAL_STEPS = 500_000
CHECKPOINTS = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def process_pids() -> list[int]:
    try:
        result = subprocess.run(["pgrep", "-f", PROCESS_PATTERN], check=False, capture_output=True, text=True)
        return sorted({int(value) for value in result.stdout.split() if value.isdigit() and int(value) != os.getpid()})
    except FileNotFoundError:
        return []


def read_last_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def latest_checkpoint() -> int | None:
    values = [int(m.group(1)) for p in MODEL_DIR.glob("*ckpt*.zip") if (m := re.search(r"ckpt(\d+)\.zip$", p.name))]
    return max(values) if values else None


def terminal_result() -> dict[str, Any]:
    for path in sorted(OUT.glob("*result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") in {"completed", "failed", "partial"}:
            return {"path": str(path), "status": payload.get("status"), "phase": payload.get("phase"), "error": payload.get("error", "")[-500:]}
    return {}


def snapshot() -> dict[str, Any]:
    pids = process_pids()
    last = read_last_json(LOG_JSONL)
    checkpoint = latest_checkpoint()
    log_age = None
    if LOG_JSONL.exists():
        log_age = max(0.0, time.time() - LOG_JSONL.stat().st_mtime)
    terminal = terminal_result()
    if pids:
        state = "running"
    elif terminal.get("status") == "completed":
        state = "completed"
    elif terminal:
        state = "failed_or_partial"
    else:
        state = "crashed_or_not_started"
    if pids and log_age is not None and log_age > 900:
        state = "stalled_log"
    rss = float(last.get("rss_mb", 0.0) or 0.0)
    if rss >= 2048:
        state = "memory_stop_threshold_seen"
    step = int(last.get("num_timesteps", 0) or 0)
    return {
        "timestamp": now(), "state": state, "training_process_pids": pids,
        "training_process_running": bool(pids), "step": step, "total_steps": TOTAL_STEPS,
        "progress_percent": round(100.0 * step / TOTAL_STEPS, 4), "latest_checkpoint": checkpoint,
        "next_checkpoint": next((v for v in CHECKPOINTS if checkpoint is None or v > checkpoint), None),
        "scenario_coverage": last.get("scenario_coverage"), "source_year_coverage": last.get("source_year_coverage"),
        "variant_coverage": last.get("variant_coverage"), "rss_mb": rss,
        "log_age_seconds": round(log_age, 1) if log_age is not None else None,
        "mean_recent_episode_irrigation_mm": last.get("mean_recent_episode_irrigation_mm"),
        "mean_recent_episode_n_kg_ha": last.get("mean_recent_episode_n_kg_ha"), "terminal_result": terminal,
    }


def write_status(status: dict[str, Any]) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{status['timestamp']}] [218YCA watchdog] {status['state']} "
        f"step={status['step']}/{status['total_steps']} ({status['progress_percent']:.2f}%) "
        f"ckpt={status['latest_checkpoint']} next={status['next_checkpoint']} "
        f"coverage={status['scenario_coverage']}/60 rss_mb={status['rss_mb']:.1f} "
        f"log_age_s={status['log_age_seconds']}"
    )
    with WATCHDOG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    WATCHDOG_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--report-seconds", type=int, default=600)
    args = parser.parse_args()
    poll_seconds = max(15, int(args.poll_seconds))
    report_seconds = max(poll_seconds, int(args.report_seconds))
    last_report = 0.0
    while True:
        status = snapshot()
        current = time.time()
        important = status["state"] != "running"
        if not last_report or current - last_report >= report_seconds or important:
            write_status(status)
            last_report = current
        if status["state"] in {"completed", "failed_or_partial", "crashed_or_not_started", "stalled_log", "memory_stop_threshold_seen"}:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()

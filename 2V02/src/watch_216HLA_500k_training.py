"""Watchdog for the detached 216HLA 500K training process.

The watchdog is intentionally read-only with respect to training artifacts:
it reads process/log/checkpoint state and writes only two watchdog files under
the 216HLA output directory. It never kills or changes the training process.
"""

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
OUT = ROOT / "benchmark_results" / "216HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
LOG_JSONL = OUT / "logs" / "216HLA_training_update_metrics.jsonl"
MODEL_DIR = OUT / "models" / "HLA"
WATCHDOG_LOG = OUT / "logs" / "216HLA_watchdog.log"
WATCHDOG_STATUS = OUT / "logs" / "216HLA_watchdog_status.json"
PROCESS_PATTERN = "run_216HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k.py --formal"
TOTAL_STEPS = 500_000
CHECKPOINTS = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def process_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", PROCESS_PATTERN],
            check=False,
            capture_output=True,
            text=True,
        )
        pids = []
        for value in result.stdout.split():
            try:
                pid = int(value)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)
        return sorted(set(pids))
    except FileNotFoundError:
        pids = []
        proc_root = Path("/proc")
        if not proc_root.exists():
            return pids
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
            except OSError:
                continue
            if PROCESS_PATTERN in command and int(entry.name) != os.getpid():
                pids.append(int(entry.name))
        return sorted(set(pids))


def read_last_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            last = b""
            for line in handle:
                if line.strip():
                    last = line.strip()
        return json.loads(last.decode("utf-8")) if last else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def latest_checkpoint() -> int | None:
    if not MODEL_DIR.exists():
        return None
    values: list[int] = []
    for path in MODEL_DIR.glob("*ckpt*.zip"):
        match = re.search(r"ckpt(\d+)\.zip$", path.name)
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else None


def terminal_result() -> dict[str, Any]:
    if not OUT.exists():
        return {}
    candidates = sorted(OUT.glob("*result.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") in {"completed", "failed", "partial"}:
            return {"path": str(path), "status": payload.get("status"), "phase": payload.get("phase")}
    return {}


def snapshot() -> dict[str, Any]:
    pids = process_pids()
    last = read_last_json(LOG_JSONL)
    checkpoint = latest_checkpoint()
    log_age = None
    if LOG_JSONL.exists():
        try:
            log_age = max(0.0, time.time() - LOG_JSONL.stat().st_mtime)
        except OSError:
            pass
    terminal = terminal_result()
    running = bool(pids)
    if running:
        state = "running"
    elif terminal.get("status") == "completed":
        state = "completed"
    elif terminal:
        state = "failed"
    else:
        state = "crashed_or_not_started"
    step = int(last.get("num_timesteps", last.get("training_steps_seen", 0)) or 0)
    rss = float(last.get("rss_mb", 0.0) or 0.0)
    if running and log_age is not None and log_age > 900:
        state = "stalled_log"
    if rss >= 2048:
        state = "memory_stop_threshold_seen"
    return {
        "timestamp": now(),
        "state": state,
        "training_process_pids": pids,
        "training_process_running": running,
        "step": step,
        "total_steps": TOTAL_STEPS,
        "progress_percent": round(100.0 * step / TOTAL_STEPS, 4),
        "latest_checkpoint": checkpoint,
        "next_checkpoint": next((value for value in CHECKPOINTS if checkpoint is None or value > checkpoint), None),
        "scenario_coverage": last.get("scenario_coverage"),
        "source_year_coverage": last.get("source_year_coverage"),
        "variant_coverage": last.get("variant_coverage"),
        "rss_mb": rss,
        "log_age_seconds": round(log_age, 1) if log_age is not None else None,
        "mean_recent_episode_irrigation_mm": last.get("mean_recent_episode_irrigation_mm"),
        "mean_recent_episode_n_kg_ha": last.get("mean_recent_episode_n_kg_ha"),
        "terminal_result": terminal,
    }


def write_status(status: dict[str, Any]) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{status['timestamp']}] [216HLA watchdog] {status['state']} "
        f"step={status['step']}/{status['total_steps']} "
        f"({status['progress_percent']:.2f}%) "
        f"ckpt={status['latest_checkpoint']} "
        f"coverage={status['scenario_coverage']}/60 "
        f"rss_mb={status['rss_mb']:.1f} "
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
    terminal_seen = False
    while True:
        status = snapshot()
        current = time.time()
        important = status["state"] not in {"running"}
        if not terminal_seen and (not last_report or current - last_report >= report_seconds or important):
            write_status(status)
            last_report = current
        if status["state"] in {"completed", "failed", "crashed_or_not_started", "stalled_log", "memory_stop_threshold_seen"}:
            if not terminal_seen:
                write_status(status)
                terminal_seen = True
            if status["state"] != "running":
                return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()

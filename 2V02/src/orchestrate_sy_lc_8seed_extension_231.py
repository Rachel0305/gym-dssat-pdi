"""Sequential, resumable SY/LC 8-seed expansion with per-seed smoke gates."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sy_lc_8seed_extension_231"
LOGS = OUT / "logs"
EVENTS = OUT / "orchestrator_events.jsonl"
RUNNER = ROOT / "src/run_sy_lc_8seed_extension.py"


def emit(event: str, **data: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **data}) + "\n")


def path_for(site: str, seed: int, smoke: bool) -> Path:
    name = "sya_originIC_expanded_action_maskableppo" if site == "SY" else "lca_lowIC_expanded_action_maskableppo"
    suffix = "_smoke2k" if smoke else ""
    return ROOT / "benchmark_results" / f"231_{site.lower()}_seed{seed}_{name}_seed{seed}_extension{suffix}"


def gate_status(site: str, seed: int, smoke: bool) -> bool:
    root = path_for(site, seed, smoke)
    phase = "smoke" if smoke else "formal"
    result_path = root / f"231_{site.lower()}_seed{seed}_{phase}_result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = result.get("smoke_gate" if smoke else ("action_audit" if site == "SY" else "action_gate"), {})
    # Action diversity is a scientific outcome: weak/no-action seeds must be
    # retained in the eight-seed cohort. Stop only on technical invalidity.
    technical = (
        gate.get("expected_validation_rows") == 10
        and gate.get("observed_validation_rows") == 10
        and gate.get("all_daily_files_exist") is True
        and gate.get("all_actions_on_declared_grid") is True
        and gate.get("positive_actions_transmitted") is True
    )
    if not technical:
        raise RuntimeError(f"{site} seed{seed} {phase} technical audit failed")
    inventory = root / "evaluation/032_22_training_checkpoint_inventory.csv"
    frame = pd.read_csv(inventory, keep_default_na=False)
    wanted = 2000 if smoke else 100000
    matches = frame[pd.to_numeric(frame.checkpoint_step).eq(wanted)]
    if len(matches) != 1 or int(matches.iloc[0].seed) != seed:
        raise RuntimeError(f"{site} seed{seed} {phase} checkpoint or seed mismatch")
    model = ROOT / str(matches.iloc[0].model_path)
    if not model.is_file():
        raise FileNotFoundError(model)
    if not smoke:
        evaluation = pd.read_csv(root / "evaluation/032_22_checkpoint_validation_summary.csv", keep_default_na=False)
        if len(evaluation) != 40 or not evaluation.run_status.astype(str).eq("ok").all():
            raise RuntimeError(f"{site} seed{seed} formal validation incomplete or failed")
    return True


def run_one(site: str, seed: int, smoke: bool) -> None:
    phase = "smoke" if smoke else "formal"
    if gate_status(site, seed, smoke):
        emit("already_technically_passed", site=site, seed=seed, phase=phase)
        return
    target = path_for(site, seed, smoke)
    if target.exists():
        raise FileExistsError(f"Incomplete existing output retained for review: {target}")
    LOGS.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(RUNNER), "--site", site, "--seed", str(seed), f"--{phase}"]
    emit("started", site=site, seed=seed, phase=phase, command=command)
    log = LOGS / f"{site.lower()}_seed{seed}_{phase}.log"
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        emit("failed", site=site, seed=seed, phase=phase, returncode=completed.returncode, log=str(log.relative_to(ROOT)))
        raise RuntimeError(f"{site} seed{seed} {phase} failed; see {log}")
    if not gate_status(site, seed, smoke):
        emit("failed_gate", site=site, seed=seed, phase=phase)
        raise RuntimeError(f"{site} seed{seed} {phase} ended without a passing gate")
    emit("technically_passed", site=site, seed=seed, phase=phase, output=str(target.relative_to(ROOT)))


def main() -> None:
    for site in ("SY", "LC"):
        for seed in range(1, 8):
            run_one(site, seed, smoke=True)
            run_one(site, seed, smoke=False)
    emit("complete", passed_formal=14)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("orchestrator_stopped", error=str(exc))
        raise

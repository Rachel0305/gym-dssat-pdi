"""Isolated deterministic FQA WGEN heldout rollouts for frozen 2K and 10K models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE_045 = ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/run_gate.py"
PROMPT = ROOT / "prompt_02/048_fqa_wgen_10k_heldout_pair_diagnostic.md"
MODELS = {
    "2k": ROOT / "results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip",
    "10k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/fqa_multiyear_wgen_ppo_seed0_10k.zip",
}
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 300

spec = importlib.util.spec_from_file_location("fqa_gate_045_for_048", SOURCE_045)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
yc = gate.yc
smoke = gate.smoke


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    label = parser.parse_args().model
    model_path = MODELS[label]
    out = BASE / label
    if out.exists():
        raise FileExistsError(out)
    if not model_path.is_file() or not PROMPT.is_file():
        raise FileNotFoundError("model or prompt missing")
    if json.loads((ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/final_gate.json").read_text(encoding="utf-8"))["status"] != "PASS_10K_ARCHIVE_ONLY":
        raise RuntimeError("047 gate missing")
    out.mkdir(parents=True)
    temp = out / "temp"
    temp.mkdir()
    os.environ["TMPDIR"] = str(temp)
    tempfile.tempdir = str(temp)
    plan = [{"episode_index": i, "historical_year": 2007, "weather_seed": seed, "pool": "heldout"} for i, seed in enumerate((1081, 1100), 1)]
    write_json(out / "schedule.json", plan)
    smoke.OUT = out
    cfg, env_cfg, preflight = smoke.preflight()
    frozen = json.loads((ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/effective_config.json").read_text(encoding="utf-8"))
    for key in ("ppo", "reward", "discrete_actions", "action_safety"):
        if cfg[key] != frozen[key]:
            raise RuntimeError(f"frozen contract drift: {key}")
    write_json(out / "preflight.json", {"canonical": preflight, "model": model_path.relative_to(ROOT).as_posix(), "model_sha256": sha(model_path), "prompt_sha256": sha(PROMPT), "source_045_sha256": sha(SOURCE_045), "frozen_contract_keys": ["ppo", "reward", "discrete_actions", "action_safety"]})

    def make_env(*args, **kwargs):
        env = smoke.make_fq_env(*args, **kwargs)
        dssat = yc.find_dssat_instance(env)
        dssat._pilot_bootstrap_seed = int(args[-1] if args else kwargs["bootstrap_weather_seed"])
        return env

    yc.make_weather_env = make_env
    yc.TASK_ROOT = out
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    started = time.monotonic()
    env = None
    result = {"status": "FAILED", "model_label": label, "model_sha256": sha(model_path)}
    try:
        model = MaskablePPO.load(str(model_path), device="cpu")
        env = gate.ArchivedEval(cfg, env_cfg, plan, "heldout", out)
        metrics = []
        resources = []
        for row in plan:
            obs, _ = env._gym_env.reset(seed=None)
            days = 0
            while True:
                action, _ = model.predict(obs, deterministic=True, action_masks=env._gym_env.action_masks())
                obs, reward, done, truncated, _ = env._gym_env.step(action)
                days += 1
                if days > 260 or time.monotonic() - started >= WALL_LIMIT_S:
                    raise RuntimeError("step or wall time limit")
                if done or truncated:
                    break
            ep = env.episode_rows[-1]
            archive = env.archive_rows[-1]
            rss = yc._tree_rss_mb()
            resources.append({"episode_index": row["episode_index"], "process_tree_rss_mb": round(rss, 2), "elapsed_seconds": round(time.monotonic() - started, 2)})
            if rss >= RSS_LIMIT_MB:
                raise MemoryError(f"RSS {rss} MB")
            if int(ep["episode_days"]) != days or int(archive["days"]) != days:
                raise RuntimeError("episode day closure mismatch")
            y = ep["yield"]
            n = ep["fertilizer"]
            irr = ep["irrigation"]
            if any(v is None for v in (y, n, irr)):
                raise RuntimeError("missing crop/resource endpoint")
            metrics.append({"model": label, "episode_index": row["episode_index"], "year": 2007, "weather_seed": row["weather_seed"], "weather_sha256": archive["weather_sha256"], "days": days, "yield_kg_ha": y, "irrigation_mm": irr, "nitrogen_kg_ha": n, "pfp_n_kg_kg": y / n if n > 0 else "", "episode_return": ep["episode_return"], "endpoint_source": "ScheduledEpisodeEnv.episode_rows; wrapper executed cumulative inputs"})
            print(f"[{label}] seed={row['weather_seed']} days={days} yield={y:.1f} I={irr:.1f} N={n:.1f} rss={rss:.1f}", flush=True)
        write_csv(out / "endpoint_metrics.csv", metrics)
        write_csv(out / "resource_usage.csv", resources)
        result.update({"status": "PASS_DIAGNOSTIC_INPUT", "episodes": len(metrics), "weather_days": sum(x["days"] for x in metrics), "peak_rss_mb": max(x["process_tree_rss_mb"] for x in resources)})
    except Exception:
        result["error"] = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(result["error"], encoding="utf-8")
    finally:
        if env is not None:
            env.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS_DIAGNOSTIC_INPUT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

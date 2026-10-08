"""Run one frozen 047 checkpoint on two paired heldout FQA WGEN realizations."""
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
PROMPT = ROOT / "prompt_02/049_fqa_047_checkpoint_learning_signal_smoke.md"
MODELS = {
    "5k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_5000.zip",
    "10k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_10000.zip",
}
SOURCE_048 = ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/run_eval.py"
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 300

spec = importlib.util.spec_from_file_location("fqa_eval_048_for_049", SOURCE_048)
eval048 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval048)
gate, yc, smoke = eval048.gate, eval048.yc, eval048.smoke


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    try:
        import numpy as np
        result = float(np.asarray(value).item())
        return result if __import__("math").isfinite(result) else ""
    except Exception:
        return ""


def main() -> int:
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(MODELS), required=True)
    label = parser.parse_args().checkpoint
    model_path = MODELS[label]
    out = BASE / label
    if out.exists():
        raise FileExistsError(out)
    if not model_path.is_file() or not PROMPT.is_file():
        raise FileNotFoundError("checkpoint or prompt missing")
    gate047 = json.loads((ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/final_gate.json").read_text(encoding="utf-8"))
    gate048 = json.loads((ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/final_gate.json").read_text(encoding="utf-8"))
    if gate047["status"] != "PASS_10K_ARCHIVE_ONLY" or gate048["status"] != "PASS_PAIRED_DIAGNOSTIC_LIMITED":
        raise RuntimeError("047/048 input gate missing")
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
    write_json(out / "preflight.json", {"canonical": preflight, "checkpoint": model_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(model_path), "prompt_sha256": sha(PROMPT), "source_048_sha256": sha(SOURCE_048), "frozen_contract_keys": ["ppo", "reward", "discrete_actions", "action_safety"]})

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
    outcome = {"status": "FAILED", "checkpoint": label, "checkpoint_sha256": sha(model_path)}
    try:
        model = MaskablePPO.load(str(model_path), device="cpu")

        class TraceEval(gate.ArchivedEval):
            def __init__(self, *args, **kwargs):
                self.daily_rows = []
                super().__init__(*args, **kwargs)

            def _step(self, action):
                transition = super()._step(action)
                env_now = self.current_env
                ai = dict(getattr(env_now, "last_action_info", {}) or {})
                state = dict(getattr(env_now, "last_obs_dict", {}) or {})
                try:
                    action_index = int(np.asarray(action).item())
                except Exception:
                    action_index = ""
                row = {
                    "episode_index": int(self.current_schedule["episode_index"]),
                    "historical_year": int(self.current_schedule["historical_year"]),
                    "weather_seed": int(self.current_schedule["weather_seed"]),
                    "step_dap": int(self.episode_days),
                    "policy_action_index": action_index,
                    "safe_action_amir_mm": finite(ai.get("safe_action_amir")),
                    "safe_action_anfer_kg_ha": finite(ai.get("safe_action_anfer")),
                    "grnwt_kg_ha": finite(state.get("grnwt")),
                    "topwt_kg_ha": finite(state.get("topwt")),
                    "swfac": finite(state.get("swfac")),
                    "nstres": finite(state.get("nstres")),
                }
                self.daily_rows.append(row)
                return transition

        env = TraceEval(cfg, env_cfg, plan, "heldout", out)
        endpoint_rows = []
        resource_rows = []
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
            resource_rows.append({"episode_index": row["episode_index"], "process_tree_rss_mb": round(rss, 2), "elapsed_seconds": round(time.monotonic() - started, 2)})
            if rss >= RSS_LIMIT_MB or int(ep["episode_days"]) != days or int(archive["days"]) != days:
                raise RuntimeError(f"resource/day closure failed rss={rss} days={days}")
            trace = [x for x in env.daily_rows if int(x["episode_index"]) == int(row["episode_index"])]
            action_counts = {}
            positive_i_days = positive_n_days = 0
            for item in trace:
                key = str(item["policy_action_index"])
                action_counts[key] = action_counts.get(key, 0) + 1
                positive_i_days += int(float(item["safe_action_amir_mm"] or 0) > 0)
                positive_n_days += int(float(item["safe_action_anfer_kg_ha"] or 0) > 0)
            y, n, irrigation = ep["yield"], ep["fertilizer"], ep["irrigation"]
            endpoint_rows.append({"checkpoint": label, "episode_index": row["episode_index"], "year": 2007, "weather_seed": row["weather_seed"], "weather_sha256": archive["weather_sha256"], "days": days, "yield_kg_ha": y, "irrigation_mm_wrapper": irrigation, "nitrogen_kg_ha_wrapper": n, "pfp_n_wrapper_basis": y / n if n else "", "episode_return": ep["episode_return"], "distinct_policy_action_indices": len(action_counts), "action_index_counts_json": json.dumps(action_counts, sort_keys=True), "positive_irrigation_days": positive_i_days, "positive_nitrogen_days": positive_n_days})
            print(f"[{label}] wgen={row['weather_seed']} yield={y:.1f} I={irrigation:.1f} N={n:.1f} actions={len(action_counts)} rss={rss:.1f}", flush=True)
        write_csv(out / "endpoint_and_action_summary.csv", endpoint_rows)
        write_csv(out / "daily_action_state_trace.csv", env.daily_rows)
        write_csv(out / "resource_usage.csv", resource_rows)
        outcome.update({"status": "PASS_CHECKPOINT_DIAGNOSTIC_INPUT", "episodes": len(endpoint_rows), "trace_rows": len(env.daily_rows), "weather_days": sum(int(row["days"]) for row in endpoint_rows), "peak_rss_mb": max(row["process_tree_rss_mb"] for row in resource_rows)})
    except Exception:
        outcome["error"] = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(outcome["error"], encoding="utf-8")
    finally:
        if env is not None:
            env.close()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(out / "result.json", outcome)
    print(json.dumps({key: value for key, value in outcome.items() if key != "error"}, ensure_ascii=False), flush=True)
    return 0 if outcome["status"] == "PASS_CHECKPOINT_DIAGNOSTIC_INPUT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import (
    OUT_DIR,
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)


OUT = OUT_DIR / "daily_step_timing" / "2010"


def main() -> None:
    install_official_reward_module()
    import gym
    from sb3_wrapper import GymDssatWrapper

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_case_at(2010, OUT)
    env_args = json.loads((OUT / "env_args.json").read_text(encoding="utf-8"))

    timings: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    raw_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    env = GymDssatWrapper(raw_env)
    timings.append({"phase": "make_env", "step": -1, "elapsed_s": time.perf_counter() - t0})

    snapshot = OUT / "pdi_tmp_snapshot"
    try:
        t0 = time.perf_counter()
        obs, info = env.reset()
        timings.append({"phase": "reset", "step": -1, "elapsed_s": time.perf_counter() - t0})

        for step in range(10):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step))))
            real = {"amir": 0.0, "anfer": 165.0 if dap_before == 1 else 0.0}
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.perf_counter() - t0
            latest_after = latest_observation_dict(env, obs, info)
            done = bool(terminated or truncated)
            timings.append(
                {
                    "phase": "step",
                    "step": step,
                    "elapsed_s": elapsed,
                    "dap_before": dap_before,
                    "dap_after": scalar(latest_after.get("dap")),
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "reward": repr(reward),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    df = pd.DataFrame(timings)
    df.to_csv(OUT / "daily_step_timing.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    step_times = pd.to_numeric(df.loc[df["phase"] == "step", "elapsed_s"], errors="coerce")
    summary = {
        "n_step_rows": int((df["phase"] == "step").sum()),
        "mean_step_elapsed_s": float(step_times.mean()) if len(step_times) else np.nan,
        "max_step_elapsed_s": float(step_times.max()) if len(step_times) else np.nan,
        "min_step_elapsed_s": float(step_times.min()) if len(step_times) else np.nan,
        **events,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# 011_05 HLA daily linked step timing diagnosis",
        "",
        "This is not PPO training. It measures raw daily linked env step cost.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


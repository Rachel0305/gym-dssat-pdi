from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

from ppo_evaluate import append_evaluation_rows, evaluate_model, make_env
from ppo_experiment_plan import find_year, validation_years_for
from ppo_safe_rendering import DEFAULT_CONFIG, PROJECT_ROOT, load_yaml


def ppo_kwargs(config: dict, debug: bool) -> dict:
    from stable_baselines3 import PPO

    _ = PPO
    source = config.get("debug", {}).get("ppo_overrides", {}) if debug else config.get("ppo", {})
    allowed = ["learning_rate", "gamma", "n_steps", "batch_size", "ent_coef", "clip_range"]
    return {key: value for key, value in source.items() if key in allowed and value is not None}


def run_pretrain_smoke_check(config: dict, station: str, train_year: int, seed: int, row_metadata: dict | None = None) -> tuple[bool, Path]:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    smoke_root = output_root / "smoke_checks"
    smoke_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    env = os.environ.copy()
    env["SMOKE_TEST_OUTPUT_ROOT"] = str(smoke_root)
    for policy in ["null_zero", "fixed_low_input"]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "run_smoke_tests.py"),
            "--single",
            "--station",
            station,
            "--year",
            str(train_year),
            "--policy",
            policy,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                timeout=int(config.get("runtime", {}).get("episode_timeout_seconds", 300)),
                text=True,
                capture_output=True,
            )
            single_path = smoke_root / "evaluation" / "single" / f"{station}_{train_year}_{policy}.json"
            if completed.returncode == 0 and single_path.exists():
                data = pd.read_json(single_path, typ="series").to_dict()
                row = {
                        "station": station,
                        "train_year": train_year,
                        "policy_name": policy,
                        "run_status": data.get("run_status", "unknown"),
                        "episode_completed": data.get("episode_completed", False),
                        "error_message": data.get("error_message", ""),
                        "daily_csv_path": data.get("daily_csv_path", ""),
                        "notes": "pretrain_smoke_check",
                    }
                if row_metadata:
                    row.update(row_metadata)
                rows.append(row)
            else:
                row = {
                        "station": station,
                        "train_year": train_year,
                        "policy_name": policy,
                        "run_status": "failed",
                        "episode_completed": False,
                        "error_message": (completed.stderr or completed.stdout)[-1000:],
                        "daily_csv_path": "",
                        "notes": "subprocess_failed",
                    }
                if row_metadata:
                    row.update(row_metadata)
                rows.append(row)
        except subprocess.TimeoutExpired:
            row = {
                    "station": station,
                    "train_year": train_year,
                    "policy_name": policy,
                    "run_status": "timeout",
                    "episode_completed": False,
                    "error_message": "pretrain_smoke_check_timeout",
                    "daily_csv_path": "",
                    "notes": "subprocess_timeout",
                }
            if row_metadata:
                row.update(row_metadata)
            rows.append(row)
    out = smoke_root / "pretrain_smoke_check_summary.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, new], ignore_index=True)
        duplicate_keys = ["station", "train_year", "policy_name"]
        if "cap_name" in df.columns:
            duplicate_keys.append("cap_name")
        df = df.drop_duplicates(duplicate_keys, keep="last")
    else:
        df = new
    df.to_csv(out, index=False, encoding="utf-8-sig")
    ok = all(row["run_status"] == "ok" and bool(row["episode_completed"]) for row in rows)
    return ok, out


def train_one_policy(
    station: str,
    train_year: int,
    seed: int,
    total_timesteps: int,
    config_path: Path = DEFAULT_CONFIG,
    debug: bool = False,
    policy_tag: str = "",
    row_metadata: dict | None = None,
) -> dict:
    from stable_baselines3 import PPO

    config = load_yaml(config_path)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    train_info = find_year(config, station, train_year)
    safety_enabled = bool(config.get("action_safety", {}).get("enabled", False))
    if debug and safety_enabled:
        suffix = "_action_safe_debug"
    elif debug:
        suffix = "_debug"
    elif safety_enabled:
        suffix = "_action_safe"
    else:
        suffix = ""
    safe_policy_tag = policy_tag if policy_tag.startswith("_") or not policy_tag else f"_{policy_tag}"
    policy_name = f"{station}_train{train_year}{safe_policy_tag}_seed{seed}{suffix}"
    if config.get("safety", {}).get("run_pretrain_smoke_check", True):
        smoke_ok, smoke_summary = run_pretrain_smoke_check(config, station, train_year, seed, row_metadata=row_metadata)
        if not smoke_ok:
            raise RuntimeError(f"pretrain smoke check failed: {smoke_summary}")
    env = make_env(config, station, train_year, seed, run_tag=f"{policy_name}_train", evaluation=False, action_safety_enabled=safety_enabled)
    model_dir = output_root / "models" / station
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_root / "tensorboard" / station
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / policy_name
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tensorboard_dir),
            **ppo_kwargs(config, debug=debug),
        )
        model.learn(total_timesteps=int(total_timesteps), progress_bar=False)
        model.save(str(model_path))
    finally:
        try:
            env.close()
        except Exception:
            pass
    saved_model_path = model_path.with_suffix(".zip")
    eval_rows = []
    eval_years = [train_info] + validation_years_for(config, station, train_year)
    for eval_info in eval_years:
        eval_rows.append(
            evaluate_model(
                model=model,
                config=config,
                station=station,
                train_year=train_year,
                train_year_label=train_info["label"],
                eval_year=int(eval_info["year"]),
                eval_year_label=eval_info["label"],
                seed=seed,
                model_path=saved_model_path,
                policy_name=policy_name,
                action_safety_enabled=safety_enabled,
                row_metadata=row_metadata,
            )
        )
    evaluation_summary = append_evaluation_rows(eval_rows, config)
    train_log = output_root / "logs" / station / f"{policy_name}_training_summary.csv"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "station": station,
                "train_year": train_year,
                "train_year_label": train_info["label"],
                "validation_years": ",".join(str(item["year"]) for item in validation_years_for(config, station, train_year)),
                "seed": seed,
                "total_timesteps": int(total_timesteps),
                "model_path": str(saved_model_path.relative_to(PROJECT_ROOT)),
                "evaluation_summary_path": str(evaluation_summary.relative_to(PROJECT_ROOT)),
                "debug": debug,
            }
        ]
    ).to_csv(train_log, index=False, encoding="utf-8-sig")
    return {
        "model_path": saved_model_path,
        "evaluation_summary": evaluation_summary,
        "training_summary": train_log,
        "eval_rows": eval_rows,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--train-year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    try:
        result = train_one_policy(args.station, args.train_year, args.seed, args.timesteps, config_path=Path(args.config), debug=args.debug)
        print(result["model_path"].relative_to(PROJECT_ROOT))
        print(result["evaluation_summary"].relative_to(PROJECT_ROOT))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()

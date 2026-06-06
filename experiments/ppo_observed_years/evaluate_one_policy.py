from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import PPO

from ppo_evaluate import append_evaluation_rows, evaluate_model
from ppo_experiment_plan import find_year
from ppo_safe_rendering import DEFAULT_CONFIG, PROJECT_ROOT as ROOT, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--train-year", type=int, required=True)
    parser.add_argument("--eval-year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy-name", default=None)
    args = parser.parse_args()
    config = load_yaml(DEFAULT_CONFIG)
    train_info = find_year(config, args.station, args.train_year)
    eval_info = find_year(config, args.station, args.eval_year)
    model_path = ROOT / args.model_path
    model = PPO.load(str(model_path))
    policy_name = args.policy_name or f"{args.station}_train{args.train_year}_seed{args.seed}"
    row = evaluate_model(
        model=model,
        config=config,
        station=args.station,
        train_year=args.train_year,
        train_year_label=train_info["label"],
        eval_year=args.eval_year,
        eval_year_label=eval_info["label"],
        seed=args.seed,
        model_path=model_path,
        policy_name=policy_name,
    )
    print(append_evaluation_rows([row], config).relative_to(ROOT))


if __name__ == "__main__":
    main()

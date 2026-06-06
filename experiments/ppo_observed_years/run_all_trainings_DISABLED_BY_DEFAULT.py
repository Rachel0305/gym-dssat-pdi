from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_experiment_plan import generate_plan
from ppo_safe_rendering import DEFAULT_CONFIG, load_yaml


if __name__ == "__main__":
    config = load_yaml(DEFAULT_CONFIG)
    plan = generate_plan(config)
    print("Batch PPO training is DISABLED BY DEFAULT.")
    print("Review the plan below and run train_one_policy.py manually for one model at a time.")
    print(plan.to_string(index=False))

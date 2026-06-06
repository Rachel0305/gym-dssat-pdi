from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_experiment_plan import generate_plan
from ppo_safe_rendering import DEFAULT_CONFIG, load_yaml


if __name__ == "__main__":
    config = load_yaml(DEFAULT_CONFIG)
    plan = generate_plan(config)
    print("Batch PPO evaluation is DISABLED BY DEFAULT.")
    print("After each model is trained, run evaluate_one_policy.py for selected eval years.")
    print(plan[["station", "train_year", "train_year_label", "validation_years", "validation_year_labels"]].to_string(index=False))

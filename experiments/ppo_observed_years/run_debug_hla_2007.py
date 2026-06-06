from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_experiment_plan import write_plan
from ppo_safe_rendering import DEFAULT_CONFIG, load_yaml
from ppo_train import train_one_policy


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    write_plan(DEFAULT_CONFIG)
    debug = config["debug"]
    result = train_one_policy(
        station=debug["station"],
        train_year=int(debug["train_year"]),
        seed=int(config["seed"]),
        total_timesteps=int(debug["total_timesteps"]),
        config_path=DEFAULT_CONFIG,
        debug=True,
    )
    print(result["model_path"].relative_to(PROJECT_ROOT))
    print(result["evaluation_summary"].relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

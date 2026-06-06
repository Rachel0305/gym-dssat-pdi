from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_experiment_plan import write_plan
from ppo_safe_rendering import load_yaml
from ppo_train import train_one_policy


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_ppo_action_safe_debug.yaml"


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    write_plan(CONFIG_PATH)
    debug = config["debug"]
    result = train_one_policy(
        station=debug["station"],
        train_year=int(debug["train_year"]),
        seed=int(config["seed"]),
        total_timesteps=int(debug["total_timesteps"]),
        config_path=CONFIG_PATH,
        debug=True,
    )
    print(result["model_path"].relative_to(PROJECT_ROOT))
    print(result["evaluation_summary"].relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

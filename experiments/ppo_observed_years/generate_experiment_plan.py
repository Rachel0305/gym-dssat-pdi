from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_experiment_plan import write_plan
from ppo_safe_rendering import PROJECT_ROOT as ROOT


if __name__ == "__main__":
    print(write_plan().relative_to(ROOT))

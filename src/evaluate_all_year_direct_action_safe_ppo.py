from __future__ import annotations

from pathlib import Path

from run_all_year_direct_action_safe_ppo import CONFIG_PATH, run_pipeline


if __name__ == "__main__":
    run_pipeline(Path(CONFIG_PATH), evaluate_only=True)

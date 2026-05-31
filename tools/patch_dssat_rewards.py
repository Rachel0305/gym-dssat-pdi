"""Safely patch the installed gym_dssat_pdi rewards.py for reward sweeps.

This script writes a backup next to the installed file before changing it. It is intended
for reproducible experiments where train_hl.py/evaluate_hl.py pass reward parameters via
environment variables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil


DEFAULT_REWARDS_PATH = Path(
    "/opt/gym_dssat_pdi/lib/python3.10/site-packages/"
    "gym_dssat_pdi/envs/configs/rewards.py"
)


FERTILIZATION_REWARD = '''def _reward_float(env_name, fallback):
    value = os.environ.get(env_name)
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def fertilization_reward(_previous_state, _next_state, _history, _cultivar):
    weights = {
            "maize"  : {"coef":1.0, "penality":0.5},
            "cotton" : {"coef":1.0, "penality":0.75},
            "rice"   : {"coef":1.0, "penality":0.5},
    }
    if _next_state:
        last_action = _history['action'][-1]['anfer']
        cultivar_weights = weights[_cultivar]
        penality = _reward_float(
            'GYM_DSSAT_REWARD_PENALITY',
            _reward_float('GYM_DSSAT_REWARD_PENALTY', cultivar_weights["penality"])
        )
        coef = _reward_float('GYM_DSSAT_REWARD_COEF', cultivar_weights["coef"])
        trnu = _next_state['trnu']
        return trnu * coef - penality * last_action
    return None
'''


ALL_REWARD = '''def all_reward(_previous_state, _next_state, _history, _cultivar):
    ferti_reward_value = fertilization_reward(_previous_state, _next_state, _history, _cultivar)
    irrig_reward_value = irrigation_reward(_previous_state, _next_state, _history, _cultivar)
    if ferti_reward_value is None or irrig_reward_value is None:
        return None
    fert_weight = _reward_float('GYM_DSSAT_ALL_FERT_WEIGHT', 1.0)
    irrig_weight = _reward_float('GYM_DSSAT_ALL_IRRIG_WEIGHT', 1.0)
    return fert_weight * ferti_reward_value + irrig_weight * irrig_reward_value
'''


def replace_function(text: str, name: str, replacement: str, next_name: str) -> str:
    pattern = rf"def {name}\([^\n]*\):.*?\n\ndef {next_name}\("
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not locate function block for {name} before {next_name}.")
    prefix = match.group(0)[: -len(f"def {next_name}(")]
    return text.replace(prefix, replacement + "\n\n", 1)


def patch_text(text: str, patch_all_reward: bool) -> str:
    if "import os" not in text:
        text = text.replace("import numpy as np\n", "import numpy as np\nimport os\n", 1)

    text = replace_function(
        text=text,
        name="fertilization_reward",
        replacement=FERTILIZATION_REWARD.rstrip(),
        next_name="irrigation_reward",
    )

    if patch_all_reward:
        text = replace_function(
            text=text,
            name="all_reward",
            replacement=ALL_REWARD.rstrip(),
            next_name="get_reward_function",
        )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards-path", type=Path, default=DEFAULT_REWARDS_PATH)
    parser.add_argument("--backup-suffix", default=".codex_backup")
    parser.add_argument("--patch-all-reward", action="store_true")
    args = parser.parse_args()

    rewards_path = args.rewards_path
    text = rewards_path.read_text(encoding="utf-8")
    if "GYM_DSSAT_REWARD_COEF" in text and (
        not args.patch_all_reward or "GYM_DSSAT_ALL_FERT_WEIGHT" in text
    ):
        print(f"Already patched: {rewards_path}")
        return

    backup_path = rewards_path.with_name(rewards_path.name + args.backup_suffix)
    if not backup_path.exists():
        shutil.copy2(rewards_path, backup_path)
        print(f"Backup written: {backup_path}")
    else:
        print(f"Backup exists: {backup_path}")

    patched = patch_text(text, patch_all_reward=args.patch_all_reward)
    rewards_path.write_text(patched, encoding="utf-8")
    print(f"Patched: {rewards_path}")


if __name__ == "__main__":
    main()

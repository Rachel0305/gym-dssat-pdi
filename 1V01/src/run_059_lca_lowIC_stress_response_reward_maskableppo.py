"""059_00: LCA/LC lowIC stress-response reward MaskablePPO branch."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stress_response_reward_058_059 import cli


DEFAULT_CONFIG = ROOT / "configs" / "059_00_lca_lowIC_stress_response_reward_maskableppo.json"
PROMPT = ROOT / "prompts" / "059_00_lca_lowIC_stress_response_reward_maskableppo.md"


if __name__ == "__main__":
    cli(DEFAULT_CONFIG, PROMPT, expected_task_id="059_00")

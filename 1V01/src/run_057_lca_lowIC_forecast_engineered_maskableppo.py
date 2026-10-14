"""057_00: LCA/LC lowIC engineered-forecast MaskablePPO branch."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from forecast_engineered_observation_056_057 import cli


DEFAULT_CONFIG = ROOT / "configs" / "057_00_lca_lowIC_forecast_engineered_maskableppo.json"
PROMPT = ROOT / "prompts" / "057_00_lca_lowIC_forecast_engineered_maskableppo.md"


if __name__ == "__main__":
    cli(DEFAULT_CONFIG, PROMPT, expected_task_id="057_00")

"""Freeze balanced HLA 80/20 WGEN weather-pool schedule before generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

base = Path(__file__).resolve().parent
source = base / "run_weather_gate.py"
target = base / "weather_pool_plan.json"
if target.exists():
    raise FileExistsError(target)
spec = importlib.util.spec_from_file_location("hla_weather_gate_053", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan = module.build_plan()
if len(plan["train"]) != 80 or len(plan["heldout"]) != 20:
    raise RuntimeError("Expected 80 train and 20 heldout schedule entries")
if {x["weather_seed"] for x in plan["train"]} != set(range(1001, 1081)):
    raise RuntimeError("Train seed pool incomplete")
if {x["weather_seed"] for x in plan["heldout"]} != set(range(1081, 1101)):
    raise RuntimeError("Heldout seed pool incomplete")
target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)

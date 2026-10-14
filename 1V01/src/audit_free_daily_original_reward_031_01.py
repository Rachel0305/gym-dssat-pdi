from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_01_free_daily_original_reward_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_01_free_daily_original_reward_smoke"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reward = config["reward"]
    safety = config["action_safety"]

    checks = {
        "gym_dssat_mode_is_all": config["runtime"]["mode"] == "all",
        "min_irrigation_interval_is_1": int(safety["min_days_between_irrigation"]) == 1,
        "min_fertilization_interval_is_1": int(safety["min_days_between_fertilization"]) == 1,
        "fertilization_latest_dap_is_90": list(safety["fertilization_allowed_dap_range"])[1] == 90,
        "topwt_removed": float(reward["topwt_delta_coef"]) == 0.0,
        "grnwt_kept": float(reward["grnwt_delta_coef"]) == 1.0,
        "water_cost_is_1": float(reward["water_cost"]) == 1.0,
        "nitrogen_cost_is_5": float(reward["nitrogen_cost"]) == 5.0,
        "reward_type_simple": reward["reward_type"] == "delta_grnwt_minus_water_nitrogen_cost",
    }
    passed = all(checks.values())
    result = {
        "task": "031_01_free_daily_original_reward_smoke",
        "config": str(CONFIG.relative_to(ROOT)),
        "passed": passed,
        "checks": checks,
        "training_or_dssat_run": False,
        "reward_formula": "delta_grnwt - 1.0*irrigation - 5.0*nitrogen",
        "min_interval_days": 1,
    }
    (OUT / "031_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 031_01 Free-daily original-reward smoke record",
        "",
        "## Config audit",
        "",
        f"- Passed: {passed}",
        "- Training/DSSAT run: no",
        "- Expert DAP windows: removed by design; this audit checks config only.",
        "- Minimum interval: 1 day.",
        "- Reward: `delta_GRNWT - 1.0 * irrigation - 5.0 * nitrogen`.",
        "- TOPWT term: removed.",
        "- Terminal bonus: not present in this config.",
        "- `/1000` scaling: not present in this config.",
        "",
        "## Checks",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Run a tiny smoke training only after confirming the runner uses this config output root and does not fall back to the old scenario selection silently.",
        ]
    )
    (OUT / "031_01_free_daily_original_reward_smoke_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

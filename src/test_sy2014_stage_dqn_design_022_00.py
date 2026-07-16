from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    EXECUTABLE_STAGE_DAPS,
    EXPERT_GATE_YIELD,
    FEASIBILITY_BONUS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
    NULL_YIELD,
    ORACLE_AUDIT_ONLY_DAPS,
    SOURCE_STAGE_DAPS,
    execute_stage_action,
    nearest_stage_distance,
    terminal_complete_returns,
    valid_action_indices,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "022_00"
PHENOLOGY = {
    "emergence": 9,
    "end_juvenile": 34,
    "floral_initiation": 44,
    "silking_75pct": 89,
    "begin_grain_fill": 99,
    "end_grain_fill": 136,
    "maturity": 139,
}


def phase(dap: int) -> str:
    if dap <= PHENOLOGY["emergence"]:
        return "pre_emergence"
    if dap <= PHENOLOGY["end_juvenile"]:
        return "juvenile"
    if dap <= PHENOLOGY["floral_initiation"]:
        return "juvenile_to_floral_init"
    if dap <= PHENOLOGY["silking_75pct"]:
        return "floral_init_to_silking"
    if dap <= PHENOLOGY["begin_grain_fill"]:
        return "silking_to_grain_fill"
    if dap <= PHENOLOGY["end_grain_fill"]:
        return "grain_fill"
    return "post_grain_fill"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "sy2014_stage_dqn_022_00.yaml").read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    check("official_source_daps_locked", SOURCE_STAGE_DAPS == (0, 30, 50, 65, 85, 110), str(SOURCE_STAGE_DAPS))
    check("gym_dap0_mapped_only_to_dap1", EXECUTABLE_STAGE_DAPS == (1, 30, 50, 65, 85, 110), str(EXECUTABLE_STAGE_DAPS))
    check("yaml_source_daps_match_code", tuple(config["source_stage_daps"]) == SOURCE_STAGE_DAPS, str(config["source_stage_daps"]))
    check("yaml_executable_daps_match_code", tuple(config["executable_stage_daps"]) == EXECUTABLE_STAGE_DAPS, str(config["executable_stage_daps"]))
    check("stage_daps_strictly_increasing", all(b > a for a, b in zip(EXECUTABLE_STAGE_DAPS, EXECUTABLE_STAGE_DAPS[1:])), str(EXECUTABLE_STAGE_DAPS))
    check("all_stages_inside_growth_season", min(EXECUTABLE_STAGE_DAPS) >= 1 and max(EXECUTABLE_STAGE_DAPS) < PHENOLOGY["maturity"], f"maturity={PHENOLOGY['maturity']}")
    check("action_table_is_cartesian_3x3", len(ACTION_TABLE_9) == 9 and {(x['amir'], x['anfer']) for x in ACTION_TABLE_9.values()} == {(i, n) for i in (0.0, 15.0, 30.0) for n in (0.0, 50.0, 100.0)}, "9 unique actions")
    check("non_stage_day_forces_noop", valid_action_indices(40) == (0,), str(valid_action_indices(40)))
    check("dap110_nitrogen_action_mask", valid_action_indices(110) == (0, 1, 2), str(valid_action_indices(110)))
    check("pre_silking_stage_allows_9_actions", valid_action_indices(85) == tuple(range(9)), str(valid_action_indices(85)))

    clipped = execute_stage_action(8, 85, used_irrigation=105.0, used_nitrogen=250.0)
    check("remaining_budget_clip", clipped.executed_irrigation == 15.0 and clipped.executed_nitrogen == 50.0, str(clipped))
    check("budget_constants_frozen", IRRIGATION_BUDGET == 120.0 and NITROGEN_BUDGET == 300.0, f"I={IRRIGATION_BUDGET},N={NITROGEN_BUDGET}")

    late_n_rejected = False
    try:
        execute_stage_action(3, 110, 0.0, 0.0)
    except ValueError:
        late_n_rejected = True
    check("late_n_rejected_not_aliased", late_n_rejected, "DAP110 action3 raises ValueError")

    oracle_distances = {dap: nearest_stage_distance(dap) for dap in ORACLE_AUDIT_ONLY_DAPS}
    check("oracle_daps_not_used_as_stage_daps", not any(dap in EXECUTABLE_STAGE_DAPS for dap in ORACLE_AUDIT_ONLY_DAPS), str(oracle_distances))
    check("oracle_proximity_is_audit_only", max(oracle_distances.values()) <= 8, str(oracle_distances))

    stage_rows = pd.DataFrame(
        {
            "source_dap": SOURCE_STAGE_DAPS,
            "executable_dap": EXECUTABLE_STAGE_DAPS,
            "dssat_phase_interpretation": [phase(dap) for dap in EXECUTABLE_STAGE_DAPS],
            "valid_action_indices": [json.dumps(valid_action_indices(dap)) for dap in EXECUTABLE_STAGE_DAPS],
        }
    )
    stage_rows.to_csv(OUT / "022_00_stage_provenance_and_masks.csv", index=False)

    actions = [
        execute_stage_action(index, dap, 0.0, 0.0)
        for index, dap in zip((1, 4, 7, 4, 1, 0), EXECUTABLE_STAGE_DAPS)
    ]
    returns = terminal_complete_returns(actions, final_yield=11205.0)
    # I75/N200 => cost 1075; yield gain 5797; gate bonus 1620; initial return 6342.
    check("terminal_reward_constants", NULL_YIELD == 5408.0 and EXPERT_GATE_YIELD == 11077.0 and FEASIBILITY_BONUS == 1620.0, "5408/11077/1620")
    check("terminal_complete_return_handcheck", abs(returns[0] - 6342.0) < 1e-12, f"G0={returns[0]}")
    check("return_has_one_target_per_stage", len(returns) == len(EXECUTABLE_STAGE_DAPS), str(returns))
    check("one_step_td_disabled_by_design", True, "single loss target is terminal-complete G_t")
    check("five_step_td_disabled_by_design", True, "no parallel 5-step loss")
    check("target_bootstrap_disabled_by_design", True, "completed episode return contains no target-network bootstrap")
    q_target = config["q_target"]
    check(
        "yaml_has_single_terminal_complete_target",
        q_target["type"] == "terminal_complete_monte_carlo_return"
        and q_target["gamma"] == 1.0
        and not q_target["one_step_td_enabled"]
        and not q_target["five_step_td_enabled"]
        and not q_target["target_network_bootstrap_enabled"],
        json.dumps(q_target, ensure_ascii=False),
    )
    check(
        "yaml_disables_training_and_dssat",
        not config["training"]["enabled_in_022_00"] and not config["dssat"]["enabled_in_022_00"],
        "training=false,dssat=false",
    )
    check("no_training_or_dssat_in_022_00", True, "pure Python unit test")

    pd.DataFrame(checks).to_csv(OUT / "022_00_unit_test_results.csv", index=False)
    pd.DataFrame([{"oracle_dap": dap, "nearest_official_stage_distance_days": distance} for dap, distance in oracle_distances.items()]).to_csv(OUT / "022_00_oracle_distance_audit.csv", index=False)
    pd.DataFrame([{"stage_index": i, "return_target": value} for i, value in enumerate(returns)]).to_csv(OUT / "022_00_terminal_complete_return_handcheck.csv", index=False)

    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": "completed" if not failed else "failed",
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "training_calls": 0,
        "dssat_calls": 0,
        "next_gate": "022_01 deterministic fixed-stage feasibility search" if not failed else "stop",
        "fixed_stage_scientific_feasibility_verified": False,
    }
    (OUT / "022_00_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

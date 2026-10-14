from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
from frozen_nstep_dqn_config_020_11 import apply_environment_constants


RUN = (
    ROOT
    / "benchmark_results"
    / "021_05"
    / "021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k__sy_2014_seed0"
)
CHECKPOINT = 10_000
OUT = RUN / "independent_reevaluation"


def main() -> None:
    apply_environment_constants(shared)
    env_args = json.loads((RUN / "dqn_env_args.json").read_text(encoding="utf-8"))
    null_yield = float(pd.read_csv(RUN / "null_evaluation" / "null_summary.csv").iloc[0]["final_grain_kg_ha"])
    spec = legacy.SiteSpec(
        code="SY",
        station="Shenyang",
        year=2014,
        treatment=2,
        input_root=RUN / "dqn_input",
        mzx_name="SY2014_benchmark_dqn.MZX",
        weather_name="",
        soil_id="",
    )
    model = DQN.load(str(RUN / "checkpoints" / f"checkpoint_{CHECKPOINT}" / "model.zip"))
    rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for repeat in (1, 2):
        destination = OUT / f"repeat_{repeat}"
        destination.mkdir(parents=True, exist_ok=False)
        daily, summary, audit = legacy.evaluate_checkpoint(
            model,
            spec,
            env_args,
            null_yield,
            CHECKPOINT,
            destination,
        )
        summary = dict(summary)
        summary["repeat"] = repeat
        summary["runtime_audit_passed"] = bool(audit["passed"])
        rows.append(summary)
        daily = daily.copy()
        daily["repeat"] = repeat
        daily_frames.append(daily)

    summary_frame = pd.DataFrame(rows)
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    summary_frame.to_csv(OUT / "021_05_independent_reevaluation_summary.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(OUT / "021_05_independent_reevaluation_daily.csv", index=False, encoding="utf-8-sig")

    numeric = [
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "action_irrigation_total_mm",
        "action_nitrogen_total_kg_ha",
        "total_reward",
        "steps",
    ]
    if not summary_frame[numeric].nunique(dropna=False).eq(1).all():
        raise RuntimeError("Independent deterministic reevaluations disagree in season summary")
    compare_columns = [
        "dap",
        "operation_dap",
        "action_index",
        "irrigation_mm",
        "fertilizer_kg_ha",
        "reward",
        "grnwt",
        "topwt",
        "swfac",
        "nstres",
    ]
    left = daily_frame[daily_frame["repeat"].eq(1)][compare_columns].reset_index(drop=True)
    right = daily_frame[daily_frame["repeat"].eq(2)][compare_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    print(summary_frame.to_string(index=False))
    print("021_05 independent deterministic reevaluation passed exactly")


if __name__ == "__main__":
    main()

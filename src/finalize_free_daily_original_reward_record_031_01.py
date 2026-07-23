from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_01_free_daily_original_reward_smoke"


def simple_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_string(index=False)


def main() -> None:
    result_path = OUT / "031_01_result.json"
    train_path = OUT / "evaluation" / "training_run_summary.csv"
    eval_path = OUT / "evaluation" / "ppo_direct_eval_summary.csv"
    diag_path = OUT / "evaluation" / "ppo_decision_reasonableness_diagnosis.csv"

    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    evals = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()

    lines = [
        "# 031_01 Free-daily original-reward smoke record",
        "",
        "## Execution status",
        "",
        "- Effective run environment: Docker container `nifty_taussig`, Python `/opt/gym_dssat_pdi/bin/python`.",
        "- Earlier host-Python attempt failed because `gym_dssat_pdi` was not registered; this is a non-scientific environment failure.",
        "- A second container run failed at env creation because `runtime.mode=smoke`; config was corrected to `runtime.mode=all` while keeping the smoke scope as SYA2014 only.",
        "- The final container run completed training and evaluation; only the first markdown writer failed because optional package `tabulate` was missing. This finalizer writes the record from existing CSV/JSON outputs without retraining.",
        "",
        "## Fixed setup",
        "",
        "- Smoke case: SYA2014 seed0.",
        "- Training timesteps: 512.",
        "- Free daily decision: yes.",
        "- Expert DAP windows: no.",
        "- Minimum operation interval: 1 day.",
        "- Safety constraints retained: daily/season caps and no fertilization after DAP90.",
        "- Reward: `delta_GRNWT - 1.0 * irrigation - 5.0 * nitrogen`.",
        "- No TOPWT term, no terminal bonus, no `/1000` scaling.",
        "",
        "## Result summary",
        "",
        f"- Training status: {train['run_status'].iloc[0] if 'run_status' in train and len(train) else 'missing'}",
        f"- Evaluation rows: {len(evals)}",
        f"- Result JSON: `{result_path.relative_to(ROOT).as_posix()}`",
        "",
        "## Training summary",
        "",
        "```text",
        simple_table(train),
        "```",
        "",
        "## Evaluation summary",
        "",
        "```text",
        simple_table(evals),
        "```",
        "",
        "## Decision diagnosis",
        "",
        "```text",
        simple_table(diag),
        "```",
        "",
        "## Interpretation",
        "",
        "- This smoke proves the fully free daily PPO code path can run in the correct container.",
        "- It does not yet prove a good management strategy.",
        "- In this short smoke, the learned deterministic policy reaches the season caps: irrigation 160 mm and nitrogen 250 kg/ha.",
        "- Therefore the next decision is whether to treat this as an expected baseline limitation of the original reward, or to add a pre-registered feasibility/efficiency objective before scaling to all years.",
    ]
    (OUT / "031_01_free_daily_original_reward_smoke_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

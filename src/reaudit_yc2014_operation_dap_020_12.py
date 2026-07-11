from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frozen_nstep_dqn_config_020_11 import apply_environment_constants
from run_yc_fq_frozen_nstep_cross_site_020_12 import (
    SITE_SPECS,
    evaluate_checkpoint,
    shared,
)


RUN_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "frozen_nstep_cross_site_020_12"
    / "YC2014"
    / "seed0_50000steps"
)
OUT_DIR = RUN_DIR / "reaudit_operation_dap_020_12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Re-evaluate saved YC models only; never call model.learn()."""

    if OUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing re-audit output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    spec = SITE_SPECS["YC"]
    env_args = json.loads((RUN_DIR / "dqn_env_args.json").read_text(encoding="utf-8"))
    null_summary = pd.read_csv(RUN_DIR / "null" / "null_summary.csv").iloc[0]
    null_yield = float(null_summary["final_grain_kg_ha"])
    original = pd.read_csv(RUN_DIR / "checkpoint_summary.csv")
    apply_environment_constants(shared)

    from stable_baselines3 import DQN

    all_daily: list[pd.DataFrame] = []
    all_summary: list[dict[str, object]] = []
    all_audit: list[dict[str, object]] = []
    model_hashes: list[dict[str, object]] = []

    for checkpoint in sorted(original["checkpoint_step"].astype(int).tolist()):
        source_model = RUN_DIR / "checkpoints" / f"checkpoint_{checkpoint}" / "model.zip"
        if not source_model.exists():
            raise FileNotFoundError(source_model)
        destination = OUT_DIR / f"checkpoint_{checkpoint}"
        destination.mkdir(parents=True, exist_ok=False)

        eval_args = dict(env_args)
        eval_args["log_saving_path"] = str(destination / "pdi_gym.log")
        model = DQN.load(str(source_model), device="cpu")
        daily, summary, audit = evaluate_checkpoint(
            model,
            spec,
            eval_args,
            null_yield,
            checkpoint,
            destination,
        )
        all_daily.append(daily)
        all_summary.append(summary)
        all_audit.append(audit)
        model_hashes.append(
            {
                "checkpoint_step": checkpoint,
                "source_model": str(source_model.relative_to(PROJECT_ROOT)),
                "size_bytes": source_model.stat().st_size,
                "sha256": sha256(source_model),
            }
        )

    daily_frame = pd.concat(all_daily, ignore_index=True)
    summary_frame = pd.DataFrame(all_summary).sort_values("checkpoint_step")
    audit_frame = pd.DataFrame(
        [
            {
                "checkpoint_step": int(item["checkpoint_step"]),
                "operation_daps": ";".join(map(str, item["operation_daps"])),
                "intervals": ";".join(map(str, item["intervals"])),
                "passed": bool(item["passed"]),
                **item["checks"],
            }
            for item in all_audit
        ]
    ).sort_values("checkpoint_step")

    compare_columns = [
        "checkpoint_step",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "action_irrigation_total_mm",
        "action_nitrogen_total_kg_ha",
        "total_reward",
        "steps",
    ]
    joined = original[compare_columns].merge(
        summary_frame[compare_columns],
        on="checkpoint_step",
        suffixes=("_original", "_corrected_reaudit"),
        validate="one_to_one",
    )
    for column in compare_columns[1:]:
        joined[f"delta_{column}"] = (
            pd.to_numeric(joined[f"{column}_corrected_reaudit"], errors="coerce")
            - pd.to_numeric(joined[f"{column}_original"], errors="coerce")
        )
    delta_columns = [column for column in joined.columns if column.startswith("delta_")]
    joined["all_summary_values_reproduced"] = np.isclose(
        joined[delta_columns].fillna(np.inf).abs().max(axis=1), 0.0, atol=1e-6
    )

    daily_frame.to_csv(OUT_DIR / "corrected_all_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(OUT_DIR / "corrected_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    audit_frame.to_csv(OUT_DIR / "corrected_runtime_audit_summary.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(OUT_DIR / "original_vs_corrected_summary_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_hashes).to_csv(OUT_DIR / "source_model_hash_audit.csv", index=False, encoding="utf-8-sig")

    selected = summary_frame.sort_values(
        ["total_reward", "checkpoint_step"], ascending=[False, True], kind="stable"
    ).iloc[0]
    report = {
        "purpose": "YC2014 saved-model-only re-audit using pre-action operation_dap",
        "training_performed": False,
        "source_run": str(RUN_DIR.relative_to(PROJECT_ROOT)),
        "output": str(OUT_DIR.relative_to(PROJECT_ROOT)),
        "checkpoint_count": int(len(summary_frame)),
        "all_runtime_audits_passed": bool(audit_frame["passed"].all()),
        "all_summary_values_reproduced": bool(joined["all_summary_values_reproduced"].all()),
        "corrected_15000_passed": bool(
            audit_frame.loc[audit_frame["checkpoint_step"].eq(15000), "passed"].iloc[0]
        ),
        "selected_checkpoint_step": int(selected["checkpoint_step"]),
        "selected_total_reward": float(selected["total_reward"]),
    }
    (OUT_DIR / "corrected_reaudit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PYTHON = "/opt/gym_dssat_pdi/bin/python"


COMBOS = [
    {
        "name": "safe60_20_pen15_icost15_ns20_nlim300_nex5_ilim120_iex10",
        "penality": 15,
        "amir_cost": 15,
        "amir_no_stress_cost": 20,
        "anfer_excess_limit": 300,
        "anfer_excess_cost": 5,
        "amir_excess_limit": 120,
        "amir_excess_cost": 10,
        "safe_anfer_cap": 60,
        "safe_amir_cap": 20,
    },
    {
        "name": "safe40_15_pen15_icost15_ns50_nlim250_nex10_ilim80_iex20",
        "penality": 15,
        "amir_cost": 15,
        "amir_no_stress_cost": 50,
        "anfer_excess_limit": 250,
        "anfer_excess_cost": 10,
        "amir_excess_limit": 80,
        "amir_excess_cost": 20,
        "safe_anfer_cap": 40,
        "safe_amir_cap": 15,
    },
    {
        "name": "safe30_10_pen20_icost20_ns50_nlim220_nex15_ilim80_iex30",
        "penality": 20,
        "amir_cost": 20,
        "amir_no_stress_cost": 50,
        "anfer_excess_limit": 220,
        "anfer_excess_cost": 15,
        "amir_excess_limit": 80,
        "amir_excess_cost": 30,
        "safe_anfer_cap": 30,
        "safe_amir_cap": 10,
    },
    {
        "name": "safe20_5_pen20_icost20_ns80_nlim180_nex20_ilim60_iex40",
        "penality": 20,
        "amir_cost": 20,
        "amir_no_stress_cost": 80,
        "anfer_excess_limit": 180,
        "anfer_excess_cost": 20,
        "amir_excess_limit": 60,
        "amir_excess_cost": 40,
        "safe_anfer_cap": 20,
        "safe_amir_cap": 5,
    },
]


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def common_reward_args(combo: dict, args: argparse.Namespace) -> list[str]:
    return [
        "--coef",
        "1.0",
        "--penality",
        str(combo["penality"]),
        "--all-fert-weight",
        "1.0",
        "--all-irrig-weight",
        "1.0",
        "--all-amir-cost",
        str(combo["amir_cost"]),
        "--all-amir-no-stress-cost",
        str(combo["amir_no_stress_cost"]),
        "--all-water-stress-threshold",
        str(args.water_stress_threshold),
        "--all-anfer-excess-limit",
        str(combo["anfer_excess_limit"]),
        "--all-anfer-excess-cost",
        str(combo["anfer_excess_cost"]),
        "--all-amir-excess-limit",
        str(combo["amir_excess_limit"]),
        "--all-amir-excess-cost",
        str(combo["amir_excess_cost"]),
    ]


def train_combo(combo: dict, args: argparse.Namespace, combo_dir: Path) -> None:
    command = [
        PYTHON,
        "train_hl_all_multisite_safe.py",
        "--site",
        "HL",
        "--data-dir",
        args.data_dir,
        *common_reward_args(combo, args),
        "--total-timesteps",
        str(args.total_timesteps),
        "--eval-freq",
        str(args.eval_freq),
        "--n-eval-episodes",
        str(args.n_eval_episodes),
        "--output-dir",
        str(combo_dir),
        "--log-dir",
        args.log_dir,
        "--ppo-n-steps",
        str(args.ppo_n_steps),
        "--ppo-batch-size",
        str(args.ppo_batch_size),
        "--ppo-n-epochs",
        str(args.ppo_n_epochs),
        "--safe-anfer-cap",
        str(combo["safe_anfer_cap"]),
        "--safe-amir-cap",
        str(combo["safe_amir_cap"]),
        "--sb3-verbose",
        "0",
    ]
    if args.resume_model:
        command.extend(["--resume-model", args.resume_model])
    run_command(command, combo_dir / "train.log")


def diagnose_combo(combo: dict, combo_dir: Path, args: argparse.Namespace) -> None:
    model_path = combo_dir / "best_model.zip"
    if not model_path.exists():
        model_path = combo_dir / "final_model.zip"
    diagnose_dir = combo_dir / "diagnose_ppo"
    command = [
        PYTHON,
        "diagnose_all_water_stress_sites.py",
        "--sites",
        "HL",
        "--agents",
        "null,expert,ppo",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(diagnose_dir),
        "--model-path",
        str(model_path),
        "--safe-anfer-cap",
        str(combo["safe_anfer_cap"]),
        "--safe-amir-cap",
        str(combo["safe_amir_cap"]),
        "--max-steps",
        "260",
    ]
    run_command(command, combo_dir / "diagnose_ppo.log")


def summarize(output_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(output_dir.glob("*/diagnose_ppo/all_water_stress_summary.csv")):
        combo_name = summary_path.parents[1].name
        df = pd.read_csv(summary_path, keep_default_na=False)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["combo"] = combo_name
            rows.append(row_dict)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    preferred_cols = [
        "combo",
        "agent",
        "PRCP",
        "ETCP",
        "swfac_stress_days_gt_0.05",
        "max_swfac",
        "mean_swfac",
        "nstres_days_gt_0.05",
        "max_nstres",
        "mean_nstres",
        "max_grnwt",
        "total_anfer",
        "total_amir",
        "final_totir",
        "total_reward",
        "terminated_normally",
    ]
    return summary[[col for col in preferred_cols if col in summary.columns]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="my_data_rain_scaling/HL/scale0p6")
    parser.add_argument("--output-dir", default="output_hl/rain_scaling_train/HL_scale0p6_safe_penalty_scan_50k")
    parser.add_argument("--log-dir", default="logs_hl")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--ppo-n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-n-epochs", type=int, default=5)
    parser.add_argument("--water-stress-threshold", type=float, default=0.05)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scan_config.json").write_text(json.dumps(COMBOS, indent=2), encoding="utf-8")

    for combo in COMBOS:
        combo_dir = output_dir / combo["name"]
        done_marker = combo_dir / "SCAN_COMPLETE"
        if done_marker.exists() and not args.force:
            print(f"SKIP completed combo: {combo['name']}", flush=True)
            continue
        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "combo_config.json").write_text(json.dumps(combo, indent=2), encoding="utf-8")
        train_combo(combo, args, combo_dir)
        diagnose_combo(combo, combo_dir, args)
        done_marker.write_text("complete\n", encoding="utf-8")

    summary = summarize(output_dir)
    if not summary.empty:
        summary_path = output_dir / "safe_penalty_scan_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(summary.round(3).to_string(index=False), flush=True)
        print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise

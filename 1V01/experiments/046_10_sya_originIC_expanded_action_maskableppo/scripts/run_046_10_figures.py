"""Run the 046_10 figure/report package from one pinned entry point.

This wrapper intentionally reuses the audited 046_05/046_06 plotting code, but
keeps the exact 046_10 config, checkpoint, PPO output root, and expected outputs
in one experiment-local place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "046_10_sya_originIC_expanded_action_maskableppo.json"
DEFAULT_AUTO_CONFIG = ROOT / "configs" / "046_11_sya_originIC_external_auto_n_rule_nstd050_minimal.json"
PPO_RUN_DIR = ROOT / "benchmark_results" / "046_10_sya_originIC_expanded_action_maskableppo"
CHECKPOINT = 100_000
VALIDATION_YEARS = list(range(2014, 2024))

DAILY_SCRIPT = ROOT / "src" / "build_sya_configured_five_scenario_daily_plots_046_06.py"
BAR_SCRIPT = ROOT / "src" / "build_sya_configured_reporting_046_05.py"

def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def auto_suffix(config: dict) -> str:
    return str(config.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()


def expected_suffix_from_config_name(path: Path) -> str:
    stem = path.stem
    marker = "external_auto_n_rule_"
    if marker in stem:
        return stem.split(marker, 1)[1]
    return ""


def validate_auto_config_identity(path: Path, config: dict) -> None:
    expected = expected_suffix_from_config_name(path)
    suffix = auto_suffix(config)
    task_name = str(config.get("task_name", ""))
    if expected and suffix != expected:
        raise ValueError(
            f"Auto config identity mismatch: file name expects output_suffix={expected!r}, "
            f"but {rel(path)} contains output_suffix={suffix!r}. "
            "Please fix external_auto_n_rule.output_suffix before running figures."
        )
    if expected and expected not in task_name:
        raise ValueError(
            f"Auto config identity mismatch: file name expects task_name to contain {expected!r}, "
            f"but {rel(path)} contains task_name={task_name!r}."
        )


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def auto_root(auto_config: dict, run_id: str = "") -> Path:
    suffix = auto_suffix(auto_config)
    name = "046_04_sya_originIC_external_auto_n_rule"
    if suffix:
        name = f"{name}_{suffix}"
    return ROOT / "benchmark_results" / f"{name}{run_suffix(run_id)}"


def daily_root(report_config: dict, run_id: str) -> Path:
    suffix = auto_suffix(report_config)
    auto_part = f"_auto_{suffix}" if suffix else ""
    return ROOT / "benchmark_results" / (
        f"046_06_sya_{report_config['input_profile']}_{report_config['task_id']}_"
        f"{report_config['task_name']}{auto_part}_five_scenario_daily_ckpt{CHECKPOINT}{run_suffix(run_id)}"
    )


def bar_root(report_config: dict, run_id: str) -> Path:
    suffix = auto_suffix(report_config)
    auto_part = f"_auto_{suffix}" if suffix else ""
    return ROOT / "benchmark_results" / (
        f"046_05_sya_{report_config['input_profile']}_{report_config['task_id']}_"
        f"{report_config['task_name']}{auto_part}_reporting_ckpt{CHECKPOINT}{run_suffix(run_id)}"
    )


def write_reporting_config(base_config: dict, auto_config: dict, auto_config_path: Path, label: str, run_label: str, auto_run_id: str) -> Path:
    report_config = dict(base_config)
    rule = dict(auto_config["external_auto_n_rule"])
    if label:
        report_config["task_name"] = f"{base_config['task_name']}_{label}"
    report_config["external_auto_n_rule"] = rule
    report_config["figure_package"] = {
        "label": run_label,
        "source_task_name": base_config["task_name"],
        "auto_config": rel(auto_config_path),
        "auto_run_id": auto_run_id,
        "auto_output_root": rel(auto_root(auto_config, auto_run_id)),
        "resolved_auto_output_suffix": rule.get("output_suffix", ""),
    }
    out_dir = ROOT / "experiments" / "046_10_sya_originIC_expanded_action_maskableppo" / "runs" / run_label / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "046_10_reporting_config.json"
    path.write_text(json.dumps(report_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def check_required_inputs(report_config: dict) -> list[str]:
    required = [
        CONFIG,
        PPO_RUN_DIR / "046_10_formal_result.json",
        PPO_RUN_DIR / "evaluation" / "046_10_checkpoint_validation_summary.csv",
        PPO_RUN_DIR / "models" / "SYA" / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip",
        ROOT / "benchmark_results" / "046_03_sya_originIC_four_baselines" / "evaluation" / "046_03_baseline_summary.csv",
        (ROOT / str(report_config["figure_package"]["auto_output_root"])) / "evaluation" / "046_04_external_auto_n_summary.csv",
        DAILY_SCRIPT,
        BAR_SCRIPT,
    ]
    return [rel(path) for path in required if not path.exists()]


def daily_command(python_exe: str, report_config_path: Path, years: list[int]) -> list[str]:
    return [
        python_exe,
        str(DAILY_SCRIPT),
        "--config",
        str(report_config_path),
        "--checkpoint",
        str(CHECKPOINT),
        "--ppo-run-dir",
        str(PPO_RUN_DIR),
        "--years",
        ",".join(map(str, years)),
    ]


def bar_command(python_exe: str, report_config_path: Path) -> list[str]:
    return [
        python_exe,
        str(BAR_SCRIPT),
        "--config",
        str(report_config_path),
        "--checkpoint",
        str(CHECKPOINT),
        "--ppo-run-dir",
        str(PPO_RUN_DIR),
    ]


def expected_outputs(report_config: dict, years: list[int]) -> dict[str, list[str] | str]:
    daily = daily_root(report_config, "")
    bars = bar_root(report_config, "")
    return {
        "daily_root": rel(daily),
        "daily_figures": [rel(daily / "figures" / f"046_06_sy{year}_five_scenario_daily.png") for year in years],
        "daily_table": rel(daily / "tables" / "046_06_sya_five_scenario_daily.csv"),
        "season_summary": rel(daily / "tables" / "046_06_sya_five_scenario_season_summary.csv"),
        "bar_root": rel(bars),
        "metric_bars": rel(bars / "figures" / "046_05_five_scenario_metrics.png"),
        "management_bars": rel(bars / "figures" / "046_05_five_scenario_management.png"),
        "metric_table": rel(bars / "tables" / "046_05_five_scenario_metric_summary.csv"),
        "gap_table": rel(bars / "tables" / "046_05_metric_gaps_vs_four_baseline_max.csv"),
    }


def missing_outputs(outputs: dict[str, list[str] | str]) -> list[str]:
    paths: list[str] = []
    for value in outputs.values():
        if isinstance(value, list):
            paths.extend(value)
        else:
            paths.append(value)
    return [path for path in paths if not (ROOT / path).exists()]


def existing_output_roots(outputs: dict[str, list[str] | str]) -> list[str]:
    roots = []
    for key in ("daily_root", "bar_root"):
        value = outputs.get(key)
        if isinstance(value, str) and (ROOT / value).exists():
            roots.append(value)
    return roots


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def parse_years(text: str) -> list[int]:
    if not text.strip():
        return VALIDATION_YEARS
    years = [int(item.strip()) for item in text.split(",") if item.strip()]
    invalid = [year for year in years if year not in VALIDATION_YEARS]
    if invalid:
        raise ValueError(f"046_10 validation years are 2014-2023; invalid years: {invalid}")
    return years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands and expected outputs without running.")
    parser.add_argument("--skip-daily", action="store_true", help="Skip 046_06 daily replay/process plots.")
    parser.add_argument("--skip-bars", action="store_true", help="Skip 046_05 metric/resource bar charts.")
    parser.add_argument("--years", default="", help="Optional comma-separated subset of validation years for daily plots.")
    parser.add_argument("--auto-config", type=Path, default=DEFAULT_AUTO_CONFIG, help="Config whose external_auto_n_rule selects the 046_04 auto source.")
    parser.add_argument("--auto-run-id", default="", help="Optional run-id suffix used when the selected auto source was generated.")
    parser.add_argument("--label", default="", help="Optional semantic output label. Empty keeps output folders compact.")
    parser.add_argument("--stamp", default="", help="Deprecated alias for --label, kept for older commands.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into existing figure output roots.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use inside the active environment.")
    args = parser.parse_args()

    years = parse_years(args.years)
    base_config = read_json(CONFIG)
    auto_config_path = args.auto_config if args.auto_config.is_absolute() else (ROOT / args.auto_config).resolve()
    auto_config = read_json(auto_config_path)
    validate_auto_config_identity(auto_config_path, auto_config)
    label = args.label.strip() or args.stamp.strip()
    run_label = label or auto_suffix(auto_config) or "default_auto"
    report_config_path = write_reporting_config(base_config, auto_config, auto_config_path, label, run_label, args.auto_run_id.strip())
    report_config = read_json(report_config_path)

    commands: list[list[str]] = []
    if not args.skip_daily:
        commands.append(daily_command(args.python, report_config_path, years))
    if not args.skip_bars:
        commands.append(bar_command(args.python, report_config_path))

    outputs = expected_outputs(report_config, years)
    payload = {
        "task": "046_10_figure_package",
        "checkpoint": CHECKPOINT,
        "years": years,
        "label": run_label,
        "report_config": rel(report_config_path),
        "auto_config": rel(auto_config_path),
        "auto_source": report_config["figure_package"]["auto_output_root"],
        "auto_run_id": args.auto_run_id,
        "commands": commands,
        "expected_outputs": outputs,
    }
    missing_inputs = check_required_inputs(report_config)
    payload["missing_inputs"] = missing_inputs
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if missing_inputs:
        raise FileNotFoundError("Missing required 046_10 figure inputs: " + "; ".join(missing_inputs))
    occupied_roots = existing_output_roots(outputs)
    if occupied_roots and not args.overwrite:
        raise FileExistsError(
            "Refusing to write into existing figure output roots without --overwrite: "
            + "; ".join(occupied_roots)
            + ". Use a new --label for a clean comparison, or pass --overwrite intentionally."
        )

    for command in commands:
        run_command(command)

    remaining = missing_outputs(outputs)
    payload["missing_outputs_after_run"] = remaining
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if remaining:
        raise RuntimeError("Some expected outputs were not created: " + "; ".join(remaining))


if __name__ == "__main__":
    main()

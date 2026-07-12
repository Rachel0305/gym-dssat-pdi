"""Unified CLI for configuration-driven DQN benchmark experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config_loader import load_config
from .evaluation_runner import evaluate_model_file
from .experiment_registry import (
    ExperimentRegistry,
    ExperimentSpec,
    ManifestStatus,
    expand_experiments,
)
from .result_registry import ResultRegistry, discover_legacy_results
from .train_runner import SimulatedInterruption, latest_checkpoint, train_case
from .validators import ConfigValidationError, raise_for_invalid_config, validate_config


LOGGER = logging.getLogger("benchmark")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv_int(value: str | None) -> list[int] | None:
    return [int(item.strip()) for item in value.split(",") if item.strip()] if value else None


def _parse_csv_text(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _ensure_layout(run_dir: Path) -> None:
    for name in (
        "configs",
        "logs",
        "checkpoints",
        "evaluations",
        "summaries",
        "figures",
        "tables",
        "reports",
        "manifests",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _mode(args: argparse.Namespace) -> str:
    active = [
        name
        for name in ("train_only", "evaluate_only", "report_only", "plot_only")
        if getattr(args, name)
    ]
    if len(active) > 1:
        raise ValueError("Only one of --train-only/--evaluate-only/--report-only/--plot-only may be used")
    return active[0].replace("_", "-") if active else "full"


def _save_config(spec: ExperimentSpec, run_dir: Path, source_path: Path) -> None:
    with (run_dir / "configs" / "merged_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec.config, handle, allow_unicode=True, sort_keys=False)
    (run_dir / "configs" / "source_config_path.txt").write_text(
        str(source_path.resolve()) + "\n", encoding="utf-8"
    )


def _registry_path(value: str | Path, root: Path) -> str:
    """Store portable project-relative paths in the result registry."""

    raw = str(value)
    for prefix in ("/workspace/", "/workspaces/gym-dssat-pdi/"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    path = Path(raw)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _load_evaluations(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_path = run_dir / "evaluations" / "season_summary.csv"
    daily_path = run_dir / "evaluations" / "daily_trajectory.csv"
    if not season_path.exists() or not daily_path.exists():
        raise FileNotFoundError(
            f"Standardized evaluation outputs are missing under {run_dir / 'evaluations'}"
        )
    return pd.read_csv(season_path), pd.read_csv(daily_path)


def _report_context(manifest: dict[str, Any], mode: str, train_result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "background": [
            "五站点 DQN 证据已形成，但训练、评估、绘图和报告入口分散。",
            "本框架在不改 reward、IC 和旧结果的前提下统一控制面。",
        ],
        "current_problems": [
            "历史脚本存在复制、全局常量耦合和输出 schema 不统一。",
            "历史模型没有 replay buffer，不能声称 exact resume。",
            "SY2014 IC 来源仍冲突，正式训练被配置层阻塞。",
        ],
        "configuration": [
            "冻结主线：9 动作、I120/N300、共享 7 DAP、n_steps=5。",
            "reward：local-null terminal yield gain - water cost - nitrogen cost。",
            "站点路径使用项目相对路径；逐年 prepared input 参与 hash。",
        ],
        "reuse": [
            "config_hash 完全匹配且 manifest completed/reused 时跳过训练。",
            "旧结果按 report/evaluation/warm_start/exact_resume 分级，不强行认定等价。",
            "020_11、020_13、LC/SY legacy 证据已写入 existing-results registry。",
        ],
        "resume": [
            "新 checkpoint 保存 model、replay buffer、RNG state 和 training_state。",
            "恢复在 episode/checkpoint 边界重建 DSSAT 环境，标记 partial reproducible resume，而非 bitwise exact continuation。",
        ],
        "training_evaluation": [
            f"运行模式：{mode}",
            f"训练结果：{train_result or '本次未训练'}",
            "所有 daily/season 字段由统一 schema adapter 生成；缺失值保持 NA。",
        ],
        "risks": [
            "旧脚本不被删除或覆盖。",
            "输出目录存在时默认拒绝覆盖；--force 生成时间戳目录。",
            "缺失 baseline/指标不会导致整个报告崩溃。",
            "模型与 replay buffer 不进入 Git。",
        ],
        "ready": [
            "配置读取、校验、哈希、manifest 和 registry。",
            "dry-run、train-only、evaluate-only、report-only、plot-only。",
            "CSV/Excel、PNG/SVG、Markdown/PPTX 输出。",
        ],
        "smoke_test": manifest.get("details", {}).get(
            "smoke_test", {"status": "not_designated", "checks": ["This run is not the 021_00 smoke configuration."]}
        ),
        "commands": (
            "python -m benchmark.benchmark_runner "
            "--config configs/experiments/021_00_framework_smoke.yaml "
            f"--{mode}" if mode != "full" else
            "python -m benchmark.benchmark_runner --config <experiment.yaml>"
        ),
        "next_steps": [
            "先完成并复核 021_00 smoke、resume、reuse 和 invalid-config tests。",
            "随后只在用户确认后 dry-run 021_01–021_08。",
        ],
    }


def _build_reports(
    spec: ExperimentSpec,
    registry: ExperimentRegistry,
    *,
    mode: str,
    train_result: dict[str, Any] | None,
) -> dict[str, Any]:
    # Reporting dependencies (openpyxl/python-pptx/matplotlib) are optional in
    # the DSSAT container and are therefore imported only in report modes.
    from .plot_builder import build_plots
    from .ppt_builder import build_pptx_report
    from .report_builder import build_markdown_report
    from .summary_builder import build_summary_outputs

    run_dir = registry.run_dir(spec.experiment_id)
    season, daily = _load_evaluations(run_dir)
    manifest = registry.load(spec.experiment_id)
    summaries = build_summary_outputs(
        season,
        daily,
        run_dir / "summaries",
        config=spec.config,
        manifest=manifest,
    )
    plots = build_plots(
        summaries["tables"]["season_summary"],
        summaries["tables"]["daily_trajectory"],
        run_dir / "figures",
        action_df=summaries["tables"].get("action_summary"),
        config=spec.config,
        manifest=manifest,
    )
    context = _report_context(manifest, mode, train_result)
    markdown = build_markdown_report(
        run_dir / "reports",
        config=spec.config,
        manifest=manifest,
        tables=summaries["tables"],
        figures=plots["artifacts"],
        context={**context, "output_paths": summaries["paths"]},
    )
    pptx = build_pptx_report(
        run_dir / "reports",
        config=spec.config,
        manifest=manifest,
        tables=summaries["tables"],
        figures=plots["artifacts"],
        context=context,
    )
    return {
        "summary_paths": {key: str(value) for key, value in summaries["paths"].items()},
        "figure_paths": plots["artifacts"],
        "plot_failures": plots["failures"],
        "markdown": str(markdown),
        "pptx": str(pptx),
    }


def _dry_run_plan(
    specs: list[ExperimentSpec], registry: ExperimentRegistry, mode: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = registry.manifest_path(spec.experiment_id)
        existing = None
        if path.exists():
            existing = registry.load(spec.experiment_id)
        reusable = bool(
            existing
            and existing.get("config_hash") == spec.config_hash
            and existing.get("status") in {"completed", "reused"}
        )
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "station": spec.station_code,
                "year": spec.year,
                "seed": spec.seed,
                "config_hash": spec.config_hash[:16],
                "action": "reuse" if reusable else (mode if mode != "full" else "train/evaluate/report"),
                "output_dir": str(registry.run_dir(spec.experiment_id)),
            }
        )
    return {
        "experiment_count": len(rows),
        "train_count": sum(row["action"] != "reuse" and "train" in row["action"] for row in rows),
        "reuse_count": sum(row["action"] == "reuse" for row in rows),
        "cases": rows,
    }


def run(args: argparse.Namespace) -> int:
    """Execute the requested benchmark mode and return a process exit code."""

    root = _project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path, root)
    config["_project_root"] = str(root)
    if args.resume:
        config["experiment"]["resume"] = True
    if args.force:
        config["experiment"]["force"] = True
    errors = validate_config(config, root)
    if errors:
        raise ConfigValidationError(errors)
    mode = _mode(args)
    specs = expand_experiments(
        config,
        root,
        sites=_parse_csv_text(args.site),
        years=_parse_csv_int(args.year),
        seeds=_parse_csv_int(args.seed),
    )
    if not specs:
        raise RuntimeError("No experiment remains after --site/--year/--seed filters")
    output_root = Path(config["experiment"].get("output_root", "benchmark_results"))
    if not output_root.is_absolute():
        output_root = root / output_root
    registry = ExperimentRegistry(output_root)

    dry_run = bool(args.dry_run or config["experiment"].get("dry_run", False))
    plan = _dry_run_plan(specs, registry, mode)
    if dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, ensure_ascii=False))
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.force:
        forced: list[ExperimentSpec] = []
        for spec in specs:
            new_id = f"{spec.experiment_id}__force_{timestamp}"
            new_config = json.loads(json.dumps(spec.config))
            new_config["experiment"]["experiment_id"] = new_id
            forced.append(replace(spec, experiment_id=new_id, config=new_config))
        specs = forced

    result_registry = ResultRegistry(output_root / "result_registry.csv")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        manifest = registry.register(spec)
        run_dir = registry.run_dir(spec.experiment_id)
        _ensure_layout(run_dir)
        _save_config(spec, run_dir, config_path)
        reuse_enabled = bool(spec.config["experiment"].get("reuse_existing_results", True))
        requested_artifact = {
            "full": run_dir / "reports" / "benchmark_report.pptx",
            "train-only": run_dir / "evaluations" / "season_summary.csv",
            "evaluate-only": run_dir / "evaluations" / "season_summary.csv",
            "report-only": run_dir / "reports" / "benchmark_report.pptx",
            "plot-only": run_dir / "figures" / "baseline_comparison.png",
        }[mode]
        if (
            manifest.get("config_hash") == spec.config_hash
            and manifest.get("status") in {"completed", "reused"}
            and reuse_enabled
            and requested_artifact.exists()
        ):
            completion_details: dict[str, Any] = {}
            if train_result is not None:
                completion_details["training"] = train_result
            if report_result is not None:
                completion_details["reporting"] = report_result
            registry.update_status(
                spec.experiment_id,
                ManifestStatus.REUSED,
                message="Exact framework manifest/config_hash reused; no training started",
                allow_restart=True,
            )
            results.append({"experiment_id": spec.experiment_id, "status": "reused"})
            continue

        registry.update_status(
            spec.experiment_id,
            ManifestStatus.RUNNING,
            message=f"mode={mode}",
            allow_restart=True,
            details={"run_command": " ".join(sys.argv)},
        )
        train_result: dict[str, Any] | None = None
        report_result: dict[str, Any] | None = None
        try:
            if mode in {"full", "train-only"}:
                train_result = train_case(
                    spec.config,
                    run_dir=run_dir,
                    experiment_id=spec.experiment_id,
                    config_hash=spec.config_hash,
                    year=spec.year,
                    seed=spec.seed,
                    resume=bool(args.resume or spec.config["experiment"].get("resume", False)),
                )
            elif mode == "evaluate-only":
                checkpoint, checkpoint_dir = latest_checkpoint(run_dir)
                if checkpoint_dir is None:
                    raise FileNotFoundError(f"No checkpoint available under {run_dir}")
                daily, season = evaluate_model_file(
                    spec.config,
                    run_dir=run_dir,
                    model_path=checkpoint_dir / "model.zip",
                    checkpoint=checkpoint,
                    experiment_id=spec.experiment_id,
                    config_hash=spec.config_hash,
                    year=spec.year,
                    seed=spec.seed,
                )
                daily.to_csv(run_dir / "evaluations" / "daily_trajectory.csv", index=False, encoding="utf-8-sig")
                season.to_csv(run_dir / "evaluations" / "season_summary.csv", index=False, encoding="utf-8-sig")

            if mode in {"full", "report-only", "plot-only"}:
                if mode == "plot-only":
                    from .plot_builder import build_plots

                    season, daily = _load_evaluations(run_dir)
                    report_result = build_plots(season, daily, run_dir / "figures", config=spec.config)
                else:
                    report_result = _build_reports(spec, registry, mode=mode, train_result=train_result)

            completion_details: dict[str, Any] = {}
            if train_result is not None:
                completion_details["training"] = train_result
            if report_result is not None:
                completion_details["reporting"] = report_result
            registry.update_status(
                spec.experiment_id,
                ManifestStatus.COMPLETED,
                message="requested mode completed",
                details=json.loads(json.dumps(completion_details, default=str)),
            )
            season_path = run_dir / "evaluations" / "season_summary.csv"
            daily_path = run_dir / "evaluations" / "daily_trajectory.csv"
            model_path = train_result.get("model_path", "") if train_result else ""
            if not model_path:
                model_path = (
                    registry.load(spec.experiment_id)
                    .get("details", {})
                    .get("training", {})
                    .get("model_path", "")
                )
            existing_rows = result_registry.find(
                config_hash=spec.config_hash,
                station_code=spec.station_code,
                year=spec.year,
                seed=spec.seed,
                scenario="dqn",
            )
            if not model_path and existing_rows:
                model_path = existing_rows[0].get("model_path", "")
            result_registry.upsert(
                {
                    "experiment_id": spec.experiment_id,
                    "config_hash": spec.config_hash,
                    "source_experiment": spec.experiment_id.split("__", 1)[0],
                    "station_code": spec.station_code,
                    "year": spec.year,
                    "seed": spec.seed,
                    "scenario": "dqn",
                    "model_path": _registry_path(model_path, root) if model_path else "",
                    "summary_path": _registry_path(season_path, root) if season_path.exists() else "",
                    "trajectory_path": _registry_path(daily_path, root) if daily_path.exists() else "",
                    "figure_path": _registry_path(run_dir / "figures", root),
                    "status": "completed",
                    "can_reuse": True,
                    "reuse_scope": "evaluation/report/partial_resume",
                    "exact_resume": False,
                    "reuse_reason": "Exact framework config_hash; checkpoint resume restarts DSSAT at episode boundary",
                    "notes": mode,
                }
            )
            results.append({"experiment_id": spec.experiment_id, "status": "completed"})
        except SimulatedInterruption as exc:
            registry.update_status(
                spec.experiment_id,
                ManifestStatus.PARTIAL,
                message=str(exc),
                details={"smoke_test": {"status": "partial", "checks": [str(exc)]}},
            )
            results.append({"experiment_id": spec.experiment_id, "status": "partial", "reason": str(exc)})
        except Exception as exc:
            traceback_path = run_dir / "logs" / "runner_failure_traceback.txt"
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            registry.update_status(
                spec.experiment_id,
                ManifestStatus.FAILED,
                message=str(exc),
                details={"error_type": type(exc).__name__, "traceback_path": str(traceback_path)},
            )
            failures.append(
                {"experiment_id": spec.experiment_id, "error_type": type(exc).__name__, "error": str(exc)}
            )
            LOGGER.exception("Experiment failed: %s", spec.experiment_id)

    print(json.dumps({"mode": mode, "results": results, "failures": failures}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configurable DQN benchmark runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--site", help="comma-separated site filter")
    parser.add_argument("--year", help="comma-separated year filter")
    parser.add_argument("--seed", help="comma-separated seed filter")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        code = run(build_parser().parse_args())
    except ConfigValidationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()

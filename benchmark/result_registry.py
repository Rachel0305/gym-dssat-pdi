"""Atomic CSV result registry and conservative legacy-result discovery."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config_loader import canonical_station_code

LOGGER = logging.getLogger(__name__)

REGISTRY_COLUMNS = [
    "experiment_id",
    "config_hash",
    "source_experiment",
    "station_code",
    "year",
    "seed",
    "scenario",
    "model_path",
    "summary_path",
    "trajectory_path",
    "figure_path",
    "status",
    "can_reuse",
    "reuse_scope",
    "exact_resume",
    "reuse_reason",
    "notes",
]

_REGISTRY_LOCK = threading.RLock()


def _serialise(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _atomic_csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({str(key) for row in rows for key in row} - set(REGISTRY_COLUMNS))
    fieldnames = [*REGISTRY_COLUMNS, *extras]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _serialise(row.get(key)) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class ResultRegistry:
    """Small durable result index with atomic upserts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)

    def list(self) -> list[dict[str, str]]:
        """Return all rows as strings, matching the on-disk CSV exactly."""

        if not self.path.is_file():
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def find(self, **criteria: Any) -> list[dict[str, str]]:
        """Return rows whose named fields equal the supplied values."""

        expected = {key: _serialise(value) for key, value in criteria.items()}
        return [
            row
            for row in self.list()
            if all(row.get(key, "") == value for key, value in expected.items())
        ]

    def upsert(
        self,
        record: Mapping[str, Any],
        *,
        key_fields: Sequence[str] = (
            "config_hash",
            "station_code",
            "year",
            "seed",
            "scenario",
        ),
    ) -> dict[str, str]:
        """Insert or atomically replace a result row by its scientific key."""

        if not record:
            raise ValueError("Cannot upsert an empty result record")
        missing = [key for key in key_fields if key not in record]
        if missing:
            raise ValueError(f"Result upsert is missing key fields: {missing}")

        target_key = tuple(_serialise(record.get(key)) for key in key_fields)
        with _REGISTRY_LOCK:
            rows: list[dict[str, Any]] = self.list()
            replacement = {str(key): value for key, value in record.items()}
            found = False
            for index, row in enumerate(rows):
                row_key = tuple(row.get(key, "") for key in key_fields)
                if row_key == target_key:
                    rows[index] = {**row, **replacement}
                    found = True
                    break
            if not found:
                rows.append(replacement)
            _atomic_csv_write(self.path, rows)

        matches = self.find(**{key: record[key] for key in key_fields})
        return matches[0]

    def write_rows(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Atomically replace the registry with supplied rows."""

        rows = [dict(record) for record in records]
        with _REGISTRY_LOCK:
            _atomic_csv_write(self.path, rows)


def _first_existing(root: Path, candidates: Iterable[Path | str]) -> Path | None:
    for candidate in candidates:
        path = candidate if isinstance(candidate, Path) else root / candidate
        if path.exists():
            return path.resolve()
    return None


def _relative_or_absolute(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as exc:
        LOGGER.warning("Could not index legacy CSV %s: %s", path, exc)
        return []


def _value(row: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _infer_seed(row: Mapping[str, str], source_path: Path) -> str:
    direct = _value(row, "seed", "dqn_seed", "random_seed")
    if direct:
        return direct
    match = re.search(r"seed[_-]?(\d+)", source_path.as_posix(), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _infer_station(row: Mapping[str, str], default: str) -> str:
    raw = _value(row, "station_code", "site", "station") or default
    try:
        return canonical_station_code(raw)
    except ValueError:
        return default


def _normalise_scenario(row: Mapping[str, str]) -> str:
    raw = _value(row, "scenario", "scenario_family", "scenario_label", "label")
    if not raw or raw.strip().lower() in {"nan", "null", "null zero"}:
        return "null"
    lowered = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "recorded": "recorded_farmer",
        "recorded_farmer_practice": "recorded_farmer",
        "official_extension_expert": "extension_expert",
    }
    return aliases.get(lowered, lowered)


def _known_artifacts(
    definition: Mapping[str, Any], row: Mapping[str, str], project_root: Path, scenario: str
) -> tuple[Path | None, Path | None, str]:
    """Resolve only model/trajectory mappings that are explicit in project records."""

    source = str(definition["source_experiment"])
    station = _infer_station(row, str(definition.get("station", "")))
    seed_raw = _infer_seed(row, Path(str(definition["root"])))
    seed = int(float(seed_raw)) if seed_raw else None
    if source.startswith("020_11"):
        daily = Path(definition["root"]) / "020_11_hla_five_scenario_daily.csv"
        model = None
        model_roots = {
            0: project_root / "DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed0_020_08/nstep5_seed0_50000steps",
            1: project_root / "DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed1_020_06/nstep5_seed1_50000steps",
            2: project_root / "DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed2_020_07/nstep5_seed2_50000steps",
        }
        selected_steps = {0: 30000, 1: 10000, 2: 20000}
        if scenario.startswith("nstep_seed") and seed in model_roots:
            candidate = model_roots[seed] / "models" / f"nstep5_checkpoint_{selected_steps[seed]}.zip"
            model = candidate if candidate.is_file() else None
        return model, daily if daily.is_file() else None, "trajectory requires year/scenario/seed filtering"

    if source.startswith("020_13") and scenario == "frozen_nstep5_dqn" and seed is not None:
        checkpoint_raw = _value(row, "checkpoint_step", "checkpoint")
        checkpoint = int(float(checkpoint_raw)) if checkpoint_raw else None
        source_file_raw = _value(row, "source_file")
        source_file = project_root / source_file_raw if source_file_raw else None
        run_dir = source_file.parent if source_file and source_file.is_file() else None
        if run_dir and checkpoint is not None:
            model_candidate = run_dir / "checkpoints" / f"checkpoint_{checkpoint}" / "model.zip"
            if station == "YC" and seed == 0:
                corrected = run_dir / "reaudit_operation_dap_020_12" / f"checkpoint_{checkpoint}" / "eval_daily.csv"
            else:
                corrected = run_dir / "selected_checkpoint_daily.csv"
            return (
                model_candidate if model_candidate.is_file() else None,
                corrected if corrected.is_file() else None,
                "selected model and corrected selected trajectory",
            )
    return None, None, "legacy mapping is report/diagnostic only"


def _legacy_source_definitions(project_root: Path) -> list[dict[str, Any]]:
    validation = project_root / "DSSAT_auto_validation"
    return [
        {
            "source_experiment": "020_11_HLA_five_scenario_nstep",
            "station": "HLA",
            "root": validation / "HLA_2004" / "hla_five_scenario_nstep_020_11",
            "summaries": ["020_11_hla_five_scenario_summary.csv"],
        },
        {
            "source_experiment": "020_13_YC_FQ_frozen_nstep",
            "station": "",
            "root": validation / "frozen_nstep_cross_site_020_12",
            "summaries": ["020_13_yc_fq_seed0_seed1_summary.csv"],
        },
        {
            "source_experiment": "017_10_LC_input_rescue",
            "station": "LC",
            "year": "2008",
            "root": validation / "lc_pdi_initialization_rescue_017_10",
            "summaries": ["017_10_lc2008_soilfix_baseline_summary.csv"],
        },
        {
            "source_experiment": "017_12_LC_DQN_smoke",
            "station": "LC",
            "year": "2010",
            "root": validation / "lc2010_baseline_relative_dqn_smoke_017_12",
            "summary_glob": "seed*_5000steps/checkpoint_summary.csv",
        },
        {
            "source_experiment": "017_08_SY_local_DQN_transfer",
            "station": "SY",
            "root": validation / "sy_local_dqn_train_cross_year_transfer_017_08",
            "summaries": [
                "017_08_sy_combined_summary.csv",
                "017_08_sy_dqn_checkpoint_summary.csv",
                "017_08_sy_dqn_transfer_summary.csv",
            ],
        },
        {
            "source_experiment": "017_09_SY2014_resource_space",
            "station": "SY",
            "year": "2014",
            "root": validation / "sy2014_dqn_resource_space_017_09",
            "summaries": ["017_09_sy2014_four_scenario_summary.csv"],
        },
    ]


def _discover_models(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        return []
    models = list(source_root.glob("**/*.zip"))
    return sorted(models, key=lambda item: (item.stat().st_mtime_ns, item.as_posix()))


def discover_legacy_results(project_root: Path | str):
    """Return a conservative pandas DataFrame index of key legacy evidence.

    The discovery covers HLA 020_11, YC/FQ 020_13, LC 017_10/017_12,
    SY 017_08/017_09, and the 020_14 five-site status note.  Legacy artifacts
    lack a framework ``config_hash`` and usually lack replay buffers, so this
    function *never* marks them as exact training resumes.  They are reusable
    only for diagnostic/report generation until input equivalence is proven.
    """

    import pandas as pd

    root = Path(project_root).expanduser().resolve()
    discovered: list[dict[str, Any]] = []
    for definition in _legacy_source_definitions(root):
        source_root: Path = definition["root"]
        summary_paths = [source_root / item for item in definition.get("summaries", [])]
        if definition.get("summary_glob") and source_root.is_dir():
            summary_paths.extend(sorted(source_root.glob(definition["summary_glob"])))
        summary_paths = [path.resolve() for path in summary_paths if path.is_file()]
        models = _discover_models(source_root)
        latest_model = models[-1] if models else None

        if not summary_paths:
            discovered.append(
                {
                    "source_experiment": definition["source_experiment"],
                    "station_code": definition.get("station", ""),
                    "year": definition.get("year", ""),
                    "seed": "",
                    "scenario": "",
                    "model_path": _relative_or_absolute(latest_model, root),
                    "summary_path": "",
                    "trajectory_path": "",
                    "figure_path": "",
                    "status": "partial",
                    "can_reuse": False,
                    "reuse_scope": "diagnostic/report",
                    "exact_resume": False,
                    "reuse_reason": "Expected legacy summary was not found",
                    "notes": _relative_or_absolute(source_root, root),
                }
            )
            continue

        for summary_path in summary_paths:
            rows = _read_csv_rows(summary_path)
            if not rows:
                rows = [{}]
            figures_dir = source_root / "figures"
            first_figure = _first_existing(
                root,
                sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else [],
            )
            for row in rows:
                trajectory_raw = _value(
                    row, "source_file", "trajectory_path", "daily_path", "daily_file"
                )
                trajectory = None
                if trajectory_raw:
                    trajectory = Path(trajectory_raw)
                    if not trajectory.is_absolute():
                        trajectory = root / trajectory
                    if not trajectory.exists():
                        trajectory = None
                scenario = _normalise_scenario(row)
                known_model, known_trajectory, mapping_note = _known_artifacts(
                    definition, row, root, scenario
                )
                model = known_model
                if known_trajectory is not None:
                    trajectory = known_trajectory
                discovered.append(
                    {
                        "source_experiment": definition["source_experiment"],
                        "station_code": _infer_station(row, definition.get("station", "")),
                        "year": _value(row, "year") or definition.get("year", ""),
                        "seed": _infer_seed(row, summary_path),
                        "scenario": scenario,
                        "model_path": _relative_or_absolute(model, root),
                        "summary_path": _relative_or_absolute(summary_path, root),
                        "trajectory_path": _relative_or_absolute(trajectory, root),
                        "figure_path": _relative_or_absolute(first_figure, root),
                        "status": "completed",
                        "can_reuse": True,
                        "reuse_scope": "diagnostic/report",
                        "exact_resume": False,
                        "reuse_reason": (
                            "Legacy artifact is usable for diagnostics/reporting; exact input, "
                            "config_hash, and replay-buffer equivalence are not proven"
                        ),
                        "notes": mapping_note + "; do not use as an exact checkpoint resume",
                    }
                )

    status_note = root / "docs" / "2026-07-11_020_14_five_site_current_status_and_priority.md"
    discovered.append(
        {
            "source_experiment": "020_14_five_site_status",
            "station_code": "MULTI",
            "year": "",
            "seed": "",
            "scenario": "evidence_status",
            "model_path": "",
            "summary_path": _relative_or_absolute(status_note if status_note.is_file() else None, root),
            "trajectory_path": "",
            "figure_path": "",
            "status": "completed" if status_note.is_file() else "partial",
            "can_reuse": status_note.is_file(),
            "reuse_scope": "diagnostic/report",
            "exact_resume": False,
            "reuse_reason": "Narrative evidence index only; not a training artifact",
            "notes": "Five-site status snapshot",
        }
    )
    return pd.DataFrame(discovered, columns=REGISTRY_COLUMNS[2:])

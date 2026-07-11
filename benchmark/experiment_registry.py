"""Experiment expansion and durable run-manifest state management."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config_loader import canonical_station_code, deep_merge
from .provenance import collect_provenance, compute_config_hash

LOGGER = logging.getLogger(__name__)


class ManifestStatus(str, Enum):
    """Allowed persistent experiment states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REUSED = "reused"
    PARTIAL = "partial"


RECOVERABLE_STATUSES = {
    ManifestStatus.PENDING.value,
    ManifestStatus.FAILED.value,
    ManifestStatus.PARTIAL.value,
}
TERMINAL_STATUSES = {
    ManifestStatus.COMPLETED.value,
    ManifestStatus.REUSED.value,
}


@dataclass
class ExperimentSpec:
    """A single station-year-seed benchmark work item."""

    experiment_id: str
    config_hash: str
    station_code: str
    year: int
    seed: int
    config: dict[str, Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable summary suitable for manifests/dry-runs."""

        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "station_code": self.station_code,
            "year": self.year,
            "seed": self.seed,
            "config": copy.deepcopy(self.config),
        }


def _normalise_sites(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(config.get("sites"), list):
        sites: list[dict[str, Any]] = []
        for item in config["sites"]:
            if isinstance(item, str):
                sites.append({"station_code": canonical_station_code(item)})
            elif isinstance(item, Mapping):
                sites.append(copy.deepcopy(dict(item)))
            else:
                raise ValueError("sites entries must be station aliases or mappings")
        return sites
    site = config.get("site")
    if not isinstance(site, Mapping):
        raise ValueError("Configuration requires a site mapping or sites list")
    return [copy.deepcopy(dict(site))]


def _normalise_years(site: Mapping[str, Any]) -> list[int]:
    years = site.get("eval_years") or site.get("train_years")
    if not isinstance(years, list) or not years:
        raise ValueError("Each site requires non-empty eval_years or train_years")
    if any(not isinstance(year, int) or isinstance(year, bool) for year in years):
        raise ValueError(f"Invalid year list: {years!r}")
    return list(dict.fromkeys(years))


def _normalise_seeds(config: Mapping[str, Any]) -> list[int]:
    evaluation = config.get("evaluation", {})
    algorithm = config.get("algorithm", {})
    seeds: Any = evaluation.get("seeds") if isinstance(evaluation, Mapping) else None
    if seeds is None and isinstance(algorithm, Mapping):
        seeds = algorithm.get("seeds")
    if seeds is None and isinstance(algorithm, Mapping):
        seeds = [algorithm.get("seed", 0)]
    if not isinstance(seeds, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ):
        raise ValueError(f"Seeds must be an integer list, received {seeds!r}")
    return list(dict.fromkeys(seeds))


def expand_experiments(
    config: Mapping[str, Any],
    project_root: Path | str | None = None,
    *,
    sites: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[ExperimentSpec]:
    """Expand one merged config into station-year-seed work items.

    Optional filters correspond to CLI ``--site``, ``--year`` and ``--seed``.
    Each expanded config retains the original training-year list and narrows
    ``eval_years`` to the active year.
    """

    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping")
    base_id = str(experiment.get("experiment_id") or "benchmark")
    site_filter = {canonical_station_code(item) for item in sites} if sites else None
    year_filter = set(years) if years else None
    seed_values = list(seeds) if seeds is not None else _normalise_seeds(config)
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_values):
        raise ValueError("Seed filters must contain integers")

    specs: list[ExperimentSpec] = []
    for site_data in _normalise_sites(config):
        station_raw = site_data.get("station_code") or site_data.get("station_name")
        if not isinstance(station_raw, str):
            raise ValueError(f"Site is missing station_code: {site_data!r}")
        station_code = canonical_station_code(station_raw)
        if site_filter is not None and station_code not in site_filter:
            continue
        site_data["station_code"] = station_code

        for year in _normalise_years(site_data):
            if year_filter is not None and year not in year_filter:
                continue
            for seed in seed_values:
                expanded = copy.deepcopy(dict(config))
                expanded.pop("sites", None)
                # Materialize exact year-specific critical files so the hash
                # contains file digests rather than only a directory name.
                inputs = site_data.get("inputs", {})
                by_year = inputs.get("by_year", {}) if isinstance(inputs, Mapping) else {}
                year_entry = by_year.get(str(year), by_year.get(year, {})) if isinstance(by_year, Mapping) else {}
                if isinstance(year_entry, Mapping):
                    raw_dir = year_entry.get("prepared_input_dir") or inputs.get("source_dir")
                    if raw_dir:
                        filex_name = year_entry.get("fileX_name")
                        weather_name = year_entry.get("weather_name")
                        critical = []
                        if filex_name:
                            critical.append((Path(raw_dir) / str(filex_name)).as_posix())
                        if weather_name:
                            critical.append((Path(raw_dir) / str(weather_name)).as_posix())
                        critical.extend(
                            [
                                (Path(raw_dir) / "MZCER048.CUL").as_posix(),
                                (Path(raw_dir) / "SOIL.SOL").as_posix(),
                            ]
                        )
                        site_data["critical_input_paths"] = critical
                expanded["site"] = deep_merge(site_data, {"eval_years": [year], "active_year": year})
                algorithm = expanded.get("algorithm", {})
                expanded["algorithm"] = deep_merge(
                    algorithm if isinstance(algorithm, Mapping) else {}, {"seed": seed}
                )
                work_id = f"{base_id}__{station_code.lower()}_{year}_seed{seed}"
                expanded["experiment"] = deep_merge(experiment, {"experiment_id": work_id})
                config_hash = compute_config_hash(expanded, project_root)
                specs.append(
                    ExperimentSpec(
                        experiment_id=work_id,
                        config_hash=config_hash,
                        station_code=station_code,
                        year=year,
                        seed=seed,
                        config=expanded,
                    )
                )
    return specs


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class ExperimentRegistry:
    """Store one atomic JSON manifest per expanded experiment."""

    def __init__(self, output_root: Path | str) -> None:
        self.output_root = Path(output_root).expanduser().resolve(strict=False)

    def run_dir(self, experiment_id: str) -> Path:
        """Return the unified result directory for *experiment_id*."""

        return self.output_root / experiment_id

    def manifest_path(self, experiment_id: str) -> Path:
        """Return the canonical manifest path for *experiment_id*."""

        return self.run_dir(experiment_id) / "manifests" / "manifest.json"

    def register(self, spec: ExperimentSpec, *, overwrite: bool = False) -> dict[str, Any]:
        """Create a pending manifest, preserving an existing one by default."""

        path = self.manifest_path(spec.experiment_id)
        if path.exists() and not overwrite:
            existing = self.load(spec.experiment_id)
            if existing.get("config_hash") != spec.config_hash:
                raise ValueError(
                    f"experiment_id collision for {spec.experiment_id}: existing config_hash "
                    f"{existing.get('config_hash')} != {spec.config_hash}"
                )
            return existing

        now = datetime.now(timezone.utc).isoformat()
        manifest: dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "config_hash": spec.config_hash,
            "station_code": spec.station_code,
            "year": spec.year,
            "seed": spec.seed,
            "status": ManifestStatus.PENDING.value,
            "created_at_utc": now,
            "updated_at_utc": now,
            "config": spec.config,
            "provenance": collect_provenance(spec.config),
            "events": [
                {"at_utc": now, "status": ManifestStatus.PENDING.value, "message": "registered"}
            ],
        }
        _atomic_json_write(path, manifest)
        return manifest

    def load(self, experiment_id: str) -> dict[str, Any]:
        """Load a manifest and fail clearly if it is absent or malformed."""

        path = self.manifest_path(experiment_id)
        if not path.is_file():
            raise FileNotFoundError(f"Experiment manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Experiment manifest is not a JSON object: {path}")
        return payload

    def update_status(
        self,
        experiment_id: str,
        status: ManifestStatus | str,
        *,
        message: str = "",
        details: Mapping[str, Any] | None = None,
        allow_restart: bool = False,
    ) -> dict[str, Any]:
        """Atomically transition a manifest to one of the six allowed states."""

        requested = ManifestStatus(status).value
        manifest = self.load(experiment_id)
        current = ManifestStatus(manifest["status"]).value
        if current in TERMINAL_STATUSES and requested != current and not allow_restart:
            raise ValueError(
                f"Refusing to transition terminal experiment {experiment_id} from "
                f"{current} to {requested} without allow_restart=True"
            )

        now = datetime.now(timezone.utc).isoformat()
        manifest["status"] = requested
        manifest["updated_at_utc"] = now
        if details:
            manifest.setdefault("details", {}).update(dict(details))
        manifest.setdefault("events", []).append(
            {"at_utc": now, "status": requested, "message": message}
        )
        _atomic_json_write(self.manifest_path(experiment_id), manifest)
        LOGGER.info("Experiment %s status: %s -> %s", experiment_id, current, requested)
        return manifest

    def list(self, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Return manifests, optionally filtered by status."""

        accepted = {ManifestStatus(item).value for item in statuses} if statuses else None
        manifests: list[dict[str, Any]] = []
        if not self.output_root.exists():
            return manifests
        for path in sorted(self.output_root.glob("*/manifests/manifest.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Skipping unreadable manifest %s: %s", path, exc)
                continue
            if accepted is None or manifest.get("status") in accepted:
                manifests.append(manifest)
        return manifests

    def find_by_hash(self, config_hash: str) -> list[dict[str, Any]]:
        """Return all manifests with an exact configuration hash."""

        return [item for item in self.list() if item.get("config_hash") == config_hash]

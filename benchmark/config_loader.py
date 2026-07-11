"""YAML configuration loading with deterministic deep inheritance.

Merge order is ``benchmark defaults -> explicit bases -> site defaults ->
experiment file``.  Mappings are merged recursively while lists are replaced,
which keeps experiment grids intentional and predictable.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

LOGGER = logging.getLogger(__name__)

SITE_ALIASES: dict[str, str] = {
    "FENGQIU": "FQ",
    "FQ": "FQ",
    "FQA": "FQ",
    "HAILUN": "HLA",
    "HL": "HLA",
    "HLA": "HLA",
    "LC": "LC",
    "LCA": "LC",
    "LUANCHENG": "LC",
    "SHENYANG": "SY",
    "SY": "SY",
    "SYA": "SY",
    "YC": "YC",
    "YCA": "YC",
    "YUCHENG": "YC",
}

_SITE_CONFIG_FILENAMES = {
    "HLA": "hla.yaml",
    "YC": "yca.yaml",
    "FQ": "fqa.yaml",
    "LC": "lca.yaml",
    "SY": "sya.yaml",
}


def canonical_station_code(value: str) -> str:
    """Return a canonical five-site station code.

    Accepted aliases include project shorthand (``YC``, ``FQ``, ``LC``,
    ``SY``), canonical DSSAT benchmark codes, and English station names.
    """

    normalised = value.strip().upper().replace("-", "").replace("_", "")
    try:
        return SITE_ALIASES[normalised]
    except KeyError as exc:
        accepted = ", ".join(sorted(SITE_ALIASES))
        raise ValueError(
            f"Unknown station code/name {value!r}; accepted aliases: {accepted}"
        ) from exc


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _infer_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / ".git").exists() or (candidate / "AGENTS.md").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return dict(payload)


def _normalise_reference_list(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, (str, Path)):
        return [str(raw)]
    if isinstance(raw, list) and all(isinstance(item, (str, Path)) for item in raw):
        return [str(item) for item in raw]
    raise ValueError("Configuration inheritance references must be a path or path list")


def _resolve_reference(reference: str, source: Path, project_root: Path) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    local = (source.parent / candidate).resolve(strict=False)
    if local.exists():
        return local
    return (project_root / candidate).resolve(strict=False)


def _load_explicit_bases(
    path: Path, project_root: Path, stack: tuple[Path, ...]
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    resolved = path.resolve(strict=False)
    if resolved in stack:
        chain = " -> ".join(item.as_posix() for item in (*stack, resolved))
        raise ValueError(f"Circular benchmark configuration inheritance: {chain}")

    raw = _read_yaml(resolved)
    references: list[str] = []
    for key in ("extends", "inherits", "inherit", "defaults_file"):
        references.extend(_normalise_reference_list(raw.pop(key, None)))
    # ``defaults`` is treated as a reference only when it is a path/list.  A
    # mapping named defaults remains ordinary experiment data.
    if isinstance(raw.get("defaults"), (str, Path, list)):
        references.extend(_normalise_reference_list(raw.pop("defaults")))

    merged: dict[str, Any] = {}
    sources: list[Path] = []
    for reference in references:
        base_path = _resolve_reference(reference, resolved, project_root)
        base_merged, _, base_sources = _load_explicit_bases(
            base_path, project_root, (*stack, resolved)
        )
        merged = deep_merge(merged, base_merged)
        sources.extend(base_sources)
    merged = deep_merge(merged, raw)
    sources.append(resolved)
    return merged, raw, sources


def _station_code_from_config(config: Mapping[str, Any]) -> str | None:
    site = config.get("site")
    if isinstance(site, Mapping):
        raw = site.get("station_code") or site.get("code") or site.get("station_name")
        if isinstance(raw, str) and raw.strip():
            return canonical_station_code(raw)
    return None


def load_config(
    path: Path | str, project_root: Path | str | None = None
) -> dict[str, Any]:
    """Load a benchmark experiment YAML and merge defaults/site overrides.

    Automatic sources, when present, are:

    1. ``configs/benchmark_defaults.yaml``;
    2. any explicit ``extends``/``defaults_file`` references;
    3. ``configs/sites/<canonical-site>.yaml``;
    4. the requested experiment file.

    Missing automatic files are harmless.  Explicitly referenced files must
    exist and raise a clear exception otherwise.
    """

    config_path = Path(path).expanduser().resolve(strict=False)
    root = Path(project_root).expanduser().resolve() if project_root else _infer_project_root(config_path)

    experiment_with_bases, experiment_raw, explicit_sources = _load_explicit_bases(
        config_path, root, ()
    )
    merged: dict[str, Any] = {}
    sources: list[Path] = []

    default_path = (root / "configs" / "benchmark_defaults.yaml").resolve(strict=False)
    if default_path.is_file() and default_path != config_path and default_path not in explicit_sources:
        default_config, _, default_sources = _load_explicit_bases(default_path, root, ())
        merged = deep_merge(merged, default_config)
        sources.extend(default_sources)

    merged = deep_merge(merged, experiment_with_bases)
    sources.extend(explicit_sources)

    explicit_site_reference = experiment_raw.pop("site_config", None)
    station_code = _station_code_from_config(merged)
    site_path: Path | None = None
    if explicit_site_reference:
        if not isinstance(explicit_site_reference, (str, Path)):
            raise ValueError("site_config must be a YAML path")
        site_path = _resolve_reference(str(explicit_site_reference), config_path, root)
    elif station_code:
        candidate = root / "configs" / "sites" / _SITE_CONFIG_FILENAMES[station_code]
        if candidate.is_file():
            site_path = candidate.resolve()

    if site_path and site_path != config_path and site_path not in explicit_sources:
        site_config, _, site_sources = _load_explicit_bases(site_path, root, ())
        # Site defaults belong below experiment-level overrides.
        without_experiment = deep_merge(
            deep_merge({}, merged), {}
        )
        # Rebuild merge order so site defaults do not override the experiment.
        base_before_experiment: dict[str, Any] = {}
        if default_path.is_file() and default_path != config_path:
            base_before_experiment, _, _ = _load_explicit_bases(default_path, root, ())
        base_before_experiment = deep_merge(base_before_experiment, site_config)
        merged = deep_merge(base_before_experiment, experiment_with_bases)
        sources = [*site_sources, *sources]
        del without_experiment

    station_code = _station_code_from_config(merged)
    if station_code and isinstance(merged.get("site"), Mapping):
        merged["site"] = deep_merge(merged["site"], {"station_code": station_code})

    unique_sources = list(dict.fromkeys(source.resolve(strict=False) for source in sources))
    LOGGER.info(
        "Loaded benchmark configuration %s from %d source(s)",
        config_path,
        len(unique_sources),
    )
    return merged

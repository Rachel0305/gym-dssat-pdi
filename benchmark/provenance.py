"""Reproducibility and provenance helpers for benchmark experiments.

The module deliberately has no DSSAT or machine-learning dependency.  It can
therefore be used by dry-runs, audits, and unit tests before a container is
available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

LOGGER = logging.getLogger(__name__)

# Runtime-only values must not change experiment identity.  In particular,
# rerunning with ``--resume`` or ``--dry-run`` still describes the same science.
_VOLATILE_KEYS = {
    "_project_root",
    "_config_sources",
    "config_hash",
    "created_at",
    "dry_run",
    "force",
    "generated_at",
    "resume",
    "reuse_existing_results",
    "run_command",
    "started_at",
    "updated_at",
    "experiment_id",
}

_INPUT_PATH_KEYS = {
    "cultivar_path",
    "filex_path",
    "filex_template_path",
    "initial_conditions_path",
    "soil_path",
    "template_path",
    "weather_path",
    "prepared_input_dir",
    "source_dir",
    "critical_input_paths",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(project_root: Path | str | None = None) -> str:
    """Return the current Git commit or ``"unavailable"`` outside a repo."""

    root = Path(project_root or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("Git command unavailable for %s: %s", root, exc)
        # The DSSAT container may not install the git executable, but the
        # mounted workspace still exposes .git. Resolve HEAD directly so the
        # same configuration receives the same hash on Windows and Linux.
        git_entry = root / ".git"
        git_dir = git_entry
        if git_entry.is_file():
            pointer = git_entry.read_text(encoding="utf-8", errors="ignore").strip()
            if pointer.lower().startswith("gitdir:"):
                git_dir = (root / pointer.split(":", 1)[1].strip()).resolve()
        head = git_dir / "HEAD"
        if head.is_file():
            value = head.read_text(encoding="ascii", errors="ignore").strip()
            if value.startswith("ref:"):
                ref_name = value.split(":", 1)[1].strip()
                ref_file = git_dir / ref_name
                if ref_file.is_file():
                    return ref_file.read_text(encoding="ascii", errors="ignore").strip()
                packed = git_dir / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="ascii", errors="ignore").splitlines():
                        if line and not line.startswith(("#", "^")):
                            commit, name = line.split(" ", 1)
                            if name == ref_name:
                                return commit
            elif value:
                return value
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def get_git_dirty(project_root: Path | str | None = None) -> bool | None:
    """Return whether tracked files are dirty, or ``None`` if Git is absent."""

    root = Path(project_root or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("Git status unavailable for %s: %s", root, exc)
        return None
    return bool(completed.stdout.strip())


def _normalise_path(path_value: str | Path, project_root: Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve(strict=False)
    try:
        display_path = path.relative_to(project_root).as_posix()
    except ValueError:
        display_path = path.as_posix()

    descriptor: dict[str, Any] = {
        "path": display_path,
        "exists": path.exists(),
        "kind": "missing",
        "sha256": None,
    }
    if path.is_file():
        descriptor.update(
            kind="file",
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    elif path.is_dir():
        # Hashing an entire Weather/Soil directory would make a single run's
        # identity depend on unrelated station files.  Configurations should
        # therefore point critical keys at concrete files.  The directory path
        # and type are still recorded explicitly for diagnostics.
        descriptor["kind"] = "directory"
    return descriptor


def _iter_input_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            dotted = f"{prefix}.{key}" if prefix else key
            key_lower = key.lower()
            if key_lower in _INPUT_PATH_KEYS or key_lower.endswith("_input_path"):
                if child not in (None, ""):
                    yield dotted, child
            else:
                yield from _iter_input_paths(child, dotted)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_input_paths(child, f"{prefix}[{index}]")


def _canonicalise(value: Any) -> Any:
    """Convert common Python values to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalise(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalise(child) for child in value]
    if isinstance(value, set):
        return sorted((_canonicalise(child) for child in value), key=repr)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def build_hash_payload(
    config: Mapping[str, Any], project_root: Path | str | None = None
) -> dict[str, Any]:
    """Build the transparent payload from which ``config_hash`` is derived."""

    root = Path(project_root or Path.cwd()).resolve()
    input_files: dict[str, dict[str, Any]] = {}
    for dotted_key, path_value in _iter_input_paths(config):
        if isinstance(path_value, (str, Path)):
            input_files[dotted_key] = _normalise_path(path_value, root)
        elif isinstance(path_value, list):
            for index, item in enumerate(path_value):
                if isinstance(item, (str, Path)):
                    input_files[f"{dotted_key}[{index}]"] = _normalise_path(item, root)

    return {
        "config": _canonicalise(config),
        "critical_inputs": _canonicalise(input_files),
        "git_commit": get_git_commit(root),
    }


def compute_config_hash(
    config: Mapping[str, Any], project_root: Path | str | None = None
) -> str:
    """Return a stable SHA-256 experiment identity.

    The identity includes the scientific configuration, critical input file
    paths and SHA-256 digests, and the current Git commit.  Runtime controls
    such as ``dry_run`` and ``resume`` are intentionally excluded.
    """

    payload = build_hash_payload(config, project_root)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_provenance(
    config: Mapping[str, Any] | None = None,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Collect lightweight software provenance for a run manifest."""

    root = Path(project_root or Path.cwd()).resolve()
    provenance: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(root),
        "git_dirty_tracked": get_git_dirty(root),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": root.as_posix(),
    }
    if config is not None:
        provenance["config_hash"] = compute_config_hash(config, root)
        provenance["critical_inputs"] = build_hash_payload(config, root)[
            "critical_inputs"
        ]
    return provenance

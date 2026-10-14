"""Archive daily weather captured from a DSSAT runtime episode.

The helper never moves or edits the runtime source file. A raw WTH is copied
only when the runtime actually emitted one; runtime-state-only captures remain
explicitly marked as such.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_COLUMNS = ("DATE", "DAP", "SRAD", "TMAX", "TMIN", "RAIN")
LEGACY_HASH_COLUMNS = ("DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN")
MANIFEST_COLUMNS = (
    "archive_weather_id", "episode_id", "weather_seed", "rseed1", "weather_set",
    "crop_year", "season_start", "station", "raw_wth_path",
    "raw_wth_sha256", "raw_wth_status", "canonical_csv_path",
    "canonical_series_sha256", "legacy_runtime_hash_sha256", "cli_sha256",
    "generator_version", "runtime_version", "source_runtime_path",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _format(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return format(float(value), ".12g")
    return str(value)


def canonical_daily_csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: (str(row.get("DATE", "")), int(row.get("DAP") or 0)))
    lines = [",".join(ARCHIVE_COLUMNS)]
    for row in ordered:
        lines.append(",".join(_format(row.get(key)) for key in ARCHIVE_COLUMNS))
    return ("\n".join(lines) + "\n").encode("utf-8")


def legacy_runtime_hash_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: (str(row.get("DATE", "")), int(row.get("DOY") or 0)))
    lines = [",".join(LEGACY_HASH_COLUMNS)]
    for row in ordered:
        values = []
        for key in LEGACY_HASH_COLUMNS:
            value = row.get(key)
            values.append("" if value is None else _format(value))
        lines.append(",".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different archive content: {path}")


def archive_runtime_weather(
    archive_root: str | Path,
    *,
    weather_seed: int,
    rseed1: int | None = None,
    weather_set: str,
    crop_year: int | str,
    season_start: str,
    station: str,
    episode_id: str,
    rows: list[dict[str, Any]],
    raw_wth_path: str | Path | None = None,
    legacy_runtime_hash_sha256: str = "",
    cli_sha256: str = "",
    generator_version: str = "",
    runtime_version: str = "",
    source_runtime_path: str = "",
) -> dict[str, Any]:
    root = Path(archive_root).resolve()
    daily_bytes = canonical_daily_csv_bytes(rows)
    canonical_hash = sha256_bytes(daily_bytes)
    legacy_hash = sha256_bytes(legacy_runtime_hash_bytes(rows))
    context = str(crop_year).replace("/", "_")
    archive_id = f"seed_{int(weather_seed)}_ctx_{context}_{canonical_hash[:16]}"
    csv_rel = Path("canonical_weather") / weather_set / f"{archive_id}.csv"
    csv_path = root / csv_rel
    _write_new(csv_path, daily_bytes)

    copied_wth: Path | None = None
    raw_hash = ""
    raw_status = "NOT_EMITTED_BY_RUNTIME"
    if raw_wth_path is not None and Path(raw_wth_path).is_file():
        source = Path(raw_wth_path).resolve()
        raw_hash = sha256_file(source)
        copied_wth = root / "recovered_weather" / weather_set / f"{archive_id}.WTH"
        copied_wth.parent.mkdir(parents=True, exist_ok=True)
        if copied_wth.exists():
            if sha256_file(copied_wth) != raw_hash:
                raise FileExistsError(f"Refusing to overwrite different WTH archive: {copied_wth}")
        else:
            shutil.copy2(source, copied_wth)
        raw_status = "COPIED_RUNTIME_WTH"

    record = {
        "archive_weather_id": archive_id,
        "episode_id": episode_id,
        "weather_seed": int(weather_seed),
        "rseed1": int(rseed1) if rseed1 is not None else "",
        "weather_set": weather_set,
        "crop_year": crop_year,
        "season_start": season_start,
        "station": station,
        "raw_wth_path": copied_wth.relative_to(root).as_posix() if copied_wth else "",
        "raw_wth_sha256": raw_hash,
        "raw_wth_status": raw_status,
        "canonical_csv_path": csv_rel.as_posix(),
        "canonical_series_sha256": canonical_hash,
        "legacy_runtime_hash_sha256": legacy_hash,
        "cli_sha256": cli_sha256,
        "generator_version": generator_version,
        "runtime_version": runtime_version,
        "source_runtime_path": source_runtime_path,
    }
    manifest = root / "weather_archive_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        same_episode = [row for row in existing if row.get("episode_id") == episode_id]
        if same_episode:
            if same_episode[0].get("canonical_series_sha256") != canonical_hash:
                raise ValueError(f"Episode already maps to another weather series: {episode_id}")
            return record
    new_file = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)
    return record

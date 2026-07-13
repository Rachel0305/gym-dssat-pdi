"""Compatibility adapters between YAML cases and legacy gym-DSSAT inputs."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedCase:
    """A non-overwriting pair of DQN and local-null input packages."""

    station_code: str
    station_name: str
    year: int
    treatment: int
    dqn_filex: Path
    null_filex: Path
    dqn_env_args: dict[str, Any]
    null_env_args: dict[str, Any]


def _project_root(config: dict[str, Any]) -> Path:
    root = config.get("_project_root") or Path(__file__).resolve().parents[1]
    return Path(root).resolve()


def _legacy_module(config: dict[str, Any]):
    root = _project_root(config)
    source = root / "src"
    for value in (root, source):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy

    return legacy


def _resolve(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_input_directory(config: dict[str, Any], year: int) -> Path:
    """Resolve a confirmed prepared-input directory for a site-year."""

    root = _project_root(config)
    site = config["site"]
    status = str(site.get("input_status", "confirmed"))
    if status == "ambiguous":
        raise RuntimeError(
            f"Input provenance is ambiguous for {site.get('station_code')}{year}; "
            "formal execution is blocked until the authoritative IC file is confirmed."
        )
    by_year = site.get("inputs", {}).get("by_year", {})
    entry = by_year.get(str(year), by_year.get(int(year)))
    if entry and entry.get("prepared_input_dir"):
        directory = _resolve(entry["prepared_input_dir"], root)
    elif site.get("inputs", {}).get("source_dir"):
        directory = _resolve(site["inputs"]["source_dir"], root)
    else:
        raise KeyError(f"No input directory configured for {site.get('station_code')}{year}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Prepared input directory does not exist: {directory}")
    return directory


def _find_filex(directory: Path, configured_name: str | None = None) -> Path:
    if configured_name:
        path = directory / configured_name
        if path.exists():
            return path
    candidates = sorted(directory.glob("*.MZX"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one MZX in {directory}, found {len(candidates)}: "
            + ", ".join(path.name for path in candidates)
        )
    return candidates[0]


def _write_input_package(
    source_dir: Path,
    source_filex: Path,
    destination: Path,
    filex_name: str,
    filex_text: str,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared input: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    output_filex = destination / filex_name
    output_filex.write_text(filex_text, encoding="latin-1", errors="ignore")
    for source in source_dir.iterdir():
        if source.is_file() and source.resolve() != source_filex.resolve():
            shutil.copy2(source, destination / source.name)
    return output_filex


def prepare_case(config: dict[str, Any], year: int, run_dir: Path) -> PreparedCase:
    """Prepare DQN/null inputs without modifying the configured source package."""

    legacy = _legacy_module(config)
    site = config["site"]
    code = str(site["station_code"]).upper()
    station = str(site.get("station_name", code))
    treatment = int(site.get("treatment", 1))
    source_dir = resolve_input_directory(config, year)
    by_year = site.get("inputs", {}).get("by_year", {})
    year_entry = by_year.get(str(year), by_year.get(int(year), {})) or {}
    source_filex = _find_filex(source_dir, year_entry.get("fileX_name") or site.get("fileX_name"))

    adapter_name = str(site.get("adapter", ""))
    if code in {"YC", "FQ"} and adapter_name.startswith(("yc_", "fq_")):
        spec = legacy.SiteSpec(
            code=code,
            station=station,
            year=int(year),
            treatment=treatment,
            input_root=source_dir,
            mzx_name=source_filex.name,
            weather_name=str(year_entry.get("weather_name") or site.get("weather_name")),
            soil_id=str(site.get("soil_id")),
        )
        dqn_text = legacy.prepare_site_text(spec, "dqn")
        null_text = legacy.prepare_site_text(spec, "null")
    elif code == "LC" and adapter_name == "lc_fixed_input_017_11":
        # The configured 017_11 package is a scenario-specific null package:
        # its MI/MF pointers and fertilizer rows have already been zeroed.
        # Reusing that text for FERTI=L makes DSSAT/PDI dereference fertfile(0).
        # Rebuild the confirmed, year-aligned LC source exactly as the validated
        # 017_12 DQN entry did, while continuing to copy auxiliaries from the
        # configured prepared-input directory.
        from run_lc_fixed_input_year_screening_017_11 import fixed_source_text

        source_text = fixed_source_text()
        dqn_text = legacy.set_management_for_treatment(source_text, treatment, "L", "L")
        null_text = legacy.prepare_text_for_scenario(source_text, treatment, "null")
        spec = legacy.SiteSpec(
            code=code,
            station=station,
            year=int(year),
            treatment=treatment,
            input_root=source_dir,
            mzx_name=source_filex.name,
            weather_name=str(year_entry.get("weather_name") or site.get("weather_name", "")),
            soil_id=str(site.get("soil_id", "")),
        )
    else:
        source_text = source_filex.read_text(encoding="latin-1", errors="ignore")
        dqn_text = legacy.set_management_for_treatment(source_text, treatment, "L", "L")
        null_text = legacy.prepare_text_for_scenario(source_text, treatment, "null")
        spec = legacy.SiteSpec(
            code=code,
            station=station,
            year=int(year),
            treatment=treatment,
            input_root=source_dir,
            mzx_name=source_filex.name,
            weather_name=str(year_entry.get("weather_name") or site.get("weather_name", "")),
            soil_id=str(site.get("soil_id", "")),
        )

    dqn_filex = _write_input_package(
        source_dir, source_filex, run_dir / "dqn_input", f"{code}{year}_benchmark_dqn.MZX", dqn_text
    )
    null_filex = _write_input_package(
        source_dir, source_filex, run_dir / "null_input", f"{code}{year}_benchmark_null.MZX", null_text
    )
    dqn_env_args = legacy.build_env_args(spec, run_dir / "dqn_runtime", dqn_filex)
    null_env_args = legacy.build_env_args(spec, run_dir / "null_runtime", null_filex)
    (run_dir / "dqn_env_args.json").write_text(
        json.dumps(dqn_env_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "null_env_args.json").write_text(
        json.dumps(null_env_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Prepared %s%s input pair in %s", code, year, run_dir)
    return PreparedCase(
        station_code=code,
        station_name=station,
        year=int(year),
        treatment=treatment,
        dqn_filex=dqn_filex,
        null_filex=null_filex,
        dqn_env_args=dqn_env_args,
        null_env_args=null_env_args,
    )


def load_prepared_case(config: dict[str, Any], year: int, run_dir: Path) -> PreparedCase:
    """Load a previously prepared non-overwriting case for resume/evaluation."""

    site = config["site"]
    code = str(site["station_code"]).upper()
    dqn_filex = _find_filex(run_dir / "dqn_input")
    null_filex = _find_filex(run_dir / "null_input")
    return PreparedCase(
        station_code=code,
        station_name=str(site.get("station_name", code)),
        year=int(year),
        treatment=int(site.get("treatment", 1)),
        dqn_filex=dqn_filex,
        null_filex=null_filex,
        dqn_env_args=json.loads((run_dir / "dqn_env_args.json").read_text(encoding="utf-8")),
        null_env_args=json.loads((run_dir / "null_env_args.json").read_text(encoding="utf-8")),
    )

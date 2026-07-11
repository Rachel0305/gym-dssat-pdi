"""Validation rules for configuration-driven benchmark experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config_loader import canonical_station_code

LOGGER = logging.getLogger(__name__)

SUPPORTED_OBSERVATIONS = {
    "amir",
    "anfer",
    "cumulative_irrigation",
    "cumulative_nitrogen",
    "dap",
    "date",
    "dtt",
    "ep",
    "grnwt",
    "istage",
    "nstres",
    "rain",
    "rtdep",
    "srad",
    "sw",
    "swfac",
    "tmax",
    "tmin",
    "topwt",
    "totir",
    "trnu",
    "vstage",
    "wtdep",
    "xlai",
}

REQUIRED_SECTIONS = (
    "experiment",
    "site",
    "algorithm",
    "action_space",
    "constraints",
    "reward",
    "observations",
)


class ConfigValidationError(ValueError):
    """Raised when one or more benchmark configuration rules fail."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("Invalid benchmark configuration:\n- " + "\n- ".join(self.errors))


def _context(config: Mapping[str, Any]) -> str:
    experiment = config.get("experiment", {})
    site = config.get("site", {})
    algorithm = config.get("algorithm", {})
    experiment_id = experiment.get("experiment_id", "unknown") if isinstance(experiment, Mapping) else "unknown"
    station = site.get("station_code", "unknown") if isinstance(site, Mapping) else "unknown"
    years: Any = None
    if isinstance(site, Mapping):
        years = site.get("eval_years") or site.get("train_years")
    seed = algorithm.get("seed", "unknown") if isinstance(algorithm, Mapping) else "unknown"
    return f"experiment_id={experiment_id} station={station} year={years} seed={seed}"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_mapping(config: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key}: required mapping is missing")
        return {}
    return value


def _validate_years(site: Mapping[str, Any], errors: list[str]) -> None:
    for key in ("train_years", "eval_years"):
        years = site.get(key)
        if not isinstance(years, list) or not years:
            errors.append(f"site.{key}: must be a non-empty list of years")
            continue
        invalid = [year for year in years if not isinstance(year, int) or isinstance(year, bool) or not 1900 <= year <= 2200]
        if invalid:
            errors.append(f"site.{key}: invalid year values {invalid!r}")


def _resolve_path(path_value: str, project_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _validate_paths(site: Mapping[str, Any], root: Path, errors: list[str]) -> None:
    # Current paper-line inputs are versioned prepared-input directories rather
    # than four fabricated top-level paths.  Keep the old form compatible, but
    # prefer the audited per-year package contract.
    inputs = site.get("inputs")
    if isinstance(inputs, Mapping):
        by_year = inputs.get("by_year", {})
        if not isinstance(by_year, Mapping):
            errors.append("site.inputs.by_year: must be a mapping")
            return
        active_years = list(dict.fromkeys([*(site.get("train_years") or []), *(site.get("eval_years") or [])]))
        for year in active_years:
            entry = by_year.get(str(year), by_year.get(year))
            if not isinstance(entry, Mapping):
                errors.append(f"site.inputs.by_year.{year}: configuration is missing")
                continue
            raw_directory = entry.get("prepared_input_dir") or inputs.get("source_dir")
            if not isinstance(raw_directory, str) or not raw_directory.strip():
                errors.append(f"site.inputs.by_year.{year}.prepared_input_dir: path is required")
                continue
            directory = _resolve_path(raw_directory, root)
            if not directory.is_dir():
                errors.append(f"site.inputs.by_year.{year}.prepared_input_dir: directory does not exist: {directory}")
                continue
            required_suffixes = {".MZX", ".WTH", ".CUL", ".SOL"}
            present = {path.suffix.upper() for path in directory.iterdir() if path.is_file()}
            missing = sorted(required_suffixes - present)
            if missing:
                errors.append(
                    f"site.inputs.by_year.{year}: prepared input lacks required suffixes {missing} in {directory}"
                )
        return

    required_paths = ("fileX_template_path", "weather_path", "soil_path", "cultivar_path")
    for key in required_paths:
        raw = site.get(key)
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"site.{key}: a non-empty path is required")
        elif not _resolve_path(raw, root).exists():
            errors.append(f"site.{key}: path does not exist: {_resolve_path(raw, root)}")


def _validate_action_and_constraints(
    action: Mapping[str, Any], constraints: Mapping[str, Any], errors: list[str]
) -> None:
    pairs = (
        ("irrigation_levels", "daily_irrigation_cap", "irrigation_budget"),
        ("nitrogen_levels", "daily_nitrogen_cap", "nitrogen_budget"),
    )
    for levels_key, cap_key, budget_key in pairs:
        levels = action.get(levels_key)
        if not isinstance(levels, list) or not levels:
            errors.append(f"action_space.{levels_key}: must be a non-empty list")
            continue
        if any(not _is_number(item) or item < 0 for item in levels):
            errors.append(f"action_space.{levels_key}: all actions must be non-negative numbers")
            continue
        cap = constraints.get(cap_key)
        budget = constraints.get(budget_key)
        if not _is_number(cap) or cap < 0:
            errors.append(f"constraints.{cap_key}: must be a non-negative number")
        elif max(levels) > cap:
            errors.append(
                f"action_space.{levels_key}: maximum {max(levels)} exceeds {cap_key}={cap}"
            )
        if not _is_number(budget) or budget < 0:
            errors.append(f"constraints.{budget_key}: must be a non-negative number")
        elif max(levels) > budget:
            errors.append(
                f"constraints.{budget_key}: {budget} is below the maximum single action {max(levels)}"
            )

    interval = constraints.get("min_interval_days")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        errors.append("constraints.min_interval_days: must be a positive integer")

    start = constraints.get("decision_window_start_dap")
    end = constraints.get("decision_window_end_dap")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        errors.append(
            "constraints.decision_window_start_dap/end_dap: require integers with 0 <= start <= end"
        )


def _validate_reward(reward: Mapping[str, Any], errors: list[str]) -> None:
    if not isinstance(reward.get("type"), str) or not reward.get("type"):
        errors.append("reward.type: a non-empty reward adapter name is required")
    for key in (
        "irrigation_cost",
        "nitrogen_cost",
        "yield_gain_coefficient",
        "leaching_cost",
    ):
        value = reward.get(key)
        if not _is_number(value):
            errors.append(f"reward.{key}: a numeric value is required")
        elif value < 0:
            errors.append(f"reward.{key}: must be non-negative")


def _validate_output(config: Mapping[str, Any], root: Path, errors: list[str]) -> None:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        return
    raw = experiment.get("output_root", "benchmark_results")
    if not isinstance(raw, str) or not raw.strip():
        errors.append("experiment.output_root: must be a non-empty path")
        return
    output_root = _resolve_path(raw, root)
    if output_root.exists() and not output_root.is_dir():
        errors.append(f"experiment.output_root: existing path is not a directory: {output_root}")

    experiment_id = experiment.get("experiment_id")
    if isinstance(experiment_id, str) and experiment_id:
        target = output_root / experiment_id
        reuse = bool(experiment.get("reuse_existing_results", False))
        resume = bool(experiment.get("resume", False))
        force = bool(experiment.get("force", False))
        if target.exists() and not (reuse or resume or force):
            errors.append(
                "experiment output already exists while reuse/resume/force are all false: "
                f"{target}"
            )


def validate_config(
    config: Mapping[str, Any], project_root: Path | str | None = None
) -> list[str]:
    """Return clear validation errors; an empty list means the config passes."""

    errors: list[str] = []
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    if not isinstance(config, Mapping):
        return ["configuration root must be a mapping"]

    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), Mapping):
            errors.append(f"{section}: required mapping is missing")

    site = _as_mapping(config, "site", errors)
    algorithm = _as_mapping(config, "algorithm", errors)
    action = _as_mapping(config, "action_space", errors)
    constraints = _as_mapping(config, "constraints", errors)
    reward = _as_mapping(config, "reward", errors)
    observations = _as_mapping(config, "observations", errors)

    raw_station = site.get("station_code")
    if not isinstance(raw_station, str) or not raw_station.strip():
        errors.append("site.station_code: a station code is required")
    else:
        try:
            canonical_station_code(raw_station)
        except ValueError as exc:
            errors.append(f"site.station_code: {exc}")

    _validate_years(site, errors)
    _validate_paths(site, root, errors)

    initial_mode = site.get("initial_condition_mode")
    if not isinstance(initial_mode, int) or isinstance(initial_mode, bool) or initial_mode < 0:
        errors.append("site.initial_condition_mode: must be a non-negative integer IC level")

    seed = algorithm.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("algorithm.seed: must be an integer")
    n_steps = algorithm.get("n_steps")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        errors.append("algorithm.n_steps: must be a positive integer compatible with the adapter")
    total_timesteps = algorithm.get("total_timesteps")
    if not isinstance(total_timesteps, int) or isinstance(total_timesteps, bool) or total_timesteps < 1:
        errors.append("algorithm.total_timesteps: must be a positive integer")

    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, Mapping) and "seeds" in evaluation:
        eval_seeds = evaluation["seeds"]
        if not isinstance(eval_seeds, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in eval_seeds
        ):
            errors.append("evaluation.seeds: must be a list of integers")

    _validate_action_and_constraints(action, constraints, errors)
    _validate_reward(reward, errors)

    included = observations.get("include")
    if not isinstance(included, list) or not included:
        errors.append("observations.include: must be a non-empty list")
    else:
        unknown = sorted({str(item) for item in included} - SUPPORTED_OBSERVATIONS)
        if unknown:
            errors.append(f"observations.include: unsupported names {unknown!r}")

    _validate_output(config, root, errors)

    context = _context(config)
    contextualised = [f"[{context}] {error}" for error in dict.fromkeys(errors)]
    if contextualised:
        LOGGER.error("Benchmark configuration validation failed with %d error(s)", len(contextualised))
    return contextualised


def raise_for_invalid_config(
    config: Mapping[str, Any], project_root: Path | str | None = None
) -> None:
    """Raise :class:`ConfigValidationError` if ``validate_config`` fails."""

    errors = validate_config(config, project_root)
    if errors:
        raise ConfigValidationError(errors)

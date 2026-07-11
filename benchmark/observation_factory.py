"""Observation-name validation for benchmark configurations."""

from __future__ import annotations

from typing import Any


# Union of the validated stage observations and raw identifiers used in the
# current gym-DSSAT project. The validator rejects spelling mistakes but does
# not fabricate unavailable values during evaluation.
KNOWN_OBSERVATIONS = frozenset(
    {
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
        "vstage",
        "wtdep",
        "xlai",
        "yrdoy",
    }
)


def requested_observations(config: dict[str, Any]) -> list[str]:
    """Return canonical lower-case observation names in configured order."""

    values = config.get("observations", config).get("include", [])
    return [str(value).strip().lower() for value in values]


def unknown_observations(config: dict[str, Any]) -> list[str]:
    """Return names not recognized by the current environment adapters."""

    return sorted(set(requested_observations(config)) - KNOWN_OBSERVATIONS)


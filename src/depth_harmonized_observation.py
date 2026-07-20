"""Physical-depth observation adapter for multi-site gym-DSSAT policies.

This module does not fabricate soil data below a site's simulated profile.
Missing depth is represented by an explicit coverage fraction of zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TARGET_DEPTH_EDGES_CM = np.asarray([0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0], dtype=np.float64)
NON_SOIL_BEFORE = 9
NON_SOIL_AFTER = 7
OUTPUT_DIMENSION = 28


@dataclass(frozen=True)
class HarmonizedProfile:
    sw_vwc: np.ndarray
    coverage_fraction: np.ndarray
    covered_water_storage_cm: np.ndarray


def harmonize_soil_water(
    sw_vwc: np.ndarray,
    layer_thickness_cm: np.ndarray,
    target_edges_cm: np.ndarray = TARGET_DEPTH_EDGES_CM,
) -> HarmonizedProfile:
    sw = np.asarray(sw_vwc, dtype=np.float64).reshape(-1)
    thickness = np.asarray(layer_thickness_cm, dtype=np.float64).reshape(-1)
    edges = np.asarray(target_edges_cm, dtype=np.float64).reshape(-1)
    if sw.size == 0 or sw.size != thickness.size:
        raise ValueError("sw_vwc and layer_thickness_cm must have the same nonzero length")
    if not np.isfinite(sw).all() or not np.isfinite(thickness).all() or np.any(thickness <= 0):
        raise ValueError("soil values must be finite and every source layer thickness must be positive")
    if edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("target depth edges must be finite and strictly increasing")

    source_bottom = np.cumsum(thickness)
    source_top = np.concatenate(([0.0], source_bottom[:-1]))
    target_top, target_bottom = edges[:-1], edges[1:]
    target_width = target_bottom - target_top
    weighted_storage = np.zeros(target_width.size, dtype=np.float64)
    covered_thickness = np.zeros(target_width.size, dtype=np.float64)
    for s_top, s_bottom, value in zip(source_top, source_bottom, sw):
        overlap = np.maximum(0.0, np.minimum(target_bottom, s_bottom) - np.maximum(target_top, s_top))
        weighted_storage += overlap * value
        covered_thickness += overlap
    mapped = np.divide(
        weighted_storage,
        covered_thickness,
        out=np.zeros_like(weighted_storage),
        where=covered_thickness > 0,
    )
    coverage = covered_thickness / target_width
    return HarmonizedProfile(mapped, coverage, weighted_storage)


def harmonize_flat_observation(
    raw_observation: np.ndarray,
    layer_thickness_cm: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
    thickness = np.asarray(layer_thickness_cm, dtype=np.float64).reshape(-1)
    expected = NON_SOIL_BEFORE + thickness.size + NON_SOIL_AFTER
    if raw.size != expected:
        raise ValueError(f"Expected raw observation length {expected}, got {raw.size}")
    before = raw[:NON_SOIL_BEFORE]
    sw = raw[NON_SOIL_BEFORE : NON_SOIL_BEFORE + thickness.size]
    after = raw[-NON_SOIL_AFTER:]
    profile = harmonize_soil_water(sw, thickness)
    result = np.concatenate((before, profile.sw_vwc, profile.coverage_fraction, after))
    if result.size != OUTPUT_DIMENSION:
        raise AssertionError(f"Unexpected harmonized dimension {result.size}")
    return result.astype(np.float32)


def harmonized_labels() -> list[str]:
    before = ["cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad"]
    bands = ["0_15", "15_30", "30_60", "60_90", "90_120", "120_150"]
    after = ["swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai"]
    return before + [f"sw_vwc_{band}_cm" for band in bands] + [f"sw_coverage_{band}_cm" for band in bands] + after


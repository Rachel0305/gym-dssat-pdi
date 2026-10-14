from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from depth_harmonized_observation import (
    NON_SOIL_AFTER,
    NON_SOIL_BEFORE,
    OUTPUT_DIMENSION,
    TARGET_DEPTH_EDGES_CM,
    harmonize_flat_observation,
    harmonize_soil_water,
    harmonized_labels,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark_results" / "026_10_attempt3" / "026_10_runtime_soil_water_layers.csv"
OUT = ROOT / "benchmark_results" / "026_11"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    source = pd.read_csv(SOURCE)
    checks: dict[str, bool] = {}

    synthetic = harmonize_soil_water(np.asarray([0.2, 0.4]), np.asarray([10.0, 20.0]))
    checks["synthetic_band0_weighted_mean"] = bool(abs(synthetic.sw_vwc[0] - (0.2 * 10 + 0.4 * 5) / 15) < 1e-12)
    checks["synthetic_band1_value"] = bool(abs(synthetic.sw_vwc[1] - 0.4) < 1e-12)
    checks["synthetic_deeper_bands_explicitly_missing"] = bool(
        np.all(synthetic.sw_vwc[2:] == 0) and np.all(synthetic.coverage_fraction[2:] == 0)
    )

    rows = []
    expected_depth = {"SY": 100.0, "HLA": 90.0, "YC": 90.0, "FQ": 100.0, "LC": 150.0}
    for site, frame in source.groupby("site", sort=False):
        frame = frame.sort_values("runtime_layer")
        sw = frame["reset_sw_vwc"].to_numpy(dtype=float)
        thickness = frame["thickness_cm"].to_numpy(dtype=float)
        profile = harmonize_soil_water(sw, thickness)
        source_storage = float(np.sum(sw * thickness))
        target_storage = float(np.sum(profile.covered_water_storage_cm))
        raw = np.concatenate((np.arange(NON_SOIL_BEFORE), sw, np.arange(100, 100 + NON_SOIL_AFTER)))
        transformed = harmonize_flat_observation(raw, thickness)
        checks[f"{site}_storage_conserved"] = bool(abs(source_storage - target_storage) < 1e-10)
        checks[f"{site}_output_dimension_28"] = bool(transformed.size == OUTPUT_DIMENSION)
        checks[f"{site}_nonsoil_before_preserved"] = bool(np.array_equal(transformed[:NON_SOIL_BEFORE], raw[:NON_SOIL_BEFORE]))
        checks[f"{site}_nonsoil_after_preserved"] = bool(np.array_equal(transformed[-NON_SOIL_AFTER:], raw[-NON_SOIL_AFTER:]))
        checks[f"{site}_coverage_bounded"] = bool(np.all((profile.coverage_fraction >= 0) & (profile.coverage_fraction <= 1)))
        max_depth = float(thickness.sum())
        checks[f"{site}_source_depth_preserved"] = bool(abs(max_depth - expected_depth[site]) < 1e-10)
        for band, (top, bottom, value, coverage, storage) in enumerate(
            zip(TARGET_DEPTH_EDGES_CM[:-1], TARGET_DEPTH_EDGES_CM[1:], profile.sw_vwc, profile.coverage_fraction, profile.covered_water_storage_cm),
            start=1,
        ):
            rows.append({
                "site": site, "target_band": band, "top_depth_cm": top, "bottom_depth_cm": bottom,
                "sw_vwc": value, "coverage_fraction": coverage, "covered_water_storage_cm": storage,
                "source_max_depth_cm": max_depth,
            })

    checks["all_five_sites_present"] = set(source["site"]) == {"SY", "HLA", "YC", "FQ", "LC"}
    checks["labels_count_28"] = len(harmonized_labels()) == OUTPUT_DIMENSION
    checks["no_training"] = True
    checks["no_dssat_calls"] = True
    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(OUT / "026_11_five_site_harmonized_profiles.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed_count": int(sum(checks.values())), "check_count": len(checks),
        "input_dimension": "16 + variable soil-layer count",
        "output_dimension": OUTPUT_DIMENSION,
        "target_depth_edges_cm": TARGET_DEPTH_EDGES_CM.tolist(),
        "training_steps": 0, "dssat_calls": 0,
        "next_step_allowed": bool(all(checks.values())),
        "next_step": "Integrate the adapter into a reset-only multi-site environment smoke; existing SY weights remain incompatible and must not be reused.",
    }
    (OUT / "026_11_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

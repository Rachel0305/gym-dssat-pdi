# Random-weather archive contract

This contract applies to future YC, HL, FQ, LC, and SY experiments using runtime-generated random weather. It does not modify or retroactively certify any historical experiment.

## Required per episode

- Preserve the actual runtime WTH file byte-for-byte when the runtime emits one; copy it before cleanup or reuse.
- Preserve the canonical daily series with `DATE,DAP,SRAD,TMAX,TMIN,RAIN` and document how dates/DAP were derived.
- Record raw WTH SHA256 and canonical daily-series SHA256 separately. If the runtime emits no WTH, record `NOT_EMITTED_BY_RUNTIME`; do not synthesize a file and label it as raw.
- Record weather seed, RNG seed/RSEED1, crop-year and season-start context, station, frozen CLI/config SHA256, generator/runtime version, capture timestamp, and source runtime path.
- Record episode-to-archive-weather-ID mapping. Deduplicate identical canonical series using seed + context + canonical SHA256 while retaining one manifest row per episode.
- Copy the archive before the runtime temporary directory is cleaned or another episode can overwrite it. Verify file size and both hashes after copy.

## Required gates

An experiment without the actual canonical daily series and provenance must not claim climate coverage, weather-tail coverage, or exact weather reproducibility. An experiment with runtime-state weather but no raw WTH may report the state-derived series and its limitation; it must not claim raw WTH archival. A historical hash match must use the exact historical serialization algorithm, not a newly invented normalization.

## Integration

Call `src/archive_runtime_weather.py::archive_runtime_weather` immediately after the episode weather states have been captured and before closing the environment. Pass an actual runtime WTH path only when one exists. Keep archives under the experiment's own `weather_archive/` directory; never write them into old experiment outputs.

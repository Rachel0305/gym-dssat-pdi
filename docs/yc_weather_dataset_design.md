# YC weather dataset design and feasibility verification

## Scope and status

This record executes prompt `prompt_02/002_yc_weather_dataset_design.md` for YC/YCA only. It is a dataset design and lightweight feasibility check, not a PPO training result. It does not modify `055_00`, old checkpoints, original `CNYC*.WTH`, LC/SY frozen outputs, or HL/FQ data.

Run context recorded by `results/yc_weather_dataset_design/yc_weather_dataset_design_snapshot.json`:

- Branch: `codex/sya-forecast-freeze-2026-08-16`.
- Starting HEAD: `523588e`.
- Current task: design and evidence only; no PPO training.
- New evidence directory: `results/yc_weather_dataset_design/`.

## Confirmed split and current weather path

| Item | Status | Evidence |
|---|---|---|
| Site/station | code-confirmed | `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:5-6` sets `station_code=YCA`, `site=YC`; runner refuses other station/site at `src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py:45-46`. |
| PPO seed and action grid | code-confirmed | Seed is `0` at `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:8`; action grid is 4 irrigation by 4 nitrogen levels at `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:15-16`. |
| Observation contract | code-confirmed | Normalization and weather forecast are disabled at `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:20-21`; runner enforces this at `src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py:57-58`. |
| Train years | code-confirmed and run-confirmed | `2004-2013` in `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:24`; matched by diagnostic `results/yc_weather_audit/yc_weather_reset_diagnostic.json:14-49`. |
| Validation years | code-confirmed and run-confirmed | `2014-2023` in `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:25`; matched by diagnostic `results/yc_weather_audit/yc_weather_reset_diagnostic.json:26-61`. |
| Test years | not yet confirmed | No frozen independent YC test split was found in the current `055_00` config or 001 audit. Treat final test years as a future 003 freeze, not as an inferred fact. |
| Episode weather variation | run-confirmed | `episode_weather_changes=true` and `weather_reproducible=true` in `results/yc_weather_audit/yc_weather_config_snapshot.json:13-14`. The 001 report states this is historical-year switching, not generated WGEN weather, at `docs/yc_weather_pipeline_audit.md:7` and `docs/yc_weather_pipeline_audit.md:94`. |
| DSSAT WTH path | run-confirmed | 2004 rendered example is `results/yc_weather_audit/rendered_probe/rendered_inputs/YCA/2004/yc_weather_audit_probe/CNYC0401.WTH` in `results/yc_weather_audit/yc_weather_config_snapshot.json:8-12`; runner copies historical WTH files at `src/ppo_safe_rendering.py:300-304`. |
| `random_weather` | run-confirmed | Current env args set `random_weather=false` in `src/ppo_safe_rendering.py:335-336` and `results/yc_weather_audit/yc_weather_reset_diagnostic.json:156`; 001 conclusion is `docs/yc_weather_pipeline_audit.md:73-76`. |
| YC `.CLI` | run-confirmed absent in current chain | 001 audit records no `.CLI` in current YC lowIC input path at `docs/yc_weather_pipeline_audit.md:43-45` and `docs/yc_weather_pipeline_audit.md:118`. |

## Route comparison

### Route A: gym-DSSAT WGEN via `random_weather` and `.CLI`

Current status: blocked for current YC lowIC.

The installed `gym_dssat_pdi==0.0.5` environment supports `random_weather`; the captured constructor signature includes `random_weather=True` by default and `auxiliary_file_paths` at `results/yc_weather_audit/yc_weather_reset_diagnostic.json:86-90`. However, current YC/YCA rendering explicitly passes `random_weather=false` and only cultivar, WTH, and soil are in `auxiliary_file_paths` (`docs/yc_weather_pipeline_audit.md:182-183`).

External basis: DSSAT documentation states that DSSAT can use WGEN and SIMMETEO weather generators, and that weather-generator coefficients are stored in `*.CLI` station files; WGEN needs statistics computed from daily data across multiple years, while SIMMETEO uses monthly averages. See DSSAT User's Guide Vol. 1 and Vol. 3: <https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol1.pdf>, <https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf>. DSSAT's public model overview also notes WGEN and SIMMETEO as current CSM weather generators: <https://dssat.net/models-overview/how-to-use-the-model/>.

Decision: do not enable this route now. The blocker is not code ability; it is the missing verified YC `.CLI` parameter file derived only from train years. Hand-writing `.CLI` fields or letting a package default climate leak into YC would be invalid.

### Route B: external Python weather generator

Current status: design only, pending verified implementation.

Literature basis exists. Richardson's stochastic daily weather model generates precipitation with a Markov-chain model and then generates temperature and solar radiation conditioned on wet/dry status, preserving seasonal statistics and inter-variable relationships in observed data: <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/WR017i001p00182>. This is suitable as a candidate family, but no frozen YC implementation, fitted parameter artifact, seed contract, or DSSAT smoke has been verified in this repo.

Decision: do not generate new YC synthetic WTH files from a fresh stochastic generator in 002. Implementing Route B safely requires a separate 003 pilot that fits parameters only from 2004-2013, persists fitted parameters and hashes, and proves fixed-seed reproducibility before PPO.

### Route C: existing 217YCA deterministic rain-window scenario bank

Current status: feasible as limited scenario-bank evidence, not a full weather generator.

The repo already contains a YC/YCA fixed-width rain-window scenario bank with manifest fields and hashes at `benchmark_results/217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width/217YCA_weather_scenario_manifest.csv:1`. Existing physical gate evidence reports `training_run=false`, all six scenarios completed, WTH rain matched DSSAT logs within 2 mm, all scenarios matured, positive yield, and a physical yield response at `benchmark_results/217YCA_yca_lowIC_weather_physical_gate/217YCA_result.json:4-14`.

Decision: use Route C only as small-sample feasibility evidence and manifest rehearsal. Do not call it a statistical weather generator or proof that augmentation improves PPO.

The machine-readable comparison is saved in `results/yc_weather_dataset_design/yc_weather_route_comparison.csv:1-4`.

## Manifest design

The proposed manifest schema is saved in `results/yc_weather_dataset_design/yc_weather_manifest_schema.csv`. Required core fields:

| Field | Meaning |
|---|---|
| `scenario_id` | Stable unique key, e.g. `YC_217YCA_2005_early_dry`. |
| `site`, `station_code`, `split` | YC/YCA identity and train/validation/test/diagnostic split. |
| `source_years` | Historical years used to fit, resample, or perturb the weather. For generator fitting, validation/test years must never appear here. |
| `generator_name`, `generator_version` | Implementation identity and version or commit. |
| `weather_selection_seed` | RNG selecting historical years/scenarios; separate from PPO. |
| `weather_generation_seed` | RNG producing stochastic weather after parameters are fixed; separate from selection and PPO. |
| `ppo_seed` | PPO initialization and rollout seed; not a weather seed. |
| `weather_file`, `file_sha256` | Exact generated/selected DSSAT WTH path and byte hash. |
| `climate_parameter_file`, `parameters_hash` | `.CLI` or Python fitted parameter artifact, if used, and its hash. |
| `generation_config` | Compact JSON with generator settings and source-year contract. |
| `validation_status`, `leakage_status` | Quality gate result and evidence that validation/test data were excluded. |
| `dssat_smoke_source` | Reset/single-season smoke evidence path if available. |

Future baseline vs augmented PPO comparison must keep algorithm, network, reward, observation, action space, timesteps, real test weather, and resource constraints unchanged. Only weather dataset source should vary.

## Seed design

The old `055_00` currently uses `seed=0` in the config (`configs/055_00_yca_lowIC_expanded_action_maskableppo.json:8`) and a random-year wrapper seeds `np.random.default_rng(seed)` at `src/run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py:113-120`. The 001 diagnostic records both `ppo_seed=0` and `weather_seed=0`, and also records that the current weather seed is not separate from PPO seed in that legacy path (`results/yc_weather_audit/yc_weather_reset_diagnostic.json:88-95`).

For 003, use explicit independent fields:

- `ppo_seed`: PPO initialization and action sampling.
- `weather_selection_seed`: source-year or scenario ordering.
- `weather_generation_seed`: stochastic generator draws after fitted parameters are frozen.

Keeping all three equal to zero is allowed only if the manifest records separate fields and code paths. It must not reuse one RNG instance silently.

## Small-sample feasibility evidence

002 did not generate new stochastic weather because no verified YC statistical generator is frozen in the repo. Instead, it generated a five-row small-sample manifest over existing 217YCA deterministic train-year scenarios:

- `results/yc_weather_dataset_design/yc_weather_small_sample_manifest.csv:2-6`.
- `results/yc_weather_dataset_design/yc_weather_small_sample_quality_summary.csv:2-6`.

The five samples are `original`, `early_dry`, `mid_dry`, `late_dry`, and `early_wet` for source year 2005. They are all train-source only, deterministic, and have WTH hashes recorded. Physical file checks passed for 365 rows, nonnegative rain and solar radiation, and `TMAX >= TMIN`; annual rainfall spans 584.0-745.3 mm in the quality summary. Existing DSSAT smoke evidence is reused from `benchmark_results/217YCA_yca_lowIC_weather_physical_gate/217YCA_result.json:4-14`.

What was not done:

- No new 3-5 stochastic synthetic WTH files were produced.
- No fixed/different stochastic seed reproducibility test was run.
- No new DSSAT reset/single-season smoke was run for a new generator.
- No PPO training was run.

This is intentional: the prompt requires stopping rather than fabricating "reasonable" Gaussian weather when the generator implementation is not verified.

## Quality gates for 003

Weather file gates:

- DSSAT WTH parse succeeds; required columns are present.
- Rainfall and solar radiation are nonnegative.
- `TMAX >= TMIN` on every day.
- Calendar length is valid for year and leap-year handling.
- Missing values are absent or documented with a deterministic fill rule.
- WSTA and weather filename match the rendered experiment file.
- File hash and generator parameter hash are persisted.

Statistical review gates:

- Monthly rainfall totals, wet-day ratio, mean wet-day rainfall, maximum dry spell, seasonal temperature/radiation, and variable correlations are compared with 2004-2013 training weather.
- Validation/test years are excluded from parameter fitting and hyperparameter selection.
- Outlier thresholds are treated as review flags, not scientific laws.

DSSAT smoke gates:

- Minimal reset/single-season run reads the WTH file.
- Season completes to maturity.
- Rain in DSSAT logs matches WTH within a predeclared tolerance.
- Yield, phenology, irrigation, and nitrogen behavior are checked for obvious anomalies.
- Smoke passing is not evidence of final agronomic validity.

## Temporary-directory inventory

002 inspected but did not delete inherited artifacts:

- `.codex-yc-weather-pptx-build`: untracked, 7800 files, retained.
- `results/yc_weather_audit/rendered_probe`: untracked, 2 files, retained as 001 audit evidence.
- `results/yc_weather_audit/gym_reset_probe`: untracked, 12 files, retained as 001 audit evidence.

Inventory is saved in `results/yc_weather_dataset_design/yc_weather_cleanup_inventory.csv:1-4`. No `git clean` or recursive deletion was used.

## Recommended 003 minimal work

1. Freeze the exact train/validation/test contract: keep train 2004-2013, validation 2014-2023, and define any independent test split explicitly rather than inferring it.
2. Choose one generator route:
   - A: estimate a YC `.CLI` from train years only and prove DSSAT WGEN reads it.
   - B: implement one literature-backed Python generator, likely Richardson-style WGEN, with fitted parameter artifacts and seed-separated manifest.
   - C: if the goal is only stress scenarios, formally declare use of the existing 217YCA fixed rain-window bank and avoid statistical-generator claims.
3. Generate only 3-5 new pilot WTH files in a new YC-only directory.
4. Prove fixed `weather_generation_seed` reproduces identical hashes and a different seed changes measurable weather statistics.
5. Run DSSAT smoke only; still no PPO training until the weather gate is passed.
6. Only after the weather gate, create a new experiment id for PPO so `055_00` remains frozen.

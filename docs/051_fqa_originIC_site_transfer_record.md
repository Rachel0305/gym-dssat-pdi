# 051 FQA/FQ originIC site-transfer planning record

## Why this series was opened

YCA/YC 050 fixed the static management event-chain issue, but PPO still mainly won through high PFP_N caused by low nitrogen use. It did not clearly beat the best baseline in yield or WP_ET.

FQA/FQ is therefore opened as the next station-screening attempt, using the same corrected-baseline-first workflow.

## 051 design

- Station code used by the training framework: `FQA`.
- DSSAT input/site short code: `FQ`.
- Input profile: `originIC`.
- Train years: 2005-2013.
- Validation years: 2014-2023.
- PPO method: frozen SYA 046_10 framework and parameters.
- Auto method: DSSAT native auto irrigation plus minimal external auto-N, default `NSTRES >= 0.5`, `25 kg/ha`.
- Baseline method: regenerate four non-PPO baselines with the 037_07 static level-1 correction.

## Required run order

1. `051_02` corrected four baselines.
2. `051_00` PPO smoke and formal training.
3. `051_01` auto irrigation + external auto-N.
4. `051_03` five-scenario figures and season-level bar charts.

## Acceptance checks

- `051_02_coverage_manifest.csv` should have no non-ok rows.
- `051_02_management_event_audit.csv` should show that planned static expert/recorded events are represented in DSSAT outputs as expected.
- PPO smoke must pass before formal training.
- `PFP_N` remains `N/A` when actual nitrogen is zero.

## Completed 051 result audit

Audited after `051_02` baseline rebuild, `051_00` PPO formal run, `051_01` auto run, and `051_03` five-scenario figures were available.

### Baseline event-chain status

The static level-1 correction worked for FQ. `051_02_management_event_audit.csv` reports all 40 four-baseline records as `ok`.

| scenario | ok years | event-chain interpretation |
| --- | ---: | --- |
| null | 10 | no planned or executed management events |
| dssat_auto | 10 | DSSAT automatic irrigation executes; no automatic fertilizer |
| recorded_farmer_template | 10 | planned/executed template closes: 1 irrigation event and 1 fertilizer event per year |
| official_extension_expert | 10 | planned/executed expert schedule closes: about 5-6 irrigation and 5 fertilizer events per year |

Mean static-management audit:

| scenario | planned I events | MgmtEvent I events | planned I total mm | MgmtEvent I total mm | planned N events | MgmtEvent N events | planned N kg/ha | MgmtEvent N kg/ha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_extension_expert | 5.4 | 5.5 | 209.62 | 212.68 | 4.9 | 4.9 | 244.12 | 241.7 |
| recorded_farmer_template | 1.0 | 1.0 | 75.0 | 75.0 | 1.0 | 1.0 | 144.0 | 144.0 |

The corrected baseline pipeline is therefore functioning; the weak PPO result is not caused by the old “only one expert event executed” bug.

### Five-scenario 10-year means

| scenario | yield kg/ha | irrigation mm | nitrogen kg/ha | WP_ET kg/m3 | PFP_N kg/kg | max WSPD | max NSTD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null | 6953.4 | 0.0 | 0.0 | 2.145 | N/A | 0.157 | 0.130 |
| dssat_auto_external_n | 7178.2 | 34.7 | 0.0 | 2.125 | N/A | 0.000 | 0.126 |
| official_extension_expert | 7311.9 | 212.9 | 241.7 | 2.079 | 33.156 | 0.000 | 0.012 |
| recorded_farmer_template | 7201.7 | 75.0 | 144.0 | 2.145 | 55.578 | 0.094 | 0.012 |
| rl_candidate | 7315.3 | 219.0 | 240.0 | 1.996 | 33.867 | 0.000 | 0.012 |

### PPO comparison

Against each individual baseline:

| comparison baseline | PPO yield mean gap | PPO yield wins | PPO WP_ET mean gap | PPO WP_ET wins | PPO PFP_N mean gap | PPO PFP_N wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| null | +361.9 | 5/10 | -0.149 | 2/10 | N/A | N/A |
| dssat_auto_external_n | +137.1 | 4/10 | -0.129 | 2/10 | N/A | N/A |
| official_extension_expert | +3.4 | 4/10 | -0.083 | 1/10 | +0.711 | 8/10 |
| recorded_farmer_template | +113.6 | 4/10 | -0.149 | 1/10 | -21.711 | 0/10 |

Against the best of the other four scenarios each year:

| metric | PPO wins | mean gap | min gap | max gap |
| --- | ---: | ---: | ---: | ---: |
| grain_yield_kg_ha | 3/10 | -65.6 | -406.0 | +481.0 |
| WP_ET_kg_m3 | 1/10 | -0.195 | -0.35 | +0.01 |
| PFP_N_kg_kg | 0/9 | -21.711 | -25.0 | -18.9 |

### Interpretation

FQ is another negative station-screening result, but the failure mode differs from YC.

In YC, PPO mainly looked good through very low nitrogen and very high PFP_N. In FQ, PPO instead used high irrigation and high nitrogen, close to the official expert input level, but this did not translate into robust yield superiority and it clearly harmed WP_ET. The recorded template, with much lower nitrogen, dominates PPO on PFP_N.

This confirms that the current 046_10-style PPO policy/reward/action setup is not station-robust. The same framework can produce qualitatively different behavior across sites: SY looked acceptable, YC became low-N/PFP_N-driven, and FQ became high-input with poor WP_ET. Therefore 051 should not be used as a positive result; it is evidence that station transfer is unstable under the current formulation.

The final station-screening attempt is LC/LCA under 052, keeping the same corrected-baseline-first workflow and the same PPO/auto parameters.

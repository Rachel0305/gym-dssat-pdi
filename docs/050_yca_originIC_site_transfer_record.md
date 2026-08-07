# 050 YCA/YC originIC site-transfer planning record

## Why this series was opened

The HLA/HL 049 results showed severe comparison-risk:

- static expert/recorded management appeared to execute only one event per year;
- five-scenario yield and WP_ET were therefore almost identical;
- PPO appeared mainly strong in PFP_N because all managed scenarios used very little nitrogen;
- HLA native auto irrigation and default minimal auto-N did not trigger much stress response.

Before tuning HLA auto thresholds, the safer next step is to test another station while fixing the static baseline event-chain problem.

## 050 design

- Station code used by the training framework: `YCA`.
- DSSAT input/site short code: `YC`.
- Input profile: `originIC`.
- Train years: 2004-2013.
- Validation years: 2014-2023.
- PPO method: frozen SYA 046_10 framework and parameters.
- Auto method: DSSAT native auto irrigation plus minimal external auto-N, default `NSTRES >= 0.5`, `25 kg/ha`.
- Baseline method: regenerate four non-PPO baselines with the 037_07 static level-1 correction.

## Required run order

1. `050_02` corrected four baselines.
2. `050_00` PPO smoke and formal training.
3. `050_01` auto irrigation + external auto-N.
4. `050_03` five-scenario figures and season-level bar charts.

Do not use old `034_00` baselines for final 050 figures unless the goal is explicitly diagnostic.

## Acceptance checks

- `050_02_coverage_manifest.csv` should have no non-ok rows.
- `050_02_management_event_audit.csv` should show that planned static expert/recorded events are represented in DSSAT outputs as expected.
- PPO smoke must pass before formal training.
- `PFP_N` remains `N/A` when actual nitrogen is zero.

## Completed 050 result audit

Audited after the full `050_02` baseline rebuild, `050_00` PPO formal run, `050_01` auto run, and `050_03` five-scenario figures were available.

### Baseline event-chain status

The static level-1 correction worked for YC. `050_02_management_event_audit.csv` reports all 40 four-baseline records as `ok`:

| scenario | ok years | event-chain interpretation |
| --- | ---: | --- |
| null | 10 | no planned or executed management events |
| dssat_auto | 10 | DSSAT automatic irrigation executes normally; no automatic fertilizer |
| recorded_farmer_template | 10 | planned/executed static template closes: 1 irrigation event and 2 fertilizer events per year |
| official_extension_expert | 10 | planned/executed expert schedule closes: about 5 irrigation and 5 fertilizer events per year |

Mean static-management audit:

| scenario | planned I events | MgmtEvent I events | planned I total mm | MgmtEvent I total mm | planned N events | MgmtEvent N events | planned N kg/ha | MgmtEvent N kg/ha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_extension_expert | 5.2 | 5.4 | 204.75 | 210.8 | 5.0 | 5.0 | 247.5 | 245.0 |
| recorded_farmer_template | 1.0 | 1.0 | 120.0 | 120.0 | 2.0 | 2.0 | 374.0 | 374.0 |

This means the HL problem of “expert only has one management event” was not carried into YC.

### Five-scenario 10-year means

| scenario | yield kg/ha | irrigation mm | nitrogen kg/ha | WP_ET kg/m3 | PFP_N kg/kg | max WSPD | max NSTD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null | 7758.5 | 0.0 | 0.0 | 2.204 | N/A | 0.159 | 0.384 |
| dssat_auto_external_n | 7892.5 | 42.8 | 0.0 | 2.217 | N/A | 0.000 | 0.386 |
| official_extension_expert | 8211.1 | 211.0 | 245.0 | 2.257 | 33.51 | 0.000 | 0.012 |
| recorded_farmer_template | 8215.4 | 120.0 | 374.0 | 2.308 | 21.96 | 0.000 | 0.012 |
| rl_candidate | 8164.2 | 45.0 | 40.0 | 2.295 | 204.11 | 0.000 | 0.174 |

### PPO comparison

Against each individual baseline:

| comparison baseline | PPO yield mean gap | PPO yield wins | PPO WP_ET mean gap | PPO WP_ET wins | PPO PFP_N mean gap | PPO PFP_N wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| null | +405.7 | 10/10 | +0.091 | 9/10 | N/A | N/A |
| dssat_auto_external_n | +271.7 | 10/10 | +0.078 | 9/10 | N/A | N/A |
| official_extension_expert | -46.9 | 6/10 | +0.038 | 9/10 | +170.60 | 10/10 |
| recorded_farmer_template | -51.2 | 4/10 | -0.013 | 1/10 | +182.15 | 10/10 |

Against the best of the other four scenarios each year:

| metric | PPO wins | mean gap | min gap | max gap |
| --- | ---: | ---: | ---: | ---: |
| grain_yield_kg_ha | 3/10 | -54.0 | -421.0 | +3.0 |
| WP_ET_kg_m3 | 1/10 | -0.013 | -0.110 | +0.010 |
| PFP_N_kg_kg | 10/10 | +170.60 | +136.10 | +197.90 |

### Interpretation

YC is diagnostically useful but not a strong manuscript result for the current PPO formulation.

The good news is that the comparison itself is much cleaner than HL: expert and recorded management were read and executed as multi-event schedules. The bad news is substantive: PPO again mainly wins by using very little nitrogen, not by clearly combining high yield, high WP_ET, and high PFP_N. Its mean yield is slightly below both expert and recorded, and its WP_ET does not beat the best baseline in most years.

For this reason, 050 should be treated as a station-screening result rather than a frozen positive result. The next station-screening attempt is FQ/FQA under a new 051 series, using the same corrected-baseline-first workflow.

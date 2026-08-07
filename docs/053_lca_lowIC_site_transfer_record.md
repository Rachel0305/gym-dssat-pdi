# 053 LCA/LC lowIC site-transfer result record

## Why this series was opened

LC originIC (`052`) behaved like a low-stress / low-management-response site: null already had high yield and high WP_ET, and PPO mainly won through high PFP_N from low nitrogen use. The 053 series lowers the LC initial condition profile to test whether LC becomes more management-responsive under `lowIC`.

## Design

- Station code: `LCA`.
- DSSAT site/input short code: `LC`.
- Input profile: `lowIC`.
- PPO method: frozen 046_10-style raw-observation MaskablePPO.
- Action grid: irrigation `[0, 15, 30, 45]` mm; nitrogen `[0, 40, 80, 120]` kg/ha.
- Train years: 2005-2013.
- Validation years: 2014-2023.
- Auto method: DSSAT native auto irrigation plus minimal external auto-N, default `NSTRES >= 0.5`, `25 kg/ha`.
- Baseline method: four non-PPO baselines regenerated with the 037_07 static level-1 correction.

## Completed 053 result audit

Audited after `053_02` baseline rebuild, `053_00` PPO formal run, `053_01` auto run, and `053_03` five-scenario figures were available.

### Baseline event-chain status

The static level-1 correction worked for LC lowIC. `053_02_management_event_audit.csv` reports all 40 four-baseline records as `ok`.

| scenario | ok years | event-chain interpretation |
| --- | ---: | --- |
| null | 10 | no planned or executed management events |
| dssat_auto | 10 | DSSAT automatic irrigation executes; no automatic fertilizer |
| recorded_farmer_template | 10 | planned/executed template closes: 2 irrigation events and 2 fertilizer events per year |
| official_extension_expert | 10 | planned/executed expert schedule closes: 5 irrigation and 5 fertilizer events per year |

Mean static-management audit:

| scenario | planned I events | MgmtEvent I events | planned I total mm | MgmtEvent I total mm | planned N events | MgmtEvent N events | planned N kg/ha | MgmtEvent N kg/ha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_extension_expert | 5.0 | 5.0 | 198.75 | 198.8 | 5.0 | 5.0 | 247.5 | 245.0 |
| recorded_farmer_template | 2.0 | 2.0 | 130.0 | 130.0 | 2.0 | 2.0 | 250.0 | 250.0 |

### Five-scenario 10-year means

| scenario | yield kg/ha | irrigation mm | nitrogen kg/ha | WP_ET kg/m3 | PFP_N kg/kg | max WSPD | max NSTD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null | 3191.8 | 0.0 | 0.0 | 1.495 | N/A | 0.833 | 0.102 |
| dssat_auto_external_n | 6771.2 | 177.0 | 0.0 | 1.983 | N/A | 0.008 | 0.333 |
| official_extension_expert | 7951.4 | 199.0 | 245.0 | 2.581 | 32.46 | 0.783 | 0.012 |
| recorded_farmer_template | 8328.2 | 130.0 | 250.0 | 2.572 | 33.32 | 0.358 | 0.012 |
| rl_candidate | 9140.6 | 208.5 | 240.0 | 2.723 | 38.09 | 0.310 | 0.012 |

### PPO comparison

Against each individual baseline:

| comparison baseline | PPO yield mean gap | PPO yield wins | PPO WP_ET mean gap | PPO WP_ET wins | PPO PFP_N mean gap | PPO PFP_N wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| null | +5948.8 | 10/10 | +1.228 | 10/10 | N/A | N/A |
| dssat_auto_external_n | +2369.4 | 10/10 | +0.740 | 10/10 | N/A | N/A |
| official_extension_expert | +1189.2 | 9/10 | +0.142 | 9/10 | +5.63 | 10/10 |
| recorded_farmer_template | +812.4 | 6/10 | +0.151 | 6/10 | +4.77 | 10/10 |

Against the best of the other four scenarios each year:

| metric | PPO wins | ties | mean gap | min gap | max gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| grain_yield_kg_ha | 6/10 | 0/10 | +503.6 | -51.0 | +1736.0 |
| WP_ET_kg_m3 | 6/10 | 0/10 | +0.036 | -0.09 | +0.16 |
| PFP_N_kg_kg | 10/10 | 0/10 | +3.39 | +0.80 | +8.60 |

### Management behavior

Mean management event counts:

| scenario | irrigation events/year | nitrogen events/year |
| --- | ---: | ---: |
| dssat_auto_external_n | 3.4 | 0.0 |
| official_extension_expert | 5.0 | 5.0 |
| recorded_farmer_template | 2.0 | 2.0 |
| rl_candidate | 6.3 | 6.0 |

The current default auto-N threshold (`NSTRES >= 0.5`) did not trigger any external nitrogen events in 053: total auto external N events = 0, mean auto N = 0 kg/ha. This means the current auto baseline is still irrigation-only plus DSSAT native irrigation, and a separate auto-N threshold sensitivity is needed before freezing the final LC lowIC comparison.

### Interpretation

Lowering LC to `lowIC` materially changed the experiment. Unlike 052 originIC, LC lowIC creates a clear management-response environment:

- null collapses to low yield and high water stress;
- auto irrigation improves yield but remains nitrogen-limited because no external N is triggered;
- expert and recorded improve strongly;
- PPO becomes the best integrated strategy in most years, with higher mean yield, higher mean WP_ET, and higher mean PFP_N than the managed baselines.

This is the first LC result that looks potentially useful. It is not yet final because the auto baseline is incomplete: the default `0.5` nitrogen-stress trigger never applied fertilizer. The next step should be a controlled `053_01` auto-N threshold sensitivity, changing only the auto external N threshold/dose config and regenerating `053_03` figures with a labeled auto run.

Recommended first sensitivity: lower the auto-N trigger enough to apply some nitrogen but avoid making auto an aggressively tuned teacher. Candidate thresholds: `0.3`, then `0.2` if `0.3` still gives zero N. Keep dose at `25 kg/ha` initially.

# 052 LCA/LC originIC site-transfer planning record

## Why this series was opened

SYA is currently the only station where the frozen 046_10-style PPO result looks usable. YCA and FQA both completed with corrected baselines but did not show robust PPO superiority across yield, WP_ET, and PFP_N. LC/LCA is therefore opened as the final same-framework station-screening attempt.

## 052 design

- Station code used by the training framework: `LCA`.
- DSSAT input/site short code: `LC`.
- Input profile: `originIC`.
- Train years: 2005-2013.
- Validation years: 2014-2023.
- PPO method: frozen SYA 046_10 framework and parameters.
- Auto method: DSSAT native auto irrigation plus minimal external auto-N, default `NSTRES >= 0.5`, `25 kg/ha`.
- Baseline method: regenerate four non-PPO baselines with the 037_07 static level-1 correction.

## Input note

`configs/sites/lca.yaml` documents an LC2010 prepared-input adapter from 017_11. For this 052 station-screening series, the workflow intentionally keeps the same multisite originIC input profile used by 050 and 051. Any switch to the LC-specific adapter should be a separate experiment, not an implicit change inside 052.

## Required run order

1. `052_02` corrected four baselines.
2. `052_00` PPO smoke and formal training.
3. `052_01` auto irrigation + external auto-N.
4. `052_03` five-scenario figures and season-level bar charts.

## Acceptance checks

- `052_02_coverage_manifest.csv` should have no non-ok rows.
- `052_02_management_event_audit.csv` should show that planned static expert/recorded events are represented in DSSAT outputs as expected.
- PPO smoke must pass before formal training.
- `PFP_N` remains `N/A` when actual nitrogen is zero.

## Completed 052 result audit

Audited after `052_02` baseline rebuild, `052_00` PPO formal run, `052_01` auto run, and `052_03` five-scenario figures were available.

### Baseline event-chain status

The static level-1 correction worked for LC. `052_02_management_event_audit.csv` reports all 40 four-baseline records as `ok`.

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

The comparison is therefore technically valid under the 052 workflow; the poor PPO result is not caused by missing expert/recorded management events.

### Five-scenario 10-year means

| scenario | yield kg/ha | irrigation mm | nitrogen kg/ha | WP_ET kg/m3 | PFP_N kg/kg | max WSPD | max NSTD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null | 9318.1 | 0.0 | 0.0 | 2.876 | N/A | 0.000 | 0.012 |
| dssat_auto_external_n | 9335.6 | 74.9 | 0.0 | 2.763 | N/A | 0.000 | 0.012 |
| official_extension_expert | 9354.5 | 199.0 | 245.0 | 2.772 | 38.17 | 0.000 | 0.012 |
| recorded_farmer_template | 9325.5 | 130.0 | 250.0 | 2.757 | 37.30 | 0.000 | 0.014 |
| rl_candidate | 9317.2 | 45.0 | 40.0 | 2.851 | 232.95 | 0.000 | 0.012 |

### PPO comparison

Against each individual baseline:

| comparison baseline | PPO yield mean gap | PPO yield wins | PPO WP_ET mean gap | PPO WP_ET wins | PPO PFP_N mean gap | PPO PFP_N wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| null | -0.9 | 2/10 | -0.025 | 1/10 | N/A | N/A |
| dssat_auto_external_n | -18.4 | 2/10 | +0.088 | 8/10 | N/A | N/A |
| official_extension_expert | -37.3 | 5/10 | +0.079 | 9/10 | +194.78 | 10/10 |
| recorded_farmer_template | -8.3 | 8/10 | +0.094 | 9/10 | +195.65 | 10/10 |

Against the best of the other four scenarios each year:

| metric | PPO wins | ties | mean gap | min gap | max gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| grain_yield_kg_ha | 0/10 | 7/10 | -44.8 | -256.0 | 0.0 |
| WP_ET_kg_m3 | 0/10 | 2/10 | -0.031 | -0.05 | 0.0 |
| PFP_N_kg_kg | 10/10 | 0/10 | +194.78 | +161.1 | +235.6 |

### Interpretation

LC is also not a positive station result for the current PPO framework.

The key pattern is that management has very weak marginal benefit under the current LC originIC setup. Null already has high yield and the highest mean WP_ET, while nitrogen stress is near the floor in all scenarios. PPO applies only one irrigation event and one nitrogen event on average, giving very high PFP_N, but it does not improve yield or WP_ET over the best baseline. In fact, best-of-four comparisons show zero PPO wins for yield and zero wins for WP_ET.

This is closest to a “low-stress / low-management-response” station outcome: PPO is not catastrophically over-applying resources, but it also does not discover a robust agronomic advantage. Its PFP_N advantage is mostly a denominator effect from very low nitrogen use.

Across the same fixed 046_10-style framework:

| station series | broad outcome |
| --- | --- |
| SYA / 046_10 | only currently usable stage result |
| HLA / 049 | weak management contrast; auto/expert/recorded comparison not compelling |
| YCA / 050 | low-N PPO with high PFP_N, but yield/WP_ET not robust |
| FQA / 051 | high-input PPO, poor WP_ET, PFP_N loses to recorded |
| LCA / 052 | low-stress site; PPO PFP_N wins but yield/WP_ET do not |

Conclusion: the fixed 046_10 PPO reward/action/observation formulation is not station-robust. The repeated cross-station failures are informative rather than just “bad luck”: different stations expose different failure modes. Before using this as a multi-station method, the next research step should shift from station screening to method redesign, most likely including reward alignment with yield-WP_ET-PFP_N tradeoffs, explicit action-budget/efficiency constraints, or station-conditioned policy/input treatment.

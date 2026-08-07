# 046_11 SYA minimal external auto-N

Question: if DSSAT native automatic fertilizer is not operational, what happens
when auto-N is represented by the smallest transparent external rule?

Frozen factors:

- input profile: `originIC`
- station: `SYA`
- irrigation: DSSAT native automatic irrigation
- nitrogen action channel: gym-DSSAT external scheduled action
- validation years: 2014-2023

Only nitrogen rule:

- apply `nitrogen_dose_kg_ha` whenever pre-action `NSTRES >= nitrogen_stress_threshold`

Removed relative to 046_04:

- no DAP cutoff
- no minimum days between nitrogen applications
- no seasonal nitrogen cap

The output keeps `046_04_external_auto_n_summary.csv` so the existing
`046_05` and `046_06` reporting scripts can reuse this branch by selecting the
new `external_auto_n_rule.output_suffix`.

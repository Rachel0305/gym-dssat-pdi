# 009_04 HLA 2004 joint PPO nitrogen cap tightening probe

## Purpose

Continue from 009_03. Do not broaden the experiment.

009_03 showed that lowering `water_cost` from 0.050 to 0.030 restored irrigation and yield, while increasing `nitrogen_cost` from 0.030 to 0.050 did not reduce nitrogen use. Both 009_03A and 009_03B reached about 190 kg/ha total N, but their GRNWT differed by only about 0.3%. This suggests that the 190 kg/ha nitrogen level is already on a yield plateau.

The goal of 009_04 is to test whether tightening PPO-controlled supplemental N caps can reduce ineffective high N input while preserving yield.

## Constraints

- Do not train unrestricted daily PPO.
- Do not run multi-seed.
- Do not expand to additional stations or years.
- Do not modify `my_data/`.
- Do not modify or overwrite original reward files in `site-packages`.
- Do not overwrite 008 or 009_02/009_03 outputs.
- Keep the run small and serial to reduce OOM risk.

## Experimental design

Use HLA 2004, seed 0, 5000 timesteps.

Inherit from the 009_02 / 009_03 framework:

- stage-level water-nitrogen PPO;
- no daily continuous action;
- no hard minimum irrigation;
- forecast/stress gate;
- soft SWFAC penalty;
- soft NSTRES penalty;
- S1/S2 fixed base N;
- PPO controls irrigation and supplemental N only within stage caps.

Use the successful water price from 009_03A:

```yaml
water_cost: 0.030
nitrogen_cost: 0.030
```

Change only supplemental nitrogen caps:

```text
S1 fixed N = 50 kg/ha, PPO extra N = 0
S2 fixed N = 50 kg/ha, PPO extra N = 0
S3 PPO extra N cap = 40 kg/ha
S4 PPO extra N cap = 10 kg/ha
S5 PPO extra N cap = 0 kg/ha
```

This sets the total N upper bound to about 150 kg/ha, matching the previous fixed `I120_N150` reference that achieved similar yield to the 190 kg/ha runs.

## Success criteria

009_04 is considered successful if:

- total N is around 130-160 kg/ha;
- total irrigation stays around 70-90 mm;
- GRNWT is not lower than 6900 kg/ha;
- N cap saturation is avoided, or if the new cap is reached, yield remains close to 009_03A and the report clearly states that the new cap is still binding;
- irrigation does not collapse back to the 009_02 low-irrigation pattern;
- no S1 irrigation occurs.

## Outputs

Save:

- rendered config;
- model;
- daily CSV;
- stage-decision CSV;
- summary CSV;
- process figure;
- Markdown report under `docs/`.

The report must compare 009_04 with:

- 009_02 baseline;
- 009_03A lower water cost;
- 009_03B higher nitrogen cost;
- fixed `I120_N150` reference where available.

The report must explicitly answer:

1. Did tightening N caps reduce total N?
2. Did GRNWT remain close to the 190 kg/ha runs?
3. Did irrigation remain in the recovered 70-90 mm range?
4. Is the next step more cap tightening, longer training, or seed stability?

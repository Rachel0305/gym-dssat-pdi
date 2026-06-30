# 011_11 HLA 2004 updated-cultivar native automatic fertilizer rescue

## Purpose

Rescue DSSAT native automatic fertilization (`FERTI=A`) after updating HLA cultivar parameters and moving the project to IC=1.

This is a low-cost diagnostic only:

- no PPO training;
- no long multi-year scan;
- HLA 2004 only;
- base input copied from the Windows DSSAT 4.8.0 IC=1 auto-irrigation run.

## Input basis

Base MZX:

```text
DSSAT_auto_validation/HLA_2004/run_CNHL0408_DSSAT480_2004/CNHL0408.MZX
```

Auxiliary files:

```text
DSSAT_auto_validation/HLA_2004/CNHL0401.WTH
DSSAT_auto_validation/HLA_2004/run_CNHL0404/SOIL.SOL
DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL
```

Runtime:

```text
Docker container: b2fd6726c8c1
Python: /opt/gym_dssat_pdi/bin/python
DSSAT/PDI: /opt/dssat_pdi/run_dssat
```

## Command

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2004_auto_fertilizer_rescue_011_11.py"
```

## Variants

Variants changed only native management settings around automatic nitrogen:

- `FERTI=R` auto-irrigation-only process control;
- `FERTI=A` with `FE001` and `FE005`;
- `NMTHR=50` and aggressive `NMTHR=99`;
- `NAMNT=25` and `NAMNT=50`;
- `NAOFF=GS000` and `NAOFF=GS999`;
- with `IRRIG=A` and no-irrigation `IRRIG=N` variants.
- forced low/zero initial mineral N variants, with all initial-condition `SNH4/SNO3` layer values set to `0.0` or `0.1`.

## Output

```text
DSSAT_auto_validation/HLA_2004/auto_fertilizer_rescue_2004_newcul_011_11
```

Main table:

```text
DSSAT_auto_validation/HLA_2004/auto_fertilizer_rescue_2004_newcul_011_11/summary.csv
```

## Result

No variant triggered native automatic fertilization.

Key rows:

| Variant group | Irrigation | Fertilizer events | Fertilizer amount | Yield response |
| --- | ---: | ---: | ---: | --- |
| Auto-irrigation only, `FERTI=R` | 1132.8 mm | 0 | 0 kg/ha | HWAM 9636 |
| `FERTI=A`, FE001/FE005, GS000/GS999, NMTHR 50/99 | 1132.8 mm | 0 | 0 kg/ha | HWAM 9636 |
| `IRRIG=N`, `FERTI=A`, FE001/FE005, NMTHR 99 | 0 mm | 0 | 0 kg/ha | HWAM 0 |
| `IRRIG=A`, `FERTI=A`, initial SNH4/SNO3 = 0.0 | 422.1 mm | 0 | 0 kg/ha | HWAM 374 |
| `IRRIG=A`, `FERTI=A`, initial SNH4/SNO3 = 0.1 | 424.2 mm | 0 | 0 kg/ha | HWAM 495 |

The zero/low initial-N variants produced strong nitrogen stress (`max_nstd=0.799`) and severe yield loss, but native auto-N still did not fire.

## Interpretation

The updated cultivar and IC=1 setup did not rescue native DSSAT automatic nitrogen.

This diagnostic further rules out the most obvious input-side explanations:

1. material code (`FE001` vs `FE005`);
2. threshold (`NMTHR=50` vs `NMTHR=99`);
3. amount per application (`NAMNT=25` vs `NAMNT=50`);
4. cutoff growth stage (`GS000` vs `GS999`);
5. whether automatic irrigation is enabled.
6. whether initial mineral N is too high.

Native automatic irrigation does trigger, and triggers very strongly, so the automatic-management block is being parsed. The failure is specific to the native automatic nitrogen trigger path.

## Practical conclusion

For this project, DSSAT native `FERTI=A` should not be used as a validated rule baseline unless a later Windows/XBuild manual test discovers a missing GUI-specific setting. For the current PPO work, use externally defined fertilizer rules or PPO linked actions instead.

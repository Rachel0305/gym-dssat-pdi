# Soil Water and Nitrogen Initialization Summary

Date: 2026-06-04

## Purpose

This note records the updated DSSAT/gym-DSSAT initial soil water and mineral nitrogen settings for the five maize stations: Hailun (HL), Shenyang (SY), Luancheng (LC), Yucheng (YC), and Fengqiu (FQ).

The goal is to avoid unrealistically wet initial soil water conditions and to remove missing mineral nitrogen values (`-99`) from nitrogen-enabled simulations while preserving measured or record-based values where available.

## Soil Water Formula

Initial volumetric soil water content was set by layer using:

```text
SH2O = SLLL + 0.55 * (SDUL - SLLL)
```

This uses each layer's plant-available soil water range (`SDUL - SLLL`) and sets initial water at a medium to slightly dry level. This is a single adopted initialization setting, not a sensitivity test.

## Final Initial Conditions

### Hailun HL

Mineral N values are record-based and were preserved.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.26  22.7  10.1
 1    40  0.26  10.8  11.3
 1    60  0.26  15.1  12.5
 1    90  0.25   8.2   8.1
```

### Shenyang SY

The 10-60 cm mineral N values are record-based. The 100 cm layer was missing and was filled with conservative estimated values based on the downward profile trend.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    10  0.27  4.57  2.60
 1    20  0.24  4.55  2.62
 1    30  0.24  4.07  2.54
 1    40  0.24  4.05  2.52
 1    60  0.23  3.82  2.00
 1   100  0.25  3.50  1.50
```

Estimated values: `SNH4=3.50`, `SNO3=1.50` at 100 cm.

### Luancheng LC

Mineral N values are record-based and were preserved.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.22   4.0   5.3
 1    40  0.25   4.0   4.5
 1   110  0.26   4.0   6.2
 1   150  0.25   4.0  14.6
```

### Yucheng YC

Mineral N values are record-based and were preserved.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    15  0.22   4.0  10.0
 1    30  0.25   4.0   8.0
 1    60  0.26   4.0   6.0
 1    90  0.25   4.0   5.0
```

### Fengqiu FQ

All mineral N values were missing in the original target block and were filled with conservative estimated values. These are estimates, not measurements.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    30  0.17   4.0   5.0
 1    70  0.28   4.0   6.0
 1   100  0.27   4.0   8.0
```

Estimated values: all `SNH4` and `SNO3` values for FQ.

## Mineral Nitrogen Rules

- Preserve measured or record-based `SNH4` and `SNO3` values.
- Fill missing `-99` values only when needed for nitrogen-enabled DSSAT/gym-DSSAT simulation.
- Label filled values as estimated values.
- Use conservative one- or two-decimal estimates rather than highly precise invented values.
- Use nearby station profiles, vertical profile logic, and North China Plain nitrogen literature as broad support.

## Files Modified

```text
my_data/UFGA8201-HL.jinja2
my_data/UFGA8201-SY.jinja2
my_data/UFGA8201-LC.jinja2
my_data/UFGA8201-YC.jinja2
my_data/UFGA8201-FQ.jinja2
```

The previous current versions were backed up before this update:

```text
backups/2026-06-04_soil_water_nitrogen_initialization/
```

## References

- DSSAT User's Guide, Volume 4. https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol4.pdf
- DSSAT Foundation. Soil water balance overview. https://dssat.net/soil-water/
- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration: Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56. https://www.fao.org/4/x0490e/x0490e0e.htm
- Jones, J. W., et al. DSSAT Cropping System Model. https://abe.ufl.edu/Faculty/jjones/ABE_5646/Week%207/The%20DSSAT%20Cropping%20System%20Model.pdf
- Cui et al. Soil nitrate-N levels required for high yield maize production in the North China Plain. https://www.researchgate.net/publication/226611965_Soil_nitrate-N_levels_required_for_high_yield_maize_production_in_the_North_China_Plain
- Liu et al. Nitrogen dynamics in a winter wheat-maize system in the North China Plain. https://www.sciencedirect.com/science/article/abs/pii/S0378429003000686
- Cui et al. Current nitrogen management in China. https://pmc.ncbi.nlm.nih.gov/articles/PMC3357710/

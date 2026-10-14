# HLA cultivar calibration input package

Purpose: Hailun cultivar calibration with PDI/gym-DSSAT 4.8.0.024, using 2007/2009/2011 observed management and observations.

Files:

- `CNHL0701_corrected_IC123.MZX`: corrected three-treatment file for 2007, 2009, 2011; cultivar `HY0006`; management `IRRIG=R, FERTI=R`; IC levels 1/2/3 match years.
- `CNHL0701.MZA`: observed ADAP, MDAP, CWAM, HWAM.
- `CNHL0701.MZT`: observed LAI/CWAD time-course, retained for later inspection but not used in first-pass score.
- `CNHL0701.WTH`, `CNHL0901.WTH`, `CNHL1101.WTH`: year-specific weather files.
- `SOIL.SOL`: soil file containing `HL99001200`.
- `MZCER048.CUL`: cultivar file; calibration patches only the `HY0006` line in per-run copies.

Observed targets from CNHL0701.MZA:

trno,obs_adap,obs_mdap,obs_cwam,obs_hwam
1,82,142,16814,7420.0
2,85,149,12836,8835.0
3,80,138,16110,5106.0


Important: no original input file is modified during calibration; each candidate gets a copied `MZCER048.CUL`.
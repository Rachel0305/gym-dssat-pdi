# Corrected HLA 2007/2009/2011 cultivar calibration input package

This folder is a calibration-only corrected copy. The original `input_review_package/CNHL0701.MZX` was not overwritten.

Corrections applied:

1. Treatment IC pointers are year-specific: 2007 -> IC=1, 2009 -> IC=2, 2011 -> IC=3.
2. The same initial soil water/N profile is duplicated into IC levels 1/2/3, with ICDAT = 07121, 09121, 11121 respectively.
3. Treatment 3 simulation start date was changed from SDATE=07121 to SDATE=11121.
4. Management, planting, irrigation, fertilization, soil, weather and cultivar values were otherwise preserved.

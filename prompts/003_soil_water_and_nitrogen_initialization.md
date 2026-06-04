# Codex Prompt: Soil Water and Mineral Nitrogen Initialization for Five DSSAT/gym-DSSAT Stations

## Background

This project uses gym-DSSAT / DSSAT-style Jinja2 templates for maize water and nitrogen management experiments at five Chinese stations:

- Hailun: `UFGA8201-HL.jinja2`
- Shenyang: `UFGA8201-SY.jinja2`
- Luancheng: `UFGA8201-LC.jinja2`
- Yucheng: `UFGA8201-YC.jinja2`
- Fengqiu: `UFGA8201-FQ.jinja2`

The original Jinja2 template files have already been backed up to GitHub, so the existing working files may be edited directly. However, do **not** delete, rename, or overwrite unrelated files.

The goal is to replace unrealistic or missing initial soil water and mineral nitrogen values in the `*INITIAL CONDITIONS` section, and generate a PowerPoint report documenting the analysis process, assumptions, literature support, and final values.

---

## Required Outputs

Create the following outputs:

1. A PowerPoint report documenting the analysis and final initialization values.
2. Updated Jinja2 templates for all five stations.
3. A short Markdown summary file in `docs/` recording what was changed.

Use the following directory convention:

```text
prompts/    # Markdown prompts for Codex tasks
           # Use sequential names such as 001_xxx.md, 002_xxx.md

docs/       # Project notes, analysis records, experiment design, troubleshooting records
           # Use lowercase English names with underscores
```

Save the report and summary using clear names, for example:

```text
docs/soil_water_and_nitrogen_initialization_summary.md
outputs/soil_water_and_nitrogen_initialization_report.pptx
```

If `outputs/` does not exist, create it. Do not use `outputs/` for cleaned weather data; cleaned weather data belongs in `weather_clean/`.

---

## Safety Rules

Follow these safety rules strictly:

1. Do not delete any project files.
2. Do not run destructive commands such as `rm -rf`, disk formatting commands, or commands that modify files outside the project directory.
3. Before editing each Jinja2 template, read the file and confirm it contains the expected `*INITIAL CONDITIONS` section.
4. Make only the required replacements in the `*INITIAL CONDITIONS` section.
5. Preserve all non-target sections, spacing style, dates, treatment metadata, cultivar information, weather station information, planting details, fertilizer management, irrigation settings, and simulation controls.
6. After editing, verify that all five files still contain valid DSSAT-style `*INITIAL CONDITIONS` blocks.
7. If a target template file is missing, stop and report the missing file instead of guessing.

---

## Scientific Basis

### DSSAT initial conditions

The DSSAT experiment file `*INITIAL CONDITIONS` section defines initial soil water and mineral nitrogen conditions by soil layer:

```text
@C  ICBL  SH2O  SNH4  SNO3
```

- `ICBL` = bottom depth of the soil layer, cm.
- `SH2O` = initial volumetric soil water content.
- `SNH4` = initial ammonium-N concentration for the soil layer.
- `SNO3` = initial nitrate-N concentration for the soil layer.

DSSAT examples use the same `@C ICBL SH2O SNH4 SNO3` format in the `*INITIAL CONDITIONS` section. The DSSAT manual and example experiment files show that initial conditions should be specified by layer, rather than using one universal value for all soil types.

### Soil water initialization method

Do not set all stations to `SH2O = 0.30`. The same absolute volumetric water content can represent very different water status in different soils.

Use each station's soil hydraulic parameters:

- `SLLL`: soil lower limit, close to crop wilting lower limit.
- `SDUL`: drained upper limit, approximately field capacity.
- `SSAT`: saturated soil water content.

Use one fixed, literature-supported, non-sensitivity-test setting:

```text
SH2O = SLLL + 0.55 × (SDUL - SLLL)
```

Rationale:

- `SDUL - SLLL` represents the layer's plant-available soil water range.
- A 0.55 fraction represents medium to slightly dry initial soil water status.
- This avoids unrealistically wet initial conditions while not imposing an extreme drought scenario.
- This is a single parameter setting, not a sensitivity test.

### Mineral nitrogen initialization method

The existing measured or record-based `SNO3` and `SNH4` values should be preserved when available.

Do **not** replace measured values with literature averages.

Use the following rules:

1. For layers with measured or experiment-record values, keep the original `SNH4` and `SNO3`.
2. For missing `-99` values, fill them only when necessary for nitrogen-enabled DSSAT/gym-DSSAT simulation.
3. Missing values should be filled with conservative, clearly documented estimates.
4. Filled values must be labeled in the PPT and docs as **estimated values**, not measured values.
5. Use nearby station behavior, vertical profile logic, and literature on North China Plain maize soil mineral N as support.
6. Do not invent highly precise values; use simple values with one or two decimals.

Important interpretation:

- The Luancheng and Yucheng values are record-based and plausible for North China Plain maize systems.
- Hailun has high ammonium-N, especially in the top layer. This is high but should be preserved because the user provided it from site records / soil data.
- Shenyang has missing deep-layer mineral N at 100 cm. Fill it conservatively by extrapolating downward from the 60 cm layer.
- Fengqiu has all mineral N values missing. Fill with conservative North China Plain / nearby-station estimates, and label as estimated.

---

## Literature Support to Cite in the PPT

Use these references in the PPT and summary document. Cite them on relevant slides, not only at the end.

1. DSSAT User's Guide / example experiment files show the `*INITIAL CONDITIONS` format with `@C ICBL SH2O SNH4 SNO3` specified by soil layer.  
   URL: https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol4.pdf

2. DSSAT soil water documentation explains the soil water balance and the role of rainfall, irrigation, runoff, drainage, evaporation, transpiration, and soil profile water redistribution.  
   URL: https://dssat.net/soil-water/

3. FAO-56 explains that total available soil water in the root zone is based on the difference between field capacity and wilting point. This supports using `SDUL - SLLL` as the plant-available water range.  
   URL: https://www.fao.org/4/x0490e/x0490e0e.htm

4. Jones et al. DSSAT Cropping System Model paper describes DSSAT as a cropping system model using soil, weather, management, and initial conditions to simulate crop growth and water/nitrogen dynamics.  
   URL: https://abe.ufl.edu/Faculty/jjones/ABE_5646/Week%207/The%20DSSAT%20Cropping%20System%20Model.pdf

5. Cui et al. / North China Plain maize nitrate research reports that soil nitrate-N in the top 90 cm is important for high-yield maize production, and gives a reference range around 87–180 kg N ha⁻¹ for high yield maize in the North China Plain. Use this only as broad support for plausibility, not as a direct replacement for measured layer values.  
   URL: https://www.researchgate.net/publication/226611965_Soil_nitrate-N_levels_required_for_high_yield_maize_production_in_the_North_China_Plain

6. Liu et al. nitrogen dynamics in a winter wheat–maize system in the North China Plain found that ammonium-N in the soil profile generally remained low and relatively constant except after fertilizer application or in the surface layer. This supports using conservative NH4-N estimates for missing values.  
   URL: https://www.sciencedirect.com/science/article/abs/pii/S0378429003000686

7. Cui et al. current nitrogen management in China reports high nitrogen inputs and soil inorganic nitrogen concerns in intensive wheat–maize systems. Use this as general background for why initial mineral N matters.  
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3357710/

---

## Soil Files and Final Soil Water Values

The final `SH2O` values below are calculated using:

```text
SH2O = SLLL + 0.55 × (SDUL - SLLL)
```

Rounded to two decimals for DSSAT template readability.

### Hailun HL

Soil hydraulic parameters:

```text
ICBL  SLLL  SDUL
20    0.11  0.39
40    0.11  0.38
60    0.11  0.38
90    0.11  0.36
```

Final water values:

```text
20 cm: 0.26
40 cm: 0.26
60 cm: 0.26
90 cm: 0.25
```

### Shenyang SY

Soil hydraulic parameters:

```text
ICBL  SLLL  SDUL
10    0.14  0.37
20    0.15  0.32
30    0.16  0.31
40    0.14  0.32
60    0.13  0.31
100   0.13  0.35
```

Final water values:

```text
10 cm: 0.27
20 cm: 0.24
30 cm: 0.24
40 cm: 0.24
60 cm: 0.23
100 cm: 0.25
```

### Luancheng LC

Soil hydraulic parameters:

```text
ICBL  SLLL  SDUL
20    0.09  0.33
40    0.11  0.36
110   0.12  0.37
150   0.07  0.40
```

Final water values:

```text
20 cm: 0.22
40 cm: 0.25
110 cm: 0.26
150 cm: 0.25
```

### Yucheng YC

Soil hydraulic parameters:

```text
ICBL  SLLL  SDUL
15    0.09  0.33
30    0.11  0.36
60    0.12  0.37
90    0.07  0.40
```

Final water values:

```text
15 cm: 0.22
30 cm: 0.25
60 cm: 0.26
90 cm: 0.25
```

### Fengqiu FQ

Soil hydraulic parameters:

```text
ICBL  SLLL  SDUL
30    0.12  0.21
70    0.24  0.31
100   0.15  0.36
```

Final water values:

```text
30 cm: 0.17
70 cm: 0.28
100 cm: 0.27
```

Important note for Fengqiu:

The previous value `SH2O = 0.30` in the 0–30 cm layer was higher than `SDUL = 0.21`, so it was unrealistically wet for the purpose of evaluating water stress.

---

## Final Initial Mineral Nitrogen Values

### Rule

Keep all user-provided measured / record-based `SNH4` and `SNO3` values. Fill only missing `-99` values.

### Hailun HL

Keep the original mineral N values.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.26  22.7  10.1
 1    40  0.26  10.8  11.3
 1    60  0.26  15.1  12.5
 1    90  0.25   8.2   8.1
```

Notes:

- Hailun ammonium-N is relatively high, especially at 20 cm.
- Since these values come from the provided site records / soil data, keep them.
- Do not replace them with literature averages.

### Shenyang SY

Keep measured values for 10–60 cm. Fill the missing 100 cm values conservatively by downward extrapolation.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    10  0.27  4.57  2.60
 1    20  0.24  4.55  2.62
 1    30  0.24  4.07  2.54
 1    40  0.24  4.05  2.52
 1    60  0.23  3.82  2.00
 1   100  0.25  3.50  1.50
```

Notes:

- The 100 cm `SNH4=3.50` and `SNO3=1.50` are estimated values.
- They continue the observed downward trend from the shallower layers.
- They are conservative and avoid using `-99` in nitrogen-enabled simulation.

### Luancheng LC

Keep the original mineral N values.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.22   4.0   5.3
 1    40  0.25   4.0   4.5
 1   110  0.26   4.0   6.2
 1   150  0.25   4.0  14.6
```

Notes:

- The deeper nitrate value is high but plausible in North China Plain systems because nitrate can accumulate and move downward.
- Keep these values because they are record-based.

### Yucheng YC

Keep the original mineral N values.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    15  0.22   4.0  10.0
 1    30  0.25   4.0   8.0
 1    60  0.26   4.0   6.0
 1    90  0.25   4.0   5.0
```

Notes:

- The values are reasonable for a North China Plain maize system.
- Keep them because they are record-based.

### Fengqiu FQ

The original values are missing. Fill with conservative estimated values.

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    30  0.17   4.0   5.0
 1    70  0.28   4.0   6.0
 1   100  0.27   4.0   8.0
```

Notes:

- All Fengqiu `SNH4` and `SNO3` values are estimated.
- Use `SNH4 = 4.0` as a conservative, stable ammonium-N estimate, consistent with Luancheng and Yucheng record-based values.
- Use a moderate nitrate profile: `SNO3 = 5.0 / 6.0 / 8.0`.
- This avoids leaving `-99` while staying within a conservative North China Plain range.

---

## Template Files to Modify

Modify these files directly:

```text
UFGA8201-HL.jinja2
UFGA8201-SY.jinja2
UFGA8201-LC.jinja2
UFGA8201-YC.jinja2
UFGA8201-FQ.jinja2
```

The exact directory may be the project root or a data folder such as `my_data/`. Search for the exact filenames inside the project directory if necessary.

Only update the `*INITIAL CONDITIONS` block in each file.

---

## PowerPoint Requirements

Generate a PPT report with clear slides. Suggested slide structure:

### Slide 1: Title

Title:

```text
Initial Soil Water and Mineral Nitrogen Settings for Five DSSAT/gym-DSSAT Stations
```

Include date and project name if available.

### Slide 2: Problem

Explain:

- Previous initial soil water often used `SH2O = 0.30`.
- A fixed `0.30` has different meanings across soils.
- Some sites had missing mineral N values as `-99`.
- The goal is to set one reliable, literature-supported initialization scheme, not run a sensitivity test.

### Slide 3: DSSAT Initial Conditions

Show the DSSAT block:

```text
@C  ICBL  SH2O  SNH4  SNO3
```

Explain each variable.

Cite DSSAT User's Guide.

### Slide 4: Soil Water Method

Show:

```text
SH2O = SLLL + 0.55 × (SDUL - SLLL)
```

Explain:

- `SLLL` = lower limit.
- `SDUL` = drained upper limit / field capacity approximation.
- `SDUL - SLLL` = plant-available water range.
- `0.55` = medium to slightly dry initial condition.

Cite FAO-56 and DSSAT soil water documentation.

### Slide 5: Why Fixed 0.30 Was Problematic

Use examples:

- Hailun `0.30` was medium-wet but below `SDUL`.
- Shenyang `0.30` was very close to `SDUL` in several layers.
- Fengqiu 0–30 cm `0.30` was higher than `SDUL = 0.21`.

### Slide 6: Final Soil Water Values

Include a table:

```text
Station | Layer bottom cm | Final SH2O
HL      | 20,40,60,90      | 0.26,0.26,0.26,0.25
SY      | 10,20,30,40,60,100 | 0.27,0.24,0.24,0.24,0.23,0.25
LC      | 20,40,110,150    | 0.22,0.25,0.26,0.25
YC      | 15,30,60,90      | 0.22,0.25,0.26,0.25
FQ      | 30,70,100        | 0.17,0.28,0.27
```

### Slide 7: Mineral Nitrogen Method

Explain:

- Keep record-based `SNH4` and `SNO3` values.
- Fill only missing `-99` values.
- Estimated values must be documented as estimated.
- Do not replace measured values with literature averages.

Cite DSSAT manual and nitrogen management literature.

### Slide 8: Current Mineral N Assessment

Include brief assessment:

```text
HL: high NH4, but record-based; keep.
SY: 100 cm missing; fill conservatively by downward extrapolation.
LC: record-based; keep.
YC: record-based; keep.
FQ: all mineral N missing; fill conservatively using nearby-station/NCP logic.
```

### Slide 9: Final Initial Conditions by Station

Show final `@C ICBL SH2O SNH4 SNO3` table for each station. Use compact formatting.

### Slide 10: Notes and Limitations

State clearly:

- Filled `-99` values are estimates, not measurements.
- The selected water fraction is a single adopted setting, not a sensitivity test.
- The updated templates should be checked by running a short DSSAT/gym-DSSAT simulation and inspecting `SWFAC`, `NSTRES`, `TOPWT`, and `GRNWT`.

### Slide 11: References

List all references with URLs.

---

## Markdown Summary Requirements

Create:

```text
docs/soil_water_and_nitrogen_initialization_summary.md
```

Include:

1. Purpose.
2. Formula used for soil water.
3. Final values by station.
4. Rules used for mineral N.
5. Which values were measured / record-based and which were estimated.
6. Files modified.
7. References.

---

## Verification Checklist

After editing, verify:

1. Each target Jinja2 file exists.
2. Each file has one updated `*INITIAL CONDITIONS` block.
3. No `SH2O = 0.30` remains in the target initial conditions unless it occurs outside the target block or is intentionally part of another unrelated section.
4. No `SNH4` or `SNO3` value remains as `-99` in the five target initial condition blocks.
5. All final values match the tables above.
6. PPT file exists and includes references.
7. Markdown summary file exists.

---

## Final Values to Apply Exactly

### `UFGA8201-HL.jinja2`

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.26  22.7  10.1
 1    40  0.26  10.8  11.3
 1    60  0.26  15.1  12.5
 1    90  0.25   8.2   8.1
```

### `UFGA8201-SY.jinja2`

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    10  0.27  4.57  2.60
 1    20  0.24  4.55  2.62
 1    30  0.24  4.07  2.54
 1    40  0.24  4.05  2.52
 1    60  0.23  3.82  2.00
 1   100  0.25  3.50  1.50
```

### `UFGA8201-LC.jinja2`

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.22   4.0   5.3
 1    40  0.25   4.0   4.5
 1   110  0.26   4.0   6.2
 1   150  0.25   4.0  14.6
```

### `UFGA8201-YC.jinja2`

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    15  0.22   4.0  10.0
 1    30  0.25   4.0   8.0
 1    60  0.26   4.0   6.0
 1    90  0.25   4.0   5.0
```

### `UFGA8201-FQ.jinja2`

```text
@C  ICBL  SH2O  SNH4  SNO3
 1    30  0.17   4.0   5.0
 1    70  0.28   4.0   6.0
 1   100  0.27   4.0   8.0
```

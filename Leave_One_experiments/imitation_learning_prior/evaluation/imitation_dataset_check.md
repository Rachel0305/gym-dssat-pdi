# Imitation Dataset Check

Generated at: 2026-06-06

- Original imitation_dataset.csv exists: True
- Original rows: 477
- Original stations: HLA
- Rebuilt rows from best HLA/SYA/LCA daily outputs: 1300
- Best schedules: {'HLA': 'HLA2011_S0157', 'SYA': 'SYA2012_S0443', 'LCA': 'LCA2010_S0313'}
- State variables used: station, sim_day, doy, topwt, grnwt, xlai, totir, tofer, swfac, nstres
- Targets used: expert_action_irrigation, expert_action_n
- NaN cells after cleaning: 0
- Inf cells after cleaning: 0
- Irrigation nonzero samples: 0
- Nitrogen nonzero samples: 14
- Zero-action ratio: 0.9892
- Conclusion: actions are extremely sparse; two-stage classifier-regressor is trained in addition to regression baselines.

## Rows by Station-Year

| station | year | rows |
| --- | --- | --- |
| HLA | 2007 | 149 |
| HLA | 2009 | 159 |
| HLA | 2011 | 169 |
| LCA | 2008 | 105 |
| LCA | 2009 | 102 |
| LCA | 2010 | 92 |
| LCA | 2011 | 102 |
| SYA | 2012 | 135 |
| SYA | 2014 | 144 |
| SYA | 2015 | 143 |

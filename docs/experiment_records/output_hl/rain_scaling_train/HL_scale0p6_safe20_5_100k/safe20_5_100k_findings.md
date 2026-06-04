# HL scale0.6 safe20_5 100k follow-up

Date: 2026-06-01

Configuration:
- daily fertilizer cap: 20 kg/ha
- daily irrigation cap: 5 mm
- nitrogen penalty: 20
- irrigation cost: 20
- no-water-stress irrigation cost: 80
- fertilizer excess limit/cost: 180 / 20
- irrigation excess limit/cost: 60 / 40
- training length: 100,000 timesteps

Diagnostic result:
- PPO yield: 7457 kg/ha
- PPO total fertilizer: 1350 kg/ha
- PPO total irrigation: 351 mm
- Expert yield: 6276 kg/ha
- Expert total fertilizer: 165 kg/ha
- Expert total irrigation: 30 mm

Interpretation:
- The 100k run produced almost the same policy as the 50k run.
- This suggests the remaining problem is not simply insufficient training length.
- PPO needs a seasonal budget or stronger action feasibility mechanism so that it learns timing decisions rather than frequent small daily applications.

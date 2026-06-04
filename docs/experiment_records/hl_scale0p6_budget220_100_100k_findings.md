# HL scale0.6 budget220_100 100k confirmation

Date: 2026-06-02

Configuration:
- daily fertilizer cap: 20 kg/ha
- daily irrigation cap: 5 mm
- seasonal fertilizer budget: 220 kg/ha
- seasonal irrigation budget: 100 mm
- nitrogen penalty: 20
- irrigation cost: 20
- no-water-stress irrigation cost: 80
- training length: 100,000 timesteps

Diagnostic result:
- PPO yield: 6669 kg/ha
- PPO total fertilizer: 202 kg/ha
- PPO total irrigation: 92 mm
- Expert yield: 6276 kg/ha
- Expert total fertilizer: 165 kg/ha
- Expert total irrigation: 30 mm

Interpretation:
- The 100k result is close to the 50k result for the same budget setting.
- This confirms that the seasonal-budget wrapper produces a stable and plausible all-mode policy.
- The policy improves yield over the expert baseline while using moderately more fertilizer and irrigation.
- This is the current best configuration for demonstrating a complete water-nitrogen joint optimization workflow under the rainfall-scaled Hailun drought scenario.

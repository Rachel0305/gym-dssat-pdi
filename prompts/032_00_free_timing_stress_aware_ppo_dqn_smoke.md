# 032_00 free-timing stress-aware PPO/DQN smoke

## Purpose

Test a minimal literature-motivated free-timing RL variant after the advisor rejected fixed expert-DAP decision windows.

This task is not a full all-site claim. It is a single-site-year smoke designed to answer one narrow question:

> Can coarse action levels plus a small stress-relief process reward reduce the previously observed free-timing failure modes, especially early budget dumping and frequent 6 mm micro-irrigation?

## Background frozen from 031

- 031 free-timing PPO/DQN removed expert-DAP windows.
- Daily fully free action with fine irrigation level 6 mm produced many repeated small irrigation events, especially in LC years.
- LC2010 current PPO candidate used 13 irrigation events of 6 mm each, which is agronomically suspicious because each irrigation event has real operational cost.
- 031_41 showed that adding fixed operation-event costs can suppress repeated irrigation, but the tested coefficients over-penalized water use and are not ready for all-site expansion.
- Literature checks suggest free crop RL normally includes agricultural constraints such as weekly/low-frequency decisions, single-event upper bounds, late-stage fertilization limits, operation costs, or process/constraint signals.

## Scope

- Site-year: LCA2010 only.
- Algorithms: MaskablePPO and DQN.
- Seed: 0 only.
- Training budget: 5,000 timesteps per algorithm.
- No all-year expansion.
- No parameter scan.
- No checkpoint cherry-picking: evaluate the final 5,000-step model only.

## Fixed action/constraint design

This task keeps the agent free from expert DAP windows, but limits actions to agronomically coarser levels:

- Irrigation levels: `{0, 15, 30, 45}` mm.
- Nitrogen levels: `{0, 40, 80, 120}` kg/ha.
- Seasonal irrigation cap: 160 mm.
- Seasonal nitrogen cap: 250 kg/ha.
- Minimum days between irrigation events: 7.
- Minimum days between fertilization events: 7.
- Irrigation allowed DAP range: 1-120.
- Fertilization allowed DAP range: 1-90.

These constraints are not expert-DAP windows. They define feasible operational boundaries while leaving timing choices free.

## Reward v2-smoke

Base reward remains the literature-style harvest reward:

```text
unscaled_reward =
    0.158 * final_grnwt_at_harvest
    - 1.1 * irrigation_mm
    - 1.58 * nitrogen_kg
    + stress_relief_bonus

reward = unscaled_reward * 0.001
```

Stress-relief term:

```text
stress_relief_bonus =
    10.0 * irrigation_mm * max(previous_SWFAC - current_SWFAC, 0)
    + 5.0 * nitrogen_kg * max(previous_NSTRES - current_NSTRES, 0)
```

Interpretation:

- If an operation actually reduces a water/nitrogen stress indicator, it receives a small process credit.
- If there is no stress relief, the operation still only receives resource cost.
- This is an exploratory smoke coefficient set, not a final tuned reward.

## Success/failure interpretation

Do not judge success by final yield alone.

Primary behavior checks:

- No 6 mm micro-irrigation is possible by construction.
- No DAP1-DAP10 full budget dump.
- Irrigation event count is not excessive.
- Fertilization event count is not excessive.
- Water/N operations should be more interpretable relative to stress/weather than 031 fine-grid PPO.

Metric checks:

- Compare final grain yield, total irrigation, total N, simple return, WP/PFP proxies where available.
- Compare PPO and DQN only as a smoke; do not claim algorithm superiority from one seed.

Stop rules:

- If PPO and DQN both still dump resources early or produce implausible actions, do not expand to all sites.
- If one algorithm becomes plausible, next step is multi-seed repeat on the same site-year before all-site expansion.
- Do not modify reward coefficients in-place after seeing results.


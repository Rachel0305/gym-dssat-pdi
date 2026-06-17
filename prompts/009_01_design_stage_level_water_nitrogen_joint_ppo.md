# 009_01 Design Stage-Level Water-Nitrogen Joint PPO

## Goal

Design the next-stage PPO framework that reintroduces nitrogen decisions while preserving the successful 008 soft-stress stage irrigation framework.

The purpose of this task is design and preparation only.

Do not train PPO in 009_01.

## Background

The successful 008 framework showed that PPO can produce interpretable irrigation decisions in water-stress years when:

- station-years are screened for real water-stress potential;
- PPO acts at DAP stages instead of daily;
- S1 irrigation is blocked;
- forecast/stress information gates irrigation opportunity;
- hard minimum irrigation is removed;
- soft SWFAC penalty makes water stress visible in reward;
- nitrogen is fixed at N150.

The main scientific limitation is that the current successful PPO cases do not yet perform true water-nitrogen joint optimization because nitrogen is fixed.

## Design Principle

Do not return to unrestricted daily water-nitrogen PPO.

Use:

- DAP-stage decision wrapper;
- agronomic base nitrogen;
- PPO-controlled supplemental nitrogen;
- forecast/stress-assisted irrigation gate without hard minimum;
- soft SWFAC and NSTRES penalties;
- explicit logging of water and nitrogen action contribution.

## Proposed Stage Action Design

Use five DAP stages:

| Stage | DAP | Irrigation design | Nitrogen design |
|---|---:|---|---|
| S1 establishment | 1-20 | blocked, cap 0 mm | fixed base N 50 kg ha-1, PPO extra 0 |
| S2 early vegetative | 21-45 | PPO 0-40 mm if gate allows | fixed base N 50 kg ha-1, PPO extra 0 |
| S3 mid season | 46-75 | PPO 0-40 mm if gate allows | PPO supplemental N 0-50 kg ha-1 |
| S4 late water-risk | 76-100 | PPO 0-40 mm if gate allows | PPO supplemental N 0-20 kg ha-1 |
| S5 terminal water-risk | 101-end | PPO 0-40 mm if gate allows | PPO supplemental N 0-20 kg ha-1 |

Total nitrogen range:

- minimum: 100 kg ha-1
- maximum: 190 kg ha-1

This means nitrogen is no longer fixed, but early agronomic nitrogen is protected from PPO collapse.

## Reward Design For 009_02 Smoke Test

Start from the 008_17 stronger soft-stress configuration:

- topwt_delta_coef = 0.001
- grnwt_delta_coef = 0.020
- terminal_grnwt_coef = 0.010
- water_cost = 0.050
- soft SWFAC threshold = 0.05
- swfac_excess_cost = 3.0
- swfac_day_cost = 1.0

Add a small nitrogen cost only for PPO-controlled nitrogen:

- nitrogen_cost = 0.020 to 0.050 in smoke test candidates

For the first smoke test, use:

- nitrogen_cost = 0.030

Do not tune multiple nitrogen costs in 009_01.

Add a soft nitrogen-stress penalty because nitrogen is now partly controlled by PPO:

- nstres_threshold = 0.05
- nstres_excess_cost = 1.5
- nstres_day_cost = 0.5

This first NSTRES penalty is intentionally weaker than the SWFAC penalty. Its purpose is to prevent PPO from avoiding fertilizer costs by accepting severe nitrogen stress.

## Required Logging Additions

Existing stage wrapper already records:

- raw_stage_action_irrigation
- raw_stage_action_nitrogen
- stage_action_amir
- stage_action_anfer
- stage_nitrogen_base
- stage_nitrogen_extra_cap
- stage_extra_anfer
- season_cumulative_irrigation
- season_cumulative_n
- growth_reward
- terminal_reward

The 009 framework report must explicitly summarize:

- total_irrigation
- total_n
- total_base_n
- total_ppo_extra_n
- total_ppo_extra_irrigation
- swfac_soft_penalty
- nstres_soft_penalty
- N by stage
- irrigation by stage
- final GRNWT
- SWFAC stress days
- NSTRES stress days
- first irrigation DAP
- first PPO-controlled nitrogen DAP
- whether irrigation cap saturated
- whether nitrogen cap saturated
- raw action pattern by stage

## 009_02 Smoke Test Recommendation

After 009_01 design is reviewed, run one small smoke test:

- station: HLA
- year: 2004
- seed: 0
- total_timesteps: 5000

Success signal:

- irrigation nonzero and non-saturated;
- nitrogen total between 110 and 190 kg ha-1;
- PPO extra nitrogen is nonzero in S3/S4/S5, or PPO deliberately chooses a low-N strategy with NSTRES stress days < 30;
- NSTRES stress days >= 30 should be treated as a warning unless yield remains high;
- GRNWT remains close to 008_15 HLA irrigation-only baseline or at least does not collapse;
- no S1 irrigation;
- no unreasonable early irrigation.

Failure signal:

- total irrigation = 0;
- total nitrogen close to only base 100 kg ha-1 with severe NSTRES;
- total_ppo_extra_n = 0 and NSTRES stress days >= 30;
- total nitrogen saturates 190 kg ha-1 in every run without yield justification;
- irrigation cap saturation;
- GRNWT collapse;
- S1 irrigation occurs.

## Do Not

- Do not modify site-packages reward files.
- Do not modify `my_data/`.
- Do not train in 009_01.
- Do not overwrite 008 outputs.
- Do not run multi-seed before a single smoke test passes.
- Do not remove the existing 008 HLA/FQA outputs.

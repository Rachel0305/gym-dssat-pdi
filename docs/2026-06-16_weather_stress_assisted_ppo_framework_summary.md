# Weather- and Stress-Assisted PPO Framework Summary

## Current Core Conclusion

The current project can be summarized as:

> A weather-forecast and stress-diagnosis assisted water-nitrogen management framework was constructed. After screening station-years with real water-stress potential, the framework was tested in representative dry years. PPO was shown to generate nonzero, non-saturated, and agronomically interpretable irrigation decisions in HLA 2004 and FQA 2016 under fixed adequate nitrogen supply.

This is not yet a final universal optimal policy. The current result supports a framework-level conclusion:

> PPO can produce useful irrigation decisions when the task is set in a truly water-limited year and the action space/reward design expose meaningful water-stress signals.

## Why the Research Direction Changed

The initial goal was direct water-nitrogen joint PPO optimization. Early experiments showed that this direct setup was unstable:

- Daily continuous PPO often saturated irrigation and nitrogen caps.
- In years with little or no SWFAC water stress, PPO irrigation decisions were difficult to explain agronomically.
- Hard rule gates could produce good yields, but PPO contribution became zero because the rule, not PPO, made the effective irrigation decision.
- Seed stability was not guaranteed under weak soft-stress penalties.

Therefore, the task was reframed from "unrestricted PPO water-nitrogen optimization" to:

1. Diagnose water and nitrogen stress first.
2. Select station-years where water optimization is meaningful.
3. Use weather forecast and stress state to structure the PPO decision problem.
4. Test whether PPO can produce interpretable irrigation behavior in representative dry years.

## Data and Scenario Screening

### Weather and Phenology Cleaning

Earlier tasks cleaned multi-site weather data and organized observed phenology:

- SRAD, TMAX, TMIN, and RAIN were cleaned into `weather_clean/` and QC outputs.
- Blank rainfall records were treated as zero-rain days where appropriate.
- Planting and harvest dates were organized from observed records.
- QC WTH files were generated for available station-years.

### Water-Stress and Irrigation-Response Screening

All available station-years were screened using fixed-management diagnostics:

- SWFAC stress days
- maximum SWFAC
- NSTRES stress days
- yield response to irrigation at the same nitrogen level
- nitrogen response
- growing-season rainfall

This screening showed that many observed years were nitrogen-limited or weakly water-limited. FQA 2008, previously used as a main PPO case, was found to be mainly nitrogen-limited, not an ideal irrigation optimization showcase.

The strongest water-stress candidates included:

- HLA 2004
- FQA 2016
- SYA 2017

HLA 2004 was selected as the primary dry-year demonstration case. FQA 2016 was then used as a second validation case.

## Management Scenario Comparison

For HLA 2004, four supervisor-requested management scenarios were organized:

1. `null_zero`: no irrigation and no nitrogen.
2. `expert_reference_recorded`: a single-year observed management record.
3. `dssat_auto_attempt`: an attempted gym-DSSAT automatic management baseline; retained as diagnostic because it behaved close to null.
4. `ppo_soft_stress_seed0_00815`: current HLA 2004 soft-stress PPO result.

The combined process plot and daily table were generated:

- rainfall
- SWFAC and NSTRES
- irrigation and nitrogen actions
- TOPWT and GRNWT
- daily reward

The HLA 2004 four-scenario result supports:

- HLA 2004 is strongly water-limited under poor management.
- PPO outperformed null, DSSAT auto attempt, and recorded expert in final GRNWT.
- PPO irrigation was nonzero and non-saturated.
- PPO decision timing was more interpretable than earlier FQA 2008 results.

But it does not prove universal optimality.

## PPO Framework Evolution

### Failed or Insufficient Designs

#### Daily Continuous Action PPO

Daily unrestricted action PPO often learned to:

- irrigate too early,
- saturate caps,
- remove future stress signals,
- or fail to produce agronomically interpretable actions.

This made the policy hard to defend to the supervisor.

#### Stage-Level Action With Hard Minimum Gate

A forecast/stress gate with a hard minimum irrigation amount produced high yields, but later diagnosis showed:

- pure rule replay and PPO with hard-minimum gate produced identical outcomes,
- PPO autonomous contribution was zero,
- the rule was making the effective irrigation decisions.

This was useful diagnostically but weak as an RL result.

#### Weak Soft-Stress PPO

Removing the hard minimum gate allowed PPO to act autonomously. HLA 2004 seed0 learned a good late-irrigation strategy, but seed1 initially degenerated to zero irrigation. This showed that the reward landscape had competing local optima.

#### Stronger Soft-Stress Penalty

Increasing the SWFAC penalty partially recovered seed1 from zero irrigation to a nonzero late-irrigation strategy, but did not fully match seed0. This identified seed sensitivity as a remaining limitation.

## Current Working Framework

The current framework has four key components.

### 1. Stress-Based Station-Year Selection

PPO irrigation optimization is only tested in years with meaningful water stress or irrigation response.

This avoids forcing PPO to learn irrigation in years where null or rainfed conditions already have little SWFAC stress.

### 2. Stage-Level Decisions

PPO does not act every day. It acts by DAP-defined growth stages:

- S1: establishment
- S2: early vegetative
- S3: mid season
- S4: late water-risk stage
- S5: terminal water-risk stage

This reduces the action horizon and avoids the original daily PPO tendency to make noisy or saturated actions.

### 3. Weather/Forecast Gate Without Hard Minimum

The framework still records forecast and stress information, but no longer forces a minimum irrigation amount.

In the current soft-stress version:

- S1 irrigation is blocked.
- Later stages can irrigate when allowed by the gate.
- No hard minimum irrigation is imposed.
- PPO must choose the actual irrigation amount.

This is important because it preserves PPO autonomy.

### 4. Soft-Stress Reward

The reward combines:

- growth reward,
- terminal yield reward,
- irrigation cost,
- soft SWFAC stress penalty.

The purpose is not to make PPO chase SWFAC alone, but to make water stress visible earlier than final yield loss.

The current stronger penalty configuration used in FQA 2016 validation:

- `swfac_threshold = 0.05`
- `swfac_excess_cost = 3.0`
- `swfac_day_cost = 1.0`
- fixed N = 150 kg ha-1
- PPO controls irrigation only

## Main Evidence So Far

### HLA 2004

HLA 2004 is the primary dry-year case.

Key HLA 2004 soft-stress PPO result:

- experiment: `008_15`
- seed: 0
- timesteps: 10k
- total irrigation: 80 mm
- total nitrogen: 150 kg ha-1
- final GRNWT: about 6778 kg ha-1
- about 96% of the fixed I120_N150 reference
- irrigation pattern: S4/S5 late-stage补水
- PPO autonomous irrigation contribution: 80 mm

This shows PPO can learn a nonzero, non-saturated, interpretable late-irrigation strategy when hard minimum irrigation is removed.

Limit:

- seed stability remains incomplete. Seed1 initially degenerated to zero irrigation, and stronger stress penalty only partially recovered it.

### FQA 2016

FQA 2016 was used as a second water-stress validation case.

Key FQA 2016 result:

- experiment: `008_20`
- seed: 0
- timesteps: 5k
- total irrigation: 80 mm
- total nitrogen: 150 kg ha-1
- final GRNWT: 6999.27 kg ha-1
- rule replay reference from 008_11: 6641.81 kg ha-1 with 90 mm irrigation
- GRNWT / rule replay: 1.054
- first irrigation DAP: 46
- SWFAC stress days: 9
- cap saturated: false
- debug_promising: true

Stage decisions:

- S1: no irrigation, N=50
- S2: no irrigation, N=50
- S3: irrigation 40 mm, N=50
- S4: irrigation 40 mm, N=0

This supports that the framework can work in a second water-stress year, not only HLA 2004.

## How to Explain the Current Claim

A safe claim:

> In representative water-stress years, the weather/stress-assisted stage PPO framework can produce nonzero, non-saturated, and interpretable irrigation decisions. In HLA 2004 and FQA 2016, PPO learned mid-to-late season supplemental irrigation strategies under fixed adequate nitrogen supply.

Avoid claiming:

> PPO is universally optimal.

Also avoid claiming:

> The current result fully solves water-nitrogen joint optimization.

The current result is better described as:

> A validated framework prototype for weather/stress-assisted irrigation optimization, with nitrogen controlled as adequate background management.

## Remaining Limitations

1. Seed stability is not fully solved.
2. The current successful PPO validation focuses on irrigation under fixed N150.
3. Cross-station and cross-year generalization still needs more tests.
4. DSSAT automatic management through gym-DSSAT is not yet a fully verified baseline.
5. Reward comparability across scenario families should be handled carefully because some daily reward columns come from different diagnostic contexts.

## Recommended Next Steps

### Short-Term

1. Build a two-case summary table for HLA 2004 and FQA 2016.
2. Generate paired figures showing:
   - rainfall,
   - SWFAC/NSTRES,
   - PPO irrigation and N,
   - GRNWT/TOPWT,
   - daily reward.
3. State clearly that HLA 2004 is the main demonstration case and FQA 2016 is a second validation case.

### Medium-Term

1. Run seed stability for FQA 2016 only if necessary.
2. Test SYA 2017 as a third water-stress case.
3. Decide whether to keep fixed N150 or reintroduce nitrogen PPO after irrigation behavior is stable.

### Paper Framing

The paper can be framed as:

> Weather forecast and crop-stress assisted reinforcement learning for interpretable irrigation decisions in DSSAT-based maize management.

The water-nitrogen component should be described carefully:

- water and nitrogen stress are diagnosed jointly;
- nitrogen is controlled as adequate stage supply in current PPO validation;
- irrigation decisions are optimized by PPO in water-stress years;
- full simultaneous PPO water-nitrogen optimization remains a future extension.


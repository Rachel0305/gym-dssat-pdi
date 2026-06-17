# 008_13 Official Maize Example vs Project Data PPO Gap Analysis

## Executive Summary

The official gym-DSSAT maize example and our current project are not equivalent PPO tasks. The official example is a controlled Gainesville maize experiment with clear nitrogen and irrigation treatment structure, fixed observation definitions, and relatively direct reward signals. Our project is a real multi-station, multi-year water-nitrogen management problem where water stress is uneven, nitrogen often dominates yield, soil-layer dimensions differ by site, and agronomic interpretability is required.

This means the current difficulty does not prove that PPO is unusable. It shows that direct unrestricted PPO is not enough for this real-station water-nitrogen task. At the same time, the latest 008_12 result shows that our forecast-gated design currently gives almost all decision power to the rule gate: PPO added no measurable irrigation beyond the gate minimum in HLA 2004, FQA 2008, or FQA 2016.

The safest next step is not to change real data to make PPO look better. Instead, we should build a separate official-style benchmark inside this project. That benchmark would test whether PPO can learn under cleaner, controlled conditions, then gradually add back real-station complexity.

## 1. What The Official Maize Example Is Actually Testing

The installed official maize configuration is based on `UFGA8201MZ NIT X IRR, GAINESVILLE 2N*3I`, a Gainesville maize nitrogen-by-irrigation experiment. Its DSSAT template explicitly defines rainfed, irrigated, low-nitrogen, high-nitrogen, and vegetation-stress treatments.

Important official configuration properties:

| Component | Official maize example |
| --- | --- |
| Site | University of Florida, Gainesville, USA |
| Experiment | Nitrogen x irrigation factorial maize experiment |
| Cultivar | `IB0035 McCurdy 84aa` |
| Soil profile | Soil ID `IBMZ910014`, 180 cm profile |
| Initial water | Explicit layer-by-layer initial soil water in the template |
| Action space | `anfer` 0-200 kg/ha/day and `amir` 0-50 mm/day |
| Observation space | Includes `rain`, `swfac`, `nstres`, `trnu`, `topwt`, `grnwt`, `istage`, `totir`, and fixed-size 9-layer soil arrays |
| Management structure | DSSAT template already contains treatment schedules for low/high nitrogen and rainfed/irrigated cases |
| Reward style | Fertilization: `TRNU - 0.5 * ANFER`; irrigation: `delta TOPWT - 15 * AMIR` |

Why PPO is easier here:

1. The experiment already has clear treatment contrasts.
2. Nitrogen and irrigation response signals are deliberately present.
3. Observation dimensions are fixed.
4. The baseline task is closer to a controlled algorithm benchmark than to an agronomic deployment problem.
5. PPO is not being asked to solve cross-site transfer, observed-data inconsistency, water-nitrogen interaction, and supervisor-level process interpretability at the same time.

## 2. What Our Project Is Testing

Our project is a substantially harder task:

| Component | Current project |
| --- | --- |
| Sites | HLA, SYA, LCA, YCA, FQA Chinese ecological stations |
| Years | Observed-year and all-year weather pools, including 2000-2023 weather scenarios |
| Main target | Water-nitrogen joint management |
| Data type | Real station weather, soil, phenology, and partial observed management records |
| Water signal | Many station-years have weak or no meaningful water stress under the reference scenario |
| Nitrogen signal | Often strong; FQA 2008 is mainly nitrogen-limited rather than water-limited |
| PPO requirement | Not only high yield, but also agronomically reasonable timing |
| Transfer issue | Soil-layer dimensions differ across stations, causing direct model-weight transfer to fail |

The key recent results show why this is difficult:

| Experiment | Key finding |
| --- | --- |
| 008_05 water-stress showcase selection | HLA 2004, FQA 2016, and SYA 2017 are better water-stress examples than FQA 2008 |
| 008_06 HLA 2004 scenario comparison | HLA 2004 is a strong water-limited case; fixed I120/N150 reached 7062.5 kg/ha while N-only reached 678.9 kg/ha |
| 008_11 pure forecast gate replay | The forecast gate produced reasonable water-gradient behavior across HLA 2004, FQA 2008, and FQA 2016 |
| 008_12 PPO contribution test | PPO produced identical results to the pure gate rule; PPO added 0 mm beyond gate minimum in all three tested station-years |

This latest result is important: the forecast gate made the decision agronomically explainable, but it also made PPO passive.

## 3. Difference Matrix: Why Official PPO Works More Easily

| Dimension | Official maize baseline | Our project | Effect on PPO |
| --- | --- | --- | --- |
| Main objective | Mostly baseline management optimization, especially fertilization | Water-nitrogen joint optimization | Joint control is harder because water and nitrogen interact |
| Data design | Controlled factorial experiment | Real station-year data | Real data have uneven signals and more confounding |
| Stress signal | Treatment design creates clear contrast | Many years lack useful SWFAC signal | PPO may not see consistent irrigation benefit |
| Nitrogen signal | Direct TRNU response supports reward | Nitrogen response can be delayed and crop-stage dependent | Credit assignment is harder |
| Action timing | Daily action allowed in a benchmark setting | Daily action caused early dumping and cap saturation | Needed stage/gate constraints |
| Observation space | Fixed 9-layer soil arrays | Variable soil-layer dimensions by site | Direct cross-site PPO model transfer fails |
| Reward delay | Fertilization reward has daily TRNU signal | Final yield response is delayed across stages | PPO may choose zero or excessive inputs |
| Agronomic interpretation | Less strict in baseline benchmark | Supervisor requires stress-weather-action consistency | A high-yield policy can still be unacceptable |
| Baseline expectation | Show PPO can beat simple baselines | Show process-rational management vs null/expert/DSSAT automatic | Much higher evidence burden |

## 4. Can We Modify Our Data To Become Like The Official Example?

We should not modify or falsify real data to make it behave like the official example. That would weaken the scientific value of the project.

But we can create a separate official-style diagnostic benchmark. This is not data manipulation; it is a controlled algorithm sanity check. Its purpose would be:

1. Verify that PPO can learn in our local codebase when the task has a clear response signal.
2. Separate algorithm/code problems from real-data/task-design problems.
3. Provide a bridge between the official gym-DSSAT baseline and the final real-station water-nitrogen study.

The report language should be:

> The official-style benchmark is a diagnostic control experiment. It is not used as the final agronomic result, but as evidence that the PPO implementation is functional under a controlled response setting.

## 5. Recommended Official-Style Benchmark For This Project

### Phase A: Reproduce An Official-Like Single-Task Benchmark

Start with the simplest PPO task:

- one station-year or one official Gainesville-like scenario;
- fixed observation dimension;
- fertilization-only mode;
- official reward: `TRNU - 0.5 * ANFER`;
- compare null, expert/fixed fertilization, and PPO.

Goal:

> Confirm PPO can learn a nitrogen response in this codebase.

### Phase B: Official-Style Water Response Benchmark

Use a strong water-stress station-year such as HLA 2004, but keep the problem controlled:

- fixed nitrogen, such as N150;
- PPO controls only irrigation;
- compare no irrigation, fixed I60, fixed I120, rule gate, and PPO;
- use a station-invariant observation adapter.

Goal:

> Confirm PPO can learn irrigation response when water stress is real and nitrogen is not the main confounder.

### Phase C: Controlled Water-Nitrogen Joint Benchmark

Only after A and B pass:

- allow both irrigation and fertilization actions;
- keep stage-level decisions;
- keep action caps;
- avoid a hard minimum gate that makes PPO passive;
- compare PPO's actual extra decision contribution against fixed/rule baselines.

Goal:

> Test whether PPO adds measurable value beyond agronomic rules.

### Phase D: Return To Real Multi-Year Station Data

After the controlled benchmark:

- use HLA 2004, FQA 2016, SYA 2017 as water-stress cases;
- use FQA 2008 as a nitrogen-limited diagnostic, not the main water case;
- add future rainfall information as an observation or constraint;
- evaluate process rationality: stress, rainfall, action timing, yield, water use, nitrogen use.

## 6. How To Bring PPO Back Into The Research Mainline

The current project has not drifted away from PPO for no reason. It drifted because direct PPO produced agronomically unacceptable behavior:

- daily PPO saturated caps;
- stage PPO could collapse to zero nitrogen or insufficient irrigation;
- forecast gate fixed timing logic but made PPO passive.

The next design should explicitly reserve a nontrivial decision for PPO. Three options are reasonable:

### Option 1: PPO Controls Amount, Gate Controls Eligibility

The gate only decides whether irrigation is allowed. If allowed, PPO decides the amount from 0 to a stage cap. Avoid forcing `max(PPO_action, minimum_irrigation)` because that makes the rule the real actor.

### Option 2: PPO Controls Thresholds Or Budgets

PPO chooses seasonal or stage-level budgets, while rules distribute them based on forecast/stress. This makes PPO a higher-level manager rather than a daily actuator.

### Option 3: PPO Controls Residual Around Expert/Rule Baseline

Expert or forecast rule provides a baseline, and PPO can add or subtract a bounded residual. This is more explainable, but must be tested against a pure rule baseline to prove PPO adds value.

## 7. Recommended Immediate Next Step

Create a new diagnostic task:

`008_14_official_style_ppo_sanity_benchmark`

Minimum version:

1. Do not use all five sites.
2. Do not train a large model first.
3. Use HLA 2004 because it has a strong water response.
4. Run irrigation-only PPO with fixed nitrogen N150.
5. Use no hard minimum irrigation gate.
6. Compare PPO against:
   - null/no irrigation;
   - fixed I60;
   - fixed I120;
   - pure forecast rule.
7. Report whether PPO changes irrigation timing or amount beyond fixed/rule baselines.

If PPO still fails in this cleaner task, then the issue is likely the reward/action/observation design. If PPO works there, then the problem is not PPO itself but the complexity of the full real-data water-nitrogen task.

## 8. Bottom-Line Conclusion

The official maize baseline is a useful reference, but it is not proof that direct PPO should automatically solve our real water-nitrogen joint optimization problem.

The scientifically defensible path is:

1. Keep real data unchanged.
2. Use official-style benchmarks as algorithm sanity checks.
3. Use HLA 2004 and other screened years as water-stress cases.
4. Separate rule contribution from PPO contribution.
5. Only claim PPO optimization when PPO measurably improves over rule/fixed baselines.

This framing should be acceptable for a supervisor: it admits the current limitation, preserves agronomic interpretability, and gives a concrete path to return PPO to the center of the study.

## Source Notes

- Official baseline documentation: https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Baselines/maize_baseline.html
- Installed official config inspected in Docker:
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/maize/env_config.yml`
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2`
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/rewards.py`
- Project evidence:
  - `docs/2026-06-13_008_05_water_stress_showcase_selection_report.md`
  - `docs/2026-06-13_008_06_hla2004_representative_management_comparison_report.md`
  - `docs/2026-06-14_008_11_forecast_gate_rule_replay_validation_report.md`
  - `docs/2026-06-14_008_12_ppo_contribution_under_forecast_gate_report.md`


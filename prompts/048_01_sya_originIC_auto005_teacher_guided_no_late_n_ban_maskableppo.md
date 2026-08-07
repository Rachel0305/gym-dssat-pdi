# 048_01 SYA originIC auto-0.05 teacher-guided PPO: remove late-N ban

## Question

Does the DAP90 fertilization cutoff prevent PPO from following useful portions
of the auto-0.05 stress-responsive nitrogen rule?

## Controlled change from 048_00

Only the PPO nitrogen-action window changes: `fertilization_allowed_dap_range`
is extended from `[1, 90]` to `[1, 150]`.

Retain the 048_00 teacher shaping, raw 046_02 observation, no forecast/no
normalization, 046_10 action grid, base reward, season-N cap, minimum days
between fertilization, and discrete action mask. This is a safety-envelope
change, not a pure teacher-guidance comparison.

## Fixed protocol

- station: `SYA`; input: `originIC`
- train years: 2005-2013; validation years: 2014-2023
- irrigation actions: `[0, 15, 30, 45]` mm
- nitrogen actions: `[0, 40, 80, 120]` kg/ha
- teacher: external minimal auto-N rule with `NSTRES >= 0.05`
- one 2K smoke before any 100K formal run

## Required smoke checks

Confirm that the manifest records `[1, 150]`, that actions remain on the
declared grid, no positive action is forced to DAP1 only, and more than one
nonzero action pair is used. Inspect whether any legal nitrogen opportunity is
available after DAP90; actual late-N use is evidence to report, not a required
success condition for the smoke.

## Interpretation boundary

If 048_01 fails to improve the policy, stop iterating on SYA by adding training
steps. Preserve 046_05 with auto-0.5 as the staged formal benchmark and move to
other sites or a separately preregistered reward redesign.

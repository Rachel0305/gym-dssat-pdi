# 048_00 SYA originIC auto-0.05 teacher-guided MaskablePPO

## Research question

Can a lightweight auto-0.05 teacher signal move PPO away from fixed high-input or
DAP1-only policies and toward stress-responsive nitrogen management, while keeping
the `046_10` observation, action grid, reward base, and safety constraints mostly
unchanged?

## Position in the SYA experiment sequence

- `046_10`: expanded action PPO with nitrogen `[0, 40, 80, 120]`; high yield but high water/N input.
- `047_00`: small-N action PPO with nitrogen `[0, 10, 20, 40]`; failed by collapsing to `DAP1 I45/N10`.
- `048_00`: return to the `046_10` action grid and add only an auto-0.05 teacher-shaping signal.

## Frozen factors

- station: `SYA`
- input profile: `originIC`
- train years: 2005-2013
- validation years: 2014-2023
- observation: raw `046_02` observation
- observation normalization: disabled
- weather forecast features: disabled
- action grid: irrigation `[0, 15, 30, 45]` mm; nitrogen `[0, 40, 80, 120]` kg/ha
- recorded farmer template: frozen
- base reward and safety chain: inherit `042_15 -> 046_02 -> 046_10`

## Teacher signal

The diagnostic teacher is the minimal external auto-N rule with nitrogen stress
threshold 0.05. It is not the formal DSSAT-default auto benchmark. The formal
auto benchmark remains the threshold-0.5 result.

The teacher signal is implemented as reward shaping based on the current
pre-action nitrogen stress (`NSTRES`) and DAP:

- penalize nitrogen when `NSTRES < 0.05`;
- penalize early nitrogen before the teacher-like stress window;
- reward legal nitrogen action when `NSTRES >= 0.05`;
- penalize missing nitrogen action when `NSTRES >= 0.05` during the allowed N window;
- add a small DAP1-only collapse penalty.

## Important feasibility boundary

The auto-0.05 baseline often applies nitrogen after DAP90, but the inherited PPO
safety chain forbids late nitrogen. Therefore `048_00` does not attempt exact
imitation of auto-0.05. It only tests whether a stress-triggered teacher signal
can improve PPO behavior within the existing PPO safety envelope.

If this fails, do not keep increasing training steps. The SYA stage can be frozen
at the `046_05...nstd050...` reporting package, and follow-up work should move to
other sites or to a separately preregistered safety/reward redesign.

## Required execution order

1. `--dry-run`
2. `--smoke` for 2K
3. Inspect the smoke action audit and teacher-shaping summary
4. Only if the smoke produces non-DAP1 stress-responsive actions, run `--formal`


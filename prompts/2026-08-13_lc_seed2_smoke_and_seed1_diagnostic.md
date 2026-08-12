# LC seed1 diagnostic and seed2 2K smoke

## Read-only seed1 diagnosis

Read the existing `069_01` seed1 checkpoints only.  Quantify, by checkpoint
and validation year, action sequence/signature, positive-action count,
post-DAP1 count, action-pair distribution and Shannon entropy.  Compare it to
the frozen `053_00` seed0 run without using any forecast or external-N branch.

## Seed2 smoke contract

Change only PPO random seed from 0 to 2.  Keep 053_00 lowIC input, the raw and
no-forecast observation, train 2005--2013/validation 2014--2023, 16-action grid
I `[0,15,30,45]` x N `[0,40,80,120]`, reward/safety, and PPO kwargs unchanged.
Run only 2K with 1K/2K checkpoints in `nifty_taussig`, single process.

## Gate

At 2K require 10 complete daily files and fields, legal grid, zero
request-to-safe/raw-to-safe/safe-to-DSSAT mismatch, >=3 nonzero pairs,
post-DAP1 actions in >=8 years, >=2 cross-year signatures, and no collapse.
If it fails, no long LC training is permitted.  No `WP_ET` is inferred from
smoke output and no five-scenario replay is run for LC in this task.

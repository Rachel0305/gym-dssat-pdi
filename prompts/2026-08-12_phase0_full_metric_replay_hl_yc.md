# 2026-08-12 HL/YC phase-0 full-metric replay

## Objective

Use only existing isolated checkpoints to obtain the same five-scenario metrics for HL and YC before any new tuning decision. This is a read-only replay, not training.

## Required candidates

- HL original 25K checkpoint from the frozen 054 workflow.
- HL `learning_rate=1e-4` 25K checkpoint from `064_01`.
- YC original 25K checkpoint from the frozen 055 workflow.
- YC `learning_rate=1e-4` checkpoints 5K, 10K and 25K from `065_01`.

## Fixed rules

- Run inside `nifty_taussig` with `/opt/gym_dssat_pdi/bin/python`.
- One process at a time; never modify or overwrite 054/055 or rescue outputs.
- Use the same lowIC input root, renderer root, validation years and five scenarios for every candidate.
- Report yield, `WP_ET`, `PFP_N`, irrigation, nitrogen, yearly wins, action signatures and transmission audits.
- If a same-definition ETCP/snapshot is unavailable, write `WP_ET=unavailable`; never infer it.

## Stop conditions

Stop on any renderer/input provenance mismatch, missing checkpoint, output collision, OOM, or endpoint disagreement. Save a manifest, source hashes, command, output path, and an explicit reason for every skipped metric.

## Deliverables

Write an isolated CSV/JSON comparison and `docs/2026-08-12_hl_yc_phase0_full_metric_replay.md`. Do not commit or push.

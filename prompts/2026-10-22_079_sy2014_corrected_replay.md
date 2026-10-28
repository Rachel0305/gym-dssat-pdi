# 079: Restore original SY2014 frozen replay environment

Scope: one SY2014 replay, frozen 143E2 seed0 5000-step model. No training, no weather perturbation, no additional years/sites/seeds.
Restore the exact original engine.patch_base_module invocation so the original LateIrrigationReserveMaskWrapper is used. Do not tune any constraints, rewards, doses, model weights or tolerances.
Compare against original training daily requested actions, mask valid counts, executed I/N sequences and original summary yield/I/N. Old audit trace is diagnostic only because it omitted this wrapper.
Yield tolerance remains 0.5 kg/ha; I/N tolerance 1e-6. Keep original078 failure unchanged; isolated079 output. Single-thread CPU, RSS768MB cap and180-second deadline. Preserve failure and stop if reproduction fails. Passing does not authorize training or prove forecast benefit.

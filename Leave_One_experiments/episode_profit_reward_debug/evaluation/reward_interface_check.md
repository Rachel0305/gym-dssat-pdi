# Terminal reward interface check

Generated at: 2026-06-06

## Answers required by prompt 006_05

1. The original gym-DSSAT reward function can access `_next_state`.
2. The original reward function cannot reliably judge episode done because the callback signature is `_previous_state, _next_state, _history, _cultivar` and has no done flag.
3. The original reward function can access `_history`.
4. `_history` includes actions in previous training/evaluation traces.
5. `_history` includes daily observations through the environment history used by the wrapper/evaluation utilities.
6. `final_grnwt` can be read from the final observation / history after the environment step returns done.
7. Daily `amir` and `anfer` can be accumulated reliably from `SafeActionWrapper.last_safety_result.safe_real_action`.
8. Terminal reward can be added on the last step at the gymnasium wrapper layer, after `terminated` or `truncated` is known.
9. Because the original reward callback has no done flag, terminal reward should be added in a new project-level wrapper, not in site-packages.
10. Minimal implementation: keep `site-packages` reward unchanged, keep `src/ppo_train.py` unchanged, add `src/episode_profit_reward.py` and a dedicated runner that wraps the action-safe env.

## Feasibility conclusion

Terminal reward is feasible through `EpisodeProfitRewardWrapper`. This is the least invasive design: action safety still controls the physical water/N limits, while the profit wrapper replaces the training reward with daily input cost plus terminal grain-yield profit.

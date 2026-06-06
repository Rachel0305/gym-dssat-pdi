# Daily continuous action design failure review

Generated at: 2026-06-06

## Diagnosis

The previous reward experiments show that daily continuous actions make cap saturation easy. The agent receives a chance to output irrigation and fertilizer every simulated day. Even when action safety clips daily amounts, intervals, windows, and seasonal caps, the learned behavior can become "keep asking for water and nitrogen until safety stops it".

Action safety is therefore acting as the real manager, not just a safety guard. It prevents physical over-application but also creates a simple attractor: repeatedly request positive actions and let the wrapper spend the entire seasonal cap.

Reward cost tuning and terminal profit reward did change reward scale, but did not change the action opportunity structure. As long as the agent can ask every day, saturation remains an easy policy.

Real field management is lower-frequency and stage-based. Irrigation and fertilization are normally decided in a few operational windows, not every day. Therefore this stage tests low-frequency and phenology-window action designs.

## Minimal-intrusion wrappers

- DecisionIntervalActionWrapper / ScheduledActionDesignWrapper: only allow PPO actions every N days.
- PhenologyWindowActionWrapper behavior: only allow irrigation and N in agronomic DAP windows.
- Wrapper order: GymDssatWrapper -> ActionDesignWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper.

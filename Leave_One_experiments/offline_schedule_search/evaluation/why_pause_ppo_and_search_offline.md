# Why pause PPO and search offline

Generated at: 2026-06-06

006_03 to 006_07 showed a repeated pattern: step-wise cost rewards, stronger cost coefficients, episode-level profit rewards, low-frequency/window action wrappers, and explicit budget/event action wrappers all still drove HLA PPO policies to the seasonal action-safety cap of 300 mm irrigation and 450 kg ha-1 nitrogen.

Reward cost tuning was insufficient because the learned policy still treated available seasonal input as beneficial. Terminal profit reward was insufficient because the policy continued to select cap-level actions before the terminal score could create an interpretable low-input behavior. Low-frequency, window, budget, and event wrappers changed when actions could happen, but did not provide an external prior for what a reasonable management schedule should look like.

The next methodological step is therefore non-RL schedule search. A deterministic DSSAT scenario ensemble can expose the yield-water-nitrogen trade-off, identify Pareto-efficient schedules, and construct expert schedules. These expert schedules can later support imitation learning, behavior cloning, reward calibration, or constrained PPO.

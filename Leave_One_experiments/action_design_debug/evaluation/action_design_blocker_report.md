# Action design blocker report

Generated at: 2026-06-06

Design A/B wrappers did not identify a strict or relaxed passing HLA pilot candidate. The next action-design step should not be another coefficient scan. Recommended directions:

1. Explicit seasonal budget action: PPO first decides total water/N budget, then allocates within fixed windows.
2. Scheduled discrete events: actions are event amounts at basal, jointing, pre-tasseling, silking/grain-filling windows.
3. Rule-based expert policy plus imitation learning: pretrain management timing from agronomic rules before RL fine tuning.
4. Hybrid PPO over management windows: reduce episode action count to a few stage decisions.

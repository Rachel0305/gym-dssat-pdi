# Attempt history: 011 YC seed2 25K/50K Q-value replay

- attempt: `attempt_01`
- time: `unknown / insufficient evidence`
- stage: first replay completed its first checkpoint-year episode and failed while constructing the parity comparison row
- failure reason: the audit script passed the formal CSV field name `action_index` directly to replay rows, whose field is `selected_action`; this raised `KeyError: 'action_index'`.
- fix: add an explicit formal-to-replay field mapping for `action_index -> selected_action` and `executed_action_index -> executed_action`.
- classification: engineering audit-script plumbing only; no formal DQN code, environment, scientific contract, checkpoint, or training output was changed.
- retention: the failed attempt was not used as evidence for mechanism conclusions; its runtime artifacts remain under this experiment directory, and the corrected replay produced the retained PASS outputs listed in the diagnostic report.

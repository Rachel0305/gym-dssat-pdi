# 013_05 YC2014 dynamic action transmission diagnostic

## Summary

| scenario | linked_management | forced_action | action_irrigation_total | action_fertilizer_total | mgmt_event_irrigation_total | mgmt_event_fertilizer_total | final_gwad | final_cwad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static_null_zero_action | False | False | 0 | 0 | 0 | 0 | 7825 | 17996 |
| static_forced_action | False | True | 90 | 300 | 0 | 0 | 7825 | 17996 |
| linked_null_zero_action | True | False | 0 | 0 | 0 | 0 | 7825 | 17996 |
| linked_forced_action | True | True | 90 | 300 | 90 | 300 | 9418 | 20514 |

## Conclusion template

- If static_forced_action has action total > 0 but MgmtEvent total = 0, static N/N management blocks dynamic actions.
- If linked_forced_action has MgmtEvent total > 0 and yield changes, PDI action transmission works when management is L/L.
- DQN/PPO training inputs must use IRRIG=L and FERTI=L for dynamic actions.

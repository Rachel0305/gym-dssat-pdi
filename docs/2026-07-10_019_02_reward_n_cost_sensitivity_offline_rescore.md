# 019_02 Reward N Cost Sensitivity Offline Rescore

## Conclusion

This run only rescored existing checkpoints. It did not train DQN and did not call DSSAT.

## Best Checkpoint Under Each N Cost

| site | year | seed | n_cost | checkpoint | gwad | irrigation | nitrogen | offline_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 2016 | 0 | 5 | 25000 | 7779 | 60 | 0 | 653 |
| FQ | 2016 | 0 | 10 | 25000 | 7779 | 60 | 0 | 653 |
| FQ | 2016 | 0 | 20 | 25000 | 7779 | 60 | 0 | 653 |
| FQ | 2016 | 1 | 5 | 30000 | 7995 | 60 | 0 | 869 |
| FQ | 2016 | 1 | 10 | 30000 | 7995 | 60 | 0 | 869 |
| FQ | 2016 | 1 | 20 | 30000 | 7995 | 60 | 0 | 869 |
| HLA | 2010 | 0 | 5 | 35000 | 7854 | 120 | 0 | 778 |
| HLA | 2010 | 0 | 10 | 35000 | 7854 | 120 | 0 | 778 |
| HLA | 2010 | 0 | 20 | 35000 | 7854 | 120 | 0 | 778 |
| HLA | 2010 | 1 | 5 | 25000 | 7573 | 60 | 0 | 557 |
| HLA | 2010 | 1 | 10 | 25000 | 7573 | 60 | 0 | 557 |
| HLA | 2010 | 1 | 20 | 25000 | 7573 | 60 | 0 | 557 |
| LC | 2010 | 0 | 5 | 5000 | 8739 | 90 | 0 | 598 |
| LC | 2010 | 0 | 10 | 5000 | 8739 | 90 | 0 | 598 |
| LC | 2010 | 0 | 20 | 5000 | 8739 | 90 | 0 | 598 |
| LC | 2010 | 1 | 5 | 5000 | 8739 | 120 | 300 | -932 |
| LC | 2010 | 1 | 10 | 5000 | 8739 | 120 | 300 | -2432 |
| LC | 2010 | 1 | 20 | 5000 | 8739 | 120 | 300 | -5432 |
| SY | 2014 | 0 | 5 | 15000 | 11216 | 120 | 300 | 6827 |
| SY | 2014 | 0 | 10 | 15000 | 11216 | 120 | 300 | 5327 |
| SY | 2014 | 0 | 20 | 15000 | 11216 | 120 | 300 | 2327 |
| SY | 2014 | 1 | 5 | 10000 | 11227 | 90 | 300 | 6868 |
| SY | 2014 | 1 | 10 | 10000 | 11227 | 90 | 300 | 5368 |
| SY | 2014 | 1 | 20 | 10000 | 11227 | 90 | 300 | 2368 |
| YC | 2014 | 0 | 5 | 45000 | 8939 | 90 | 100 | 524 |
| YC | 2014 | 0 | 10 | 45000 | 8939 | 90 | 100 | 24 |
| YC | 2014 | 0 | 20 | 45000 | 8939 | 90 | 100 | -976 |
| YC | 2014 | 1 | 5 | 10000 | 9418 | 120 | 300 | -27 |
| YC | 2014 | 1 | 10 | 10000 | 9418 | 120 | 300 | -1527 |
| YC | 2014 | 1 | 20 | 10000 | 9418 | 120 | 300 | -4527 |

## Site-Level Interpretation

| site | year | base_mean_n_at_cost5 | mean_n_at_cost20 | base_mean_gwad_at_cost5 | mean_gwad_at_cost20 | status | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 2016 | 0 | 0 | 7887 | 7887 | n_cost_no_selection_change | 现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。 |
| HLA | 2010 | 0 | 0 | 7713.500 | 7713.500 | n_cost_no_selection_change | 现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。 |
| LC | 2010 | 150 | 150 | 8739 | 8739 | n_cost_no_selection_change | 现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。 |
| SY | 2014 | 300 | 300 | 11221.500 | 11221.500 | n_cost_no_selection_change | 现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。 |
| YC | 2014 | 200 | 200 | 9178.500 | 9178.500 | n_cost_no_selection_change | 现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。 |

## Important Limitation

Offline rescoring can show whether existing checkpoints would be selected differently, but it cannot prove a newly trained policy would learn the same behavior. If a station shows promise here, the next step is a small retraining smoke under the new cost.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from literature_aligned_dqfd import PrioritizedDemonstrationReplay, dqfd_loss_components


SOURCE = ROOT / "benchmark_results" / "021_20" / "021_20_demonstration_transitions.npz"
OUT = ROOT / "benchmark_results" / "021_21"
DOC = ROOT / "docs" / "2026-07-15_021_21_sy2014_literature_aligned_dqfd_offline_unit_audit.md"


def table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    source = np.load(SOURCE)
    demonstrations = {name: source[name] for name in source.files}
    finite = all(np.isfinite(value).all() for value in demonstrations.values())
    replay = PrioritizedDemonstrationReplay(
        demonstrations,
        agent_capacity=20,
        alpha=0.4,
        epsilon_demo=1.0,
        epsilon_agent=0.001,
        seed=21021,
    )
    demo_hash_before = replay.demonstration_hash

    # Overflow the replaceable agent ring more than twice. Demonstrations must remain immutable.
    for index in range(50):
        transition = {
            name: np.asarray(value[index % len(value)]).copy()
            for name, value in demonstrations.items()
        }
        replay.add_agent(transition, td_error=0.1 + index / 100.0)
    demo_hash_after_overflow = replay.demonstration_hash

    # Equal TD errors must still give demonstrations a higher priority bonus.
    replay.update_priorities([0], [2.0])
    agent_global_index = replay.demo_count
    replay.update_priorities([agent_global_index], [2.0])
    demo_priority_equal_td = float(replay.demo_priorities[0])
    agent_priority_equal_td = float(replay.agent_priorities[0])

    priorities = replay.combined_priorities()
    probabilities = replay.sampling_probabilities()
    manual_probabilities = priorities ** 0.4
    manual_probabilities /= manual_probabilities.sum()
    probability_error = float(np.max(np.abs(probabilities - manual_probabilities)))
    all_weights = (replay.total_size * probabilities) ** (-0.6)
    all_weights /= all_weights.max()
    inverse_relation = bool(np.corrcoef(probabilities, all_weights)[0, 1] < 0)

    sample_a = replay.sample(64, beta=0.6, seed=12345)
    sample_b = replay.sample(64, beta=0.6, seed=12345)
    deterministic_sampling = bool(
        np.array_equal(sample_a.global_indices, sample_b.global_indices)
        and np.array_equal(sample_a.importance_weights, sample_b.importance_weights)
    )

    demo_priority_snapshot = replay.demo_priorities.copy()
    agent_priority_snapshot = replay.agent_priorities.copy()
    update_indices = np.asarray([1, replay.demo_count + 1], dtype=np.int64)
    replay.update_priorities(update_indices, [4.0, 5.0])
    changed_demo = np.flatnonzero(replay.demo_priorities != demo_priority_snapshot).tolist()
    changed_agent = np.flatnonzero(replay.agent_priorities != agent_priority_snapshot).tolist()
    identity_preserved_after_update = bool(
        changed_demo == [1]
        and changed_agent == [1]
        and replay.demonstration_hash == demo_hash_before
    )

    # Synthetic mixed batch: first two are demonstrations, last two are agents.
    q_values = torch.nn.Parameter(torch.tensor([
        [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0],
        [0.1, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2],
    ], dtype=torch.float32))
    actions = torch.tensor([0, 4, 1, 7])
    target_1 = torch.tensor([1.2, 1.0, 0.8, 1.1])
    target_n = torch.tensor([1.4, 1.2, 1.0, 1.3])
    importance = torch.tensor([1.0, 0.8, 0.7, 0.6])
    demo_mask = torch.tensor([True, True, False, False])
    losses = dqfd_loss_components(
        q_values=q_values,
        actions=actions,
        target_1_step=target_1,
        target_n_step=target_n,
        importance_weights=importance,
        demonstration_mask=demo_mask,
        l2_parameters=[q_values],
        margin=0.8,
        lambda_n_step=1.0,
        lambda_margin=1.0,
        lambda_l2=1e-5,
    )
    numeric_losses = {name: float(value.detach()) for name, value in losses.items()}
    weighted_sum = sum(numeric_losses[name] for name in (
        "td_1_weighted", "td_n_weighted", "margin_weighted", "l2_weighted"
    ))
    weighted_sum_error = abs(weighted_sum - numeric_losses["total"])

    agent_only = dqfd_loss_components(
        q_values=q_values,
        actions=actions,
        target_1_step=target_1,
        target_n_step=target_n,
        importance_weights=importance,
        demonstration_mask=torch.zeros(4, dtype=torch.bool),
        l2_parameters=[q_values],
        margin=0.8,
        lambda_n_step=1.0,
        lambda_margin=1.0,
        lambda_l2=1e-5,
    )
    agent_only_margin = float(agent_only["margin_raw"].detach())

    q_margin_ok = torch.tensor([[1.0, 0.1, 0.0]], dtype=torch.float32)
    q_margin_bad = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    common = {
        "actions": torch.tensor([0]),
        "target_1_step": torch.tensor([1.0]),
        "target_n_step": torch.tensor([1.0]),
        "importance_weights": torch.tensor([1.0]),
        "demonstration_mask": torch.tensor([True]),
        "margin": 0.8,
        "lambda_n_step": 1.0,
        "lambda_margin": 1.0,
        "lambda_l2": 0.0,
    }
    margin_zero = float(dqfd_loss_components(q_values=q_margin_ok, l2_parameters=[q_margin_ok], **common)["margin_raw"])
    margin_positive = float(dqfd_loss_components(q_values=q_margin_bad, l2_parameters=[q_margin_bad], **common)["margin_raw"])

    priority_table = pd.DataFrame({
        "global_index": np.arange(replay.total_size),
        "sample_type": ["demonstration"] * replay.demo_count + ["agent"] * replay.agent_size,
        "priority": replay.combined_priorities(),
        "sampling_probability": replay.sampling_probabilities(),
    })
    priority_table["importance_weight_beta0p6"] = (
        replay.total_size * priority_table.sampling_probability
    ) ** (-0.6)
    priority_table["importance_weight_beta0p6"] /= priority_table.importance_weight_beta0p6.max()
    priority_table.to_csv(OUT / "021_21_priority_table.csv", index=False, encoding="utf-8-sig")

    td_reference = max(abs(numeric_losses["td_1_weighted"]), 1e-12)
    loss_rows = []
    for component in ("td_1", "td_n", "margin", "l2"):
        raw = numeric_losses[f"{component}_raw"]
        weighted = numeric_losses[f"{component}_weighted"]
        loss_rows.append({
            "component": component,
            "raw_value": raw,
            "weighted_value": weighted,
            "weighted_to_td1_ratio": weighted / td_reference,
        })
    loss_frame = pd.DataFrame(loss_rows)
    loss_frame.to_csv(OUT / "021_21_loss_component_audit.csv", index=False, encoding="utf-8-sig")

    tests = {
        "source_demonstration_count": replay.demo_count,
        "source_all_fields_finite": finite,
        "agent_capacity": replay.agent_capacity,
        "agent_insertions": 50,
        "agent_active_count_after_overflow": replay.agent_size,
        "demo_hash_before": demo_hash_before,
        "demo_hash_after_overflow": demo_hash_after_overflow,
        "demonstrations_permanent_after_overflow": demo_hash_before == demo_hash_after_overflow,
        "demo_priority_equal_td": demo_priority_equal_td,
        "agent_priority_equal_td": agent_priority_equal_td,
        "demo_bonus_strictly_higher": demo_priority_equal_td > agent_priority_equal_td,
        "sampling_probability_sum": float(probabilities.sum()),
        "sampling_probability_manual_max_error": probability_error,
        "importance_weights_finite": bool(np.isfinite(all_weights).all()),
        "importance_weight_max": float(all_weights.max()),
        "importance_weight_probability_correlation": float(np.corrcoef(probabilities, all_weights)[0, 1]),
        "importance_weights_inverse_to_probability": inverse_relation,
        "fixed_seed_sampling_reproducible": deterministic_sampling,
        "priority_update_changed_demo_positions": changed_demo,
        "priority_update_changed_agent_positions": changed_agent,
        "priority_update_identity_and_content_preserved": identity_preserved_after_update,
        "large_margin_zero_case": margin_zero,
        "large_margin_positive_case": margin_positive,
        "agent_only_margin_loss": agent_only_margin,
        "all_loss_components_finite": bool(all(np.isfinite(value) for value in numeric_losses.values())),
        "weighted_component_sum_error": weighted_sum_error,
        "no_dssat_or_network_training_performed": True,
    }
    tests["passed"] = bool(
        tests["source_demonstration_count"] == 160
        and tests["source_all_fields_finite"]
        and tests["agent_active_count_after_overflow"] == 20
        and tests["demonstrations_permanent_after_overflow"]
        and tests["demo_bonus_strictly_higher"]
        and abs(tests["sampling_probability_sum"] - 1.0) <= 1e-12
        and tests["sampling_probability_manual_max_error"] <= 1e-15
        and tests["importance_weights_finite"]
        and abs(tests["importance_weight_max"] - 1.0) <= 1e-12
        and tests["importance_weights_inverse_to_probability"]
        and tests["fixed_seed_sampling_reproducible"]
        and tests["priority_update_identity_and_content_preserved"]
        and abs(tests["large_margin_zero_case"]) <= 1e-7
        and tests["large_margin_positive_case"] > 0
        and abs(tests["agent_only_margin_loss"]) <= 1e-7
        and tests["all_loss_components_finite"]
        and tests["weighted_component_sum_error"] <= 1e-6
        and tests["no_dssat_or_network_training_performed"]
    )
    (OUT / "021_21_unit_test_results.json").write_text(
        json.dumps({"tests": tests, "loss_components": numeric_losses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not tests["passed"]:
        raise RuntimeError(f"021_21 unit tests failed: {tests}")

    summary_tests = pd.DataFrame([
        {"审计项": "示范永久保留", "结果": tests["demonstrations_permanent_after_overflow"]},
        {"审计项": "示范优先级加成", "结果": tests["demo_bonus_strictly_higher"]},
        {"审计项": "采样概率公式", "结果": tests["sampling_probability_manual_max_error"] <= 1e-15},
        {"审计项": "IS权重", "结果": tests["importance_weights_inverse_to_probability"]},
        {"审计项": "固定seed复现", "结果": tests["fixed_seed_sampling_reproducible"]},
        {"审计项": "priority更新不改身份/内容", "结果": tests["priority_update_identity_and_content_preserved"]},
        {"审计项": "margin仅作用于示范", "结果": abs(tests["agent_only_margin_loss"]) <= 1e-7},
        {"审计项": "损失分解恒等式", "结果": tests["weighted_component_sum_error"] <= 1e-6},
    ])
    record = f"""# 021_21 SY2014 文献对齐 DQfD 离线组件与单元审计记录

## 目的与边界

本轮只实现并审计 Hester et al. (2018) DQfD 的关键 replay/loss 组件，没有调用DSSAT、没有训练网络、没有修改reward、IC、动作、预算或旧结果。参数在运行前写入prompt，未根据结果调整。

## 冻结参数

- alpha=0.4，beta=0.6；
- epsilon_demo=1.0，epsilon_agent=0.001；
- margin=0.8；
- lambda_n=1，lambda_margin=1，lambda_L2=1e-5；
- 项目适配gamma=0.99、n-step=5。

## 单元测试结果

{table(summary_tests)}

全部核心测试通过。160条021_20 oracle示范在50次agent插入（agent容量20、发生两轮以上覆盖）后内容哈希不变；相同TD error=2时，示范优先级为3.0，agent优先级为2.001；统一采样概率、alpha幂次和beta重要性权重与手算一致。

## 合成混合batch损失量级

{table(loss_frame.round(8))}

本表只是代码恒等式和日志链路测试，不是SY真实训练损失，不能据此调整lambda或推断5K结果。四项分量均有限，加权分项之和与总损失误差为{weighted_sum_error:.3e}；agent-only batch的margin loss严格为0。

## 判定

`completed`：优先回放、示范永久保留、示范优先级加成、IS权重、priority更新和四项损失分解具备进入下一轮真实短诊断设计的工程条件。

这不等于已经证明DQfD有效，也不授权自动运行5K。下一轮若继续，应另写prompt：先在真实网络上运行极短的损失量级诊断并保存逐update原始/加权损失；若审计未通过则停止，不能在同一任务现场调lambda。

## 方法来源

Hester, T. et al. Deep Q-learning from Demonstrations. *Proceedings of the AAAI Conference on Artificial Intelligence* 32 (2018). DOI: 10.1609/aaai.v32i1.11757。

## 输出

- `prompts/021_21_sy2014_literature_aligned_dqfd_offline_unit_audit.md`
- `src/literature_aligned_dqfd.py`
- `src/test_literature_aligned_dqfd_offline_021_21.py`
- `benchmark_results/021_21/021_21_unit_test_results.json`
- `benchmark_results/021_21/021_21_priority_table.csv`
- `benchmark_results/021_21/021_21_loss_component_audit.csv`
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps({"status": "completed", "passed": True, "output": str(OUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


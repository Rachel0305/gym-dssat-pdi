from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from mask_aware_dqn_029 import MaskAwareDQN


def run() -> None:
    model = MaskAwareDQN(4, action_dim=3, seed=17, batch_size=4)
    obs = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask = np.asarray([True, False, True])

    with torch.no_grad():
        for parameter in model.online.parameters():
            parameter.zero_()
        model.online.net[-1].bias.copy_(torch.tensor([1.0, 999.0, 2.0]))
    assert model.select_action(obs, mask, 0, deterministic=True) == 2
    assert all(model.select_action(obs, mask, 0, deterministic=False) in (0, 2) for _ in range(200))

    terminal_mask = np.zeros(3, dtype=bool)
    for index in range(8):
        done = bool(index % 2)
        model.add_transition(
            observation=obs + index,
            action=0,
            reward=float(index),
            next_observation=obs + index + 1,
            done=done,
            mask=mask,
            next_mask=terminal_mask if done else mask,
        )
    metrics = model.train_step()
    assert all(np.isfinite(value) for value in metrics.values())

    assert model.epsilon(0) == 1.0
    assert np.isclose(model.epsilon(168), 0.05)
    assert np.isclose(model.epsilon(240), 0.05)

    before = model.module_hash(model.target)
    model.sync_target()
    after = model.module_hash(model.target)
    assert before != after
    assert after == model.module_hash(model.online)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        expected_action = model.select_action(obs, mask, 60, deterministic=True)
        expected_hash = model.module_hash(model.online)
        model.save(path, global_step=60)
        restored, step = MaskAwareDQN.load(path)
        assert step == 60
        assert restored.module_hash(restored.online) == expected_hash
        assert restored.select_action(obs, mask, 60, deterministic=True) == expected_action

    # Terminal transitions must not bootstrap, even when the target network has
    # an intentionally enormous action value.
    terminal_model = MaskAwareDQN(4, action_dim=3, seed=3, batch_size=1)
    with torch.no_grad():
        for parameter in terminal_model.target.parameters():
            parameter.zero_()
        terminal_model.target.net[-1].bias.fill_(1e6)
    terminal_model.add_transition(
        observation=obs,
        action=0,
        reward=7.0,
        next_observation=obs,
        done=True,
        mask=np.ones(3, dtype=bool),
        next_mask=np.zeros(3, dtype=bool),
    )
    terminal_metrics = terminal_model.train_step()
    assert np.isclose(terminal_metrics["mean_target"], 7.0)

    print("029 mask-aware DQN unit tests passed")


if __name__ == "__main__":
    run()

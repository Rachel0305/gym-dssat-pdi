"""169: SYA weather augmentation smoke with a controlled PPO entropy override.

This wrapper reuses the validated 168 scenario runner, but applies exactly one
PPO change at runtime: ent_coef=0.02. The original 168 runner and outputs are
left untouched; this task writes to a new result root.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_168_sya_weather_aug_pilot_smoke2k_v2 as base168


TASK_ID = "169"
TASK_NAME = "sya_weather_aug_entcoef002_smoke2k"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-19_sya_weather_aug_entcoef002_smoke2k.md"
CHECKPOINT_STEPS = [1000, 2000]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_aliases() -> dict[str, str]:
    aliases = {
        "168_scenario_episode_metrics.csv": "169_scenario_episode_metrics.csv",
        "168_action_pair_detail.csv": "169_action_pair_detail.csv",
        "168_action_change_summary.csv": "169_action_change_summary.csv",
    }
    copied: dict[str, str] = {}
    evaluation = OUT / "evaluation"
    for source_name, target_name in aliases.items():
        source = evaluation / source_name
        target = evaluation / target_name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
            copied[target_name] = target.relative_to(ROOT).as_posix()
    return copied


def run() -> dict[str, Any]:
    cfg = load_json(CONFIG)
    override = float(cfg["ppo_overrides"]["ent_coef"])
    if abs(override - 0.02) > 1e-12:
        raise ValueError(f"Expected ent_coef=0.02, got {override}")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {OUT.relative_to(ROOT)}")

    original_load_config = base168.engine.load_config
    original_globals = {
        "TASK_ID": base168.TASK_ID,
        "TASK_NAME": base168.TASK_NAME,
        "DEFAULT_CONFIG": base168.DEFAULT_CONFIG,
        "OUT": base168.OUT,
        "DOC": base168.DOC,
        "PROMPT": base168.engine.PROMPT,
        "CHECKPOINT_STEPS": list(base168.CHECKPOINT_STEPS),
    }

    def tuned_load_config() -> dict[str, Any]:
        runtime = copy.deepcopy(original_load_config())
        runtime.setdefault("ppo", {})["ent_coef"] = override
        runtime.setdefault("experiment_metadata", {}).update(
            {
                "task_id": TASK_ID,
                "ent_coef_override": override,
                "inherited_runner": "168_sya_weather_aug_pilot_smoke2k_v2",
            }
        )
        return runtime

    base168.engine.load_config = tuned_load_config
    base168.TASK_ID = TASK_ID
    base168.TASK_NAME = TASK_NAME
    base168.DEFAULT_CONFIG = CONFIG
    base168.OUT = OUT
    base168.DOC = DOC
    base168.CHECKPOINT_STEPS = CHECKPOINT_STEPS

    try:
        result = base168.run(CONFIG)
    finally:
        base168.engine.load_config = original_load_config
        for key, value in original_globals.items():
            setattr(base168, key, value)

    aliases = copy_aliases()
    result = dict(result)
    result.update(
        {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "ppo_overrides": {"ent_coef": override},
            "fixed_ppo_parameters": cfg["fixed_ppo_parameters"],
            "inherited_runner": "168_sya_weather_aug_pilot_smoke2k_v2",
            "prompt": PROMPT.relative_to(ROOT).as_posix(),
            "output_aliases": aliases,
        }
    )
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {TASK_ID} SYA weather augmentation PPO ent_coef=0.02 smoke",
        "",
        "- 继承 168 的六套 2005 年 RAIN-only 情景和物理响应门槛。",
        "- 唯一 PPO 改动：`ent_coef 0.01 -> 0.02`。",
        "- reward、观测、动作空间、动作掩码、安全约束和 seed=0 保持不变。",
        "- 训练：2000 timesteps；checkpoint：1000、2000。",
        "- 不执行 2014 年验证。",
        f"- 训练状态：`{result.get('status')}`；action response passed：`{result.get('action_response_passed')}`。",
        f"- Prompt：`{PROMPT.relative_to(ROOT).as_posix()}`。",
        f"- Result：`{(OUT / f'{TASK_ID}_result.json').relative_to(ROOT).as_posix()}`。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()

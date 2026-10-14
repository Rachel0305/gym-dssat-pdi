"""170: SYA weather augmentation 20K run with ent_coef=0.02.

The validated 169 wrapper is reused. This entry point changes only the
training horizon/checkpoints and keeps the 169 entropy override and all
weather/reward/action contracts unchanged.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_169_sya_weather_aug_entcoef002_smoke2k as base169


TASK_ID = "170"
TASK_NAME = "sya_weather_aug_entcoef002_20k"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-19_sya_weather_aug_entcoef002_20k.md"
TOTAL_TIMESTEPS = 20000
CHECKPOINT_STEPS = [2000, 5000, 10000, 15000, 20000]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_aliases() -> dict[str, str]:
    aliases = {
        "168_scenario_episode_metrics.csv": "170_scenario_episode_metrics.csv",
        "168_action_pair_detail.csv": "170_action_pair_detail.csv",
        "168_action_change_summary.csv": "170_action_change_summary.csv",
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
    ent_coef = float(cfg["ppo_overrides"]["ent_coef"])
    if abs(ent_coef - 0.02) > 1e-12:
        raise ValueError(f"Expected ent_coef=0.02, got {ent_coef}")
    if int(cfg["total_timesteps"]) != TOTAL_TIMESTEPS:
        raise ValueError("Config total_timesteps does not match the 20K task")
    if [int(x) for x in cfg["checkpoint_steps"]] != CHECKPOINT_STEPS:
        raise ValueError("Config checkpoint_steps does not match the 20K task")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {OUT.relative_to(ROOT)}")

    original_patch = base169.base168.engine.patch_base_module
    original_globals = {
        "TASK_ID": base169.TASK_ID,
        "TASK_NAME": base169.TASK_NAME,
        "CONFIG": base169.CONFIG,
        "OUT": base169.OUT,
        "DOC": base169.DOC,
        "PROMPT": base169.PROMPT,
        "CHECKPOINT_STEPS": list(base169.CHECKPOINT_STEPS),
    }

    def patch_20k(out: Path, doc: Path, _ignored_timesteps: int, _ignored_checkpoints: list[int]):
        return original_patch(out, doc, TOTAL_TIMESTEPS, CHECKPOINT_STEPS)

    base169.base168.engine.patch_base_module = patch_20k
    base169.TASK_ID = TASK_ID
    base169.TASK_NAME = TASK_NAME
    base169.CONFIG = CONFIG
    base169.OUT = OUT
    base169.DOC = DOC
    base169.PROMPT = PROMPT
    base169.CHECKPOINT_STEPS = CHECKPOINT_STEPS

    try:
        base169.run()
    finally:
        base169.base168.engine.patch_base_module = original_patch
        for key, value in original_globals.items():
            setattr(base169, key, value)

    result_path = OUT / f"{TASK_ID}_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    aliases = copy_aliases()
    result.update(
        {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "ppo_overrides": {"ent_coef": ent_coef},
            "fixed_ppo_parameters": cfg["fixed_ppo_parameters"],
            "inherited_smoke_task": "169_sya_weather_aug_entcoef002_smoke2k",
            "prompt": PROMPT.relative_to(ROOT).as_posix(),
            "output_aliases": aliases,
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {TASK_ID} SYA weather augmentation PPO ent_coef=0.02 20K",
        "",
        "- 继承 169 的六套 2005 年 RAIN-only 情景和物理响应门槛。",
        "- 唯一 PPO 改动：`ent_coef 0.01 -> 0.02`。",
        "- reward、观测、动作空间、动作掩码、安全约束和 seed=0 保持不变。",
        f"- 训练步数：`{TOTAL_TIMESTEPS}`；checkpoint：{', '.join(map(str, CHECKPOINT_STEPS))}。",
        "- 不执行 2014 年验证。",
        f"- 训练状态：`{result.get('status')}`；action response passed：`{result.get('action_response_passed')}`。",
        f"- Prompt：`{PROMPT.relative_to(ROOT).as_posix()}`。",
        f"- Result：`{result_path.relative_to(ROOT).as_posix()}`。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()

from __future__ import annotations

import importlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

base = importlib.import_module("run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04")

base.TASK_ID = "035_06"
base.OUT = ROOT / "benchmark_results" / "035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail"
base.DOC = ROOT / "docs" / "035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail_record.md"
base.PROMPT = ROOT / "prompts" / "035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail.md"

_orig_load_config_and_env = base.load_config_and_env
_orig_guardrail = base.guardrail
_orig_write_record = base.write_record


def load_config_and_env():
    config, env_config = _orig_load_config_and_env()
    config["reward"]["reward_type"] = "harvest_ncost3_profit_scaled_0p001"
    config["reward"]["nitrogen_cost"] = 3.0
    base.direct_ppo.write_yaml(config, base.OUT / "configs" / "035_06_train_config.yaml")
    base.direct_ppo.write_yaml(env_config, base.OUT / "configs" / "035_06_resolved_env_config.yaml")
    return config, env_config


def guardrail(eval_df):
    out = _orig_guardrail(eval_df)
    out["ncost3_profit"] = (
        out["grain_yield_kg_ha"].astype(float)
        - 1.1 * out["actual_irrigation_mm"].astype(float)
        - 3.0 * out["actual_nitrogen_kg_ha"].astype(float)
    )
    out = out.sort_values(
        ["guardrail_pass", "ncost3_profit", "actual_nitrogen_kg_ha", "actual_irrigation_mm"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    out["guardrail_rank"] = range(1, len(out) + 1)
    out.to_csv(base.OUT / "evaluation" / "035_06_guardrail_ranking.csv", index=False, encoding="utf-8-sig")
    return out


def write_record(train, eval_df, ranked, elapsed):
    _orig_write_record(train, eval_df, ranked, elapsed)
    text = base.DOC.read_text(encoding="utf-8")
    text = text.replace("035_04 FQA2014 linked 自由时序 PPO 项目 simple_profit reward 记录", "035_06 FQA2014 linked 自由时序 PPO ncost3 reward 记录")
    text = text.replace("训练 reward 等价于 `final_GRNWT - 1.1I - 1.58N` 后乘 0.001", "训练 reward 等价于 `final_GRNWT - 1.1I - 3.0N` 后乘 0.001")
    text += "\n## 035_06 特别说明\n\n- 本任务只把 nitrogen_cost 从 1.58 提高到 3.0。\n- 动作空间和水氮上限没有收窄；PPO 仍可自由选择高氮，但必须用更高产量收益抵消更高氮成本。\n- 排名表同时保留项目原口径 project_simple_profit 与训练同口径 ncost3_profit。\n"
    base.DOC.write_text(text, encoding="utf-8")


base.load_config_and_env = load_config_and_env
base.guardrail = guardrail
base.write_record = write_record


def copy_03506_named_outputs() -> None:
    pairs = [
        ("035_04_training_checkpoints.csv", "035_06_training_checkpoints.csv"),
        ("035_04_checkpoint_eval_summary.csv", "035_06_checkpoint_eval_summary.csv"),
    ]
    eval_dir = base.OUT / "evaluation"
    for src_name, dst_name in pairs:
        src = eval_dir / src_name
        dst = eval_dir / dst_name
        if src.exists():
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    base.main()
    copy_03506_named_outputs()

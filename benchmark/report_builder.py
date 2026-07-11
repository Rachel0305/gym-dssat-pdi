"""Markdown reporting for benchmark runs and the 021_00 refactor record."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


DEFAULT_METHODS_SOURCE = (
    "Experience replay 可追溯至 Lin (1992)；DQN 与独立 target network 采用 Mnih et al. (2015)。",
    "n-step/multi-step Q-learning 依据 Peng and Williams (1996)；本项目冻结 n_steps=5 是工程配置。",
    "工程实现采用 Stable-Baselines3 DQN (Raffin et al., 2021)，并由 adapter 保留旧 gym-DSSAT 主线。",
    "gym-DSSAT / DSSAT-PDI 依据 Gautron et al. (2022) 与官方技术文档；YAML、hash、registry 和自动报告是本项目工程选择。",
)

DEFAULT_REFERENCES = (
    "Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529–533. https://doi.org/10.1038/nature14236",
    "Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning, 8, 293–321. https://doi.org/10.1007/BF00992699",
    "Peng, J., & Williams, R. J. (1996). Incremental Multi-Step Q-Learning. Machine Learning, 22, 283–290. https://doi.org/10.1007/BF00114731",
    "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research, 22(268), 1–8.",
    "Henderson, P., Islam, R., Bachman, P., et al. (2018). Deep Reinforcement Learning that Matters. AAAI, 32, 3207–3214. https://doi.org/10.1609/aaai.v32i1.11694",
    "Jones, J. W., Hoogenboom, G., Porter, C. H., et al. (2003). The DSSAT cropping system model. European Journal of Agronomy, 18, 235–265. https://doi.org/10.1016/S1161-0301(02)00107-7",
    "Gautron, R., Padrón, E. J., Preux, P., Bigot, J., Maillard, O.-A., & Emukpere, D. (2022). gym-DSSAT: a crop model turned into a Reinforcement Learning environment. Inria Research Report RR-9460, HAL hal-03711132.",
    "Tao, R., Zhao, P., Wu, J., et al. (2023). Optimizing Crop Management with Reinforcement Learning and Imitation Learning. IJCAI-23, 6228–6236. https://doi.org/10.24963/IJCAI.2023/691",
)


def _as_list(value: Any, fallback: str = "未提供") -> list[str]:
    if value is None:
        return [fallback]
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f"{key}: {item}" for key, item in value.items()]
    if isinstance(value, Sequence):
        return [str(item) for item in value] or [fallback]
    return [str(value)]


def _bullets(value: Any, fallback: str = "未提供") -> list[str]:
    return [f"- {item}" for item in _as_list(value, fallback)]


def _table_inventory(tables: Mapping[str, pd.DataFrame] | None) -> list[str]:
    if not tables:
        return ["- 未提供汇总表对象。"]
    rows = ["| 表 | 行数 | 列数 |", "|---|---:|---:|"]
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            rows.append(f"| `{name}` | {len(frame)} | {len(frame.columns)} |")
    return rows if len(rows) > 2 else ["- 未提供有效 DataFrame。"]


def _artifact_inventory(paths: Any) -> list[str]:
    rows: list[str] = []

    def collect(name: str, value: Any) -> None:
        if isinstance(value, Path):
            rows.append(f"- `{name}`: `{value}`")
        elif isinstance(value, str) and ("/" in value or "\\" in value):
            rows.append(f"- `{name}`: `{value}`")
        elif isinstance(value, Mapping):
            for child_name, child in value.items():
                collect(f"{name}.{child_name}" if name else str(child_name), child)

    collect("", paths or {})
    return rows or ["- 未提供输出路径。"]


def _implementation_table(value: Any) -> list[str]:
    rows = ["| 功能 | 状态 | 证据或说明 |", "|---|---|---|"]
    if isinstance(value, Mapping):
        for feature, details in value.items():
            if isinstance(details, Mapping):
                status = details.get("status", "未标记")
                evidence = details.get("evidence", details.get("notes", ""))
            else:
                status = str(details)
                evidence = ""
            rows.append(f"| {feature} | {status} | {evidence} |")
    if len(rows) == 2:
        rows.append("| 未提供 | 未验证 | 未提供实现状态 |")
    return rows


def build_markdown_report(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    figures: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    title: str = "021_00 Benchmark Framework Refactor 实验记录",
) -> Path:
    """Generate a Chinese, evidence-grounded Markdown experiment report.

    ``context`` may contain ``audit``, ``implementation``, ``smoke_test``,
    ``git_status``, ``solved``, ``unresolved`` and ``next_steps``.  Unknown or
    absent values are labelled as unverified rather than inferred.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = Path(output_path) if output_path is not None else destination / "benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(config or {})
    run_manifest = dict(manifest or {})
    ctx = dict(context or {})
    experiment = dict(cfg.get("experiment", {})) if isinstance(cfg.get("experiment"), Mapping) else {}
    experiment_id = experiment.get("experiment_id", run_manifest.get("experiment_id", "未提供"))
    config_hash = run_manifest.get("config_hash", "未提供")
    overall_status = run_manifest.get("status", ctx.get("status", "未验证"))

    smoke = ctx.get("smoke_test", run_manifest.get("smoke_test", {}))
    if isinstance(smoke, Mapping):
        smoke_status = smoke.get("status", "未验证")
        smoke_details = smoke.get("checks", smoke.get("details", "未提供 smoke test 证据"))
    else:
        smoke_status = str(smoke) if smoke else "未验证"
        smoke_details = "未提供 smoke test 证据"

    git_status = ctx.get("git_status", run_manifest.get("git", {}))
    if isinstance(git_status, Mapping):
        git_commit = git_status.get("commit", git_status.get("commit_hash", "未提交或未提供"))
        push_status = git_status.get("push_status", git_status.get("push", "未尝试或未提供"))
    else:
        git_commit = "未提供"
        push_status = str(git_status) if git_status else "未提供"

    architecture = ctx.get(
        "architecture",
        [
            "YAML config → validation → existing-result lookup → train/resume",
            "train/resume → evaluate → summarize → plot → Markdown/PPT → Git backup",
            "旧脚本通过 adapter 接入；底层 gym-DSSAT 与已验证 reward/IC 不在本任务中改动。",
        ],
    )
    lines: list[str] = [
        f"# {title}",
        "",
        f"- experiment_id: `{experiment_id}`",
        f"- config_hash: `{config_hash}`",
        f"- 总体状态: **{overall_status}**",
        "",
        "## 背景",
        "",
        *_bullets(ctx.get("background"), "项目进入论文实验前，需要统一可复用、可恢复、可追溯的 Benchmark 工程入口。"),
        "",
        "## 当前问题",
        "",
        *_bullets(ctx.get("current_problems"), "未提供本次审计识别的问题。"),
        "",
        "## 审计结果",
        "",
        *_bullets(ctx.get("audit"), "审计证据未注入；不得据此声称审计完成。"),
        "",
        "## 架构设计",
        "",
        *_bullets(architecture),
        "",
        "## 实现内容与状态",
        "",
        *_implementation_table(ctx.get("implementation", run_manifest.get("features"))),
        "",
        "## 配置系统",
        "",
        *_bullets(ctx.get("configuration"), "配置详情见本次运行保存的 YAML/manifest；本报告未接收到配置摘要。"),
        "",
        "## 配置哈希与结果复用",
        "",
        *_bullets(ctx.get("reuse"), "结果复用状态未提供；不能推断旧结果与新配置相同。"),
        "",
        "## Checkpoint resume 与运行模式",
        "",
        *_bullets(ctx.get("resume"), "checkpoint/resume 验证状态未提供。"),
        "",
        "## 训练与评估适配",
        "",
        *_bullets(ctx.get("training_evaluation"), "训练/评估适配状态未提供。"),
        "",
        "## 统计模块",
        "",
        "- WP_ET = yield / ET / 10；IWP_gross = yield / irrigation / 10。",
        "- PFP_N = yield / applied N；NUtE = yield / crop N uptake。",
        "- 分母为 0 或缺失时记录为 NA，不生成无穷值。",
        "- 小样本仅报告 mean、standard deviation、minimum、maximum、median 与跨 seed CV，不强行进行显著性检验。",
        "",
        "### 已生成汇总表",
        "",
        *_table_inventory(tables),
        "",
        "## 绘图模块",
        "",
        "- Python/matplotlib 后端；PNG 300 dpi 与可编辑 SVG 成对输出。",
        "- 图中英文标题与单位；缺失 baseline 或缺失指标时显示不可用，不伪造数值。",
        "",
        "## 报告模块",
        "",
        "- Markdown 与 PPTX 从同一 config、manifest、表格和图形清单生成。",
        "- PPTX 至少包含 config → validation → reuse → train/resume → evaluate → report → Git 的流程图。",
        "",
        "## Smoke test",
        "",
        f"- 状态: **{smoke_status}**",
        *_bullets(smoke_details),
        "",
        "## 实际运行命令",
        "",
        *_bullets(ctx.get("commands"), "未提供。"),
        "",
        "## 输出文件",
        "",
        *_artifact_inventory({"tables": ctx.get("output_paths", {}), "figures": figures or {}}),
        "",
        "## 已解决问题",
        "",
        *_bullets(ctx.get("solved"), "未提供已验证解决项。"),
        "",
        "## 未解决问题与已知限制",
        "",
        *_bullets(ctx.get("unresolved"), "未提供；不等于不存在限制。"),
        "",
        "## 对已有结果的影响",
        "",
        *_bullets(ctx.get("legacy_impact"), "未提供影响评估。"),
        "",
        "## 后续实验计划",
        "",
        *_bullets(ctx.get("next_steps"), "待审计与 smoke test 结果确认后再启动 021_01–021_08。"),
        "",
        "## Methods Source",
        "",
        *_bullets(ctx.get("methods_source", DEFAULT_METHODS_SOURCE)),
        "",
        "## References",
        "",
        *[f"{index}. {item}" for index, item in enumerate(_as_list(ctx.get("references", DEFAULT_REFERENCES)), 1)],
        "",
        "## Git 状态",
        "",
        f"- commit: `{git_commit}`",
        f"- push: `{push_status}`",
        "",
        "## 最终结论",
        "",
        *_bullets(ctx.get("conclusion"), f"当前任务状态为 {overall_status}；只把有测试证据的功能视为已完成。"),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Markdown report written to %s", report_path)
    return report_path


# Backwards-friendly alias.
build_report = build_markdown_report

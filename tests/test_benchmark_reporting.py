"""Report package tests using tiny in-memory benchmark tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation

from benchmark.plot_builder import build_plots
from benchmark.ppt_builder import audit_pptx, build_pptx_report
from benchmark.report_builder import build_markdown_report
from benchmark.summary_builder import build_summary_outputs


def _season() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "smoke",
                "config_hash": "abc",
                "station_code": "HLA",
                "year": 2007,
                "seed": 0,
                "scenario": "null",
                "checkpoint": 0,
                "yield_kg_ha": 5000.0,
                "biomass_kg_ha": 13000.0,
                "irrigation_mm": 0.0,
                "nitrogen_kg_ha": 0.0,
                "et_mm": 300.0,
                "nitrogen_uptake_kg_ha": 120.0,
                "reward_total": 0.0,
            },
            {
                "experiment_id": "smoke",
                "config_hash": "abc",
                "station_code": "HLA",
                "year": 2007,
                "seed": 0,
                "scenario": "dqn",
                "checkpoint": 100,
                "yield_kg_ha": 6000.0,
                "biomass_kg_ha": 14500.0,
                "irrigation_mm": 30.0,
                "nitrogen_kg_ha": 0.0,
                "et_mm": 320.0,
                "nitrogen_uptake_kg_ha": 140.0,
                "reward_total": 970.0,
            },
        ]
    )


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["smoke", "smoke"],
            "config_hash": ["abc", "abc"],
            "station_code": ["HLA", "HLA"],
            "year": [2007, 2007],
            "seed": [0, 0],
            "scenario": ["dqn", "dqn"],
            "checkpoint": [100, 100],
            "dap": [10, 20],
            "action_irrigation": [30.0, 0.0],
            "action_nitrogen": [0.0, 0.0],
            "cumulative_irrigation": [30.0, 30.0],
            "cumulative_nitrogen": [0.0, 0.0],
            "reward": [-30.0, 1000.0],
            "terminal_reward": [0.0, 1000.0],
            "final_yield": [6000.0, 6000.0],
        }
    )


def test_summary_plot_markdown_and_pptx_outputs(tmp_path: Path) -> None:
    config = {"experiment": {"experiment_id": "smoke"}, "reporting": {"dpi": 300}}
    manifest = {"experiment_id": "smoke", "config_hash": "abc", "status": "completed"}
    summaries = build_summary_outputs(_season(), _daily(), tmp_path / "summaries", config=config, manifest=manifest)
    assert (tmp_path / "summaries" / "benchmark_summary.xlsx").exists()
    assert pd.isna(summaries["tables"]["season_summary"].loc[0, "IWP_gross_kg_m3"])
    plots = build_plots(_season(), _daily(), tmp_path / "figures", config=config, manifest=manifest)
    assert plots["artifacts"]
    assert list((tmp_path / "figures").glob("*.png"))
    assert list((tmp_path / "figures").glob("*.svg"))
    report = build_markdown_report(
        tmp_path / "reports",
        config=config,
        manifest=manifest,
        tables=summaries["tables"],
        figures=plots["artifacts"],
        context={"audit": ["completed"], "smoke_test": {"status": "passed", "checks": ["pipeline"]}},
    )
    assert "Methods Source" in report.read_text(encoding="utf-8")
    pptx = build_pptx_report(
        tmp_path / "reports",
        config=config,
        manifest=manifest,
        tables=summaries["tables"],
        figures=plots["artifacts"],
        context={"smoke_test": {"status": "passed", "checks": ["pipeline"]}},
    )
    assert len(Presentation(pptx).slides) >= 14
    qa = audit_pptx(pptx)
    assert "high_severity_defects: 0" in qa


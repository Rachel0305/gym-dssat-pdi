from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_lc2010_seed0_50k_cross_year_transfer_032_07 as transfer


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_08_lc2010_seed0_50k_remaining_year_transfer.md"
OUT = ROOT / "benchmark_results" / "032_08_lc2010_seed0_50k_remaining_year_transfer"
DOC = ROOT / "docs" / "032_08_lc2010_seed0_50k_remaining_year_transfer_record.md"
TARGET_YEARS = [2005, 2006, 2007, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def configure_transfer_module() -> None:
    transfer.PROMPT = PROMPT
    transfer.OUT = OUT
    transfer.DOC = DOC
    transfer.TARGET_YEARS = TARGET_YEARS
    transfer.direct_ppo.OUTPUT_ROOT = OUT


def postprocess_record() -> None:
    for path in sorted(OUT.rglob("*032_07*")):
        if path.is_file():
            new_path = path.with_name(path.name.replace("032_07", "032_08"))
            if new_path != path:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, new_path)

    old_doc = DOC
    if old_doc.exists():
        text = old_doc.read_text(encoding="utf-8")
        text = text.replace("# 032_07 LC2010 seed0/50k cross-year transfer record", "# 032_08 LC2010 seed0/50k remaining-year transfer record")
        text = text.replace("Target years: LC2012-LC2015.", "Target years: LC2005-LC2007 and LC2016-LC2023.")
        text = text.replace("LC2011 is excluded because current completed-baseline daily files do not contain DSSAT auto for that year.", "LC2008, LC2009, and LC2011 are excluded because current completed-baseline daily files do not contain DSSAT auto for those years.")
        text = text.replace("032_07", "032_08")
        old_doc.write_text(text, encoding="utf-8")
        (OUT / DOC.name).write_text(text, encoding="utf-8")

    result_path = OUT / "032_07_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["task"] = "032_08_lc2010_seed0_50k_remaining_year_transfer"
        result["target_years"] = TARGET_YEARS
        for key in ["eval_summary", "comparison", "manifest", "record_md"]:
            if key in result:
                result[key] = str(result[key]).replace("032_07", "032_08")
        (OUT / "032_08_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison = OUT / "tables" / "032_08_transfer_vs_expert_comparison.csv"
    if comparison.exists():
        df = pd.read_csv(comparison)
        print("\n032_08 comparison:")
        print(df.to_string(index=False))


def main() -> None:
    configure_transfer_module()
    transfer.main()
    postprocess_record()
    print(
        json.dumps(
            {
                "task": "032_08_lc2010_seed0_50k_remaining_year_transfer",
                "training_run": False,
                "dssat_run": True,
                "target_years": TARGET_YEARS,
                "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
                "out_dir": str(OUT.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

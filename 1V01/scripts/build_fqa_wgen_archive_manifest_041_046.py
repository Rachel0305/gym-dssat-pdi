"""Build a bounded SHA-256 inventory for the FQA 041-046 GitHub backup."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/fqa_wgen_041_046_file_manifest.csv"
RESULT_DIRS = {
    41: "fqa_weather_resume_041",
    42: "fqa_runtime_wgen_smoke_042",
    43: "fqa_archived_weather_pilot_043",
    44: "fqa_wgen_ppo_smoke_044",
    45: "fqa_wgen_multiyear_heldout_gate_045",
    46: "fqa_wgen_full_pool_qc_046",
}
DOCS = {
    "fqa_weather_resume_041.md",
    "fqa_runtime_wgen_smoke_042.md",
    "fqa_yc_coordinate_and_weather_archive_043.md",
    "fqa_wgen_ppo_smoke_044.md",
    "fqa_wgen_multiyear_heldout_gate_045.md",
    "fqa_wgen_full_pool_qc_046.md",
    "fqa_wgen_041_046_archive_index.md",
}
SCRIPTS = {
    "verify_fqa_wgen_archive_041_046.py",
    "build_fqa_wgen_archive_manifest_041_046.py",
}
EXCLUDED_DIRS = {"temp", "tmp", "__pycache__", "rendered_inputs", "runtime_templates", "tensorboard"}
MODEL = "results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip"


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
        return False
    if "models" in path.parts and rel != MODEL:
        return False
    if path.suffix.lower() in {".pyc", ".wth", ".sol", ".out"}:
        return False
    if path.suffix.lower() == ".zip" and rel != MODEL:
        return False
    if path.stat().st_size > 1_000_000:
        raise RuntimeError(f"Archive file exceeds 1 MB review cap: {rel}")
    return True


def main() -> None:
    selected: list[tuple[Path, str]] = []
    for step in range(41, 47):
        matches = sorted((ROOT / "prompt_02").glob(f"{step:03d}_*.md"))
        matches = [p for p in matches if p.name in {
            "041_fqa_d222_weather_resume_static_cli.md",
            "042_fqa_runtime_wgen_weather_archive_smoke.md",
            "043_fqa_yc_coordinate_control_and_archived_weather_pilot.md",
            "044_fqa_wgen_ppo_2k_episode_archive_smoke.md",
            "045_fqa_wgen_multiyear_heldout_archive_gate.md",
            "046_fqa_wgen_full_pool_climate_qc.md",
        }]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one prompt for step {step}: {matches}")
        selected.append((matches[0], f"prompt_{step:03d}"))
        directory = ROOT / "results" / RESULT_DIRS[step]
        for path in sorted(directory.rglob("*")):
            if path.is_file() and included(path):
                selected.append((path, f"result_{step:03d}"))
    for name in sorted(DOCS):
        selected.append((ROOT / "docs" / name, "research_record"))
    for name in sorted(SCRIPTS):
        selected.append((ROOT / "scripts" / name, "archive_tool"))
    paths = [path.relative_to(ROOT).as_posix() for path, _ in selected]
    if len(paths) != len(set(paths)) or any(not path.is_file() for path, _ in selected):
        raise RuntimeError("Duplicate or missing archive source")
    rows = []
    for path, role in sorted(selected, key=lambda x: x[0].relative_to(ROOT).as_posix()):
        data = path.read_bytes()
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest().upper(), "role": role})
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "role"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"archive_files={len(rows)} total_bytes={sum(row['bytes'] for row in rows)} manifest={OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

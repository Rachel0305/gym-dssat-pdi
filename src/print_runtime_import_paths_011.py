from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def safe_file(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__file__", ""))
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"IMPORT_ERROR: {exc}"


def main() -> None:
    out = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "gym_dssat_pdi": safe_file("gym_dssat_pdi"),
        "stable_baselines3": safe_file("stable_baselines3"),
        "gym": safe_file("gym"),
        "gymnasium": safe_file("gymnasium"),
        "sb3_wrapper": safe_file("sb3_wrapper"),
        "torch": safe_file("torch"),
    }
    output_path = Path("DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/runtime_import_paths.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

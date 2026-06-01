from __future__ import annotations

import argparse
from pathlib import Path
import shutil


BASE = Path("/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs")
MAIZE_TEMPLATE = BASE / "configs/maize/dssat_pdi.jinja2"
MAIZE_CONFIG = BASE / "configs/maize/env_config.yml"
UTILS = BASE / "utils/utils.py"


def backup(path: Path) -> None:
    backup_path = path.with_name(path.name + ".before_turfac_patch")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"Backup written: {backup_path}")
    else:
        print(f"Backup exists: {backup_path}")


def patch_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"Already patched: {path}")
        return
    if old not in text:
        raise RuntimeError(f"Pattern not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched: {path}")


def patch_template() -> None:
    backup(MAIZE_TEMPLATE)
    patch_once(
        MAIZE_TEMPLATE,
        "      SWFAC: |\n        swfac = SWFAC\n",
        "      SWFAC: |\n        swfac = SWFAC\n      TURFAC: |\n        turfac = TURFAC\n",
    )
    patch_once(
        MAIZE_TEMPLATE,
        "          'swfac': swfac,\n          'pcngrn': pcngrn,\n",
        "          'swfac': swfac,\n          'turfac': turfac,\n          'pcngrn': pcngrn,\n",
    )


def patch_config() -> None:
    backup(MAIZE_CONFIG)
    patch_once(
        MAIZE_CONFIG,
        "  swfac:\n    type: float\n    low: 0\n    high: 1\n    info: index of plant water stress (unitless)\n",
        "  swfac:\n    type: float\n    low: 0\n    high: 1\n    info: index of plant water stress (unitless)\n  turfac:\n    type: float\n    low: 0\n    high: 1\n    info: index of plant transpiration water stress (unitless)\n",
    )
    patch_once(
        MAIZE_CONFIG,
        "      - swfac\n      - nstres\n",
        "      - swfac\n      - turfac\n      - nstres\n",
    )


def patch_utils() -> None:
    backup(UTILS)
    patch_once(
        UTILS,
        "            state['swfac'] = 1 - state['swfac']\n",
        "            state['swfac'] = 1 - state['swfac']\n            state['turfac'] = 1 - state['turfac']\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental patch for exposing TURFAC in maize all mode. "
            "In the current DSSAT-PDI build this can make maize reset return an empty observation, "
            "so use --apply only for controlled debugging and --restore immediately if reset breaks."
        )
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        for path in [MAIZE_TEMPLATE, MAIZE_CONFIG, UTILS]:
            backup_path = path.with_name(path.name + ".before_turfac_patch")
            if not backup_path.exists():
                raise FileNotFoundError(f"Missing backup: {backup_path}")
            shutil.copy2(backup_path, path)
            print(f"Restored: {path}")
        return

    if not args.apply:
        print("No changes made. Pass --apply to test the experimental TURFAC patch or --restore to revert it.")
        return

    patch_template()
    patch_config()
    patch_utils()
    print("Done. Restart Python processes before creating new gym environments.")


if __name__ == "__main__":
    main()

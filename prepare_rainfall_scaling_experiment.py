from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dssat_site_config import SITE_CONFIGS


def scale_token(scale: float) -> str:
    return f"scale{scale:g}".replace(".", "p")


def scale_weather(source: Path, destination: Path, scale: float) -> dict:
    total_before = 0.0
    total_after = 0.0
    out_lines = []
    for line in source.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            rain = float(parts[4])
            scaled_rain = max(0.0, rain * scale)
            total_before += rain
            total_after += scaled_rain
            out_lines.append(
                f"{parts[0]:>7} {float(parts[1]):5.1f} {float(parts[2]):5.1f} "
                f"{float(parts[3]):5.1f} {scaled_rain:5.1f}"
            )
        else:
            out_lines.append(line)
    destination.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "weather_source": source.as_posix(),
        "weather_scaled": destination.as_posix(),
        "annual_rain_before": round(total_before, 3),
        "annual_rain_after": round(total_after, 3),
    }


def copy_required_inputs(site: str, source_dir: Path, target_dir: Path) -> dict:
    config = SITE_CONFIGS[site]
    files = [
        config["template"],
        config["soil"],
        "MZCER048.CUL",
    ]
    copied = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = source_dir / name
        dst = target_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append(dst.as_posix())
    return {"copied_inputs": copied}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="HL")
    parser.add_argument("--scales", default="1.0,0.8,0.6,0.4")
    parser.add_argument("--source-dir", default="my_data")
    parser.add_argument("--output-root", default="my_data_rain_scaling")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    sites = [value.strip().upper() for value in args.sites.split(",") if value.strip()]
    scales = [float(value.strip()) for value in args.scales.split(",") if value.strip()]
    manifest = []

    for site in sites:
        if site not in SITE_CONFIGS:
            raise ValueError(f"Unknown site: {site}")
        weather_name = SITE_CONFIGS[site]["weather"]
        for scale in scales:
            target_dir = output_root / site / scale_token(scale)
            entry = {
                "site": site,
                "scale": scale,
                "data_dir": target_dir.as_posix(),
                "weather_file": weather_name,
            }
            entry.update(copy_required_inputs(site, source_dir, target_dir))
            entry.update(scale_weather(source_dir / weather_name, target_dir / weather_name, scale))
            manifest.append(entry)
            print(f"{site} {scale:g}: {entry['annual_rain_before']} -> {entry['annual_rain_after']} mm")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "rain_scaling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

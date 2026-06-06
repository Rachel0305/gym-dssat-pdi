from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "ppo_observed_years"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_ppo_observed_years.yaml"

SITE_INFO = {
    "HLA": {"short": "HL", "template": "UFGA8201-HL.jinja2", "soil": "HL.SOL", "weather_prefix": "HLA", "expected_weather": "CNHL0701.WTH"},
    "SYA": {"short": "SY", "template": "UFGA8201-SY.jinja2", "soil": "SY.SOL", "weather_prefix": "SYA", "expected_weather": "CNSY1201.WTH"},
    "LCA": {"short": "LC", "template": "UFGA8201-LC.jinja2", "soil": "LC.SOL", "weather_prefix": "LCA", "expected_weather": "CNLC0801.WTH"},
    "YCA": {"short": "YC", "template": "UFGA8201-YC.jinja2", "soil": "YC.SOL", "weather_prefix": "YCA", "expected_weather": "CNYC0801.WTH"},
    "FQA": {"short": "FQ", "template": "UFGA8201-FQ.jinja2", "soil": "FQ.SOL", "weather_prefix": "FQA", "expected_weather": "CNFQ0701.WTH"},
}


def load_yaml(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def yyddd(date_text: str) -> str:
    ts = pd.Timestamp(date_text)
    return f"{ts.year % 100:02d}{ts.dayofyear:03d}"


def force_management_levels_on(text: str) -> str:
    return re.sub(
        r"(Sim\d{4}\s+1\s+1\s+0\s+0\s+1\s+)0(\s+)0",
        r"\g<1>1\g<2>1",
        text,
    )


def ensure_irrigation_section(text: str, safe_yyddd: str, year: int) -> str:
    if "*IRRIGATION AND WATER MANAGEMENT" in text:
        return text
    section = (
        "*IRRIGATION AND WATER MANAGEMENT\n"
        "@I  EFIR  IDEP  ITHR  IEPT  IOFF  IAME  IAMT IRNAME\n"
        f" 1     1    30    50   100 GS000 IR001    10 {year}\n"
        "@I IDATE  IROP IRVAL\n"
        f" 1 {safe_yyddd} IR001     0\n\n"
    )
    return text.replace("*FERTILIZERS (INORGANIC)", section + "*FERTILIZERS (INORGANIC)")


def ensure_fertilizer_section(text: str, safe_yyddd: str, year: int) -> str:
    if "*FERTILIZERS (INORGANIC)" in text:
        return text
    section = (
        "*FERTILIZERS (INORGANIC)\n"
        "@F FDATE  FMCD FACD FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME\n"
        f" 1 {safe_yyddd} FE005 AP002     5     0   -99   -99   -99   -99   -99 {year}\n\n"
    )
    return text + "\n" + section


def reset_static_application_rows(text: str, safe_yyddd: str, year: int) -> str:
    output: list[str] = []
    skip_application_rows = False
    pending_zero_row: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("*"):
            if pending_zero_row is not None:
                output.append(pending_zero_row)
                pending_zero_row = None
            skip_application_rows = False
        if line.startswith("@I IDATE") or line.startswith("@F FDATE"):
            skip_application_rows = True
            output.append(line)
            if line.startswith("@I IDATE"):
                pending_zero_row = f" 1 {safe_yyddd} IR001     0"
            else:
                pending_zero_row = f" 1 {safe_yyddd} FE005 AP002     5     0   -99   -99   -99   -99   -99 {year}"
            continue
        if skip_application_rows:
            if stripped.startswith("@") or stripped.startswith("*"):
                if pending_zero_row is not None:
                    output.append(pending_zero_row)
                    pending_zero_row = None
                skip_application_rows = False
            elif stripped == "":
                continue
            elif re.match(r"^\s*\d+\s+\d{5}\b", line):
                continue
        output.append(line)
    if pending_zero_row is not None:
        output.append(pending_zero_row)
    return "\n".join(output) + "\n"


def safe_render_template(
    station: str,
    year: int,
    planting_date: str,
    output_root: Path | None = None,
    run_tag: str = "ppo",
) -> Path:
    output_root = output_root or PPO_ROOT
    info = SITE_INFO[station]
    src = PROJECT_ROOT / "my_data" / info["template"]
    out_dir = output_root / "rendered_inputs" / station / str(year) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{station}_{year}_{run_tag}.jinja2"

    text = src.read_text(encoding="utf-8", errors="replace")
    base_year_match = re.search(r"\b(20\d{2})\b", text)
    base_year = int(base_year_match.group(1)) if base_year_match else year
    old_yy = f"{base_year % 100:02d}"
    new_yy = f"{year % 100:02d}"
    planting = pd.Timestamp(planting_date)
    start = planting - pd.Timedelta(days=4)
    emergence = planting + pd.Timedelta(days=7)
    safe_date = yyddd(planting_date)

    text = re.sub(rf"\b{old_yy}(\d{{3}})\b", rf"{new_yy}\1", text)
    text = re.sub(r"\b20\d{2}\b", str(year), text)
    text = re.sub(r"\bSim20\d{2}\b", f"Sim{year}", text)
    text = re.sub(r"CN([A-Z]{2})20\d{2}", lambda m: f"CN{m.group(1)}{year}", text)
    text = re.sub(r"(@P PDATE EDATE[^\n]*\n\s*1\s+)(\d{5})(\s+)(\d{5})", rf"\g<1>{safe_date}\g<3>{yyddd(emergence.strftime('%Y-%m-%d'))}", text)
    text = re.sub(r"(\sS\s+)(\d{5})(\s+2150)", rf"\g<1>{yyddd(start.strftime('%Y-%m-%d'))}\g<3>", text)
    text = re.sub(r"(\sMZ\s+)(\d{5})(\s+100)", rf"\g<1>{yyddd(start.strftime('%Y-%m-%d'))}\g<3>", text)
    text = force_management_levels_on(text)
    text = ensure_fertilizer_section(text, safe_date, year)
    text = ensure_irrigation_section(text, safe_date, year)
    text = reset_static_application_rows(text, safe_date, year)
    out.write_text(text, encoding="utf-8")
    return out


def copy_weather_to_rendered_dir(station: str, year: int, template: Path, config: dict) -> Path:
    info = SITE_INFO[station]
    wth_dir = PROJECT_ROOT / config["paths"]["wth_dir"]
    source_weather = wth_dir / station / f"{info['weather_prefix']}{year}.WTH"
    rendered_weather = template.parent / info["expected_weather"]
    if source_weather.exists():
        shutil.copyfile(source_weather, rendered_weather)
    return rendered_weather


def build_env_args(
    station: str,
    year: int,
    planting_date: str,
    seed: int,
    config: dict,
    run_tag: str,
    evaluation: bool = False,
    mode: str = "all",
) -> dict:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    info = SITE_INFO[station]
    template = safe_render_template(station, year, planting_date, output_root, run_tag)
    weather = copy_weather_to_rendered_dir(station, year, template, config)
    cultivar = PROJECT_ROOT / config["paths"]["cultivar_file"]
    soil = PROJECT_ROOT / config["paths"]["my_data_dir"] / info["soil"]
    missing = [p for p in [template, weather, cultivar, soil] if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p) for p in missing))
    log_dir = output_root / "logs" / station
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "log_saving_path": str(log_dir / f"{station}_{year}_{run_tag}.log"),
        "mode": mode,
        "seed": int(seed),
        "random_weather": False,
        "evaluation": evaluation,
        "fileX_template_path": str(template),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(cultivar), str(weather), str(soil)],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }


def check_rendered_input(template: Path, weather: Path | None = None) -> list[dict]:
    text = template.read_text(encoding="utf-8", errors="replace") if template.exists() else ""
    checks = [
        ("template_exists", template.exists(), str(template)),
        ("wth_section_present", "@W" in text or ".WTH" in text, ""),
        ("irrigation_section_present", "*IRRIGATION AND WATER MANAGEMENT" in text, ""),
        ("fertilizer_section_present", "*FERTILIZERS (INORGANIC)" in text, ""),
        ("simulation_controls_present", "*SIMULATION CONTROLS" in text, ""),
        ("planting_section_present", "*PLANTING DETAILS" in text, ""),
        ("mi_mf_enabled", bool(re.search(r"Sim\d{4}\s+1\s+1\s+0\s+0\s+1\s+1\s+1", text)), ""),
    ]
    if weather is not None:
        checks.append(("weather_exists", weather.exists(), str(weather)))
    return [
        {"check_item": name, "status": "pass" if passed else "fail", "details": details}
        for name, passed, details in checks
    ]


def write_render_check(rows: list[dict], config: dict) -> Path:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    out = output_root / "evaluation" / "rendered_input_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out

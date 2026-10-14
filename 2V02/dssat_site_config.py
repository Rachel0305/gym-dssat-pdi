from pathlib import Path


SITE_CONFIGS = {
    "HL": {
        "template": "UFGA8201-HL.jinja2",
        "weather": "CNHL0701.WTH",
        "soil": "HL.SOL",
    },
    "SY": {
        "template": "UFGA8201-SY.jinja2",
        "weather": "CNSY1201.WTH",
        "soil": "SY.SOL",
    },
    "YC": {
        "template": "UFGA8201-YC.jinja2",
        "weather": "CNYC0801.WTH",
        "soil": "YC.SOL",
    },
    "LC": {
        "template": "UFGA8201-LC.jinja2",
        "weather": "CNLC0801.WTH",
        "soil": "MY_SOIL.SOL",
    },
    "FQ": {
        "template": "UFGA8201-FQ.jinja2",
        "weather": "CNFQ0701.WTH",
        "soil": "FQ.SOL",
    },
}


def _suffix_path(path: Path, suffix_token: str | None) -> Path:
    if not suffix_token:
        return path
    candidate = path.with_name(f"{path.stem}{suffix_token}{path.suffix}")
    return candidate if candidate.exists() else path


def build_env_args(
    *,
    site: str = "HL",
    mode: str = "all",
    seed: int = 123,
    data_dir: str = "./my_data",
    prefer_suffix: str | None = None,
    log_saving_path: str | None = None,
    run_dssat_location: str = "/opt/dssat_pdi/run_dssat",
) -> dict:
    site_key = site.upper()
    if site_key not in SITE_CONFIGS:
        valid = ", ".join(sorted(SITE_CONFIGS))
        raise ValueError(f"Unknown site {site!r}. Valid sites: {valid}")

    base_dir = Path(data_dir)
    config = SITE_CONFIGS[site_key]
    template = _suffix_path(base_dir / config["template"], prefer_suffix)
    cultivar = _suffix_path(base_dir / "MZCER048.CUL", prefer_suffix)
    weather = _suffix_path(base_dir / config["weather"], prefer_suffix)
    soil = _suffix_path(base_dir / config["soil"], prefer_suffix)

    required_paths = [template, cultivar, weather, soil]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DSSAT input files: " + ", ".join(missing))

    env_args = {
        "mode": mode,
        "seed": seed,
        "random_weather": False,
        "evaluation": False,
        "fileX_template_path": str(template),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(cultivar), str(weather), str(soil)],
        "run_dssat_location": run_dssat_location,
    }
    if log_saving_path:
        env_args["log_saving_path"] = log_saving_path
    return env_args


def describe_env_args(env_args: dict) -> str:
    paths = [env_args["fileX_template_path"], *env_args["auxiliary_file_paths"]]
    return "\n".join(f"  - {path}" for path in paths)

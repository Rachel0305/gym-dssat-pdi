from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppo_safe_rendering import DEFAULT_CONFIG, PROJECT_ROOT, load_yaml


def cross_validation_type(n_years: int) -> str:
    if n_years == 2:
        return "two_year_cross_validation"
    if n_years == 3:
        return "three_year_leave_one"
    if n_years == 4:
        return "four_year_leave_one"
    return f"{n_years}_year_leave_one"


def generate_plan(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for station, years in config["observed_years"].items():
        n_years = len(years)
        cv_type = cross_validation_type(n_years)
        for train in years:
            validations = [item for item in years if int(item["year"]) != int(train["year"])]
            rows.append(
                {
                    "station": station,
                    "experiment_group": f"{station}_train{train['year']}_seed{config['seed']}",
                    "train_year": int(train["year"]),
                    "train_year_label": train["label"],
                    "validation_years": ",".join(str(item["year"]) for item in validations),
                    "validation_year_labels": ",".join(item["label"] for item in validations),
                    "num_observed_years": n_years,
                    "cross_validation_type": cv_type,
                    "notes": "limited_two_year_cross_validation" if n_years == 2 else "observed_year_leave_one",
                }
            )
    return pd.DataFrame(rows)


def write_plan(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_yaml(config_path)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    out = output_root / "configs" / "ppo_observed_year_experiment_plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_plan(config).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def find_year(config: dict, station: str, year: int) -> dict:
    for item in config["observed_years"][station]:
        if int(item["year"]) == int(year):
            return item
    raise KeyError(f"{station} {year} not found in observed_years")


def validation_years_for(config: dict, station: str, train_year: int) -> list[dict]:
    return [item for item in config["observed_years"][station] if int(item["year"]) != int(train_year)]


def main() -> None:
    print(write_plan().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

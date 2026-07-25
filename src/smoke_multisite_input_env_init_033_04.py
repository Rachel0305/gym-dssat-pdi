from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
from run_multisite_input_enabled_five_site_half_split_maskableppo_rerun_033_04 import (
    OUT,
    load_split_available_weather,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    config = batch.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    split = load_split_available_weather()
    selection = batch.build_selection(split)
    env_config = batch.direct_ppo.build_env_config(config, selection)

    rows: list[dict] = []
    for station, group in split.groupby("station_code"):
        year = int(group.sort_values(["split", "year"])["year"].iloc[0])
        env = None
        row = {"station_code": station, "year": year, "status": "ok"}
        try:
            env = base.make_env(
                config=config,
                env_config=env_config,
                station=station,
                year=year,
                seed=0,
                run_tag=f"{station}_{year}_033_04_smoke",
                evaluation=True,
            )
            obs, info = env.reset()
            row["obs_shape"] = str(getattr(obs, "shape", ""))
            row["action_space"] = str(env.action_space)
            if hasattr(env, "action_masks"):
                row["mask"] = ";".join(map(str, env.action_masks().astype(int).tolist()))
        except Exception:
            row["status"] = "failed"
            row["notes"] = traceback.format_exc()[-4000:]
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        rows.append(row)

    out_csv = OUT / "configs" / "033_04_env_init_smoke.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_csv)
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

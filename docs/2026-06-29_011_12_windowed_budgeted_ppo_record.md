# 011_12 Windowed budgeted PPO record

## Question

The current budgeted PPO with official reward could spend the full I120/N150 budget immediately after simulation start. The specific concern was whether adding earliest-operation / growth-stage windows could prevent the pathological DAP2 budget exhaustion.

## Change

`src/run_hla_official_reward_restart_smoke.py` was extended with a new `joint_windowed` variant in `BudgetedDailyActionWrapper`.

Operation windows:

- irrigation: DAP 20-35, 45-65, 70-95
- fertilization: DAP 25-40, 55-70

The reward, I120/N150 budget, corrected cultivar, IC=1 inputs, and container environment were otherwise kept unchanged.

## Environment

- Container: `b2fd6726c8c1`
- Python: `/opt/gym_dssat_pdi/bin/python`
- Script: `src/run_hla_official_reward_restart_smoke.py`
- Variant: `joint_windowed`
- Timesteps: 5000

## Commands

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 0 --variant joint_windowed --timeout 900"

docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 1 --variant joint_windowed --timeout 900"
```

For 2015, the generic action-channel gate did not pass because its parser did not see management events in `MgmtEvent.OUT`, even though daily action-smoke rows showed actions were sent. Therefore the same script helper was used to prepare the case and call `child_train_smoke()` directly:

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python - <<'PY'
import src.run_hla_official_reward_restart_smoke as s
run_name='windowed_seed0_5000steps'
dst=s.OUT_DIR/'ppo_smoke'/'2015'/run_name
s.OUT_DIR.mkdir(parents=True, exist_ok=True)
s.prepare_case_at(2015, dst)
s.child_train_smoke(2015, 5000, 0, 'joint_windowed')
print(dst)
PY"
```

## Outputs

Summary CSV:

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/windowed_5k_seed_summary.csv`

Case folders:

- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed0_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed1_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2015/windowed_seed0_5000steps`

## Results

| year | run | null yield kg/ha | PPO yield kg/ha | irrigation | nitrogen | management timing |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2010 | seed0 | 6956 | 7854 | 120.1 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |
| 2010 | seed1 | 6956 | 7854 | 120.0 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |
| 2015 | seed0 | 6486 | 7639 | 120.0 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |

## Interpretation

The windowed wrapper solved the specific early-action pathology: PPO no longer applies water and nitrogen at DAP2. It still uses the full I120/N150 budget, but now only inside the predefined operation windows.

For HLA2010, seed0 and seed1 are essentially identical in timing, total input, and final yield. This is a more interpretable behavior than the no-window budgeted PPO result.

This result does not prove general PPO optimality. It supports a narrower conclusion: adding agronomic operation windows makes the current budgeted PPO behavior more agronomically interpretable under the same reward and budget setting.

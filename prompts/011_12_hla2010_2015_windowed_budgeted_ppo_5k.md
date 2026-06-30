# 011_12 HLA2010/2015 windowed budgeted PPO 5K

## Purpose

Current budgeted PPO with the official reward could use the full I120/N150 budget immediately after simulation start, e.g. DAP2/DAP9/DAP16. This is not agronomically interpretable even if the final yield is high.

This test adds an operation timing filter to the current budget wrapper:

- irrigation allowed only in DAP 20-35, 45-65, 70-95
- fertilization allowed only in DAP 25-40, 55-70

The goal is not to tune reward or increase budget. The goal is only to test whether an earliest-operation / growth-stage window can prevent early DAP2 budget exhaustion while keeping the rest of the training setup unchanged.

## Environment

- Container: `b2fd6726c8c1`
- Python: `/opt/gym_dssat_pdi/bin/python`
- Script: `src/run_hla_official_reward_restart_smoke.py`
- Variant: `joint_windowed`
- Timesteps: 5000
- Main test: HLA2010 seed0
- Extension after seed0 looked reasonable: HLA2010 seed1 and HLA2015 seed0

## Commands

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 0 --variant joint_windowed --timeout 900"

docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 1 --variant joint_windowed --timeout 900"
```

For 2015, the generic `action_channel_smoke` gate did not pass because its event parser did not see management events in `MgmtEvent.OUT`, even though the daily action-smoke CSV showed the actions were sent to the environment. Therefore the case was prepared with the same script helper and the same child training function was called directly:

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

Main summary:

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/windowed_5k_seed_summary.csv`

Case folders:

- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed0_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed1_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2015/windowed_seed0_5000steps`

## Result summary

| year | run | null yield kg/ha | PPO yield kg/ha | irrigation | nitrogen | management timing |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2010 | seed0 | 6956 | 7854 | 120.1 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |
| 2010 | seed1 | 6956 | 7854 | 120.0 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |
| 2015 | seed0 | 6486 | 7639 | 120.0 mm, 5 events | 150 kg/ha, 3 events | I at DAP20/27/34/45/52; N at DAP27/34/59 |

## Interpretation

The windowed budget wrapper solved the specific early-action pathology: the model no longer applies water and nitrogen at DAP2. It still uses the full I120/N150 budget, but now inside predefined agronomic operation windows.

For HLA2010, seed0 and seed1 are essentially identical in timing, total input, and final yield. This is a much cleaner result than the previous no-window budgeted PPO, where the high-yield behavior was obtained through immediate early budget use.

This does not prove PPO is generally optimal. It only shows that adding operation timing constraints makes the current budgeted PPO behavior more agronomically interpretable under the same reward/budget setting.

# 012_01 HLA2010 DQN discrete action probe

## Purpose

Start a new algorithm line after the PPO windowed-budget experiments.

Question:

> Is the problematic PPO behavior mainly caused by PPO/continuous actions, or will a discrete-action algorithm also spend the full water-nitrogen budget?

This probe tests DQN with a tiny, interpretable discrete action set on HLA2010 only.

## Design

Keep the same corrected HLA2010 input basis used by the current PPO restart line:

- corrected cultivar
- IC=1 input package
- PDI/gym-DSSAT 4.8.0 container environment
- official gym-DSSAT reward scalarized by summing components
- I120/N150 budget
- operation windows

Change only the algorithm/action structure:

- PPO continuous action -> DQN discrete action

## Discrete action set

| action | meaning |
|---:|---|
| 0 | no operation |
| 1 | irrigation 30 mm |
| 2 | fertilization 50 kg N/ha |
| 3 | irrigation 30 mm + fertilization 50 kg N/ha |

Safety filters remain active:

- irrigation budget <= 120 mm
- nitrogen budget <= 150 kg N/ha
- irrigation windows: DAP 20-35, 45-65, 70-95
- nitrogen windows: DAP 25-40, 55-70
- minimum interval between operations: 7 days

## First run

HLA2010 seed0 only.

Run a small smoke test first. If it is stable, run 5K.

## Evaluation

Compare against current `joint_windowed` PPO seed0:

| metric | question |
|---|---|
| total irrigation | does DQN also use full 120 mm? |
| total nitrogen | does DQN also use full 150 kg/ha? |
| no-op frequency | does DQN learn to choose no operation? |
| first operation DAP | does DQN avoid DAP2? |
| yield | does DQN remain competitive? |
| stress | does DQN control WSPD/NSTD reasonably? |

## Constraint

Do not modify current PPO outputs.
Do not run multi-year or multi-seed until HLA2010 seed0 shows whether this line is worth continuing.

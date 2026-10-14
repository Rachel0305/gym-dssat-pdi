# 047_00 SYA originIC small-N action MaskablePPO

This folder is the task-local entry point for the 047_00 controlled experiment.

## What changes relative to 046_10

Only the nitrogen dose grid changes:

```text
irrigation: [0, 15, 30, 45] mm
nitrogen:   [0, 10, 20, 40] kg/ha
```

Teacher guidance is disabled. The auto-0.05 teacher should be introduced only in a later experiment if this action-space-only test is understood.

## Run commands

From the container:

```bash
cd /workspace/src/047_sya_originIC_small_n_action_ppo

python run_047_00_sya_originIC_small_n_action_maskableppo.py --dry-run
python run_047_00_sya_originIC_small_n_action_maskableppo.py --smoke
python run_047_00_sya_originIC_small_n_action_maskableppo.py --formal
```

The formal run is refused unless the 2K smoke result exists and passes the action audit.

## Main outputs

```text
benchmark_results/047_00_sya_originIC_small_n_action_maskableppo_smoke2k/
benchmark_results/047_00_sya_originIC_small_n_action_maskableppo/
docs/047_00_sya_originIC_small_n_action_maskableppo_record.md
```


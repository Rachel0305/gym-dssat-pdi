# 048_00 SYA originIC auto-0.05 teacher-guided MaskablePPO

This task-local folder contains the 048 teacher-guided PPO entry point.

048 returns to the `046_10` action grid and adds a small teacher-shaping reward
based on the auto-0.05 nitrogen stress rule.

## Run commands

From the container:

```bash
cd /workspace/src/048_sya_originIC_auto005_teacher_guided_ppo

python run_048_00_sya_originIC_auto005_teacher_guided_maskableppo.py --dry-run
python run_048_00_sya_originIC_auto005_teacher_guided_maskableppo.py --smoke
python run_048_00_sya_originIC_auto005_teacher_guided_maskableppo.py --formal
```

Run `--formal` only after smoke shows non-DAP1, stress-responsive actions.

## Main outputs

```text
benchmark_results/048_00_sya_originIC_auto005_teacher_guided_maskableppo_smoke2k/
benchmark_results/048_00_sya_originIC_auto005_teacher_guided_maskableppo/
docs/048_00_sya_originIC_auto005_teacher_guided_maskableppo_record.md
```


# 048_01 SYA originIC auto-0.05 teacher-guided PPO, no late-N ban

This task-local folder contains the 048 teacher-guided PPO entry point.

048_01 retains 048_00's teacher shaping and `046_10` action grid. Its only
additional change is expanding PPO's legal N-action window from DAP 1-90 to
DAP 1-150. The season-N cap, fertilizer interval and action mask remain active.

## Run commands

From the container:

```bash
cd /workspace/src/048_sya_originIC_auto005_teacher_guided_no_late_n_ban_ppo

python run_048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo.py --dry-run
python run_048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo.py --smoke
python run_048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo.py --formal
```

Run `--formal` only after smoke shows non-DAP1, stress-responsive actions.

## Main outputs

```text
benchmark_results/048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo_smoke2k/
benchmark_results/048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo/
docs/048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo_record.md
```

# 010_15 HLA 2004 DSSAT native automatic fertilizer NMTHR diagnostic

Purpose: reuse candidate IC 0.55 + 0.25N and DSSAT native `IRRIG=A, FERTI=A`, changing only `NMTHR`.

| NMTHR | HWAM | CWAM | MDAT | IR#M | IRCM | NI#M | NICM | irrig_events | fert_events | fert_amount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 10 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 25 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 50 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 75 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 90 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |
| 99 | 2.5e+03 | 8.29e+03 | 2e+06 | 7 | 329 | 0 | 0 | 7 | 0 | 0 |

Interpretation rule:

- If any NMTHR value produces `NI#M>0`, then the original `NMTHR=50` likely missed the internal trigger.
- If all NMTHR values keep `NI#M=0`, then the issue is not simply the numeric threshold; native `FERTI=A` is not triggering in this setup despite being parsed.

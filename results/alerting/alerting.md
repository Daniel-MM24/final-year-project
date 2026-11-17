# Stage 5 - Alerting policy

Thresholds come from a **budget** (alerts an analyst can review per day), not a
probability cutoff. Within each validation day the top-N rows by calibrated Rung 2
score are alerted; the Nth score is that day's threshold. The calibrator is
fitted on the previous validation day, so 2022-09-03 (no earlier day) is excluded
from every budget number.

Chain definition: account-contiguous transfers (`a -> b -> c`), consecutive hops
at most 1 days 00:00:00 apart. `chained edges lost` counts laundering edges on
a chain where this budget alerts **none** of the chain's edges.

**The operative metric is neighbourhood recall.** Only 178 of 3,565 laundering
edges have a laundering successor within 24h, so most 'chains' are a laundering hop
followed by ordinary transfers and the chain metric saturates. Neighbourhood recall
asks the question an analyst actually has: of the laundering edges this budget
misses, what share share an account with an edge it *did* alert that day? A high
value means the alert queue puts the analyst next to the rest of the path even when
it does not flag it directly.

## Pooled over usable days

| Budget | alerts | true positives | precision | recall | neighbourhood recall | chained lost |
|---|---|---|---|---|---|---|
| 50 | 350 | 49 | 0.1400 | 0.0198 | 0.0712 (173/2430) | 610 |
| 200 | 1,400 | 127 | 0.0907 | 0.0512 | 0.1131 (266/2352) | 589 |
| 1000 | 7,000 | 425 | 0.0607 | 0.1714 | 0.1558 (320/2054) | 513 |

## Per day

### Budget 50

| day | n | pos | threshold | TP | precision | recall | nbr recall | chained lost |
|---|---|---|---|---|---|---|---|---|
| 2022-09-04 | 282,476 | 291 | 0.017818 | 1 | 0.0200 | 0.0034 | 0.0483 (14/290) | 62/62 |
| 2022-09-05 | 657,397 | 370 | 0.071680 | 6 | 0.1200 | 0.0162 | 0.0797 (29/364) | 97/100 |
| 2022-09-06 | 657,170 | 380 | 0.178540 | 11 | 0.2200 | 0.0289 | 0.0759 (28/369) | 95/100 |
| 2022-09-07 | 658,504 | 368 | 0.112784 | 2 | 0.0400 | 0.0054 | 0.0656 (24/366) | 99/99 |
| 2022-09-08 | 657,938 | 402 | 0.124054 | 8 | 0.1600 | 0.0199 | 0.0812 (32/394) | 113/115 |
| 2022-09-09 | 891,573 | 367 | 0.176182 | 15 | 0.3000 | 0.0409 | 0.0824 (29/352) | 89/93 |
| 2022-09-10 | 282,877 | 301 | 0.098069 | 6 | 0.1200 | 0.0199 | 0.0576 (17/295) | 55/56 |

### Budget 200

| day | n | pos | threshold | TP | precision | recall | nbr recall | chained lost |
|---|---|---|---|---|---|---|---|---|
| 2022-09-04 | 282,476 | 291 | 0.011387 | 11 | 0.0550 | 0.0378 | 0.0536 (15/280) | 59/62 |
| 2022-09-05 | 657,397 | 370 | 0.060737 | 17 | 0.0850 | 0.0459 | 0.1275 (45/353) | 95/100 |
| 2022-09-06 | 657,170 | 380 | 0.133605 | 22 | 0.1100 | 0.0579 | 0.1369 (49/358) | 88/100 |
| 2022-09-07 | 658,504 | 368 | 0.089399 | 9 | 0.0450 | 0.0245 | 0.1170 (42/359) | 97/99 |
| 2022-09-08 | 657,938 | 402 | 0.100783 | 22 | 0.1100 | 0.0547 | 0.1158 (44/380) | 111/115 |
| 2022-09-09 | 891,573 | 367 | 0.133536 | 26 | 0.1300 | 0.0708 | 0.1085 (37/341) | 86/93 |
| 2022-09-10 | 282,877 | 301 | 0.071597 | 20 | 0.1000 | 0.0664 | 0.1210 (34/281) | 53/56 |

### Budget 1000

| day | n | pos | threshold | TP | precision | recall | nbr recall | chained lost |
|---|---|---|---|---|---|---|---|---|
| 2022-09-04 | 282,476 | 291 | 0.005694 | 27 | 0.0270 | 0.0928 | 0.1061 (28/264) | 55/62 |
| 2022-09-05 | 657,397 | 370 | 0.038724 | 61 | 0.0610 | 0.1649 | 0.1780 (55/309) | 88/100 |
| 2022-09-06 | 657,170 | 380 | 0.060241 | 73 | 0.0730 | 0.1921 | 0.1792 (55/307) | 78/100 |
| 2022-09-07 | 658,504 | 368 | 0.037144 | 57 | 0.0570 | 0.1549 | 0.1511 (47/311) | 86/99 |
| 2022-09-08 | 657,938 | 402 | 0.046111 | 72 | 0.0720 | 0.1791 | 0.1606 (53/330) | 91/115 |
| 2022-09-09 | 891,573 | 367 | 0.070824 | 63 | 0.0630 | 0.1717 | 0.1349 (41/304) | 75/93 |
| 2022-09-10 | 282,877 | 301 | 0.031768 | 72 | 0.0720 | 0.2392 | 0.1790 (41/229) | 40/56 |


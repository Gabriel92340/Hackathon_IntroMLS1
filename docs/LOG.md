# Submission log, Team 23

One line per Kaggle submission, hypothesis written **before** looking at the score.

| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 15:14 | VTAE | ExtraTrees (500 trees) on 109 features: raw + per-user and per-(user, sport) ranks / z-scores / deviations from median | Target is a per-person tercile: comparing a workout to the person's own history should beat raw values (Spearman of within-user speed rank 0.35 vs 0.12 raw) | 0.590 | — |
| 15:27 | VTAE | HGB + ExtraTrees averaged, + local features (speed vs same person ±2 months), + out-of-fold encoding of labels of temporally close workouts of the same person, + assignment under exact per-user quotas (Hungarian algorithm) | Labels are terciles per person over train+test, so we know how many easy/moderate/hard each person has in the test; fitness drifts over time | 0.611 | — |
| 15:42 | VTAE | + ExtraTrees regressor in the assignment cost (−log p + (k − score)²); quotas per (user, sport) instead of per user | The regressor adds ordering; multi-sport users are balanced 1/3 within each sport | 0.614 | — |
| 15:46 | VTAE | + "personal model" features: weighted vote of the 15 most similar known workouts of the same (user, sport), distances to the person's easy/moderate/hard profiles (out-of-fold) | Each person "suffers" differently; resembling their known hard workouts is the best signal (Spearman 0.556) | 0.617 | 0.6345 |
| 16:31 | VTAE | + "routine" features (labels of same weekday / time slot / month for the person) and "same route" features (speed and labels on routes with same distance, altitude, climb), out-of-fold | Athletes have routines and repeat loops; these features correlate with the previous model's residuals (≈ 0.10) | 0.619 | 0.640 |
| 16:50 | VTAE | Stacking: logistic regression on out-of-fold log-probabilities of 4 classifiers + regressor; rounding convention fixed (extra workout goes to "hard", the only convention consistent with the data) | Meta-model learns which model to trust; the old rounding convention was provably impossible for one group | 0.624 | 0.64963 |
| 16:51 | VTAE | Same stacking, base models refit on 100% of train | Encodings see 25% more known neighbours | 0.624 (not measurable) | 0.64553 |
| 17:14 | VTAE | + "expected speed" (residual of a per-(user, sport) ridge regression of speed on duration, slope, climb/km, date, season, hour) + **pairwise ranking model** (HGB on pairs of workouts of the same person: "which one is harder?", then position of each workout vs the person's known easy/moderate/hard workouts) | Fast on a long or hilly workout is harder; the pairwise model sees the problem as within-person ranking, which is exactly what the target is | 0.626 (5/5 folds > previous) | 0.65234 |
| 17:34 | VTAE | Geometric mean of 4 near-equivalent stackings | Lower variance | 0.6263 (= previous) | — |
| 17:47 | VTAE | **sub10_stack_locaux (selected)**: previous stack + (1) a **local logistic regression per (user, sport)**, (2) a **k-NN (k = 25) within each group**, (3) an **HGB "moderate vs rest"**; meta logistic regression with strong regularisation (C = 0.05), then exact-quota assignment | Our global tree models were saturated; models that make different errors (a person-specific rule, a local neighbourhood, a specialist for the weakest class) still add information | 0.6277 (4/5 folds) | **0.65456** |

## Gate 2: confusion matrix (5-fold cross-validation, 22,728 workouts)

| | pred easy | pred moderate | pred hard |
| --- | --- | --- | --- |
| **true easy** | 5032 | 1924 | 573 |
| **true moderate** | 1780 | 3908 | 1915 |
| **true hard** | 608 | 1761 | 5227 |

macro-F1 = 0.623, F1 easy 0.673, moderate 0.514, hard 0.683.

**Error analysis:** errors are almost always between neighbouring classes. "moderate" is the most confused class (F1 0.51) because it sits between easy and hard, while easy ↔ hard confusions are rare (5% of workouts). The target is a tercile of a continuous measure: the model ranks each person's workouts well, and the difficulty is placing the two cut-offs.

## Ideas tested and rejected (no submission, rule: must win on ≥ 4 of 5 folds)

Ordinal regression alone; HGB tuning; seed averaging; robust z-scores; HGB meta-model; per-user calibration; exact time slot encoding; extra quota for moderate (δ 0.02 to 0.12); assignment objective −p instead of −log p; time-weighted pairs; full anchors for pairs (flat curve); larger stacks of 7 to 9 models (meta overfits); transductive smoothing between unknown workouts (0.618); 20-fold inner encodings (better single models, not better stack); label propagation graph per group (3/5 folds); ordinal penalty on top of the stack (2/5 folds). Deliberately refused: external data (the original dataset with heart rate) and tuning on the public leaderboard.

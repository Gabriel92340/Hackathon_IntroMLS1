# Workout intensity prediction: ML hackathon (Kaggle, 6 h)

Predict whether a workout was `easy`, `moderate` or `hard` *for the person who did it*. Metric: macro-F1. Constraints: scikit-learn only, no neural networks, no external data, 15 submissions maximum.

**Result:** public macro-F1 0.65456 (final submission `sub10_stack_locaux`), 5-fold CV 0.6277.

## Key ideas

1. **The target is a per-(user, sport) tercile** of a hidden sensor measure, computed over train + test. Everything is built relative to the person.
2. **Exact class quotas in the test set.** Each group has exactly 1/3 of each class, and the train labels are known, so the number of easy / moderate / hard left in the test is known. Predictions are assigned with the Hungarian algorithm under these quotas (about +0.006 on its own).
3. **Out-of-fold label encodings** (temporal form, personal k-NN, routines, same-route), never using the workout's own label.
4. **Pairwise ranking model** ("which of two workouts of the same person was harder?").
5. **Stacking** (logistic regression on out-of-fold probabilities), then local per-group models for diversity.

## Layout

| Path | Content |
| --- | --- |
| `commun.py`, `features*.py`, `encodage*.py` | Data loading, relative features, label encodings |
| `etape7_modeles.py`, `paires*.py`, `locaux.py` | Base models, pairwise model, local models |
| `stacking.py`, `quotas.py` | Stacking and quota-constrained assignment |
| `sub10_stack_locaux.py` | Final submission script |
| `notebook_final.ipynb` | End-to-end notebook reproducing the sub09 stack |
| `docs/LOG.md` | Submission log with hypotheses, CV and public scores |
| `docs/JOURNAL.md` | Full step-by-step journal (French) |

## Data

The competition data is not included (competition rules). Place `train.csv`, `test.csv` and `sample_submission.csv` in `data/`.

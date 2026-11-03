import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluer import plis, y, u_te, total, connus_de
from quotas import affecter_cout
from commun import charger, CLASSES

sortie = sys.argv[1]
CLF = sys.argv[2].split("+")
REG = sys.argv[3].split("+")
source = sys.argv[4]


def bloc_fold(f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    for r in REG:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


def bloc_test():
    if source == "complet":
        cols = [np.log(np.clip(np.load(f"preds/{c}_complet_test.npy"), 1e-6, 1)) for c in CLF]
        for r in REG:
            s = np.load(f"preds/{r}_complet_test.npy")
            cols.append(np.column_stack([s, s ** 2]))
        return np.hstack(cols)
    return None


meta = np.zeros((len(y), bloc_fold(0, "val").shape[1]))
for f, (a, b) in enumerate(plis):
    meta[b] = bloc_fold(f, "val")
lr = LogisticRegression(C=1.0, max_iter=3000).fit(meta, y)

if source == "complet":
    p_test = lr.predict_proba(bloc_test())
else:
    p_test = sum(lr.predict_proba(bloc_fold(f, "test")) for f in range(5)) / 5

train, test = charger()
pred = affecter_cout(-np.log(np.clip(p_test, 1e-9, 1)), u_te, total, connus_de(np.arange(len(y))))
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]})
sub.to_csv(sortie, index=False)
np.save(sortie.replace(".csv", "_proba.npy"), p_test)
print(sub["intensity"].value_counts(normalize=True).round(3).to_dict())

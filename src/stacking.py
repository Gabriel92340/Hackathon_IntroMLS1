import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout

CLF = sys.argv[1].split("+")
REG = sys.argv[2].split("+")
C = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
n = len(y)


def bloc(f, partie):
    cols = []
    for c in CLF:
        p = np.load(f"preds/{c}_f{f}_{partie}.npy")
        cols.append(np.log(np.clip(p, 1e-6, 1)))
    for r in REG:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


meta_oof = np.zeros((n, bloc(0, "val").shape[1]))
for f, (a, b) in enumerate(plis):
    meta_oof[b] = bloc(f, "val")

pred_oof = np.zeros(n, dtype=int)
for f, (a, b) in enumerate(plis):
    lr = LogisticRegression(C=C, max_iter=3000).fit(meta_oof[a], y[a])
    pv = lr.predict_proba(meta_oof[b])
    pt = lr.predict_proba(bloc(f, "test"))
    cout = -np.log(np.clip(np.vstack([pv, pt]), 1e-9, 1))
    pred = affecter_cout(cout, np.concatenate([u_tr[b], u_te]), total, connus_de(a))
    pred_oof[b] = pred[:len(b)]
scores = [f1_score(y[b], pred_oof[b], average="macro") for a, b in plis]
print(f"stacking C={C} | {'+'.join(CLF)} | {'+'.join(REG)} : {f1_score(y, pred_oof, average='macro'):.4f}  folds {np.round(scores, 4)}")

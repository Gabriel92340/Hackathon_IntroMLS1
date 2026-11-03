import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout
from commun import charger, CLASSES

B = ["hgb_clf_pra", "hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1", "et_clf_ra_f1"]
CONFIGS = [
    (B, ["et_reg", "paires"], 1.0),
    (B, ["et_reg", "paires", "paires2"], 1.0),
    (B + ["hgb_clf_prc"], ["et_reg", "paires", "paires2"], 1.0),
    (B, ["et_reg", "paires"], 0.3),
    (["hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1"], ["et_reg"], 1.0),
]


def bloc(clf, reg, f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in clf]
    for r in reg:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


def probas(clf, reg, C):
    M = np.zeros((len(y), bloc(clf, reg, 0, "val").shape[1]))
    for f, (a, b) in enumerate(plis):
        M[b] = bloc(clf, reg, f, "val")
    P = {}
    for f, (a, b) in enumerate(plis):
        lr = LogisticRegression(C=C, max_iter=3000).fit(M[a], y[a])
        P[f] = np.vstack([lr.predict_proba(M[b]), lr.predict_proba(bloc(clf, reg, f, "test"))])
    lr = LogisticRegression(C=C, max_iter=3000).fit(M, y)
    T = sum(lr.predict_proba(bloc(clf, reg, f, "test")) for f in range(5)) / 5
    return P, T


def score(P):
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        oof[b] = affecter_cout(-np.log(np.clip(P[f], 1e-9, 1)), np.concatenate([u_tr[b], u_te]), total, connus_de(a))[:len(b)]
    return f1_score(y, oof, average="macro"), np.array([f1_score(y[b], oof[b], average="macro") for a, b in plis])


res = [probas(*c) for c in CONFIGS]
ref, ref_f = score(res[0][0])
print(f"sub09 seul : {ref:.4f} {np.round(ref_f, 4)}")
for k in [2, 3, 4, 5]:
    P = {f: sum(np.log(np.clip(r[0][f], 1e-9, 1)) for r in res[:k]) / k for f in range(5)}
    P = {f: np.exp(v) / np.exp(v).sum(1, keepdims=True) for f, v in P.items()}
    s, sf = score(P)
    print(f"moyenne des {k} premiers stacks : {s:.4f} {np.round(sf, 4)} gagnants {(sf > ref_f).sum()}/5")
    if len(sys.argv) > 1 and int(sys.argv[1]) == k:
        T = sum(np.log(np.clip(r[1], 1e-9, 1)) for r in res[:k]) / k
        T = np.exp(T) / np.exp(T).sum(1, keepdims=True)
        train, test = charger()
        pred = affecter_cout(-np.log(np.clip(T, 1e-9, 1)), u_te, total, connus_de(np.arange(len(y))))
        pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]}).to_csv("soumissions/sub10_moyenne_stacks.csv", index=False)
        print("écrit sub10")

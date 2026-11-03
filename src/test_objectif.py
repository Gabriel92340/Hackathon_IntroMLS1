import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout
CLF = ["hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1"]
REG = ["et_reg"]
def bloc(f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    for r in REG:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy"); cols.append(np.column_stack([s, s ** 2]))
    return np.hstack(cols)
M = np.zeros((len(y), bloc(0, "val").shape[1]))
for f, (a, b) in enumerate(plis): M[b] = bloc(f, "val")
P = {}
for f, (a, b) in enumerate(plis):
    lr = LogisticRegression(max_iter=3000).fit(M[a], y[a])
    P[f] = np.vstack([lr.predict_proba(M[b]), lr.predict_proba(bloc(f, "test"))])
for nom, fn in [("-log p (actuel)", lambda p: -np.log(np.clip(p, 1e-9, 1))), ("-p", lambda p: -p),
                ("-p^0.5", lambda p: -np.sqrt(p)), ("-p^2", lambda p: -p ** 2)]:
    oof = np.zeros(len(y), int)
    for f, (a, b) in enumerate(plis):
        oof[b] = affecter_cout(fn(P[f]), np.concatenate([u_tr[b], u_te]), total, connus_de(a))[:len(b)]
    print(f"{nom:<18} {f1_score(y, oof, average='macro'):.4f}  folds {np.round([f1_score(y[b], oof[b], average='macro') for a, b in plis], 4)}")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout
from commun import charger

CLF = ["hgb_clf_pra", "hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1", "et_clf_ra_f1"]
REG = ["et_reg", "paires"]
train, test = charger()
sp_tr = np.where(train.sport.isin(["running", "cycling"]), train.sport, "autre")
sp_te = np.where(test.sport.isin(["running", "cycling"]), test.sport, "autre")
taille = pd.Series(np.concatenate([u_tr, u_te])).value_counts()
ln_tr = np.log(taille.reindex(u_tr).values)
ln_te = np.log(taille.reindex(u_te).values)


def bloc(f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    for r in REG:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


M = np.zeros((len(y), bloc(0, "val").shape[1]))
for f, (a, b) in enumerate(plis):
    M[b] = bloc(f, "val")
Mt = {f: bloc(f, "test") for f in range(5)}


def interactions(X, ln):
    z = (ln - 4.5) / 1.2
    return np.hstack([X, X * z[:, None], z[:, None]])


def evaluer(mode, C=1.0):
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        pv = np.zeros((len(b), 3))
        pt = np.zeros((len(u_te), 3))
        if mode == "global":
            lr = LogisticRegression(C=C, max_iter=3000).fit(M[a], y[a])
            pv, pt = lr.predict_proba(M[b]), lr.predict_proba(Mt[f])
        elif mode == "sport":
            for s in ["running", "cycling", "autre"]:
                ia, ib, it = a[sp_tr[a] == s], b[sp_tr[b] == s], np.where(sp_te == s)[0]
                lr = LogisticRegression(C=C, max_iter=3000).fit(M[ia], y[ia])
                pv[sp_tr[b] == s] = lr.predict_proba(M[ib])
                pt[it] = lr.predict_proba(Mt[f][it])
        elif mode == "taille":
            lr = LogisticRegression(C=C, max_iter=5000).fit(interactions(M[a], ln_tr[a]), y[a])
            pv, pt = lr.predict_proba(interactions(M[b], ln_tr[b])), lr.predict_proba(interactions(Mt[f], ln_te))
        oof[b] = affecter_cout(-np.log(np.clip(np.vstack([pv, pt]), 1e-9, 1)), np.concatenate([u_tr[b], u_te]), total, connus_de(a))[:len(b)]
    return f1_score(y, oof, average="macro"), np.array([f1_score(y[b], oof[b], average="macro") for a, b in plis])


ref, rf = evaluer("global")
print(f"global (sub09) : {ref:.4f} {np.round(rf, 4)}")
for mode, C in [("sport", 1.0), ("sport", 0.3), ("taille", 1.0), ("taille", 0.3)]:
    s, sf = evaluer(mode, C)
    print(f"{mode} C={C} : {s:.4f} {np.round(sf, 4)} gagnants {(sf > rf).sum()}/5", flush=True)

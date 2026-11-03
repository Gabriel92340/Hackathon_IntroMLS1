import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout
from commun import charger, CLASSES

for f in range(5):
    for p in ["val", "test"]:
        np.save(f"preds/paires_moy_f{f}_{p}.npy", (np.load(f"preds/paires_f{f}_{p}.npy") + np.load(f"preds/paires_s2_f{f}_{p}.npy")) / 2)

CLF = ["hgb_clf_pra", "hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1", "et_clf_ra_f1"]


def bloc(reg, f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    for r in reg:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


def evaluer(reg):
    M = np.zeros((len(y), bloc(reg, 0, "val").shape[1]))
    for f, (a, b) in enumerate(plis):
        M[b] = bloc(reg, f, "val")
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        lr = LogisticRegression(max_iter=3000).fit(M[a], y[a])
        p = np.vstack([lr.predict_proba(M[b]), lr.predict_proba(bloc(reg, f, "test"))])
        oof[b] = affecter_cout(-np.log(np.clip(p, 1e-9, 1)), np.concatenate([u_tr[b], u_te]), total, connus_de(a))[:len(b)]
    return M, f1_score(y, oof, average="macro"), np.array([f1_score(y[b], oof[b], average="macro") for a, b in plis])


_, ref, ref_f = evaluer(["et_reg", "paires"])
print(f"sub09 : {ref:.4f} {np.round(ref_f, 4)}")
meilleur = None
for reg in (["et_reg", "paires_moy"], ["et_reg", "paires", "paires_s2"]):
    M, s, sf = evaluer(reg)
    g = (sf > ref_f).sum()
    print(f"{'+'.join(reg)} : {s:.4f} {np.round(sf, 4)} gagnants {g}/5 gain {s - ref:+.4f}")
    if g >= 4 and s - ref >= 0.0015 and (meilleur is None or s > meilleur[1]):
        meilleur = (reg, s, M)
if meilleur:
    reg, s, M = meilleur
    lr = LogisticRegression(max_iter=3000).fit(M, y)
    T = sum(lr.predict_proba(bloc(reg, f, "test")) for f in range(5)) / 5
    train, test = charger()
    pred = affecter_cout(-np.log(np.clip(T, 1e-9, 1)), u_te, total, connus_de(np.arange(len(y))))
    pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]}).to_csv("soumissions/sub11_paires_moyennees.csv", index=False)
    print("SOUMISSION ÉCRITE :", "+".join(reg), round(s, 4))
else:
    print("AUCUNE VARIANTE NE PASSE LE CRITÈRE : garder sub09")

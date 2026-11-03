import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from commun import charger, CLASSES
from quotas import affecter, quotas_entiers

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
u_tr = (train["user_id"].astype(str) + "_" + train["sport"]).values
u_te = (test["user_id"].astype(str) + "_" + test["sport"]).values
total = pd.Series(np.concatenate([u_tr, u_te])).value_counts().to_dict()
plis = list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y))


def score(nom, f, partie):
    p = np.load(f"preds/{nom}_f{f}_{partie}.npy")
    return p[:, 1] + 2 * p[:, 2] if p.ndim == 2 else p


def tri_quotas(s, users, connus):
    pred = np.zeros(len(s), dtype=int)
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        q = quotas_entiers(len(idx), total[u], connus[u])
        ordre = idx[np.argsort(s[idx], kind="stable")]
        pred[ordre] = np.repeat(np.arange(3), q)
    return pred


def connus_de(a):
    c = pd.crosstab(u_tr[a], y[a]).reindex(columns=[0, 1, 2], fill_value=0)
    return {u: c.loc[u].values.astype(float) if u in c.index else np.zeros(3) for u in total}


def evaluer(noms, poids=None, methode="tri"):
    poids = poids or [1] * len(noms)
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        connus = connus_de(a)
        users = np.concatenate([u_tr[b], u_te])
        if methode == "tri":
            sv = sum(w * score(nm, f, "val") for nm, w in zip(noms, poids)) / sum(poids)
            st = sum(w * score(nm, f, "test") for nm, w in zip(noms, poids)) / sum(poids)
            pred = tri_quotas(np.concatenate([sv, st]), users, connus)
        else:
            pv = sum(w * np.load(f"preds/{nm}_f{f}_val.npy") for nm, w in zip(noms, poids)) / sum(poids)
            pt = sum(w * np.load(f"preds/{nm}_f{f}_test.npy") for nm, w in zip(noms, poids)) / sum(poids)
            pred = affecter(np.vstack([pv, pt]), users, total, connus)
        oof[b] = pred[:len(b)]
    return f1_score(y, oof, average="macro"), oof


if __name__ == "__main__":
    combos = [a.split("+") for a in sys.argv[1:]]
    for c in combos:
        f1, _ = evaluer(c)
        ligne = f"{'+'.join(c):<35} tri-quotas {f1:.4f}"
        if all("clf" in n for n in c):
            f2, _ = evaluer(c, methode="hongrois")
            ligne += f"   hongrois {f2:.4f}"
        print(ligne, flush=True)

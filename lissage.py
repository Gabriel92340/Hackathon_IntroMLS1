import sys
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
t_tr = (train["year"] * 12 + train["month"]).values.astype(float)
t_te = (test["year"] * 12 + test["month"]).values.astype(float)
w_tr = train["weekday"].values
w_te = test["weekday"].values
h_tr = train["start_hour_utc"].values
h_te = test["start_hour_utc"].values


def bloc(f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    for r in REG:
        s = np.load(f"preds/{r}_f{f}_{partie}.npy")
        cols.append(s if s.ndim == 2 else np.column_stack([s, s ** 2]))
    return np.hstack(cols)


def lisser(p, groupes, t, wd, hr, lam, sigma, bonus_creneau=0.0):
    out = p.copy()
    for g in np.unique(groupes):
        idx = np.where(groupes == g)[0]
        if len(idx) < 2:
            continue
        dt = t[idx][:, None] - t[idx][None, :]
        w = np.exp(-0.5 * (dt / sigma) ** 2)
        if bonus_creneau:
            meme = (wd[idx][:, None] == wd[idx][None, :]) & (np.abs(hr[idx][:, None] - hr[idx][None, :]) <= 1) & (dt == 0)
            w = w + bonus_creneau * meme
        np.fill_diagonal(w, 0)
        lp = np.log(np.clip(p[idx], 1e-9, 1))
        voisin = (w @ lp) / (w.sum(1, keepdims=True) + 1e-9)
        poids = lam * w.sum(1, keepdims=True) / (w.sum(1, keepdims=True) + 1)
        lo = lp + poids * (voisin - lp)
        e = np.exp(lo - lo.max(1, keepdims=True))
        out[idx] = e / e.sum(1, keepdims=True)
    return out


M = np.zeros((len(y), bloc(0, "val").shape[1]))
for f, (a, b) in enumerate(plis):
    M[b] = bloc(f, "val")
P = {}
for f, (a, b) in enumerate(plis):
    lr = LogisticRegression(max_iter=3000).fit(M[a], y[a])
    P[f] = np.vstack([lr.predict_proba(M[b]), lr.predict_proba(bloc(f, "test"))])

ref = None
for lam, sigma, bc in [(0, 1, 0), (0.2, 1.0, 0), (0.4, 1.0, 0), (0.6, 1.0, 0), (0.4, 0.5, 0), (0.4, 2.0, 0), (0.4, 1.0, 2.0), (0.6, 1.0, 2.0)]:
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        grp = np.concatenate([u_tr[b], u_te])
        t = np.concatenate([t_tr[b], t_te])
        wd = np.concatenate([w_tr[b], w_te])
        hr = np.concatenate([h_tr[b], h_te])
        ps = lisser(P[f], grp, t, wd, hr, lam, sigma, bc)
        oof[b] = affecter_cout(-np.log(np.clip(ps, 1e-9, 1)), grp, total, connus_de(a))[:len(b)]
    fs = np.array([f1_score(y[b], oof[b], average="macro") for a, b in plis])
    if ref is None:
        ref = fs
    print(f"lambda {lam} sigma {sigma} bonus créneau {bc}: {f1_score(y, oof, average='macro'):.4f}  folds {np.round(fs, 4)}  gagnants {(fs > ref).sum()}/5", flush=True)

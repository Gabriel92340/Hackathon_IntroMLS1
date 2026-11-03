import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from commun import charger, nettoyer, CLASSES
from evaluer import plis
from features_attendu import residus_attendus

rng = np.random.default_rng(12345)
train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
n, nt = len(train), len(test)
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
R = residus_attendus(tout)
F = pd.DataFrame({
    "lv": np.log(tout["speed_kmh"] + 0.1), "lvmax": np.log(tout["max_speed_kmh"] + 0.1),
    "lvstd": np.log(tout["speed_std_kmh"] + 0.1), "ldur": np.log(tout["duration_min"]),
    "ldist": np.log(tout["distance_km"] + 0.1),
    "gkm": np.log1p(np.clip(tout["altitude_gain_m"] / (tout["distance_km"] + 0.1), 0, None)),
    "pente": (tout["altitude_gain_m"] + tout["altitude_loss_m"]) / (tout["distance_km"] * 1000 + 1),
    "alt": tout["altitude_mean_m"] / 100, "t": (tout["year"] * 12 + tout["month"]).astype(float),
    "ms": np.sin(2 * np.pi * tout["month"] / 12), "mc": np.cos(2 * np.pi * tout["month"] / 12),
    "h": tout["start_hour_utc"].astype(float), "wd": tout["weekday"].astype(float),
    "vres": R["v_res_z"], "vmres": R["vmax_res_z"],
}).values.astype(np.float32)
G = (tout["user_id"].astype(str) + "_" + tout["sport"]).values
MAXK = 400


def paires_entrainement(idx, yk, par_ligne=12):
    gi = G[idx]
    I, J = [], []
    for g in np.unique(gi):
        loc = np.where(gi == g)[0]
        if len(loc) < 4:
            continue
        for i in loc:
            cand = loc[yk[loc] != yk[i]]
            if len(cand) == 0:
                continue
            j = rng.choice(cand, size=min(par_ligne, len(cand)), replace=False)
            I.append(np.full(len(j), i))
            J.append(j)
    I, J = np.concatenate(I), np.concatenate(J)
    Xp = np.hstack([F[idx[I]] - F[idx[J]], np.abs(F[idx[I], 8:9] - F[idx[J], 8:9])])
    return Xp, (yk[I] > yk[J]).astype(int)


def comparer(modele, idx_k, yk, idx_q):
    out = np.full((len(idx_q), 4), np.nan)
    gk, gq = G[idx_k], G[idx_q]
    for g in np.unique(gq):
        lq = np.where(gq == g)[0]
        lk = np.where(gk == g)[0]
        if len(lk) == 0:
            continue
        if len(lk) > MAXK:
            lk = rng.choice(lk, MAXK, replace=False)
        qi = np.repeat(idx_q[lq], len(lk))
        ki = np.tile(idx_k[lk], len(lq))
        Xp = np.hstack([F[qi] - F[ki], np.abs(F[qi, 8:9] - F[ki, 8:9])])
        p = modele.predict_proba(Xp)[:, 1].reshape(len(lq), len(lk))
        yl = yk[lk]
        cols = [p[:, yl == c].mean(1) if (yl == c).any() else np.full(len(lq), 0.5) for c in range(3)]
        cols.append(p.mean(1))
        out[lq] = np.column_stack(cols)
    return np.nan_to_num(out, nan=0.5)


for f, (a, b) in enumerate(plis):
    Xp, yp = paires_entrainement(a, y[a])
    m = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200, max_leaf_nodes=31, random_state=0).fit(Xp, yp)
    cv = comparer(m, a, y[a], b)
    ct = comparer(m, a, y[a], np.arange(n, n + nt))
    np.save(f"preds/paires_s2_f{f}_val.npy", cv)
    np.save(f"preds/paires_s2_f{f}_test.npy", ct)
    sp = pd.Series(cv[:, 0] + cv[:, 1] + cv[:, 2]).corr(pd.Series(y[b]), method="spearman")
    print(f"paires_s2 fold {f} ok — {len(yp)} paires, Spearman(score, y) {sp:.3f}", flush=True)

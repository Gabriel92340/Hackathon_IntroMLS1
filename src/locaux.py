import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from commun import charger, nettoyer, CLASSES
from evaluer import plis
from features_attendu import residus_attendus

warnings.filterwarnings("ignore")
train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
n, nt = len(train), len(test)
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
R = residus_attendus(tout)
t = (tout["year"] * 12 + tout["month"]).astype(float)
brut = pd.DataFrame({
    "lv": np.log(tout["speed_kmh"] + 0.1), "lvm": np.log(tout["max_speed_kmh"] + 0.1),
    "ld": np.log(tout["duration_min"]), "ldist": np.log(tout["distance_km"] + 0.1),
    "gkm": np.log1p(np.clip(tout["altitude_gain_m"] / (tout["distance_km"] + 0.1), 0, None)),
    "pente": (tout["altitude_gain_m"] + tout["altitude_loss_m"]) / (tout["distance_km"] * 1000 + 1),
    "t": t, "h": tout["start_hour_utc"].astype(float), "vres": R["v_res_z"],
})
G = (tout["user_id"].astype(str) + "_" + tout["sport"]).values
Z = np.zeros((len(tout), brut.shape[1] + 1))
for g in np.unique(G):
    idx = np.where(G == g)[0]
    v = brut.values[idx]
    z = (v - v.mean(0)) / (v.std(0) + 1e-6)
    Z[idx, :-1] = np.clip(z, -4, 4)
    Z[idx, -1] = Z[idx, 6] ** 2


def predire(idx_k, yk, idx_q, C=0.3, mini=30):
    out = np.full((len(idx_q), 3), 1 / 3)
    gk, gq = G[idx_k], G[idx_q]
    for g in np.unique(gq):
        lq = np.where(gq == g)[0]
        lk = np.where(gk == g)[0]
        if len(lk) < mini or len(np.unique(yk[lk])) < 3:
            continue
        m = LogisticRegression(C=C, max_iter=500).fit(Z[idx_k[lk]], yk[lk])
        out[lq] = m.predict_proba(Z[idx_q[lq]])
    return out


for f, (a, b) in enumerate(plis):
    np.save(f"preds/local_clf_f{f}_val.npy", predire(a, y[a], b))
    np.save(f"preds/local_clf_f{f}_test.npy", predire(a, y[a], np.arange(n, n + nt)))
    print("local fold", f, "ok", flush=True)

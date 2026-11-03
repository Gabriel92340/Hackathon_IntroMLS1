import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

COLS = ["speed_kmh_z_us", "max_speed_kmh_z_us", "duration_min_z_us", "distance_km_z_us",
        "altitude_gain_m_z_us", "pente_z_us", "ratio_max_z_us", "t_z_us",
        "speed_kmh_zloc", "max_speed_kmh_zloc"]
K = 15


def encoder(ref, y_ref, req, cols=COLS):
    g_ref = (ref["user_id"].astype(str) + "_" + ref["sport"]).values
    g_req = (req["user_id"].astype(str) + "_" + req["sport"]).values
    A, B = ref[cols].values.astype(float), req[cols].values.astype(float)
    A = np.clip(A, -4, 4)
    B = np.clip(B, -4, 4)
    sortie = np.full((len(req), 9), np.nan)
    onehot = np.eye(3)[y_ref]
    for g in np.unique(g_req):
        iq = np.where(g_req == g)[0]
        ir = np.where(g_ref == g)[0]
        if len(ir) < 3:
            continue
        d2 = ((B[iq][:, None, :] - A[ir][None, :, :]) ** 2).sum(-1)
        k = min(K, len(ir))
        voisins = np.argsort(d2, axis=1)[:, :k]
        dv = np.take_along_axis(d2, voisins, 1)
        w = 1.0 / (1.0 + dv)
        vote = (w[:, :, None] * onehot[ir][voisins]).sum(1) / w.sum(1, keepdims=True)
        cols_out = [vote[:, 0], vote[:, 2], vote[:, 2] - vote[:, 0]]
        dist = []
        for c in range(3):
            m = y_ref[ir] == c
            if m.sum() == 0:
                dist.append(np.full(len(iq), np.nan))
            else:
                centre = A[ir][m].mean(0)
                dist.append(np.sqrt(((B[iq] - centre) ** 2).sum(1)))
        dist = np.column_stack(dist)
        cols_out += [dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 0] - dist[:, 2]]
        proj = A[ir][y_ref[ir] == 2].mean(0) - A[ir][y_ref[ir] == 0].mean(0) if (y_ref[ir] == 2).any() and (y_ref[ir] == 0).any() else np.zeros(A.shape[1])
        s_req = B[iq] @ proj
        s_ref = A[ir] @ proj
        rang = (s_ref[None, :] < s_req[:, None]).mean(1)
        cols_out += [s_req, rang]
        sortie[iq] = np.column_stack(cols_out)
    noms = ["pk_easy", "pk_hard", "pk_diff", "pc_d_easy", "pc_d_mod", "pc_d_hard", "pc_d_diff", "pp_score", "pp_rang"]
    return pd.DataFrame(sortie, columns=noms, index=req.index)


def encoder_oof(df, y, seed=7):
    morceaux = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        morceaux.append(encoder(df.iloc[a], y[a], df.iloc[b]))
    return pd.concat(morceaux).loc[df.index]

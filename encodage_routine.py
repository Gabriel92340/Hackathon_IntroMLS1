import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

GROUPES = {"r_jour": ["weekday"], "r_heure": ["hbin"], "r_jour_heure": ["weekday", "hbin"], "r_mois": ["month"]}
COLS_PARC = ["distance_km", "altitude_mean_m", "altitude_gain_m", "altitude_loss_m"]


def preparer(df):
    df = df.copy()
    df["hbin"] = pd.cut(df["start_hour_utc"], [-1, 5, 9, 12, 15, 18, 21, 24], labels=False).astype(int)
    return df


def coords_parcours(df):
    v = df[COLS_PARC].values.astype(float)
    return np.column_stack([np.log1p(v[:, 0]) / 0.05, v[:, 1] / 20, np.log1p(np.clip(v[:, 2], 0, None)) / 0.3,
                            np.log1p(np.clip(v[:, 3], 0, None)) / 0.3])


def encoder(ref, y_ref, req, lissage=5, k=8):
    ref, req = preparer(ref), preparer(req)
    ref = ref.assign(y=y_ref)
    sortie = pd.DataFrame(index=req.index)
    base = ["user_id", "sport"]
    prior = req[base].merge(ref.groupby(base)["y"].mean().rename("pu"), left_on=base, right_index=True, how="left")["pu"].fillna(1).values
    for nom, extra in GROUPES.items():
        cles = base + extra
        g = ref.groupby(cles)["y"].agg(["sum", "count"])
        m = req[cles].merge(g, left_on=cles, right_index=True, how="left")
        sortie[nom] = (m["sum"].fillna(0).values + lissage * prior) / (m["count"].fillna(0).values + lissage) - prior
        sortie[nom + "_n"] = m["count"].fillna(0).values
    rang = np.full(len(req), np.nan)
    lab = np.full(len(req), np.nan)
    dens = np.zeros(len(req))
    cA, cB = coords_parcours(ref), coords_parcours(req)
    gr = (ref["user_id"].astype(str) + "_" + ref["sport"]).values
    gq = (req["user_id"].astype(str) + "_" + req["sport"]).values
    vr, vq, yr = ref["speed_kmh"].values, req["speed_kmh"].values, ref["y"].values
    for g in np.unique(gq):
        iq = np.where(gq == g)[0]
        ir = np.where(gr == g)[0]
        if len(ir) < 5:
            continue
        d2 = ((cB[iq][:, None, :] - cA[ir][None, :, :]) ** 2).sum(-1)
        kk = min(k, len(ir))
        vois = np.argsort(d2, 1)[:, :kk]
        w = np.exp(-np.take_along_axis(d2, vois, 1) / 4)
        sw = w.sum(1) + 1e-9
        rang[iq] = ((vq[iq][:, None] > vr[ir][vois]) * w).sum(1) / sw
        lab[iq] = (yr[ir][vois] * w).sum(1) / sw
        dens[iq] = sw
    sortie["parc_rang_vitesse"] = rang
    sortie["parc_labels"] = lab
    sortie["parc_densite"] = dens
    return sortie


def encoder_oof(df, y, seed=5):
    morceaux = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        morceaux.append(encoder(df.iloc[a], y[a], df.iloc[b]))
    return pd.concat(morceaux).loc[df.index]

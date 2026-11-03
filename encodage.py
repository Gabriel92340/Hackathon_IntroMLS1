import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SIGMAS = [0.5, 1.5, 4.0]


def temps_mois(df):
    return (df["year"] * 12 + df["month"] - 1).values.astype(float)


def noyau_labels(ref, y_ref, requete, sigmas=SIGMAS):
    sortie = np.zeros((len(requete), len(sigmas) * 3 + 1))
    t_ref = temps_mois(ref)
    t_req = temps_mois(requete)
    u_ref = ref["user_id"].values
    u_req = requete["user_id"].values
    onehot = np.eye(3)[y_ref]
    for u in np.unique(u_req):
        iq = np.where(u_req == u)[0]
        ir = np.where(u_ref == u)[0]
        if len(ir) == 0:
            sortie[iq] = np.nan
            continue
        prior = onehot[ir].mean(0)
        d = t_req[iq][:, None] - t_ref[ir][None, :]
        cols = []
        for s in sigmas:
            w = np.exp(-0.5 * (d / s) ** 2)
            somme = w.sum(1, keepdims=True)
            p = (w @ onehot[ir] + prior) / (somme + 1.0)
            cols += [p[:, 0], p[:, 2], p[:, 2] - p[:, 0]]
        same = (d == 0).sum(1)
        cols.append(same)
        sortie[iq] = np.column_stack(cols)
    noms = []
    for s in sigmas:
        noms += [f"k{s}_easy", f"k{s}_hard", f"k{s}_diff"]
    noms.append("n_meme_mois")
    return pd.DataFrame(sortie, columns=noms, index=requete.index)


def noyau_labels_vitesse(ref, y_ref, requete, s_t=3.0, s_v=0.5):
    sortie = np.full((len(requete), 3), np.nan)
    t_ref, t_req = temps_mois(ref), temps_mois(requete)
    v_ref, v_req = ref["vz"].values, requete["vz"].values
    k_ref = (ref["user_id"].astype(str) + "_" + ref["sport"]).values
    k_req = (requete["user_id"].astype(str) + "_" + requete["sport"]).values
    onehot = np.eye(3)[y_ref]
    for k in np.unique(k_req):
        iq = np.where(k_req == k)[0]
        ir = np.where(k_ref == k)[0]
        if len(ir) == 0:
            continue
        prior = onehot[ir].mean(0)
        dt = (t_req[iq][:, None] - t_ref[ir][None, :]) / s_t
        dv = (v_req[iq][:, None] - v_ref[ir][None, :]) / s_v
        w = np.exp(-0.5 * (dt ** 2 + dv ** 2))
        p = (w @ onehot[ir] + prior) / (w.sum(1, keepdims=True) + 1.0)
        sortie[iq] = np.column_stack([p[:, 0], p[:, 2], p[:, 2] - p[:, 0]])
    return pd.DataFrame(sortie, columns=["kv_easy", "kv_hard", "kv_diff"], index=requete.index)


def tous_encodages(ref, y_ref, requete, avec_vitesse=False):
    morceaux = [noyau_labels(ref, y_ref, requete)]
    if avec_vitesse:
        morceaux.append(noyau_labels_vitesse(ref, y_ref, requete))
    return pd.concat(morceaux, axis=1)


def encodage_oof(df_tr, y_tr, n_splits=5, seed=1, avec_vitesse=False):
    morceaux = []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for a, b in skf.split(df_tr, y_tr):
        morceaux.append(tous_encodages(df_tr.iloc[a], y_tr[a], df_tr.iloc[b], avec_vitesse))
    return pd.concat(morceaux).loc[df_tr.index]

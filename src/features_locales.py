import numpy as np
import pandas as pd

COLS_LOC = ["speed_kmh", "max_speed_kmh", "duration_min", "distance_km", "altitude_gain_m"]


def relatives_locales(tout, sigma=2.0, colonnes=COLS_LOC):
    t = (tout["year"] * 12 + tout["month"] - 1).values.astype(float)
    cle = tout["user_id"].astype(str) + "_" + tout["sport"]
    sortie = np.zeros((len(tout), len(colonnes) * 2 + 1))
    vals = tout[colonnes].values.astype(float)
    for g, idx in tout.groupby(cle).indices.items():
        d = t[idx][:, None] - t[idx][None, :]
        w = np.exp(-0.5 * (d / sigma) ** 2)
        np.fill_diagonal(w, 0)
        sw = w.sum(1, keepdims=True) + 1e-9
        v = vals[idx]
        moy = (w @ v) / sw
        var = (w @ (v ** 2)) / sw - moy ** 2
        et = np.sqrt(np.clip(var, 1e-6, None))
        sortie[idx, :len(colonnes)] = v - moy
        sortie[idx, len(colonnes):2 * len(colonnes)] = (v - moy) / et
        sortie[idx, -1] = sw[:, 0]
    noms = [f"{c}_dloc" for c in colonnes] + [f"{c}_zloc" for c in colonnes] + ["densite_loc"]
    return pd.DataFrame(sortie, columns=noms, index=tout.index)

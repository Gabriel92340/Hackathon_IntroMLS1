import numpy as np
import pandas as pd


def residus_attendus(tout, alpha=1.0):
    t = (tout["year"] * 12 + tout["month"]).values.astype(float)
    ld = np.log(tout["duration_min"].values)
    pente = (tout["altitude_gain_m"].values + tout["altitude_loss_m"].values) / (tout["distance_km"].values * 1000 + 1)
    gain_km = tout["altitude_gain_m"].values / (tout["distance_km"].values + 0.1)
    heure = tout["start_hour_utc"].values.astype(float)
    cibles = {"v": np.log(tout["speed_kmh"].values + 0.1), "vmax": np.log(tout["max_speed_kmh"].values + 0.1),
              "vstd": np.log(tout["speed_std_kmh"].values + 0.1)}
    sortie = {f"{k}_res": np.zeros(len(tout)) for k in cibles}
    sortie.update({f"{k}_res_z": np.zeros(len(tout)) for k in cibles})
    sortie["v_res_rang"] = np.zeros(len(tout))
    cle = (tout["user_id"].astype(str) + "_" + tout["sport"]).values
    for g in np.unique(cle):
        idx = np.where(cle == g)[0]
        tc = (t[idx] - t[idx].mean()) / (t[idx].std() + 1)
        X = np.column_stack([np.ones(len(idx)), ld[idx] - ld[idx].mean(), pente[idx], np.log1p(np.clip(gain_km[idx], 0, None)),
                             tc, tc ** 2, np.sin(2 * np.pi * tout["month"].values[idx] / 12), np.cos(2 * np.pi * tout["month"].values[idx] / 12),
                             (heure[idx] - 12) / 12])
        A = X.T @ X + alpha * np.eye(X.shape[1])
        A[0, 0] -= alpha
        for k, v in cibles.items():
            beta = np.linalg.solve(A, X.T @ v[idx])
            r = v[idx] - X @ beta
            sortie[f"{k}_res"][idx] = r
            sortie[f"{k}_res_z"][idx] = r / (r.std() + 1e-6)
        sortie["v_res_rang"][idx] = pd.Series(sortie["v_res"][idx]).rank(pct=True).values
    return pd.DataFrame(sortie, index=tout.index)

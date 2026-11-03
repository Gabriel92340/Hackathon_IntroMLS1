import numpy as np
import pandas as pd

BASE_REL = ["speed_kmh", "max_speed_kmh", "speed_std_kmh", "duration_min", "distance_km",
            "altitude_gain_m", "altitude_loss_m", "altitude_mean_m", "n_points",
            "pente", "ratio_max", "cv_vitesse", "denivele_min", "t"]
BASE_US = ["speed_kmh", "max_speed_kmh", "speed_std_kmh", "duration_min", "distance_km",
           "altitude_gain_m", "pente", "ratio_max", "t"]


def ajouter_derivees(tout):
    tout = tout.copy()
    tout["t"] = tout["year"] + (tout["month"] - 1) / 12
    tout["pente"] = (tout["altitude_gain_m"] + tout["altitude_loss_m"]) / (tout["distance_km"] * 1000 + 1)
    tout["ratio_max"] = tout["max_speed_kmh"] / (tout["speed_kmh"] + 0.1)
    tout["cv_vitesse"] = tout["speed_std_kmh"] / (tout["speed_kmh"] + 0.1)
    tout["denivele_min"] = tout["altitude_gain_m"] / tout["duration_min"]
    return tout


def relatives(tout, groupe, colonnes, suffixe):
    g = tout.groupby(groupe)
    sortie = {}
    for c in colonnes:
        moy = g[c].transform("mean")
        et = g[c].transform("std").fillna(0) + 1e-6
        med = g[c].transform("median")
        sortie[f"{c}_rg_{suffixe}"] = g[c].rank(pct=True)
        sortie[f"{c}_z_{suffixe}"] = (tout[c] - moy) / et
        sortie[f"{c}_dmed_{suffixe}"] = tout[c] - med
    return pd.DataFrame(sortie, index=tout.index)


def contexte_user(tout):
    g = tout.groupby("user_id")
    ctx = pd.DataFrame(index=tout.index)
    for c in ["speed_kmh", "duration_min", "distance_km", "altitude_gain_m", "t"]:
        ctx[f"user_moy_{c}"] = g[c].transform("mean")
        ctx[f"user_et_{c}"] = g[c].transform("std").fillna(0)
    ctx["user_n"] = g["speed_kmh"].transform("size")
    ctx["user_n_sport"] = tout.groupby(["user_id", "sport"])["speed_kmh"].transform("size")
    return ctx


def construire(tout, avec_user=True, avec_us=True, avec_ctx=True):
    tout = ajouter_derivees(tout)
    morceaux = [tout[["year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km",
                      "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m",
                      "altitude_gain_m", "altitude_loss_m", "n_points", "t", "pente",
                      "ratio_max", "cv_vitesse", "denivele_min"]]]
    morceaux.append(pd.get_dummies(tout["sport"], prefix="sport", dtype=float))
    morceaux.append(pd.get_dummies(tout["gender"], prefix="gender", dtype=float))
    if avec_user:
        morceaux.append(relatives(tout, "user_id", BASE_REL, "u"))
    if avec_us:
        morceaux.append(relatives(tout, ["user_id", "sport"], BASE_US, "us"))
    if avec_ctx:
        morceaux.append(contexte_user(tout))
    return pd.concat(morceaux, axis=1)

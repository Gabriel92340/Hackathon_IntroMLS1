import numpy as np
import pandas as pd

DOSSIER_DONNEES = "/tmp/claude-0/-home-claude/375147e1-a29a-5e1b-b428-9aaecf4407fd/scratchpad/data"
ORDRE = {"easy": 0, "moderate": 1, "hard": 2}
INVERSE = {0: "easy", 1: "moderate", 2: "hard"}
NUM_BRUTES = [
    "duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
    "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points",
]
CALENDRIER = ["year", "month", "weekday", "start_hour_utc"]
SPORTS = ["cycling", "running", "commute_cycling", "mtb", "walking", "orienteering", "indoor_cycling"]


def charger():
    train = pd.read_csv(f"{DOSSIER_DONNEES}/train.csv")
    test = pd.read_csv(f"{DOSSIER_DONNEES}/test.csv")
    y = train["intensity"].map(ORDRE).values
    return train, test, y


def nettoyer(df):
    df = df.copy()
    df.loc[df["altitude_mean_m"] < -500, "altitude_mean_m"] = np.nan
    df["altitude_mean_m"] = df["altitude_mean_m"].fillna(df["altitude_mean_m"].median())
    df["altitude_loss_m"] = df["altitude_loss_m"].abs()
    return df


def features_brutes(df):
    df = nettoyer(df)
    X = df[NUM_BRUTES + CALENDRIER].copy()
    for s in SPORTS:
        X["sport_" + s] = (df["sport"] == s).astype(int)
    X["genre_f"] = (df["gender"] == "female").astype(int)
    X["genre_inconnu"] = (df["gender"] == "unknown").astype(int)
    return X.reset_index(drop=True)


def features_relatives(train, test):
    tout = pd.concat([train.drop(columns=["intensity"]), test], ignore_index=True)
    tout = nettoyer(tout)
    n_train = len(train)
    tout["t"] = tout["year"] + (tout["month"] - 1) / 12.0
    tout["pente"] = tout["altitude_gain_m"] / tout["distance_km"].clip(lower=0.2)
    tout["ratio_max_moy"] = tout["max_speed_kmh"] / tout["speed_kmh"].clip(lower=0.5)
    tout["cv_vitesse"] = tout["speed_std_kmh"] / tout["speed_kmh"].clip(lower=0.5)
    tout["log_duree"] = np.log1p(tout["duration_min"])
    tout["log_dist"] = np.log1p(tout["distance_km"])
    tout["tronque"] = (tout["n_points"] < 500).astype(int)

    base = [
        "speed_kmh", "max_speed_kmh", "speed_std_kmh", "duration_min", "distance_km",
        "altitude_gain_m", "pente", "ratio_max_moy", "cv_vitesse", "altitude_mean_m", "n_points",
    ]
    X = pd.DataFrame(index=tout.index)
    for c in NUM_BRUTES + CALENDRIER + ["pente", "ratio_max_moy", "cv_vitesse", "tronque"]:
        X[c] = tout[c]
    for s in SPORTS:
        X["sport_" + s] = (tout["sport"] == s).astype(int)
    X["genre_f"] = (tout["gender"] == "female").astype(int)
    X["genre_inconnu"] = (tout["gender"] == "unknown").astype(int)

    g_user = tout.groupby("user_id")
    g_us = tout.groupby(["user_id", "sport"])
    for c in base:
        X[c + "_rk_u"] = g_user[c].rank(pct=True)
        X[c + "_rk_us"] = g_us[c].rank(pct=True)
        med_us = g_us[c].transform("median")
        X[c + "_rel_us"] = tout[c] / med_us.clip(lower=1e-6) if c not in ("altitude_mean_m",) else tout[c] - med_us
        mu = g_us[c].transform("mean")
        sd = g_us[c].transform("std").fillna(0).clip(lower=1e-6)
        X[c + "_z_us"] = (tout[c] - mu) / sd

    X["t_rk_u"] = g_user["t"].rank(pct=True)
    X["t_rk_us"] = g_us["t"].rank(pct=True)
    X["t_rel_u"] = tout["t"] - g_user["t"].transform("median")
    X["n_user"] = g_user["t"].transform("size")
    X["n_user_sport"] = g_us["t"].transform("size")
    X["part_sport_user"] = X["n_user_sport"] / X["n_user"]
    X["vit_med_us"] = g_us["speed_kmh"].transform("median")
    X["vit_med_u"] = g_user["speed_kmh"].transform("median")

    X["vit_x_t"] = X["speed_kmh_rk_us"] * X["t_rk_u"]
    X["vit_x_dist"] = X["speed_kmh_rk_us"] * X["distance_km_rk_us"]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    cle = tout[["user_id", "sport"]].reset_index(drop=True)
    return X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True), cle.iloc[:n_train].reset_index(drop=True), cle.iloc[n_train:].reset_index(drop=True)

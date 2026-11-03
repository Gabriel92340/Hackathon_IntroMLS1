import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

CLASSES = ["easy", "moderate", "hard"]
NUM_BRUTES = ["year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km",
              "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m",
              "altitude_gain_m", "altitude_loss_m", "n_points"]


def charger(dossier="data"):
    train = pd.read_csv(f"{dossier}/train.csv")
    test = pd.read_csv(f"{dossier}/test.csv")
    return train, test


def nettoyer(df):
    df = df.copy()
    df.loc[df["altitude_mean_m"] < -500, "altitude_mean_m"] = np.nan
    df["altitude_mean_m"] = df["altitude_mean_m"].fillna(df["altitude_mean_m"].median())
    return df


def features_brutes(tout):
    X = tout[NUM_BRUTES].copy()
    X = pd.concat([X, pd.get_dummies(tout["sport"], prefix="sport", dtype=float),
                   pd.get_dummies(tout["gender"], prefix="gender", dtype=float)], axis=1)
    return X


def cv_oof(modele_fn, X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros((len(X), 3))
    scores = []
    for tr, va in skf.split(X, y):
        m = modele_fn()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])
        proba[va] = p
        scores.append(f1_score(y[va], p.argmax(1), average="macro"))
    return proba, np.array(scores)


def resume(nom, proba, scores, y):
    f1 = f1_score(y, proba.argmax(1), average="macro")
    print(f"{nom:<45} macro-F1 OOF = {f1:.4f}  (folds {scores.mean():.4f} ± {scores.std():.4f})")
    return f1


def matrice(y, pred):
    cm = confusion_matrix(y, pred)
    return pd.DataFrame(cm, index=[f"vrai_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])

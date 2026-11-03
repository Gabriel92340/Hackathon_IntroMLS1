import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE = ["user_id", "sport", "year", "month"]
GROUPES = {"c_jour_heure": ["weekday", "start_hour_utc"], "c_jour_tranche": ["weekday", "hbin"],
           "c_jour": ["weekday"], "c_heure": ["start_hour_utc"]}


def preparer(df):
    df = df.copy()
    df["hbin"] = pd.cut(df["start_hour_utc"], [-1, 5, 9, 12, 15, 18, 21, 24], labels=False).astype(int)
    return df


def encoder(ref, y_ref, req, lissage=2):
    ref, req = preparer(ref).assign(y=y_ref), preparer(req)
    sortie = pd.DataFrame(index=req.index)
    pu = ref.groupby(["user_id", "sport"])["y"].mean().rename("pu")
    prior = req[["user_id", "sport"]].merge(pu, left_on=["user_id", "sport"], right_index=True, how="left")["pu"].fillna(1).values
    onehot = pd.get_dummies(ref["y"]).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
    ref = pd.concat([ref, onehot.add_prefix("o")], axis=1)
    for nom, extra in GROUPES.items():
        cles = BASE + extra
        g = ref.groupby(cles).agg(s=("y", "sum"), n=("y", "size"), e=("o0", "sum"), h=("o2", "sum"))
        m = req[cles].merge(g, left_on=cles, right_index=True, how="left").fillna(0)
        sortie[nom] = (m["s"].values + lissage * prior) / (m["n"].values + lissage) - prior
        sortie[nom + "_easy"] = (m["e"].values + lissage / 3) / (m["n"].values + lissage)
        sortie[nom + "_hard"] = (m["h"].values + lissage / 3) / (m["n"].values + lissage)
        sortie[nom + "_n"] = m["n"].values
    return sortie


def encoder_oof(df, y, seed=9):
    morceaux = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        morceaux.append(encoder(df.iloc[a], y[a], df.iloc[b]))
    return pd.concat(morceaux).loc[df.index]

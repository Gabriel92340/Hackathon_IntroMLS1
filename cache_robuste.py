import pickle
import numpy as np
import pandas as pd
from commun import charger, nettoyer
from features import ajouter_derivees, BASE_REL, BASE_US

train, test = charger()
tout = ajouter_derivees(nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True)))
sortie = {}
for groupe, cols, suf in [("user_id", BASE_REL, "u"), (["user_id", "sport"], BASE_US, "us")]:
    g = tout.groupby(groupe)
    for c in cols:
        med = g[c].transform("median")
        iqr = g[c].transform(lambda s: s.quantile(0.75) - s.quantile(0.25)) + 1e-6
        sortie[f"{c}_zr_{suf}"] = np.clip((tout[c] - med) / iqr, -6, 6)
R = pd.DataFrame(sortie)
n = len(train)
Rtr, Rte = R.iloc[:n].reset_index(drop=True), R.iloc[n:].reset_index(drop=True)
for f in range(5):
    with open(f"cache/fold{f}.pkl", "rb") as fh:
        d = pickle.load(fh)
    with open(f"cache/robuste{f}.pkl", "wb") as fh:
        pickle.dump({"Xa": Rtr.iloc[d["a"]].reset_index(drop=True), "Xb": Rtr.iloc[d["b"]].reset_index(drop=True), "Xt": Rte}, fh)
print("ok", R.shape)

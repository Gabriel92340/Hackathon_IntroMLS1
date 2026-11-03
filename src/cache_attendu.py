import pickle
import pandas as pd
from commun import charger, nettoyer
from features_attendu import residus_attendus

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
R = residus_attendus(tout)
n = len(train)
Rtr, Rte = R.iloc[:n].reset_index(drop=True), R.iloc[n:].reset_index(drop=True)
for f in range(5):
    with open(f"cache/fold{f}.pkl", "rb") as fh:
        d = pickle.load(fh)
    with open(f"cache/attendu{f}.pkl", "wb") as fh:
        pickle.dump({"Xa": Rtr.iloc[d["a"]].reset_index(drop=True), "Xb": Rtr.iloc[d["b"]].reset_index(drop=True), "Xt": Rte}, fh)
print("ok")

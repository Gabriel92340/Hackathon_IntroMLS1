import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from commun import charger, nettoyer, CLASSES
from features import construire
from features_locales import relatives_locales
from encodage import tous_encodages, encodage_oof

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = pd.concat([construire(tout), relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
n = len(train)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
brut_tr = tout.iloc[:n].reset_index(drop=True)
brut_te = tout.iloc[n:].reset_index(drop=True)
Xtr_all = X.iloc[:n].reset_index(drop=True)
Xte_all = X.iloc[n:].reset_index(drop=True)

skf = StratifiedKFold(5, shuffle=True, random_state=0)
for f, (a, b) in enumerate(skf.split(Xtr_all, y)):
    te_a = encodage_oof(brut_tr.iloc[a], y[a], avec_vitesse=True).reset_index(drop=True)
    te_b = tous_encodages(brut_tr.iloc[a], y[a], brut_tr.iloc[b], avec_vitesse=True).reset_index(drop=True)
    te_t = tous_encodages(brut_tr.iloc[a], y[a], brut_te, avec_vitesse=True).reset_index(drop=True)
    Xa = pd.concat([Xtr_all.iloc[a].reset_index(drop=True), te_a], axis=1).fillna(-1)
    Xb = pd.concat([Xtr_all.iloc[b].reset_index(drop=True), te_b], axis=1).fillna(-1)
    Xt = pd.concat([Xte_all, te_t], axis=1).fillna(-1)
    with open(f"cache/fold{f}.pkl", "wb") as fh:
        pickle.dump({"a": a, "b": b, "Xa": Xa, "Xb": Xb, "Xt": Xt}, fh)
    print("fold", f, "ok", flush=True)

te_full_tr = encodage_oof(brut_tr, y, seed=11, avec_vitesse=True).reset_index(drop=True)
te_full_te = tous_encodages(brut_tr, y, brut_te, avec_vitesse=True).reset_index(drop=True)
with open("cache/complet.pkl", "wb") as fh:
    pickle.dump({"Xtr": pd.concat([Xtr_all, te_full_tr], axis=1).fillna(-1),
                 "Xte": pd.concat([Xte_all, te_full_te], axis=1).fillna(-1), "y": y}, fh)
print("complet ok", flush=True)

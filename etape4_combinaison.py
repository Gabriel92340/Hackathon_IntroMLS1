import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from commun import charger, nettoyer, CLASSES
from features import construire
from features_locales import relatives_locales
from encodage import tous_encodages, encodage_oof

avec_loc = "loc" in sys.argv
avec_vit = "vit" in sys.argv
nom = "_".join(["hgb", "temp"] + (["loc"] if avec_loc else []) + (["vit"] if avec_vit else []))

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
if avec_loc:
    X = pd.concat([X, relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
n = len(train)
Xtr = X.iloc[:n].reset_index(drop=True)
brut_tr = tout.iloc[:n].reset_index(drop=True)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values


def modele():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


skf = StratifiedKFold(5, shuffle=True, random_state=0)
proba = np.zeros((n, 3))
for k, (a, b) in enumerate(skf.split(Xtr, y)):
    te_a = encodage_oof(brut_tr.iloc[a], y[a], avec_vitesse=avec_vit)
    te_b = tous_encodages(brut_tr.iloc[a], y[a], brut_tr.iloc[b], avec_vitesse=avec_vit)
    Xa = pd.concat([Xtr.iloc[a].reset_index(drop=True), te_a.reset_index(drop=True)], axis=1)
    Xb = pd.concat([Xtr.iloc[b].reset_index(drop=True), te_b.reset_index(drop=True)], axis=1)
    m = modele().fit(Xa, y[a])
    proba[b] = m.predict_proba(Xb)
    print(f"fold {k}: {f1_score(y[b], proba[b].argmax(1), average='macro'):.4f}", flush=True)

print(nom, "OOF macro-F1:", round(f1_score(y, proba.argmax(1), average="macro"), 4))
np.save(f"oof_{nom}.npy", proba)

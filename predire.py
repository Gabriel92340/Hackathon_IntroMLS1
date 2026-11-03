import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from commun import charger, nettoyer, CLASSES
from features import construire
from features_locales import relatives_locales
from encodage import tous_encodages, encodage_oof

avec_loc = "loc" in sys.argv
avec_vit = "vit" in sys.argv
sortie = sys.argv[1]
graines = [0, 1, 2]

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
if avec_loc:
    X = pd.concat([X, relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
n = len(train)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
brut_tr = tout.iloc[:n].reset_index(drop=True)
brut_te = tout.iloc[n:].reset_index(drop=True)

proba = np.zeros((len(test), 3))
te_test = tous_encodages(brut_tr, y, brut_te, avec_vitesse=avec_vit).reset_index(drop=True)
Xte = pd.concat([X.iloc[n:].reset_index(drop=True), te_test], axis=1)
for g in graines:
    te_tr = encodage_oof(brut_tr, y, seed=g + 10, avec_vitesse=avec_vit).reset_index(drop=True)
    Xtr = pd.concat([X.iloc[:n].reset_index(drop=True), te_tr], axis=1)
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=g)
    m.fit(Xtr, y)
    proba += m.predict_proba(Xte[Xtr.columns]) / len(graines)

np.save(sortie.replace(".csv", "_proba.npy"), proba)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in proba.argmax(1)]})
sub.to_csv(sortie, index=False)
print(sub["intensity"].value_counts(normalize=True).round(3))

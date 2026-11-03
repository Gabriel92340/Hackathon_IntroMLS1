import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from commun import charger, nettoyer, CLASSES
from features import construire
from encodage import noyau_labels, encodage_oof

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
n = len(train)
Xtr = X.iloc[:n].reset_index(drop=True)
brut_tr = tout.iloc[:n].reset_index(drop=True)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values


def modele():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


skf = StratifiedKFold(5, shuffle=True, random_state=0)
proba_base = np.zeros((n, 3))
proba_te = np.zeros((n, 3))
for k, (a, b) in enumerate(skf.split(Xtr, y)):
    te_a = encodage_oof(brut_tr.iloc[a], y[a])
    te_b = noyau_labels(brut_tr.iloc[a], y[a], brut_tr.iloc[b])
    m0 = modele().fit(Xtr.iloc[a], y[a])
    proba_base[b] = m0.predict_proba(Xtr.iloc[b])
    Xa = pd.concat([Xtr.iloc[a].reset_index(drop=True), te_a.reset_index(drop=True)], axis=1)
    Xb = pd.concat([Xtr.iloc[b].reset_index(drop=True), te_b.reset_index(drop=True)], axis=1)
    m1 = modele().fit(Xa, y[a])
    proba_te[b] = m1.predict_proba(Xb)
    print(f"fold {k}: base {f1_score(y[b], proba_base[b].argmax(1), average='macro'):.4f}  +temporel {f1_score(y[b], proba_te[b].argmax(1), average='macro'):.4f}", flush=True)

print("HGB relatif          :", round(f1_score(y, proba_base.argmax(1), average="macro"), 4))
print("HGB relatif+temporel :", round(f1_score(y, proba_te.argmax(1), average="macro"), 4))
np.save("oof_hgb_temporel.npy", proba_te)

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from commun import charger, nettoyer, CLASSES
from encodage_perso import encoder as enc_perso, encoder_oof as enc_perso_oof
from encodage_routine import encoder as enc_rout, encoder_oof as enc_rout_oof

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
cles = ["user_id", "sport"]
with open("cache/complet.pkl", "rb") as fh:
    d = pickle.load(fh)
Xtr, Xte = d["Xtr"], d["Xte"]

tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
brut_tr = tout.iloc[:len(train)].reset_index(drop=True)
brut_te = tout.iloc[len(train):].reset_index(drop=True)

ref = pd.concat([Xtr, train[cles].reset_index(drop=True)], axis=1)
P_tr = enc_perso_oof(ref, y).reset_index(drop=True)
P_te = enc_perso(ref, y, pd.concat([Xte, test[cles].reset_index(drop=True)], axis=1)).reset_index(drop=True)
R_tr = enc_rout_oof(brut_tr, y).reset_index(drop=True)
R_te = enc_rout(brut_tr, y, brut_te).reset_index(drop=True)
with open("cache/complet_extra.pkl", "wb") as fh:
    pickle.dump({"P_tr": P_tr, "P_te": P_te, "R_tr": R_tr, "R_te": R_te}, fh)
print("encodages complets ok", flush=True)

jeux = {
    "": (Xtr, Xte),
    "_r": (pd.concat([Xtr, R_tr], axis=1).fillna(-1), pd.concat([Xte, R_te], axis=1).fillna(-1)),
    "_pr": (pd.concat([Xtr, P_tr, R_tr], axis=1).fillna(-1), pd.concat([Xte, P_te, R_te], axis=1).fillna(-1)),
}
MODELES = {
    "hgb_clf_pr": ("_pr", lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)),
    "et_clf": ("", lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0)),
    "et_clf_r": ("_r", lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0)),
    "et_clf_r_f1": ("_r", lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0)),
    "et_clf_r_m15": ("_r", lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.15, n_jobs=-1, random_state=0)),
    "et_reg": ("", lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0)),
}
for nom in sys.argv[1:] or list(MODELES):
    jeu, fab = MODELES[nom]
    A, B = jeux[jeu]
    m = fab().fit(A, y)
    p = m.predict_proba(B) if "clf" in nom else m.predict(B)
    np.save(f"preds/{nom}_complet_test.npy", p)
    print(nom, "complet ok", flush=True)

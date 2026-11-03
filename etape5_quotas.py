import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from commun import charger, nettoyer, CLASSES, matrice
from features import construire
from features_locales import relatives_locales
from encodage import tous_encodages, encodage_oof
from quotas import affecter

avec_loc = "loc" in sys.argv
avec_vit = "vit" in sys.argv
avec_te = "te" in sys.argv
nom = "_".join(["q"] + [o for o in ["te", "loc", "vit"] if o in sys.argv])

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
if avec_loc:
    X = pd.concat([X, relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
n, nt = len(train), len(test)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
brut_tr = tout.iloc[:n].reset_index(drop=True)
brut_te = tout.iloc[n:].reset_index(drop=True)
Xtr_all = X.iloc[:n].reset_index(drop=True)
Xte_all = X.iloc[n:].reset_index(drop=True)
u_tr = train["user_id"].values
u_te = test["user_id"].values
total = pd.concat([train["user_id"], test["user_id"]]).value_counts().to_dict()

skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = np.zeros((n, 3))
oof_q = np.zeros(n, dtype=int)
proba_test = np.zeros((nt, 3))
for k, (a, b) in enumerate(skf.split(Xtr_all, y)):
    Xa, Xb, Xt = Xtr_all.iloc[a].reset_index(drop=True), Xtr_all.iloc[b].reset_index(drop=True), Xte_all
    if avec_te:
        te_a = encodage_oof(brut_tr.iloc[a], y[a], avec_vitesse=avec_vit).reset_index(drop=True)
        te_b = tous_encodages(brut_tr.iloc[a], y[a], brut_tr.iloc[b], avec_vitesse=avec_vit).reset_index(drop=True)
        te_t = tous_encodages(brut_tr.iloc[a], y[a], brut_te, avec_vitesse=avec_vit).reset_index(drop=True)
        Xa, Xb, Xt = pd.concat([Xa, te_a], axis=1), pd.concat([Xb, te_b], axis=1), pd.concat([Xt, te_t], axis=1)
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0).fit(Xa, y[a])
    pb, pt = m.predict_proba(Xb), m.predict_proba(Xt)
    oof[b] = pb
    proba_test += pt / 5
    connus = pd.crosstab(u_tr[a], y[a]).reindex(columns=[0, 1, 2], fill_value=0)
    connus = {u: connus.loc[u].values.astype(float) if u in connus.index else np.zeros(3) for u in total}
    p_inc = np.vstack([pb, pt])
    u_inc = np.concatenate([u_tr[b], u_te])
    pred = affecter(p_inc, u_inc, total, connus)
    oof_q[b] = pred[:len(b)]
    print(f"fold {k}: argmax {f1_score(y[b], pb.argmax(1), average='macro'):.4f}  quotas {f1_score(y[b], oof_q[b], average='macro'):.4f}", flush=True)

print(nom, "OOF argmax:", round(f1_score(y, oof.argmax(1), average="macro"), 4),
      " OOF quotas:", round(f1_score(y, oof_q, average="macro"), 4))
print(matrice(y, oof_q))
np.save(f"oof_{nom}.npy", oof)
np.save(f"test_{nom}.npy", proba_test)

connus_full = pd.crosstab(u_tr, y).reindex(columns=[0, 1, 2], fill_value=0)
connus_full = {u: connus_full.loc[u].values.astype(float) for u in total}
pred_test = affecter(proba_test, u_te, total, connus_full)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred_test]})
sub.to_csv(f"soumissions/sub_{nom}.csv", index=False)
print(sub["intensity"].value_counts(normalize=True).round(3))

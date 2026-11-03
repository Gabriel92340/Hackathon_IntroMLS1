import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from commun import charger, nettoyer, CLASSES, matrice
from features import construire
from features_locales import relatives_locales
from encodage import tous_encodages, encodage_oof
from quotas import affecter

nom = sys.argv[1] if len(sys.argv) > 1 else "final"

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = pd.concat([construire(tout), relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
n, nt = len(train), len(test)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
brut_tr = tout.iloc[:n].reset_index(drop=True)
brut_te = tout.iloc[n:].reset_index(drop=True)
Xtr_all = X.iloc[:n].reset_index(drop=True)
Xte_all = X.iloc[n:].reset_index(drop=True)
u_tr, u_te = train["user_id"].values, test["user_id"].values
total = pd.concat([train["user_id"], test["user_id"]]).value_counts().to_dict()


def modeles():
    return {
        "hgb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0),
        "et": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3,
                                   n_jobs=-1, random_state=0),
    }


skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = {k: np.zeros((n, 3)) for k in modeles()}
test_p = {k: np.zeros((nt, 3)) for k in modeles()}
oof_q = np.zeros(n, dtype=int)
for f, (a, b) in enumerate(skf.split(Xtr_all, y)):
    te_a = encodage_oof(brut_tr.iloc[a], y[a], avec_vitesse=True).reset_index(drop=True)
    te_b = tous_encodages(brut_tr.iloc[a], y[a], brut_tr.iloc[b], avec_vitesse=True).reset_index(drop=True)
    te_t = tous_encodages(brut_tr.iloc[a], y[a], brut_te, avec_vitesse=True).reset_index(drop=True)
    Xa = pd.concat([Xtr_all.iloc[a].reset_index(drop=True), te_a], axis=1)
    Xb = pd.concat([Xtr_all.iloc[b].reset_index(drop=True), te_b], axis=1)
    Xt = pd.concat([Xte_all, te_t], axis=1)
    Xa_f, Xb_f, Xt_f = Xa.fillna(-1), Xb.fillna(-1), Xt.fillna(-1)
    ligne = f"fold {f}:"
    pt_fold = np.zeros((nt, 3))
    for k, m in modeles().items():
        m.fit(Xa_f, y[a])
        oof[k][b] = m.predict_proba(Xb_f)
        p_t = m.predict_proba(Xt_f)
        test_p[k] += p_t / 5
        pt_fold += p_t / 2
        ligne += f" {k} {f1_score(y[b], oof[k][b].argmax(1), average='macro'):.4f}"
    pb = sum(oof[k][b] for k in oof) / len(oof)
    connus = pd.crosstab(u_tr[a], y[a]).reindex(columns=[0, 1, 2], fill_value=0)
    connus = {u: connus.loc[u].values.astype(float) if u in connus.index else np.zeros(3) for u in total}
    pred = affecter(np.vstack([pb, pt_fold]), np.concatenate([u_tr[b], u_te]), total, connus)
    oof_q[b] = pred[:len(b)]
    ligne += f" | moyenne {f1_score(y[b], pb.argmax(1), average='macro'):.4f} | quotas {f1_score(y[b], oof_q[b], average='macro'):.4f}"
    print(ligne, flush=True)

moy = sum(oof.values()) / len(oof)
for k in oof:
    print(k, "OOF:", round(f1_score(y, oof[k].argmax(1), average="macro"), 4))
print("moyenne OOF:", round(f1_score(y, moy.argmax(1), average="macro"), 4))
print("moyenne + quotas OOF:", round(f1_score(y, oof_q, average="macro"), 4))
print(matrice(y, oof_q))
for k in oof:
    np.save(f"oof6_{k}.npy", oof[k])
    np.save(f"test6_{k}.npy", test_p[k])
np.save("oof6_quotas.npy", oof_q)

pt = sum(test_p.values()) / len(test_p)
connus_full = pd.crosstab(u_tr, y).reindex(columns=[0, 1, 2], fill_value=0)
connus_full = {u: connus_full.loc[u].values.astype(float) for u in total}
pred_test = affecter(pt, u_te, total, connus_full)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred_test]})
sub.to_csv(f"soumissions/sub_{nom}.csv", index=False)
sub_sans = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pt.argmax(1)]})
sub_sans.to_csv(f"soumissions/sub_{nom}_sans_quotas.csv", index=False)
print(sub["intensity"].value_counts(normalize=True).round(3))

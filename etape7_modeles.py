import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier)
from commun import charger, CLASSES

train, _ = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values

MODELES = {
    "hgb_clf": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "hgb_reg": lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "hgb_lent": lambda: HistGradientBoostingClassifier(learning_rate=0.025, max_iter=900, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=2.0, random_state=1),
    "hgb_petit": lambda: HistGradientBoostingClassifier(learning_rate=0.04, max_iter=700, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, max_features=0.5, random_state=2),
    "et_fin": lambda: ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features=0.4, n_jobs=-1, random_state=3),
    "hgb_clf_p": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_p": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg_p": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "hgb_clf_pr": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_r": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg_r": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "hgb_clf_prz": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_r_f1": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0),
    "et_clf_r_f6": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=6, max_features=0.3, n_jobs=-1, random_state=0),
    "et_clf_r_m5": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0),
    "et_clf_r_m15": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.15, n_jobs=-1, random_state=0),
    "rf_clf_r": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.2, n_jobs=-1, random_state=0),
    "hgb_big_pr": lambda: HistGradientBoostingClassifier(learning_rate=0.04, max_iter=1000, max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=0),
    "et_big_r_f1": lambda: ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=1),
    "hgb_clf_prc": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_rc_f1": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0),
    "et_clf_rc": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg_rc": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "hgb_clf_pra": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_ra_f1": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg_a": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "hgb_clf_v20": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
    "et_clf_v20_f1": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0),
    "et_clf_v20c_f1": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg_v20": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "et_reg": lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
}
choix = sys.argv[1:] or list(MODELES)

for nom in choix:
    base, graine = (nom.split("@")[0], int(nom.split("@")[1])) if "@" in nom else (nom, 0)
    for f in range(5):
        with open(f"cache/fold{f}.pkl", "rb") as fh:
            d = pickle.load(fh)
        BLOCS = {"hgb_clf_v20": ["perso", "routine20", "creneau20", "attendu"], "et_clf_v20_f1": ["routine20", "attendu"],
                 "et_clf_v20c_f1": ["routine20", "creneau20", "attendu"], "et_reg_v20": ["routine20", "attendu"]}
        if base in BLOCS:
            for nomc in BLOCS[base]:
                with open(f"cache/{nomc}{f}.pkl", "rb") as fh:
                    r = pickle.load(fh)
                r = {k: v.add_prefix(nomc + "_") for k, v in r.items()}
                for cle in ["Xa", "Xb", "Xt"]:
                    d[cle] = pd.concat([d[cle], r[cle]], axis=1)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = d[cle].fillna(-1)
        elif base in ("hgb_clf_pra", "et_clf_ra_f1", "et_reg_a"):
            blocs = {"hgb_clf_pra": ["perso", "routine", "attendu"], "et_clf_ra_f1": ["routine", "attendu"], "et_reg_a": ["attendu"]}[base]
            for nomc in blocs:
                with open(f"cache/{nomc}{f}.pkl", "rb") as fh:
                    r = pickle.load(fh)
                for cle in ["Xa", "Xb", "Xt"]:
                    d[cle] = pd.concat([d[cle], r[cle]], axis=1)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = d[cle].fillna(-1)
        elif "_rc" in base or base.endswith("_prc"):
            blocs = ["routine", "creneau"] + (["perso"] if base.endswith("_prc") else [])
            for nomc in blocs:
                with open(f"cache/{nomc}{f}.pkl", "rb") as fh:
                    r = pickle.load(fh)
                for cle in ["Xa", "Xb", "Xt"]:
                    d[cle] = pd.concat([d[cle], r[cle]], axis=1)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = d[cle].fillna(-1)
        elif base.endswith("_r") or "_r_" in base:
            with open(f"cache/routine{f}.pkl", "rb") as fh:
                r = pickle.load(fh)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = pd.concat([d[cle], r[cle]], axis=1).fillna(-1)
        if base.endswith("_prz"):
            for nomc in ["perso", "routine", "robuste"]:
                with open(f"cache/{nomc}{f}.pkl", "rb") as fh:
                    r = pickle.load(fh)
                for cle in ["Xa", "Xb", "Xt"]:
                    d[cle] = pd.concat([d[cle], r[cle]], axis=1)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = d[cle].fillna(-1)
        if base.endswith("_pr"):
            with open(f"cache/perso{f}.pkl", "rb") as fh:
                p = pickle.load(fh)
            with open(f"cache/routine{f}.pkl", "rb") as fh:
                r = pickle.load(fh)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = pd.concat([d[cle], p[cle], r[cle]], axis=1).fillna(-1)
        if base.endswith("_p"):
            with open(f"cache/perso{f}.pkl", "rb") as fh:
                p = pickle.load(fh)
            for cle in ["Xa", "Xb", "Xt"]:
                d[cle] = pd.concat([d[cle], p[cle]], axis=1).fillna(-1)
        m = MODELES[base]()
        m.set_params(random_state=graine)
        m.fit(d["Xa"], y[d["a"]])
        if "clf" in nom:
            pb, pt = m.predict_proba(d["Xb"]), m.predict_proba(d["Xt"])
        else:
            pb, pt = m.predict(d["Xb"]), m.predict(d["Xt"])
        np.save(f"preds/{nom}_f{f}_val.npy", pb)
        np.save(f"preds/{nom}_f{f}_test.npy", pt)
        print(nom, "fold", f, "ok", flush=True)

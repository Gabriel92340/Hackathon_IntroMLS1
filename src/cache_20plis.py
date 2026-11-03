import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from commun import charger, nettoyer, CLASSES
import encodage_routine as er
import encodage_creneau as ec

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
tr = tout.iloc[:len(train)].reset_index(drop=True)
te = tout.iloc[len(train):].reset_index(drop=True)
tr_brut = train.drop(columns="intensity").reset_index(drop=True)
te_brut = test.reset_index(drop=True)


def oof20(module, df, yy, seed):
    morceaux = []
    for a, b in StratifiedKFold(20, shuffle=True, random_state=seed).split(df, yy):
        morceaux.append(module.encoder(df.iloc[a], yy[a], df.iloc[b]))
    return pd.concat(morceaux).loc[df.index]


for f in range(5):
    with open(f"cache/fold{f}.pkl", "rb") as fh:
        d = pickle.load(fh)
    a, b = d["a"], d["b"]
    for nom, module, A, T, seed in [("routine20", er, tr, te, 5), ("creneau20", ec, tr_brut, te_brut, 9)]:
        Aa = A.iloc[a].reset_index(drop=True)
        ra = oof20(module, Aa, y[a], seed)
        rb = module.encoder(Aa, y[a], A.iloc[b].reset_index(drop=True))
        rt = module.encoder(Aa, y[a], T)
        with open(f"cache/{nom}{f}.pkl", "wb") as fh:
            pickle.dump({"Xa": ra.reset_index(drop=True), "Xb": rb.reset_index(drop=True), "Xt": rt.reset_index(drop=True)}, fh)
    print("fold", f, "ok", flush=True)

import pickle
import pandas as pd
from commun import charger, nettoyer, CLASSES
from encodage_routine import encoder, encoder_oof

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
tr = tout.iloc[:len(train)].reset_index(drop=True)
te = tout.iloc[len(train):].reset_index(drop=True)
for f in range(5):
    with open(f"cache/fold{f}.pkl", "rb") as fh:
        d = pickle.load(fh)
    a, b = d["a"], d["b"]
    ra = encoder_oof(tr.iloc[a].reset_index(drop=True), y[a])
    rb = encoder(tr.iloc[a].reset_index(drop=True), y[a], tr.iloc[b].reset_index(drop=True))
    rt = encoder(tr.iloc[a].reset_index(drop=True), y[a], te)
    with open(f"cache/routine{f}.pkl", "wb") as fh:
        pickle.dump({"Xa": ra.reset_index(drop=True), "Xb": rb.reset_index(drop=True), "Xt": rt.reset_index(drop=True)}, fh)
    print("routine fold", f, "ok", flush=True)

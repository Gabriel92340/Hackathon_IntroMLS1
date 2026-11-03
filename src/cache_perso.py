import pickle
import numpy as np
import pandas as pd
from commun import charger, CLASSES
from encodage_perso import encoder, encoder_oof

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
cles = ["user_id", "sport"]

for f in range(5):
    with open(f"cache/fold{f}.pkl", "rb") as fh:
        d = pickle.load(fh)
    a, b = d["a"], d["b"]
    ref = pd.concat([d["Xa"], train.iloc[a][cles].reset_index(drop=True)], axis=1)
    val = pd.concat([d["Xb"], train.iloc[b][cles].reset_index(drop=True)], axis=1)
    tst = pd.concat([d["Xt"], test[cles].reset_index(drop=True)], axis=1)
    pa = encoder_oof(ref, y[a]).reset_index(drop=True)
    pb = encoder(ref, y[a], val).reset_index(drop=True)
    pt = encoder(ref, y[a], tst).reset_index(drop=True)
    with open(f"cache/perso{f}.pkl", "wb") as fh:
        pickle.dump({"Xa": pa, "Xb": pb, "Xt": pt}, fh)
    print("perso fold", f, "ok", flush=True)

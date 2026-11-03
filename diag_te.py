import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from commun import charger, CLASSES

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
train["y"] = y
train["hard"] = (y == 2).astype(float)
train["easy"] = (y == 0).astype(float)
train["ym"] = train["year"] * 12 + train["month"]

skf = StratifiedKFold(5, shuffle=True, random_state=0)
for cles in [["user_id"], ["user_id", "year"], ["user_id", "ym"], ["user_id", "sport", "year"], ["user_id", "year", "weekday"]]:
    te = np.full(len(train), np.nan)
    cnt = np.zeros(len(train))
    for tr, va in skf.split(train, y):
        st = train.iloc[tr].groupby(cles)["y"].agg(["mean", "size"])
        j = train.iloc[va][cles].merge(st, left_on=cles, right_index=True, how="left")
        te[va] = j["mean"].values
        cnt[va] = j["size"].fillna(0).values
    m = ~np.isnan(te)
    rho = pd.Series(te[m]).corr(pd.Series(y[m]), method="spearman")
    print(f"{str(cles):<35} couverture {m.mean():.3f}  n_moy {cnt[m].mean():.1f}  spearman {rho:.3f}")

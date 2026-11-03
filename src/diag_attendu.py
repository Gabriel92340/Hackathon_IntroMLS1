import numpy as np
import pandas as pd
from commun import charger, nettoyer, CLASSES
from evaluer import plis
from features_attendu import residus_attendus

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
R = residus_attendus(tout).iloc[:len(train)].reset_index(drop=True)
CLF = ["hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1"]
ens = np.zeros(len(y))
for f, (a, b) in enumerate(plis):
    p = sum(np.load(f"preds/{c}_f{f}_val.npy") for c in CLF) / 4
    ens[b] = p[:, 1] + 2 * p[:, 2]
res = y - ens
rg = tout.iloc[:len(train)].groupby(["user_id", "sport"])["speed_kmh"].rank(pct=True).values
print("référence : rang de vitesse brut  Spearman(y)", round(pd.Series(rg).corr(pd.Series(y), method="spearman"), 3))
for c in R.columns:
    print(f"{c:<12} Spearman(y) {pd.Series(R[c]).corr(pd.Series(y), method='spearman'):+.3f}   Spearman(résidu modèle) {pd.Series(R[c]).corr(pd.Series(res), method='spearman'):+.3f}")

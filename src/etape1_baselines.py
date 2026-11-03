import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from commun import charger, nettoyer, features_brutes, cv_oof, resume, CLASSES

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = features_brutes(tout)
n = len(train)
Xtr = X.iloc[:n].reset_index(drop=True)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values

modeles = {
    "LogReg brut": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "RandomForest brut (300 arbres)": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "HistGradientBoosting brut": lambda: HistGradientBoostingClassifier(random_state=0),
}
for nom, fn in modeles.items():
    p, s = cv_oof(fn, Xtr, y)
    resume(nom, p, s, y)

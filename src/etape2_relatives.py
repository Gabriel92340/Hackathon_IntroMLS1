import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from commun import charger, nettoyer, cv_oof, resume, CLASSES
from features import construire

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
n = len(train)
Xtr = X.iloc[:n].reset_index(drop=True)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
print("Nombre de features:", X.shape[1])

modeles = {
    "LogReg relatif": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
    "RandomForest relatif": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "ExtraTrees relatif": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0),
    "HGB relatif": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
}
for nom, fn in modeles.items():
    p, s = cv_oof(fn, Xtr, y)
    resume(nom, p, s, y)
    np.save(f"oof_{nom.replace(' ', '_')}.npy", p)

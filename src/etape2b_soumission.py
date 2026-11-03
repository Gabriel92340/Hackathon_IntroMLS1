import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from commun import charger, nettoyer, CLASSES
from features import construire

train, test = charger()
tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X = construire(tout)
n = len(train)
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
m = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=0)
m.fit(X.iloc[:n], y)
pred = m.predict(X.iloc[n:])
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]})
sub.to_csv("soumissions/sub01_extratrees_relatif.csv", index=False)
print(sub["intensity"].value_counts(normalize=True))

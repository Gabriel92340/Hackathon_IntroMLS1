import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from commun import charger, CLASSES
from quotas import affecter_cout

sortie = sys.argv[1]
CLF = sys.argv[2].split("+") if len(sys.argv) > 2 else ["hgb_clf", "et_clf"]
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
REG = sys.argv[3].split("+") if len(sys.argv) > 3 else ["et_reg"]
train, test = charger()
g_tr = (train["user_id"].astype(str) + "_" + train["sport"]).values
g_te = (test["user_id"].astype(str) + "_" + test["sport"]).values
total = pd.Series(np.concatenate([g_tr, g_te])).value_counts().to_dict()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
c = pd.crosstab(g_tr, y).reindex(columns=[0, 1, 2], fill_value=0)
connus = {g: c.loc[g].values.astype(float) if g in c.index else np.zeros(3) for g in total}

p_test = np.zeros((len(test), 3))
s_test = np.zeros(len(test))
for f in range(5):
    p_test += sum(np.load(f"preds/{c}_f{f}_test.npy") for c in CLF) / len(CLF) / 5
    s_test += sum(np.load(f"preds/{r}_f{f}_test.npy") for r in REG) / len(REG) / 5
cout = -np.log(np.clip(p_test, 1e-9, 1)) + lam * (np.arange(3)[None, :] - s_test[:, None]) ** 2
pred = affecter_cout(cout, g_te, total, connus)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]})
sub.to_csv(sortie, index=False)
print(sub["intensity"].value_counts(normalize=True).round(3))
chk = pd.crosstab(g_te, pred).reindex(columns=[0, 1, 2], fill_value=0)
ecarts = [np.abs(chk.loc[g].values + connus[g] - total[g] / 3).max() for g in chk.index]
print("écart max aux tertiles exacts après affectation:", round(max(ecarts), 2))

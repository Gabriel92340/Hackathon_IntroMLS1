import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from commun import charger, CLASSES
from quotas import affecter

train, _ = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
u = train["user_id"].values
p = np.load("oof_HGB_relatif.npy")
tot = pd.Series(u).value_counts().to_dict()
zero = {k: np.zeros(3) for k in tot}
print("argmax                       ", round(f1_score(y, p.argmax(1), average="macro"), 4))
print("hongrois log-proba (oracle q)", round(f1_score(y, affecter(p, u, tot, zero), average="macro"), 4))
s = p[:, 2] - p[:, 0]
pred = np.zeros(len(y), int)
for k in np.unique(u):
    i = np.where(u == k)[0]
    r = s[i].argsort().argsort() / len(i)
    pred[i] = np.digitize(r, [1/3, 2/3])
print("tri par p_hard - p_easy      ", round(f1_score(y, pred, average="macro"), 4))

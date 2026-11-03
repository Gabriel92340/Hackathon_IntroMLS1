import numpy as np
import pandas as pd
from commun import charger, CLASSES

train, test = charger()
ct = pd.crosstab(train["user_id"], train["intensity"])[CLASSES]
nt = test.groupby("user_id").size()
tot = ct.sum(1) + nt
quota = pd.DataFrame({c: tot / 3 - ct[c] for c in CLASSES})
print("quotas test par user (tot/3 - train) : min", quota.min().round(2).to_dict())
print("part des (user,classe) avec quota < -1 :", (quota < -1).values.mean().round(3))
print("part des (user,classe) avec quota < 0  :", (quota < 0).values.mean().round(3))
print("somme quotas == n_test ?", np.allclose(quota.sum(1), nt))
print(quota.describe().round(2))
rng = np.random.default_rng(0)
sim = []
for _ in range(20):
    lab = []
    for u, k in tot.items():
        l = np.repeat([0, 1, 2], int(np.ceil(k / 3)))[:k]
        rng.shuffle(l)
        lab.append((ct.loc[u].sum(), l))
    q = []
    for ntr, l in lab:
        tr = l[:ntr]
        q += [len(l) / 3 - (tr == c).sum() for c in range(3)]
    sim.append((np.array(q) < -1).mean())
print("simulation si labels aléatoires (sans structure tertile par split) — part quota<-1 :", np.mean(sim).round(3))

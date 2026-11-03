import numpy as np
import pandas as pd
from commun import charger, CLASSES
from evaluer import plis

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
tr = train.copy()
tr["y"] = y
tr["hbin"] = pd.cut(tr["start_hour_utc"], [-1, 5, 9, 12, 15, 18, 21, 24], labels=False)
CLF = ["hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1"]
ens = np.zeros(len(y))
for f, (a, b) in enumerate(plis):
    p = sum(np.load(f"preds/{c}_f{f}_val.npy") for c in CLF) / 4
    ens[b] = p[:, 1] + 2 * p[:, 2]
res = y - ens


def te(cles, lissage=2):
    out = np.full(len(y), np.nan)
    cnt = np.zeros(len(y))
    for a, b in plis:
        ref = tr.iloc[a]
        g = ref.groupby(cles)["y"].agg(["sum", "count"])
        pu = ref.groupby(["user_id", "sport"])["y"].mean()
        m = tr.iloc[b][cles].merge(g, left_on=cles, right_index=True, how="left")
        pr = tr.iloc[b][["user_id", "sport"]].merge(pu.rename("pu"), left_on=["user_id", "sport"], right_index=True, how="left")["pu"].fillna(1).values
        out[b] = (m["sum"].fillna(0).values + lissage * pr) / (m["count"].fillna(0).values + lissage) - pr
        cnt[b] = m["count"].fillna(0).values
    return out, cnt


base = ["user_id", "sport", "year", "month"]
for nom, extra in [("même mois+jour+heure exacte", ["weekday", "start_hour_utc"]),
                   ("même mois+jour+tranche", ["weekday", "hbin"]),
                   ("même mois+jour", ["weekday"]),
                   ("même mois+heure exacte", ["start_hour_utc"])]:
    v, c = te(base + extra)
    m = c > 0
    print(f"{nom:<30} couverture {m.mean():.2f}  Spearman(résidu) tout {pd.Series(v).corr(pd.Series(res), method='spearman'):+.3f}  quand présent {pd.Series(v[m]).corr(pd.Series(res[m]), method='spearman'):+.3f}")

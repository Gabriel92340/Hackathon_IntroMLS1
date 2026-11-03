import numpy as np
import pandas as pd
from commun import charger, nettoyer, CLASSES
from evaluer import plis

train, test = charger()
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
tr = nettoyer(train.drop(columns="intensity")).reset_index(drop=True)
tr["y"] = y
tr["hbin"] = pd.cut(tr["start_hour_utc"], [-1, 5, 9, 12, 15, 18, 21, 24], labels=False)
tr["we"] = (tr["weekday"] >= 5).astype(int)

ens = np.zeros(len(y))
for f, (a, b) in enumerate(plis):
    p = (np.load(f"preds/hgb_clf_p_f{f}_val.npy") + np.load(f"preds/et_clf_f{f}_val.npy")) / 2
    ens[b] = p[:, 1] + 2 * p[:, 2]
residu = y - ens
print("Spearman(score actuel, y) =", round(pd.Series(ens).corr(pd.Series(y), method="spearman"), 3))


def te(cles, lissage=5):
    out = np.full(len(y), np.nan)
    for a, b in plis:
        ref = tr.iloc[a]
        g = ref.groupby(cles)["y"].agg(["sum", "count"])
        moy_u = ref.groupby(["user_id", "sport"])["y"].mean()
        j = tr.iloc[b][cles + ["user_id", "sport"]].copy() if "user_id" not in cles else tr.iloc[b][cles].copy()
        m = j.merge(g, left_on=cles, right_index=True, how="left")
        prior = tr.iloc[b][["user_id", "sport"]].merge(moy_u.rename("pu"), left_on=["user_id", "sport"], right_index=True, how="left")["pu"].fillna(1).values
        out[b] = ((m["sum"].fillna(0).values + lissage * prior) / (m["count"].fillna(0).values + lissage)) - prior
    return out


def parcours(k=8):
    out = np.full(len(y), np.nan)
    out_lab = np.full(len(y), np.nan)
    cols = ["distance_km", "altitude_mean_m", "altitude_gain_m", "altitude_loss_m"]
    for a, b in plis:
        ref, req = tr.iloc[a], tr.iloc[b]
        for (u, s), ib in req.groupby(["user_id", "sport"]).indices.items():
            ia = np.where((ref["user_id"].values == u) & (ref["sport"].values == s))[0]
            if len(ia) < 5:
                continue
            A = ref.iloc[ia][cols].values.astype(float)
            B = req.iloc[ib][cols].values.astype(float)
            echelle = np.array([0.05, 1, 0.1, 0.1])
            dA = np.column_stack([np.log1p(A[:, 0]) / echelle[0], A[:, 1] / 20, np.log1p(A[:, 2]) / 0.3, np.log1p(A[:, 3]) / 0.3])
            dB = np.column_stack([np.log1p(B[:, 0]) / echelle[0], B[:, 1] / 20, np.log1p(B[:, 2]) / 0.3, np.log1p(B[:, 3]) / 0.3])
            d2 = ((dB[:, None, :] - dA[None, :, :]) ** 2).sum(-1)
            kk = min(k, len(ia))
            vois = np.argsort(d2, 1)[:, :kk]
            dv = np.take_along_axis(d2, vois, 1)
            w = np.exp(-dv / 4)
            v_ref = ref.iloc[ia]["speed_kmh"].values[vois]
            y_ref = ref.iloc[ia]["y"].values[vois]
            v = req.iloc[ib]["speed_kmh"].values
            out[b[ib]] = ((v[:, None] > v_ref) * w).sum(1) / (w.sum(1) + 1e-9)
            out_lab[b[ib]] = (y_ref * w).sum(1) / (w.sum(1) + 1e-9)
    return out, out_lab


cand = {
    "routine jour (u,sport,weekday)": te(["user_id", "sport", "weekday"]),
    "routine heure (u,sport,tranche)": te(["user_id", "sport", "hbin"]),
    "routine jour+heure": te(["user_id", "sport", "weekday", "hbin"]),
    "saison (u,sport,mois calendaire)": te(["user_id", "sport", "month"]),
}
rang_parc, lab_parc = parcours()
cand["parcours: rang vitesse vs même parcours"] = rang_parc
cand["parcours: étiquettes même parcours"] = lab_parc
for nom, v in cand.items():
    m = ~np.isnan(v)
    print(f"{nom:<42} couverture {m.mean():.2f}  Spearman(y) {pd.Series(v[m]).corr(pd.Series(y[m]), method='spearman'):+.3f}  Spearman(résidu) {pd.Series(v[m]).corr(pd.Series(residu[m]), method='spearman'):+.3f}")

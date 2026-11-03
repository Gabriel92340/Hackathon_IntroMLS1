"""sub10_stack_locaux.py : produit sub10_stack_locaux.csv (CV 0,6277 sur 4/5 folds, public 0,65456).

Idée : sub09 empilé avec des modèles qui se trompent autrement que nos arbres globaux :
  - une régression logistique locale (C=0,3) par (user, sport), si au moins 30 séances connues ;
  - un k-NN (k=25, pondéré par la distance) dans chaque (user, sport) ;
  - un HGB « moderate contre le reste » sur les mêmes features et les probabilités de sub09.
Features locales : z-scores intra-groupe de log(vitesse), log(vitesse max), log(durée), log(distance),
D+/km, pente, date centrée, date², heure. Méta-modèle : LogisticRegression(C=0,05) sur les log-probabilités
hors fold. Décision : affectation hongroise sous quotas exacts de tertiles par (user, sport).

À lancer depuis le dossier du code de l'équipe (evaluer.py, quotas.py, commun.py, preds/,
sub09_oof_probas.csv, sub09_test_probas.csv, data/train.csv, data/test.csv) :
    python3 sub10_stack_locaux.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de, train, test
from quotas import affecter_cout
CLASSES = ["easy", "moderate", "hard"]
n, m = len(train), len(test)
fold = np.zeros(n, int)
for f, (a, b) in enumerate(plis): fold[b] = f
oofcsv = pd.read_csv("sub09_oof_probas.csv")
assert (oofcsv["workout_id"].values == train["workout_id"].values).all() and (oofcsv["fold"].values == fold).all()

# --- sub09 reconstruit par fold (val + test) ---
CLF = "hgb_clf_pra+hgb_clf_pr+et_clf+et_clf_r+et_clf_r_f1+et_clf_ra_f1".split("+")
def bloc(f, partie):
    cols = [np.log(np.clip(np.load(f"preds/{c}_f{f}_{partie}.npy"), 1e-6, 1)) for c in CLF]
    s = np.load(f"preds/et_reg_f{f}_{partie}.npy"); cols += [np.column_stack([s, s**2]), np.load(f"preds/paires_f{f}_{partie}.npy")]
    return np.hstack(cols)
meta = np.zeros((n, bloc(0, "val").shape[1]))
for f, (a, b) in enumerate(plis): meta[b] = bloc(f, "val")
S_val = np.zeros((n, 3)); S_te = {}
for f, (a, b) in enumerate(plis):
    lr = LogisticRegression(max_iter=3000).fit(meta[a], y[a])
    S_val[b] = lr.predict_proba(meta[b]); S_te[f] = lr.predict_proba(bloc(f, "test"))
S_final = pd.read_csv("sub09_test_probas.csv").set_index("workout_id").loc[test["workout_id"]].values

# --- features intra-groupe ---
tout = pd.concat([train, test], ignore_index=True)
g = np.concatenate([u_tr, u_te])
t = tout["year"] + tout["month"] / 12
km = tout["distance_km"].clip(lower=0.05)
F = pd.DataFrame({"lv": np.log(tout["speed_kmh"].clip(lower=0.1)), "lvm": np.log(tout["max_speed_kmh"].clip(lower=0.1)),
    "ld": np.log(tout["duration_min"].clip(lower=1)), "lkm": np.log(km), "dpk": tout["altitude_gain_m"] / km,
    "pente": (tout["altitude_gain_m"] - tout["altitude_loss_m"]) / km, "t": t, "h": tout["start_hour_utc"]})
Z = F.groupby(g).transform(lambda c: (c - c.mean()) / (c.std() + 1e-9)).fillna(0)
Z["t2"] = Z["t"] ** 2
Z = Z.clip(-5, 5).values
ytt = np.concatenate([y, np.full(m, -1)])
fold_all = np.concatenate([fold, np.full(m, -1)])
groupes = {u: np.where(g == u)[0] for u in np.unique(g)}

def modele_local(type_, connu):  # connu : masque sur n+m des lignes étiquetées utilisables
    out = np.full((n + m, 3), 1 / 3)
    for u, idx in groupes.items():
        tr = idx[connu[idx]]
        if len(tr) < 30 or len(np.unique(ytt[tr])) < 2: continue
        mdl = LogisticRegression(C=0.3, max_iter=2000) if type_ == "lr" else KNeighborsClassifier(min(25, len(tr)), weights="distance")
        mdl.fit(Z[tr], ytt[tr])
        p = np.zeros((len(idx), 3)); p[:, mdl.classes_] = mdl.predict_proba(Z[idx])
        out[idx] = p
    return np.clip(out, 1e-3, 1) / np.clip(out, 1e-3, 1).sum(1, keepdims=True)

def decider(pv, pt, f):
    a, b = plis[f]
    cout = -np.log(np.clip(np.vstack([pv, pt]), 1e-9, 1))
    return affecter_cout(cout, np.concatenate([u_tr[b], u_te]), total, connus_de(a))[:len(b)]

def score(pred):
    return f1_score(y, pred, average="macro"), np.array([f1_score(y[b], pred[b], average="macro") for a, b in plis])

base_pred = np.zeros(n, int)
for f, (a, b) in enumerate(plis): base_pred[b] = decider(S_val[b], S_te[f], f)
base, base_f = score(base_pred)
print(f"sub09 reconstruit : {base:.4f} {np.round(base_f, 4)}", flush=True)


from sklearn.ensemble import HistGradientBoostingClassifier

# Probabilités hors fold des deux modèles locaux (pour apprendre le méta-modèle) et version entraînée sur tout le train (pour le test)
Lv, Lf = {}, {}
for type_ in ["lr", "knn"]:
    Lv[type_] = np.zeros((n, 3))
    for f in range(5):
        P = modele_local(type_, (fold_all != f) & (fold_all >= 0)); Lv[type_][fold == f] = P[:n][fold == f]
    Lf[type_] = modele_local(type_, np.concatenate([np.ones(n, bool), np.zeros(m, bool)]))[n:]

# HGB moderate contre le reste, hors fold puis sur tout le train
X_all = np.hstack([Z[:n], np.log(S_val)])
M_val = np.zeros(n)
for f, (a, b) in enumerate(plis):
    h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X_all[a], (y[a] == 1).astype(int))
    M_val[b] = h.predict_proba(X_all[b])[:, 1]
h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X_all, (y == 1).astype(int))
M_f = h.predict_proba(np.hstack([Z[n:], np.log(S_final)]))[:, 1]

# Méta-modèle très régularisé, puis affectation sous quotas exacts sur le test
lg = lambda q: np.column_stack([np.log(np.clip(q, 1e-4, 1)), np.log(np.clip(1 - q, 1e-4, 1))])
X_val = np.hstack([np.log(S_val), np.log(Lv["lr"]), np.log(Lv["knn"]), lg(M_val)])
mlr = LogisticRegression(C=0.05, max_iter=3000).fit(X_val, y)
p = mlr.predict_proba(np.hstack([np.log(S_final), np.log(Lf["lr"]), np.log(Lf["knn"]), lg(M_f)]))
c = pd.crosstab(u_tr, y).reindex(columns=[0, 1, 2], fill_value=0)
con = {u: c.loc[u].values.astype(float) if u in c.index else np.zeros(3) for u in total}
pred = affecter_cout(-np.log(np.clip(p, 1e-9, 1)), u_te, total, con)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": np.array(CLASSES)[pred]})
sortie = sys.argv[1] if len(sys.argv) > 1 else "sub10_stack_locaux.csv"
sub.to_csv(sortie, index=False)
print("Écrit", sortie, sub["intensity"].value_counts().to_dict())

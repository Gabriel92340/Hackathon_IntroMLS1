import re
import nbformat as nbf


def corps(fichier):
    lignes = open(fichier).read().splitlines()
    garde = [l for l in lignes if not re.match(r"^(import|from) ", l)]
    return "\n".join(garde).strip()


nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Hackathon ML 2026 — Will this workout be hard for this person?

**Résultat : macro-F1 en validation croisée 5 folds = 0,617**

**Idée centrale :** `intensity` est le **tertile, par (personne, sport), d'une mesure de capteur cachée** (probablement le cardio), calculé sur train + test réunis. Conséquences :
1. on compare chaque séance **aux habitudes de la personne**, pas aux valeurs absolues ;
2. on connaît **le nombre exact de séances easy / moderate / hard de chaque (personne, sport) dans le test**, et on répartit les prédictions par affectation optimale (algorithme hongrois) sous ces quotas.

Pipeline : features relatives + locales + encodages d'étiquettes hors fold → HistGradientBoosting + ExtraTrees (classifieurs) + ExtraTrees (régresseur) → affectation sous quotas.""")

code("""import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor)

DOSSIER = "data" if os.path.exists("data/train.csv") else "."
train = pd.read_csv(f"{DOSSIER}/train.csv")
test = pd.read_csv(f"{DOSSIER}/test.csv")
CLASSES = ["easy", "moderate", "hard"]
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
n, nt = len(train), len(test)
print(train.shape, test.shape)""")

md("""## 1. Comprendre la cible

Les trois classes sont équilibrées **à l'intérieur de chaque personne**, et la vitesse brute est peu liée à la cible alors que la vitesse **relative à la personne** l'est beaucoup plus.""")

code("""part = pd.crosstab(train["user_id"], train["intensity"], normalize="index")
print(part.describe().round(3))
rang = train.groupby("user_id")["speed_kmh"].rank(pct=True)
print("Spearman vitesse brute :", round(pd.Series(y).corr(train["speed_kmh"], method="spearman"), 3))
print("Spearman rang de vitesse intra-personne :", round(pd.Series(y).corr(rang, method="spearman"), 3))""")

md("""**Test des quotas.** Si l'étiquette est un tertile exact sur train + test, alors `total/3 − nombre train de la classe` est le nombre de séances test de cette classe : jamais négatif, et la somme vaut le nombre de séances test. On le vérifie par (personne, sport).""")

code("""g_tr = (train["user_id"].astype(str) + "_" + train["sport"]).values
g_te = (test["user_id"].astype(str) + "_" + test["sport"]).values
ct = pd.crosstab(g_tr, train["intensity"])[CLASSES]
nb_test = pd.Series(g_te).value_counts().reindex(ct.index).fillna(0)
quota = ct.rsub((ct.sum(axis=1) + nb_test) / 3, axis=0)
print("quota minimum :", quota.min().round(2).to_dict())
print("somme des quotas == nb séances test :", np.allclose(quota.sum(axis=1), nb_test))""")

md("""## 2. Baselines : 3 familles sur variables brutes

Validation : `StratifiedKFold(5, shuffle=True, random_state=0)`. Le test est un tirage aléatoire de séances des mêmes personnes, donc ce découpage reproduit la situation réelle.""")

code("""NUM_BRUTES = ["year", "month", "weekday", "start_hour_utc", "duration_min", "distance_km",
              "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m",
              "altitude_gain_m", "altitude_loss_m", "n_points"]

def nettoyer(df):
    df = df.copy()
    df.loc[df["altitude_mean_m"] < -500, "altitude_mean_m"] = np.nan
    df["altitude_mean_m"] = df["altitude_mean_m"].fillna(df["altitude_mean_m"].median())
    return df

tout = nettoyer(pd.concat([train.drop(columns="intensity"), test], ignore_index=True))
X_brut = pd.concat([tout[NUM_BRUTES], pd.get_dummies(tout["sport"], prefix="sport", dtype=float),
                    pd.get_dummies(tout["gender"], prefix="gender", dtype=float)], axis=1).iloc[:n]
plis = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X_brut, y))

def score_cv(fabrique, X):
    pred = np.zeros(n, dtype=int)
    for a, b in plis:
        pred[b] = fabrique().fit(X.iloc[a], y[a]).predict(X.iloc[b])
    return f1_score(y, pred, average="macro")

baselines = {
    "Régression logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0),
}
for nom, fab in baselines.items():
    print(f"{nom:<25} macro-F1 = {score_cv(fab, X_brut):.4f}")""")

md("""## 3. Features relatives à la personne (sans étiquette)

- dérivées : temps continu `t`, pente, ratio vitesse max / moyenne, coefficient de variation de vitesse, D+ par minute ;
- pour chaque variable, par personne puis par (personne, sport) : rang percentile, z-score, écart à la médiane ;
- contexte de la personne (moyennes, écarts-types, volume) ;
- **features locales** : écart à la moyenne de la même (personne, sport) **à ±2 mois** (noyau gaussien), car la forme évolue.

Calculées sur train + test : elles n'utilisent aucune étiquette, donc pas de fuite.""")

code(corps("features.py") + "\n\n" + corps("features_locales.py") + """

X = pd.concat([construire(tout), relatives_locales(tout)], axis=1)
tout["vz"] = X["speed_kmh_z_us"]
print("features :", X.shape[1])""")

md("""## 4. Encodages d'étiquettes (calculés hors fold)

Ces features utilisent les **étiquettes connues** de la même personne :
- **forme du moment** : part de easy / hard des séances de la personne, pondérée par la proximité dans le temps (σ = 0,5 / 1,5 / 4 mois) ;
- **voisins temps × vitesse** : idem, pondéré à la fois par l'écart de date et l'écart de vitesse relative ;
- **modèle personnel** : vote des 15 séances connues les plus ressemblantes de la même (personne, sport) sur 10 dimensions, et distances aux profils moyens easy / moderate / hard.

**Anti-fuite :** une séance n'est jamais encodée avec sa propre étiquette. Les séances d'entraînement d'un fold sont encodées par un second découpage interne en 5.""")

code(corps("encodage.py") + "\n\n" + corps("encodage_perso.py"))

md("""## 5. Affectation sous quotas (algorithme hongrois)

Pour chaque (personne, sport), on connaît le nombre de séances de chaque classe parmi les séances inconnues. On répartit ces séances en minimisant le coût total, sous la contrainte des quotas :

coût(séance, classe k) = −log p(k) + λ·(k − note de dureté)², avec λ = 1

où p vient des classifieurs et la note du régresseur.""")

code(corps("quotas.py"))

md("""## 6. Validation croisée du pipeline complet

Pour chaque fold, les séances « inconnues » sont le fold de validation **et** le test. Les quotas sont calculés avec les seules étiquettes des folds d'entraînement, et on ne score que la validation. C'est la situation réelle, avec plus d'inconnues : l'estimation est légèrement pessimiste.""")

code("""total = pd.Series(np.concatenate([g_tr, g_te])).value_counts().to_dict()
brut_tr = tout.iloc[:n].reset_index(drop=True)
brut_te = tout.iloc[n:].reset_index(drop=True)
Xtr_all = X.iloc[:n].reset_index(drop=True)
Xte_all = X.iloc[n:].reset_index(drop=True)
cles = ["user_id", "sport"]
LAMBDA = 1.0

def comptes_connus(idx):
    c = pd.crosstab(g_tr[idx], y[idx]).reindex(columns=[0, 1, 2], fill_value=0)
    return {g: c.loc[g].values.astype(float) if g in c.index else np.zeros(3) for g in total}

def cout_affectation(p, s):
    return -np.log(np.clip(p, 1e-9, 1)) + LAMBDA * (np.arange(3)[None, :] - s[:, None]) ** 2

oof = np.zeros(n, dtype=int)
p_test, s_test = np.zeros((nt, 3)), np.zeros(nt)
for f, (a, b) in enumerate(plis):
    te_a = encodage_oof(brut_tr.iloc[a], y[a], avec_vitesse=True).reset_index(drop=True)
    te_b = tous_encodages(brut_tr.iloc[a], y[a], brut_tr.iloc[b], avec_vitesse=True).reset_index(drop=True)
    te_t = tous_encodages(brut_tr.iloc[a], y[a], brut_te, avec_vitesse=True).reset_index(drop=True)
    Xa = pd.concat([Xtr_all.iloc[a].reset_index(drop=True), te_a], axis=1).fillna(-1)
    Xb = pd.concat([Xtr_all.iloc[b].reset_index(drop=True), te_b], axis=1).fillna(-1)
    Xt = pd.concat([Xte_all, te_t], axis=1).fillna(-1)
    ref = pd.concat([Xa, train.iloc[a][cles].reset_index(drop=True)], axis=1)
    pa = encoder_oof(ref, y[a]).reset_index(drop=True)
    pb = encoder(ref, y[a], pd.concat([Xb, train.iloc[b][cles].reset_index(drop=True)], axis=1)).reset_index(drop=True)
    pt = encoder(ref, y[a], pd.concat([Xt, test[cles].reset_index(drop=True)], axis=1)).reset_index(drop=True)
    Xa_p, Xb_p, Xt_p = [pd.concat([u, v], axis=1).fillna(-1) for u, v in [(Xa, pa), (Xb, pb), (Xt, pt)]]

    hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0).fit(Xa_p, y[a])
    etc = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3,
                               n_jobs=-1, random_state=0).fit(Xa, y[a])
    etr = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.3,
                              n_jobs=-1, random_state=0).fit(Xa, y[a])

    pv = (hgb.predict_proba(Xb_p) + etc.predict_proba(Xb)) / 2
    pt_f = (hgb.predict_proba(Xt_p) + etc.predict_proba(Xt)) / 2
    sv, st_f = etr.predict(Xb), etr.predict(Xt)
    p_test += pt_f / 5
    s_test += st_f / 5

    pred = affecter_cout(cout_affectation(np.vstack([pv, pt_f]), np.concatenate([sv, st_f])),
                         np.concatenate([g_tr[b], g_te]), total, comptes_connus(a))
    oof[b] = pred[:len(b)]
    print(f"fold {f} : macro-F1 = {f1_score(y[b], oof[b], average='macro'):.4f}")

scores = [f1_score(y[b], oof[b], average="macro") for a, b in plis]
print(f"moyenne = {np.mean(scores):.4f} (écart-type {np.std(scores):.4f})")""")

md("## 7. Analyse d'erreur")

code("""cm = pd.DataFrame(confusion_matrix(y, oof), index=[f"vrai_{c}" for c in CLASSES],
                  columns=[f"pred_{c}" for c in CLASSES])
print(cm)
print("F1 par classe :", dict(zip(CLASSES, np.round(f1_score(y, oof, average=None), 3))))""")

md("""Les erreurs se font presque toujours entre **classes voisines**. `moderate` est la classe la plus confondue, parce qu'elle est coincée entre les deux autres. Les confusions easy ↔ hard sont rares. La cible est bien une échelle ordonnée : le modèle ordonne bien les séances, la difficulté est de placer les deux coupures.""")

md("""## 8. Soumission

Prédictions test = moyenne des 5 modèles de fold. Affectation sous les quotas exacts calculés avec **toutes** les étiquettes du train.""")

code("""pred_test = affecter_cout(cout_affectation(p_test, s_test), g_te, total, comptes_connus(np.arange(n)))
submission = pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred_test]})
submission.to_csv("submission.csv", index=False)
print(submission["intensity"].value_counts(normalize=True).round(3))
submission.head()""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebook_equipe.ipynb")
print("ok", len(C), "cellules")

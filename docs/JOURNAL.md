# Journal de bord — Hackathon ML 2026 « Will this workout be hard for this person? »

Objectif : maximiser le macro-F1 privé (3 classes : easy / moderate / hard), modèle scikit-learn non neuronal, explicable sur papier.

---

## Étape 0 — Lecture du sujet et EDA (14:48)

**Ce qu'on a fait :** lecture des 22 slides, du starter notebook, du template LOG, exploration descriptive de train/test.

**Constats :**
- 22 728 séances train, 7 669 test, 183 utilisateurs, tous présents dans train ET test (split aléatoire par séance).
- Classes à ~33 % chacune, et ~33 % **à l'intérieur de chaque utilisateur** (écart-type ≈ 1,5 pt sur les users ≥ 100 séances).
- Corrélation de Spearman avec la cible : vitesse brute 0,12 → rang de vitesse **dans l'historique de l'utilisateur** 0,35.
- Effet temporel fort : 2008-09 ≈ 60 % de hard, 2015 ≈ 30 %. Persiste en rang intra-user (−0,12).
- `workout_id` sans lien avec la cible (pas de fuite).
- 3 lignes avec `altitude_mean_m = -9999` (capteur absent).

**Conclusion / hypothèse de travail :** `intensity` = tertile, calculé par utilisateur, d'une mesure de capteur cachée (probablement la fréquence cardiaque). Ce qui compte n'est donc pas « la séance est rapide » mais « la séance est rapide **pour cette personne** ». → Les features doivent être **relatives à l'utilisateur**.

**Protocole de validation retenu :** validation croisée stratifiée 5 folds sur le train (le test est un tirage aléatoire de séances des mêmes utilisateurs, donc un split aléatoire reproduit fidèlement la situation). Les statistiques par utilisateur sont calculées sur train + test concaténés : elles n'utilisent aucune étiquette, donc pas de fuite, et elles sont plus stables.

---

## Étape 1 — Baselines sur variables brutes (15:08)

**Ce qu'on a fait :** 13 variables numériques brutes + one-hot `sport` et `gender`, nettoyage des altitudes −9999 (remplacées par la médiane). Trois familles de modèles, CV stratifiée 5 folds, macro-F1 out-of-fold.

| Modèle | macro-F1 CV |
| --- | --- |
| Régression logistique (standardisée) | 0,423 |
| Random forest (300 arbres, min_samples_leaf=3) | **0,551** |
| HistGradientBoosting (défaut) | 0,522 |

**Conclusions :**
- La régression logistique plafonne : la relation est non linéaire et dépend de la personne. Une même vitesse est « hard » pour l'un et « easy » pour l'autre, un modèle linéaire global ne peut pas le représenter.
- La random forest bat HGB, ce qui est inhabituel. Explication probable : avec des arbres profonds, la forêt **reconnaît implicitement l'utilisateur** via des combinaisons de variables (même sport, même altitude moyenne, même plage de vitesses…) et compare la séance à ses voisines. Elle reconstruit à la main ce que des features relatives donneraient directement.
- Ça confirme l'hypothèse de l'étape 0 : le signal est **relatif à l'utilisateur**.

---

## Étape 2 — Features relatives à l'utilisateur (15:12)

**Ce qu'on a fait (109 features) :**
- Variables dérivées : `t` = année + mois/12 (temps continu), `pente` = (D+ + D−)/distance, `ratio_max` = vitesse max / vitesse moyenne, `cv_vitesse` = écart-type / moyenne de vitesse, `denivele_min` = D+ par minute.
- Pour chaque variable, par **utilisateur** puis par **(utilisateur, sport)** : rang percentile, z-score, écart à la médiane. Ces statistiques sont calculées sur train + test (sans étiquette, donc sans fuite).
- Contexte utilisateur : moyenne et écart-type de ses vitesses, durées, distances, D+, période d'activité, nombre de séances.

| Modèle | Brut | Relatif | Gain |
| --- | --- | --- | --- |
| Régression logistique | 0,423 | 0,494 | +0,07 |
| Random forest | 0,551 | 0,585 | +0,03 |
| ExtraTrees | — | **0,590** | — |
| HGB | 0,522 | 0,581 | +0,06 |

**Conclusions :**
- Toutes les familles progressent, et ce sont les modèles qui ne pouvaient pas « deviner » l'utilisateur seuls (LogReg, HGB) qui gagnent le plus. Ça confirme que la normalisation par personne est le bon levier.
- La CV est très stable (écart-type entre folds ≈ 0,004), donc un écart de 0,01 entre deux versions est significatif.

**Soumission 1 (15:14) :** ExtraTrees 500 arbres sur les 109 features → `sub01_extratrees_relatif.csv` (CV 0,590). Prédictions : 36 % easy / 34 % moderate / 30 % hard, proche de l'équilibre attendu.

---

## Étape 3 — Dérive temporelle : encodage des étiquettes voisines dans le temps (15:15)

**Diagnostic préalable (hors fold, 5 folds) :** pour chaque séance, la moyenne des étiquettes des *autres* séances du train de la même personne :

| Groupe | Corrélation de Spearman avec la cible |
| --- | --- |
| utilisateur seul | −0,08 (normal : chaque personne est équilibrée à 1/3) |
| utilisateur × année | 0,19 |
| utilisateur × (année, mois) | **0,26** |
| utilisateur × année × jour de semaine | 0,22 |

**Interprétation :** une personne traverse des **périodes** où la plupart de ses séances sont « hard » et d'autres où elles sont « easy ». C'est cohérent avec une mesure cachée de type fréquence cardiaque : la forme évolue, il y a des reprises après une coupure, voire un changement de capteur. Ce signal ne se voit pas dans la vitesse de la séance elle-même, il est donc **complémentaire** des features relatives.

**Ce qu'on a fait :** pour chaque séance, une moyenne des étiquettes (part de easy, part de hard, différence) des séances du train de la même personne, **pondérée par un noyau gaussien sur l'écart de temps** (σ = 0,5, 1,5 et 4 mois), lissée vers la répartition globale de la personne. Ça gère les mois avec peu de séances. On ajoute aussi le nombre de séances le même mois.

**Garde-fou anti-fuite :** ce sont des features construites à partir des étiquettes, donc :
- en validation, les séances du fold de validation sont encodées uniquement avec les étiquettes des folds d'entraînement ;
- les séances d'entraînement sont encodées par un **second découpage en 5 folds interne** (jamais avec leur propre étiquette), sinon le modèle apprendrait à lire sa propre réponse ;
- pour le test final : train encodé hors fold, test encodé avec tout le train. On n'utilise que les étiquettes fournies, aucune donnée externe.

**Résultat partiel (3 folds sur 5, HGB) :** relatif seul 0,5765 / 0,5803 / 0,5868 → avec l'encodage temporel 0,5902 / 0,5927 / 0,5895. **Gain d'environ +0,01**, régulier sur chaque fold. Run arrêté au fold 3 pour libérer le CPU, la tendance étant claire.

---

## Étape 4 — Découverte clé : quotas exacts par utilisateur (15:17)

**Question :** si `intensity` est un tertile calculé par personne sur **toutes** ses séances (train + test), alors pour chaque personne :
`nombre de séances test de la classe c = (séances train + test) / 3 − séances train de la classe c`.

**Vérification :** on calcule cette quantité pour les 183 utilisateurs × 3 classes.
- Elle n'est **jamais négative** au-delà de l'arrondi d'un tiers (min : easy 0, moderate −0,33, hard −0,67).
- La somme sur les 3 classes retombe **exactement** sur le nombre de séances test de la personne.
- Or dans le train seul, la part de chaque classe varie de 17 % à 56 % selon la personne. Les petits utilisateurs ont des comptes train très déséquilibrés, et c'est le test qui les complète exactement.

**Conclusion :** on connaît, pour chaque personne, **combien** de ses séances test sont easy / moderate / hard. Il ne reste qu'à décider **lesquelles**.

**Méthode :** pour chaque utilisateur, on résout un problème d'affectation optimale (algorithme hongrois, `scipy.optimize.linear_sum_assignment`). Chaque séance test reçoit une classe en maximisant la somme des log-probabilités du modèle, sous la contrainte que chaque classe reçoive exactement son quota.

Intuition : les probabilités du modèle servent à **ordonner** les séances de la personne de la plus facile à la plus dure, et les quotas disent où placer les coupures.

**Validation honnête :** en CV, l'ensemble « inconnu » d'un fold = fold de validation + test. Les quotas de chaque personne sont calculés à partir des étiquettes des seuls folds d'entraînement. On affecte conjointement validation + test, puis on score seulement la validation. C'est exactement la situation de la soumission finale, avec plus d'inconnues, donc l'estimation est même légèrement pessimiste.

**Légitimité :** on n'utilise que les étiquettes fournies du train et la composition du test (quelles personnes y figurent), connue de tous. Aucune donnée externe, aucune exploitation du leaderboard. C'est la conséquence directe de « mesuré séparément pour chaque personne, trois classes à peu près équiprobables » (slide 6) : c'est la nature de la cible qu'on exploite.

**Résultats de validation (HGB, quotas calculés sans les étiquettes de validation) :**

| Fold | argmax | avec quotas |
| --- | --- | --- |
| 0 | 0,5902 | 0,5938 |
| 1 | 0,5927 | 0,5959 |
| 2 | 0,5895 | 0,6005 |
| 3 | 0,5952 | 0,6018 |
| 4 | 0,5854 | 0,5923 |
| **OOF global** | **0,5906** | **0,5969** |

Gain de **+0,006** en validation, positif sur les 5 folds. Ce gain est **sous-estimé** par rapport à la soumission réelle : en CV, l'ensemble inconnu fait 40 % des séances de chaque personne (validation + test), contre 25 % pour la vraie soumission (test seul). Avec plus d'étiquettes connues, les quotas sont plus contraignants.

Contre-vérification : si on n'a **aucune** étiquette connue (quotas = 1/3 partout), l'affectation n'apporte rien (0,5809 → 0,5802). Le gain vient bien de l'information « il reste X séances hard pour cette personne ».

---

## Étape 5 — Features locales dans le temps et voisinage temps × vitesse (15:20)

**Idée :** la forme d'une personne évolue, donc « rapide pour elle » doit se juger **par rapport à ses séances de la même période**, pas à toute sa carrière.
- **Features locales (sans étiquette) :** pour chaque séance, écart et z-score de la vitesse, vitesse max, durée, distance et D+ par rapport à la moyenne de la même personne et du même sport, **pondérée par un noyau gaussien de 2 mois**.
- **Encodage temps × vitesse (avec étiquettes, hors fold) :** moyenne des étiquettes des séances de la même personne et du même sport, proches à la fois **dans le temps** (σ = 3 mois) **et en vitesse relative** (σ = 0,5 écart-type). C'est un k plus proches voisins lissé, à l'intérieur de chaque personne.

| Fold | HGB + temporel | + locales + temps × vitesse |
| --- | --- | --- |
| 0 | 0,5902 | 0,5955 |
| 1 | 0,5927 | 0,5979 |

**Gain d'environ +0,005**, régulier. Run arrêté après 2 folds pour lancer le pipeline final.

**Ensemble de modèles (sur les OOF de l'étape 2) :** HGB 0,581, ExtraTrees 0,590, moyenne des probabilités HGB + ExtraTrees **0,597**. Les deux familles font des erreurs différentes (boosting séquentiel vs arbres très aléatoires moyennés), d'où le gain. Ajouter la LogReg dégrade (0,588) : elle est trop faible.

---

## Étape 6 — Pipeline final (15:22)

Features relatives + locales + encodages temporels → HGB et ExtraTrees → moyenne des probabilités → affectation sous quotas par utilisateur. CV 5 folds, et les prédictions test sont la moyenne des 5 modèles de fold (bagging).

**Matrice de confusion (HGB + encodage temporel + quotas, OOF) :**

| | prédit easy | prédit moderate | prédit hard |
| --- | --- | --- | --- |
| vrai easy | 4 884 | 1 966 | 679 |
| vrai moderate | 1 943 | 3 723 | 1 937 |
| vrai hard | 768 | 1 870 | 4 958 |

**Lecture :** les erreurs se concentrent sur les **classes adjacentes**. `moderate` est la classe la plus confondue (F1 le plus bas, ~49 % de rappel) parce qu'elle est coincée entre les deux autres. Les confusions extrêmes easy ↔ hard sont rares (~9 %). Ça confirme la nature **ordinale** de la cible (tertiles d'une mesure continue) : le modèle ordonne bien les séances, et la difficulté est de placer les deux coupures.

**Résultats (CV 5 folds, OOF) :**

| Fold | HGB | ExtraTrees | Moyenne | + quotas |
| --- | --- | --- | --- | --- |
| 0 | 0,5955 | 0,6060 | 0,6054 | 0,6129 |
| 1 | 0,5979 | 0,6028 | 0,6091 | 0,6151 |
| 2 | 0,5983 | 0,6039 | 0,6125 | 0,6116 |
| 3 | 0,5920 | 0,5979 | 0,5979 | 0,6133 |
| 4 | 0,5894 | 0,5961 | 0,6015 | 0,6028 |
| **OOF** | **0,5946** | **0,6013** | **0,6053** | **0,6111** |

**Conclusions :**
- Chaque brique apporte un gain : features relatives (+0,04 sur HGB), encodage temporel (+0,01), features locales et temps × vitesse (+0,005), ensemble (+0,005 à +0,01), quotas (+0,006 en CV, probablement plus en réel).
- Prédictions test : 34,5 % easy / 32,9 % moderate / 32,6 % hard, conforme aux quotas.

**Soumission 2 (15:27) :** `sub_final.csv`, CV 0,611.

**Progression de la CV :** 0,423 (LogReg brute) → 0,551 (RF brute) → 0,590 (relatif) → 0,597 (+ temporel + quotas, HGB seul) → **0,611** (pipeline complet).

---

## Étape 7 — Formulation ordinale ? (15:45)

**Hypothèse :** la cible est ordonnée (tertiles d'une mesure continue). Un **régresseur** qui prédit un score de dureté continu, puis un tri des séances de chaque personne découpé selon les quotas, pourrait mieux ordonner qu'un classifieur à 3 classes indépendantes.

**Mise en cache :** les matrices de features (dont les encodages hors fold) sont sauvegardées par fold, pour tester chaque variante en quelques minutes.

| Variante (mêmes features, mêmes folds, quotas) | macro-F1 OOF |
| --- | --- |
| Régresseur HGB, tri par score | 0,5945 |
| Classifieur HGB, tri par score espéré (p_moderate + 2·p_hard) | 0,6000 |
| Régresseur + classifieur, tri | 0,6011 |
| **Classifieur HGB, affectation hongroise sur les log-probabilités** | **0,6050** |

**Conclusion : hypothèse rejetée.** Résumer la séance en un seul score de dureté perd de l'information. Les 3 probabilités disent aussi **à quel point le modèle hésite** (par exemple « sûrement pas hard, mais easy ou moderate ? »), et l'affectation hongroise s'en sert pour placer au mieux chaque séance dans les quotas. On garde **classifieur + affectation hongroise**.

---

## Étape 8 — Combinaison classifieurs + régresseur, quotas par (personne, sport) (15:40)

**Ensemble avec affectation hongroise (quotas, OOF) :**

| Combinaison | tri par score | hongrois |
| --- | --- | --- |
| ExtraTrees classifieur | 0,6021 | 0,6089 |
| HGB + ExtraTrees classifieurs | 0,6056 | **0,6112** |
| ExtraTrees régresseur seul | 0,6056 | — |

**Coût mixte :** coût d'affecter une séance à la classe k = −log p(k) (moyenne HGB + ET classifieurs) + λ·(k − score du régresseur ET)².

| λ | 0 | 0,5 | 1 | 2 | 4 |
| --- | --- | --- | --- | --- | --- |
| macro-F1 | 0,6112 | 0,6128 | **0,6138** | 0,6137 | 0,6110 |

Le régresseur apporte une information d'**ordre** que les classifieurs exploitent mal. λ = 1 est retenu (le plateau 1-2 est stable).

**Découverte : les tertiles sont par (personne, sport), pas par personne.** Les 10 utilisateurs qui ont deux sports sont équilibrés à 1/3 **dans chaque sport**, même avec vélo + marche (utilisateur 1087 : marche 18/12/14, vélo 21/27/29). Avec des tertiles par personne, la marche serait presque toujours « easy ». Exemple de l'utilisateur 1026, course : train 10/8/4 avec 8 séances test, donc les quotas test sont 0 easy, 2 moderate, 6 hard. Les quotas par personne (tous sports confondus) étaient faux pour ces 10 utilisateurs.
→ Quotas par (personne, sport) : **0,6142** (λ = 1). Gain faible en CV car seuls 10 utilisateurs sont concernés, mais c'est la bonne structure.

---

## Étape 9 — « Modèle personnel » : voisins connus de la même personne (15:48)

**Idée :** chaque personne a sa propre façon de « souffrir » : chez l'une c'est la vitesse qui fait monter le cardio, chez une autre la durée ou le dénivelé. Pour chaque séance, on regarde à quelles séances **connues** de la même personne et du même sport elle ressemble, sur 10 dimensions standardisées : vitesse, vitesse max, durée, distance, D+, pente, ratio max/moyenne, date, vitesse et vitesse max locales.

**Features créées (hors fold, comme les autres encodages d'étiquettes) :**
- vote pondéré des 15 plus proches voisins → part easy, part hard, différence ;
- distance de la séance au profil moyen des séances easy, moderate et hard de la personne ;
- projection sur l'axe « profil hard − profil easy » de la personne, et rang de la séance sur cet axe.

**Pouvoir prédictif individuel (Spearman avec la cible) :** `pk_diff` (vote des voisins) **0,556**, la meilleure feature du projet, devant `pp_score` 0,490, `kv_diff` 0,489 et le rang de vitesse 0,353.

**Résultats (quotas par (personne, sport), affectation hongroise) :**

| Ensemble | macro-F1 OOF |
| --- | --- |
| HGB + ET (étape 8) | 0,6112 |
| HGB_perso + ET | 0,6160 |
| **HGB_perso + ET + régresseur ET (λ = 1)** | **0,6172** |
| HGB_perso + ET_perso + régresseur ET (λ = 1) | 0,6170 |
| HGB_perso + ET + régresseur ET_perso | 0,6155 |

**Conclusions :**
- Le gain est réel mais **plus faible que ne le laissait espérer la corrélation** (+0,005). Une grande partie de l'information était déjà captée par les features relatives et l'encodage temps × vitesse : le modèle savait déjà, indirectement, à quelles séances une séance ressemble.
- Les features personnelles aident HGB mais pas les ExtraTrees. Explication probable : avec `max_features = 0.3`, les arbres très aléatoires les tirent rarement, et elles ajoutent du bruit (encodage hors fold dans un fold, donc calculé sur 64 % des données).
- On arrive sur un **plateau autour de 0,617** : plusieurs combinaisons différentes donnent 0,616-0,617.

**Soumission 4 (15:46) :** `sub04_perso_ensemble_quotas.csv`, HGB_perso + ET classifieurs + régresseur ET, λ = 1, quotas par (personne, sport). CV **0,617**. Soumise pour le Gate 1.

---

## Étape 10 — Moyenne de plusieurs graines (16:00)

**Hypothèse :** moyenner les modèles entraînés avec des graines aléatoires différentes réduit la variance des prédictions.
**Résultat :** 2 graines → 0,6170 contre 0,6172 avec une seule. **Aucun gain** : les modèles sont déjà stables (400 arbres, ensemble de 3 modèles). Le run de la 3e graine a été arrêté.

**Notebook d'équipe :** `notebook_equipe.ipynb` reproduit `sub04` de bout en bout en un seul fichier (EDA → test des quotas → baselines 3 familles → features → encodages hors fold → CV du pipeline complet → matrice de confusion → soumission). Exécution complète en cours pour vérifier la reproductibilité.

**Score public de sub04 : 0,6345** (CV 0,617). Plus haut que la validation, comme attendu : sur Kaggle, seul le test (25 % des séances de chaque personne) est inconnu, contre 40 % en validation. Les quotas sont donc plus précis. L'écart est **cohérent et dans le bon sens**, la validation reste fiable pour choisir entre versions.

---

## Étape 11 — Routines et même parcours (16:25)

**Méthode de sélection :** avant de coder, chaque idée est testée sur les **erreurs** du modèle actuel (résidu = vraie classe − classe espérée prédite, hors fold). Une feature corrélée au résidu apporte de l'information que le modèle n'a pas encore.

| Piste | Spearman avec la cible | Spearman avec le résidu |
| --- | --- | --- |
| Routine jour (personne, sport, jour de semaine) | 0,13 | 0,07 |
| Routine heure (personne, sport, tranche horaire) | 0,20 | 0,11 |
| Routine jour + heure | 0,21 | **0,11** |
| Saison (personne, sport, mois calendaire) | 0,15 | 0,06 |
| Rang de vitesse sur le même parcours | 0,26 | 0,07 |
| Étiquettes des séances sur le même parcours | 0,31 | **0,10** |

**Interprétation :**
- **Routines :** les sportifs ont des créneaux types (fractionné du mardi soir, sortie longue du dimanche matin).
- **Même parcours :** une boucle refaite (même distance, altitude moyenne, D+ et D−) permet de comparer la vitesse à conditions égales.

**Résultats (hors fold, quotas) :**

| Modèle | Avant | Avec routines + parcours |
| --- | --- | --- |
| HGB | 0,6058 | 0,6087 |
| ExtraTrees classifieur | 0,6086 | 0,6133 |
| ExtraTrees régresseur (tri) | 0,6057 | 0,6087 |
| **Ensemble** (HGB + ET + ET_routine, régresseur ET, λ = 0,5) | 0,6172 | **0,6194** |

**Soumission 5 (16:31) :** `sub05_routines_parcours.csv`, CV **0,619**, **public 0,64** (sub04 : 0,6345).

---

## Étape 12 — Z-scores robustes (16:32)

**Hypothèse :** un bug GPS isolé (par exemple une course à 76 km/h) gonfle l'écart-type de la personne et écrase les z-scores de toutes ses autres séances. On teste la médiane et l'écart interquartile à la place de la moyenne et de l'écart-type.
**Résultat :** HGB 0,6087 → 0,6096, ensemble 0,6194 → 0,6181. **Pas de gain, hypothèse rejetée.** Les rangs, robustes par nature, portaient déjà l'information, et les ExtraTrees découpent sur des seuils, donc ils sont peu sensibles à l'échelle des z-scores.

---

## Étape 13 — Stacking et réglage de l'ExtraTrees (16:40)

**Stacking :** au lieu de moyenner les modèles avec des poids fixés à la main, une régression logistique (le « méta-modèle ») apprend à combiner les log-probabilités des classifieurs et la note du régresseur (et son carré). Elle est entraînée sur les prédictions **hors fold**. Pour l'évaluation, le méta-modèle de chaque fold n'est appris que sur les autres folds. Les quotas sont ensuite appliqués par l'algorithme hongrois.

**Réglage de l'ExtraTrees (avec routines + parcours, hongrois) :**

| Variante | macro-F1 |
| --- | --- |
| min_samples_leaf = 3, max_features = 0,3 (référence) | 0,6133 |
| **min_samples_leaf = 1** | **0,6144** |
| min_samples_leaf = 6 | 0,6079 |
| max_features = 0,15 | 0,6146 |

Les arbres profonds sont meilleurs : la cible dépend de **combinaisons fines** (personne × période × vitesse relative).

**Résultats du stacking :**

| Combinaison | macro-F1 OOF | folds |
| --- | --- | --- |
| sub05 (moyenne fixe, λ = 0,5) | 0,6194 | 0,619 / 0,621 / 0,622 / 0,622 / 0,613 |
| Stacking HGB_pr + ET + ET_r, régresseur ET | 0,6218 | 0,623 / 0,628 / 0,622 / 0,628 / 0,608 |
| **Stacking HGB_pr + ET + ET_r + ET_r_f1, régresseur ET** | **0,6239** | 0,623 / 0,631 / 0,628 / 0,626 / 0,612 |
| + ET_r_m15 ou + ET_r_f6 | 0,6236-0,6239 | — |
| Stacking de 8 modèles (avec HGB de base et régresseurs HGB) | 0,6210 | — |

Plus de modèles ne veut pas dire mieux : le méta-modèle sur-apprend quand on empile des modèles trop similaires.

**Soumission 6 :** `sub06_stacking_folds.csv`, CV **0,624**.

## Étape 14 — Réentraînement sur 100 % du train (16:45)

**Constat :** jusqu'ici, la prédiction test était la moyenne des 5 modèles de fold. Chacun calculait les encodages du test (forme du moment, voisins, routines, parcours) avec **80 % des étiquettes du train** seulement. On se privait donc de 20 % des séances connues de chaque personne, alors que tous nos meilleurs signaux viennent de ces voisins.

**Correction :** réentraîner chaque modèle sur 100 % du train (lignes d'entraînement encodées hors fold), et encoder le test avec **toutes** les étiquettes du train. C'est la même structure que celle validée en CV (entraînement encodé hors fold, prédiction encodée avec toutes les étiquettes disponibles). Le gain n'est pas mesurable en CV mais il est logiquement positif.

## Étape 15 — Avocat du diable : la convention d'arrondi des tertiles (16:47)

**Question :** quand une personne a 31 séances dans un sport, 31/3 ne tombe pas juste, et une classe reçoit 11 séances au lieu de 10. Laquelle ? Notre code donnait l'extra à **easy** par défaut (premier indice), sans l'avoir vérifié.

**Vérification :** pour chaque convention possible, on compte les groupes (personne, sport) où elle est **impossible**, c'est-à-dire où le train contient déjà plus de séances d'une classe que ce que la convention autorise au total.

| Convention | Groupes impossibles (sur 193) |
| --- | --- |
| Extra vers easy (notre code) | 1 |
| **Extra vers hard** (reste 1 → hard ; reste 2 → moderate + hard) | **0** |
| Extra vers moderate | 1 |
| Autres combinaisons | 1-2 |
| `np.digitize` sur le rang percentile (déséquilibre aussi les multiples de 3) | 2 |

**Conclusion :** notre convention était **prouvée fausse** pour au moins un groupe. « Extra vers hard » est la seule compatible avec toutes les données. C'est le comportement d'un découpage où les séances à la frontière basculent dans la classe supérieure.

**Effet en validation :** 0,6239 → 0,6234, non mesurable. En validation, l'ensemble inconnu (validation + test) est grand et l'arrondi d'une séance s'y dilue. Sur Kaggle, où seul le test est inconnu, la correction porte sur environ 120 groupes, à une séance chacun. On **garde la convention vérifiée**, puisque l'autre est prouvée impossible.

**Autres points examinés :**
- *Sur-apprentissage de la validation à force de tester des variantes ?* Une trentaine de configurations ont été testées sur les mêmes folds. Les gains de +0,001 à +0,002 sont donc à prendre avec prudence. Les gains structurels (features relatives, quotas, stacking ≈ +0,005) sont réguliers sur les folds et confirmés par le public (sub04 0,6345 → sub05 0,64).
- *Méta-modèle appris sur des modèles de fold puis appliqué à des modèles réentraînés sur tout le train ?* Le calibrage des probabilités peut légèrement différer. C'est un risque faible : les mêmes hyperparamètres sur 25 % de données en plus produisent des probabilités du même ordre.
- *Variantes ExtraTrees (max_features = 0,5) et random forest :* arrêtées pour libérer le CPU. Vu le plateau, le gain attendu était faible.

**Scores publics :** sub06 (stacking, modèles de fold) **0,64963** ; sub07 (réentraîné sur 100 % du train) **0,64553**.

**Conclusion : le réentraînement complet est rejeté.** Le risque identifié s'est réalisé. Le méta-modèle a appris ses poids sur les probabilités des modèles de fold, or les modèles complets sont **plus confiants** (probabilité max moyenne 0,628 contre 0,609), parce que leurs encodages s'appuient sur plus de voisins. Le méta-modèle, calibré pour des modèles moins sûrs d'eux, sur-pondère alors ces probabilités.

Les deux soumissions sont notées sur la même moitié publique et diffèrent sur 9,5 % des séances, donc l'écart de 0,004 est fiable. **Leçon :** ne déployer que ce qui a la même structure que ce qui a été validé. La version validée en CV (sub06) gagne.

**Progression publique :** sub04 0,6345 → sub05 0,640 → **sub06 0,6496**.

---

## Étape 16 — Le « créneau exact » (16:55)

**Constat :** deux séances de la même personne au même créneau précis (même année, même mois, même jour de semaine, même heure) partagent l'étiquette dans **51 %** des cas (hasard : 33 %). C'est probablement le même programme d'une semaine à l'autre, ou une même sortie enregistrée en deux morceaux. Corrélation avec le résidu du modèle : +0,08 en moyenne, **+0,17** quand le créneau existe (32 % des séances).

**Résultats :**

| Modèle | Sans créneau | Avec créneau |
| --- | --- | --- |
| HGB | 0,6085 | **0,6142** |
| ExtraTrees (min_samples_leaf = 1) | 0,6129 | 0,6091 |
| **Stacking** | 0,6234 | 0,6235 |

**Conclusion : rejeté pour l'ensemble.** Le signal est réel (HGB +0,006), mais il était déjà capté par le stacking via les autres encodages temporels, et il dégrade l'ExtraTrees. Explication probable : les compteurs du créneau n'ont pas la même distribution à l'entraînement (encodage interne sur 64 % des données) et en prédiction (80 %), et les arbres profonds découpent sur ces compteurs.

**Grand HGB (63 feuilles, 1 000 itérations) :** inexploitable, une erreur de nommage a fait sauvegarder des classes au lieu de probabilités. Abandonné faute de temps.

**Gate 2 :** matrice de confusion de sub06 ajoutée au LOG (macro-F1 CV 0,623 ; F1 easy 0,673 · moderate 0,514 · hard 0,683 ; confusions easy ↔ hard : 5 % des séances).

---

## Étape 17 — Revue externe, vitesse attendue et comparaison par paires (17:00-17:14)

**Revue externe** (autre conversation, sur le code et le brief) : aucune fuite trouvée.
- *Points retenus :* (1) l'écart public entre sub06 et sub07 ne valait qu'environ 1 écart-type, donc notre rejet de sub07 était trop affirmatif ; (2) décalage de densité dans les encodages par `groupby` (entraînement 0,77× la prédiction) : corrigé avec 20 plis internes (0,91×), mais pas intégré faute de temps ; (3) la slide 21 parle d'**une** soumission sélectionnée : à vérifier auprès du prof.
- *Testé et rejeté :* surplus de quota pour `moderate` (δ = 0,02 à 0,12) : jamais mieux sur plus de 2 folds sur 5. Objectif de l'affectation (−p, −√p au lieu de −log p) : identique.

**Vitesse attendue (sans étiquette) :** pour chaque (personne, sport), une régression ridge prédit log(vitesse) à partir de log(durée), de la pente, du D+/km, de la date (et son carré), de la saison et de l'heure. La feature est le résidu, c'est-à-dire « plus rapide que d'habitude **pour ce type de séance** ». Corrélation avec le résidu du modèle : +0,10. Résultats : HGB 0,6085 → 0,6121, ExtraTrees 0,6129 → 0,6163, stacking 0,6243.

**Comparaison par paires :** un HGB est entraîné sur environ 216 000 paires de séances de la même (personne, sport) aux étiquettes différentes. Cible : « la séance i est-elle plus dure que la j ? ». Features : écarts de log-vitesse, vitesse max, durée, distance, pente, D+/km, altitude, date, saison, heure, jour, résidus de vitesse, et l'écart de date absolu. Chaque séance à prédire est comparée à toutes les séances connues de la personne (400 au maximum), puis on moyenne P(plus dure) face aux séances connues easy, moderate et hard. Ce score seul a une corrélation de 0,52-0,53 avec la cible.

| Stacking | macro-F1 CV | folds |
| --- | --- | --- |
| sub06 | 0,6234 | 0,623 / 0,631 / 0,627 / 0,627 / 0,609 |
| + paires seules | 0,6239 | — |
| **+ HGB et ET avec vitesse attendue + paires** | **0,6262** | 0,626 / 0,634 / 0,629 / 0,628 / 0,614 (5/5 > sub06) |

**Soumission 9 (17:14) :** `sub09_stacking_paires.csv`, CV **0,626**, **public 0,65234** (sub06 : 0,64963). Le gain public (+0,0027) correspond au gain de validation (+0,0028).

**Progression publique :** 0,6345 → 0,640 → 0,6496 → **0,6523**.

## Étape 18 — Dernières tentatives et gel du modèle (17:35)

- **Paires enrichies** (comparaisons pondérées par la proximité temporelle, écart d'heure, même jour de semaine, écarts de rangs intra-personne) : stacking 0,6242, contre 0,6262 avec les paires d'origine. **Rejeté** : la pondération temporelle ajoute du bruit au score des paires.
- **Surplus de quota pour moderate sur le stack de sub09** (δ = 0,03 / 0,06) : 0,6262 / 0,6255. **Rejeté**, confirmé.
- **Ancres complètes pour les paires à l'inférence** (proposition de la revue externe) : légitime, puisque les écarts des paires n'utilisent aucune étiquette, mais non testée faute de temps (il faudrait réentraîner les modèles de paires pour une courbe 60 / 80 / 100 % d'ancres).

**Décision :** **sub09 est la soumission finale** (CV 0,626, gagnante sur les 5 folds face à sub06 ; public 0,6523). Le notebook `notebook_equipe_final.ipynb` reproduit ce pipeline de bout en bout, exécution de vérification en cours.
- **Ancres complètes pour les paires — testé (17:22) :** corrélation du score des paires sur le fold 0 avec 60 % / 80 % / 100 % des ancres et un plafond de 1 500 ancres : 0,5281 / 0,5293 / 0,5290 / 0,5291. **Courbe plate, abandonné** (règle : gain ≥ 0,002 entre 80 et 100 %).

**Vérification du notebook final (17:36) :** `notebook_equipe_final.ipynb` exécuté de bout en bout sans erreur. CV 0,6261 (folds 0,626 / 0,634 / 0,628 / 0,628 / 0,614). Sa soumission est identique à sub09 sur **99,97 %** des lignes (2 séances sur 7 669 diffèrent, à cause du non-déterminisme du calcul parallèle des arbres).

**Soumission 10 (17:34) :** moyenne géométrique de 4 stackings quasi équivalents, CV 0,6263 (égale à sub09), accord de 98,4 % avec sub09. C'est une variante plus robuste, pas une amélioration.

## Étape 19 — Derniers tests et conclusion (17:46)

- **Encodages à 20 plis internes** (routine et créneau ; densité d'encodage entraînement / prédiction 0,77 → 0,91) : HGB 0,6121 → **0,6165** (meilleur HGB de la journée), ExtraTrees 0,6163 → 0,6168, mais stacking 0,6249-0,6261. Rejeté pour l'ensemble.
- **Session Claude Code parallèle** (même pack, mêmes folds, reproduit notre 0,6262) : propagation d'étiquettes par graphe dans chaque (user, sport), +0,001 et 3/5 folds, rejetée ; pénalité ordinale du régresseur dans l'affectation, 2/5 folds, rejetée.
- **Pistes refusées délibérément :** retrouver le jeu de données d'origine (avec le cardio) sur internet, ce qui serait des données externes, interdites ; régler les modèles sur le classement public, ce qui n'aide pas le privé.

**Conclusion générale :** le score est piloté par trois idées structurelles :
1. comparer chaque séance à la personne et au sport ;
2. exploiter les étiquettes connues voisines (dans le temps, en ressemblance, en routine, en parcours, en paires) ;
3. les **quotas exacts** déduits de la nature de la cible (tertiles sur train + test) avec l'affectation hongroise.

Au-delà, l'information est saturée.

**Soumission finale : sub09** (CV 0,6262, public 0,65234), reproduite par `notebook_equipe.ipynb` (99,97 % identique).

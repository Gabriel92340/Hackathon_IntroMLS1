# Fiche équipe — à lire par tout le monde

On sera **tous** interrogés sur papier à 18:20 : expliquer le modèle final vaut **6 points individuels**, plus que la performance du modèle. Cette fiche est le minimum à savoir expliquer **sans ordinateur**.

---

## 1. Le problème en 3 lignes

- On prédit si une séance de sport a été **easy**, **moderate** ou **hard** pour la personne qui l'a faite.
- La note est le **macro-F1** : le F1 de chaque classe, puis la moyenne. Les 3 classes comptent autant. Répondre toujours la même classe donne 0,17.
- On a 22 728 séances avec la réponse (train) et 7 669 à prédire (test), pour 183 personnes.

## 2. L'idée principale : « dur pour qui ? »

- Courir à 12 km/h, c'est facile pour un marathonien et dur pour un débutant.
- L'appli a pris **toutes les séances d'une personne pour un sport**, les a classées de la plus facile à la plus dure (sans doute d'après le cardio, qu'on n'a pas), puis les a coupées en **3 tas égaux**.
- **Preuve :** chaque personne a 1/3 de chaque classe, même les personnes qui font deux sports, sport par sport.
- **Donc :** on compare chaque séance **aux habitudes de la personne**. On ne donne pas « 12 km/h » au modèle, mais « plus rapide que 80 % de ses séances ». La vitesse brute est très peu liée à la réponse (0,12), la vitesse comparée aux habitudes trois fois plus (0,35).

## 3. Les quotas : notre meilleure trouvaille

- Si une personne a 30 courses, elle a **exactement 10 easy, 10 moderate, 10 hard**.
- On connaît déjà les réponses de ses séances du train, donc on sait **combien de chaque classe il reste dans le test**.
- Exemple réel : 30 courses, dont 22 dans le train (10 easy, 8 moderate, 4 hard), et 8 dans le test. Les 8 du test sont forcément **0 easy, 2 moderate, 6 hard**.
- Le modèle n'a plus qu'à décider **lesquelles** sont les plus dures. La répartition est faite par l'**algorithme hongrois**, un algorithme d'affectation optimale : comme placer des invités à des tables dont on connaît le nombre de places, en respectant au mieux leurs préférences, ici les probabilités du modèle.

## 4. Les autres ingrédients

| Ingrédient | En clair | Pourquoi ça aide |
| --- | --- | --- |
| Features relatives | Rang, z-score, écart à la médiane de chaque variable dans l'historique de la personne | La cible est relative à la personne |
| Features locales | Pareil, mais comparé aux séances de la personne **à ±2 mois** | La forme évolue avec le temps |
| Forme du moment | Part de séances hard et easy de la personne **aux mêmes dates**, d'après les étiquettes connues | Il y a des périodes où tout est dur (reprise, fatigue…) |
| Modèle personnel | Vote des 15 séances connues **les plus ressemblantes** de la même personne | Chacun « souffre » à sa façon : vitesse, durée ou dénivelé |
| Routines | Étiquettes connues de la personne le **même jour de semaine / à la même heure** | Fractionné du mardi soir, sortie longue du dimanche |
| Même parcours | Vitesse et étiquettes des séances connues sur un parcours de même distance, altitude et dénivelé | Comparer à conditions égales |
| Vitesse attendue | Petite régression par (personne, sport) qui prédit la vitesse habituelle selon durée, pente, date ; on garde l'écart | Courir vite sur une longue sortie ou en côte est plus dur |

**Anti-fuite :** les features qui utilisent des réponses connues sont toujours calculées **sans la réponse de la séance elle-même**, par validation croisée interne. Sinon le modèle apprendrait à recopier la réponse et s'effondrerait sur le test.

## 5. Les modèles (soumission finale sub09)

- **HistGradientBoosting** (×2) : des arbres de décision construits les uns après les autres, chacun corrigeant les erreurs des précédents.
- **ExtraTrees** (×4 classifieurs + 1 régresseur) : des centaines d'arbres très aléatoires qui votent. Les classifieurs donnent les probabilités des 3 classes, le régresseur une note de dureté de 0 à 2.
- **Comparaison par paires :** un modèle apprend sur des paires de séances de la même personne à dire **laquelle est la plus dure**. Chaque séance est ensuite comparée à toutes les séances connues de la personne. C'est exactement la nature de la cible : un classement à l'intérieur de chaque personne.
- **Stacking :** une régression logistique (le « chef d'orchestre ») apprend à combiner tous ces avis à partir des prédictions hors fold.
- **Décision :** l'algorithme hongrois répartit les séances de chaque (personne, sport) en respectant les quotas exacts. Quand le total n'est pas divisible par 3, le tiers en trop va à hard, la seule convention compatible avec les données.
- **Familles comparées (règle « au moins 3 ») :** régression logistique, random forest, HistGradientBoosting, ExtraTrees.

### Soumission finale : sub10_stack_locaux (public 0,65456)
C'est sub09 + 3 modèles qui se trompent autrement que nos arbres, combinés par une régression logistique très régularisée (C = 0,05), puis les quotas :
- une **régression logistique locale par (personne, sport)**, soit une règle « vitesse, durée, dénivelé → dureté » propre à chaque personne ;
- un **k-NN (25 voisins) à l'intérieur de chaque (personne, sport)** ;
- un **HGB spécialisé « moderate contre le reste »**, pour notre classe la plus difficile.

Validation : 0,6277 (+0,0015 sur sub09, gain sur 4 folds sur 5), public 0,65456 (+0,0022).

## 6. Résultats

- Progression en validation croisée : 0,42 (régression logistique) → 0,55 (forêt brute) → 0,59 (relatif) → 0,611 (ensemble + quotas) → 0,617 → 0,624 (stacking) → **0,626** (paires).
- Progression publique Kaggle : 0,6345 → 0,640 → 0,6496 → **0,6523**.
- F1 par classe : easy 0,67 | moderate 0,51 | hard 0,68.
- **Erreurs :** presque toujours entre classes voisines. `moderate` est la plus difficile, parce qu'elle est coincée au milieu. Easy ↔ hard est rarement confondu (8 %).

## 7. Testé et rejeté (à savoir citer)

- **Une note de dureté seule (formulation ordinale) :** moins bonne (0,594 contre 0,605). Trois probabilités contiennent l'hésitation du modèle, une note unique la perd.
- **Régler les hyperparamètres de HGB :** pas de gain.
- **Ajouter la régression logistique à l'ensemble :** ça dégrade, elle est trop faible.
- **Z-scores robustes, plusieurs graines, créneau exact, surplus de quota pour moderate :** aucun gain.
- **Réentraîner sur 100 % du train (sub07) :** moins bon en public. Le chef d'orchestre avait appris ses poids sur des modèles moins confiants.

## 8. Les 5 phrases à savoir dire

1. « La cible est le tertile, par personne et par sport, d'une mesure cachée, donc on compare chaque séance aux habitudes de la personne. »
2. « Comme ce sont des tiers exacts sur train + test, on connaît le nombre de séances de chaque classe par personne dans le test, et on répartit avec l'algorithme hongrois. »
3. « On ajoute la forme du moment et les séances ressemblantes connues de la personne, calculées hors fold pour éviter la fuite. »
4. « Ensemble HistGradientBoosting + ExtraTrees + comparaison par paires, combiné par stacking, validé en validation croisée 5 folds : macro-F1 de 0,626, et 0,652 en public. »
5. « Les erreurs sont surtout entre classes voisines, et `moderate` est la plus difficile. »

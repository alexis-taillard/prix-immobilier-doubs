# Prédiction des prix immobiliers dans le Doubs (DVF 2021–2025)

Modèle de machine learning supervisé qui prédit le prix de vente d'une maison ou d'un appartement dans le Doubs (25) à partir de sa surface, de son nombre de pièces, de son terrain et de sa localisation.

**Résultat principal :** sur l'année 2025, jamais vue pendant l'entraînement, le meilleur modèle (LightGBM) atteint une erreur absolue moyenne de **≈ 41 000 €** (erreur relative médiane de **18 %**, R² = **0,64**), soit **−13 %** d'erreur par rapport à une baseline « prix médian de la commune ».

## Données

- Source : [Demandes de valeurs foncières géolocalisées](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/) (data.gouv.fr), département 25, ventes de 2021 à 2025.
- 184 916 lignes brutes (une ligne = un local ou une parcelle d'une vente).
- Après nettoyage : **32 148 ventes** (une ligne = une vente d'un logement).

Les cinq fichiers (le `25.csv.gz` de chaque année, renommé) sont fournis dans le dossier `data/` du dépôt, le projet tourne donc directement après un clonage :

| Année | Fichier |
|---|---|
| 2021 | `data/25_1.csv.gz` |
| 2022 | `data/25_2.csv.gz` |
| 2023 | `data/25_3.csv.gz` |
| 2024 | `data/25_4.csv.gz` |
| 2025 | `data/25_5.csv.gz` |

Pour mettre les données à jour, télécharge les nouveaux fichiers `25.csv.gz` (département 25) sur data.gouv.fr (lien ci-dessus) et renomme-les de la même façon.

## Méthode

1. **Nettoyage.** Seules les ventes simples contenant exactement un logement (maison ou appartement) sont gardées, pour que le prix corresponde bien à ce logement. Suppression des valeurs aberrantes (prix, surface, pièces, terrain) pour définir le périmètre de l'étude. Le filtre sur le prix au m², qui dépend du prix à prédire, est appliqué au jeu d'entraînement uniquement : le test n'est donc pas flatté.
2. **Variables.** Surface, pièces, terrain, dépendances, type de bien, coordonnées GPS, distances (Besançon, Montbéliard, Pontarlier, frontière suisse), date de la vente, et prix au m² médian de la commune (lissé).
3. **Pas de fuite via l'encodage de la commune.** Le prix moyen de la commune est calculé uniquement sur les données d'entraînement (en validation croisée pour le jeu d'entraînement lui-même).
4. **Évaluation temporelle.** Entraînement sur 2021–2024 (25 873 ventes), test sur 2025 (6 137 ventes). Cela simule un usage réel : prédire des ventes futures.
5. **Modèles.** Les modèles prédisent le log du prix. LightGBM est réglé par recherche aléatoire d'hyperparamètres (validation croisée à 5 plis sur le jeu d'entraînement uniquement, score = MAE sur le log du prix).

## Résultats (test sur 2025)

| Modèle | MAE (€) | RMSE (€) | R² | Erreur relative médiane | Gain MAE vs baseline |
|---|---|---|---|---|---|
| Baseline (médiane commune × surface) | 47 098 | 70 105 | 0,58 | 22,3 % | – |
| Ridge | 46 284 | 68 407 | 0,60 | 21,7 % | −1,7 % |
| Random Forest | 41 855 | 65 532 | 0,63 | 18,7 % | −11,1 % |
| **LightGBM** | **41 080** | **64 353** | **0,64** | **18,0 %** | **−12,8 %** |

> Le test 2025 est filtré uniquement sur le périmètre de l'étude (biens de 20 000 € à 1,5 M€, surface, pièces, terrain). Le filtre sur le prix au m² ne s'applique qu'à l'entraînement.

Par type de bien (LightGBM) :

| Type | MAE (€) | R² | Ventes testées |
|---|---|---|---|
| Appartement | 29 168 | 0,60 | 3 453 |
| Maison | 56 405 | 0,57 | 2 684 |

## Interprétation (SHAP)

Importance moyenne des variables dans le modèle final :

| Variable | Importance |
|---|---|
| Surface (log) | 30,9 % |
| Prix au m² de la commune | 15,1 % |
| Terrain (log) | 14,8 % |
| Latitude | 8,5 % |
| Distance à Montbéliard | 4,8 % |
| Maison / appartement | 4,2 % |

- La **surface** est le premier facteur, suivie de la **localisation** (prix de la commune et position géographique) et du **terrain**.
- La latitude et la distance à Montbéliard captent sans doute l'écart entre le nord du département (prix plus bas) et le sud, le Haut-Doubs (prix plus élevés).
- La distance à la frontière suisse pèse peu (1,7 %) : son effet est probablement déjà contenu dans le prix de la commune.

## Limites

- **Les biens chers sont sous-estimés** (au-dessus de 400 000 €) : effet classique de retour vers la moyenne des modèles.
- **Léger biais vers le bas** (pic d'erreur autour de −12 %) : l'entraînement s'arrête en 2024, le modèle ne voit sans doute pas la hausse des prix de 2025 (et les modèles d'arbres n'extrapolent pas la variable de temps).
- **Les ventes à prix très bas sont surestimées** (queue d'erreurs supérieures à +100 %) : probablement des biens à rénover ou des ventes atypiques, dont rien n'est décrit dans DVF.
- **Les maisons ont une erreur en euros presque deux fois plus élevée** que les appartements (56 k€ contre 29 k€), sans doute parce qu'elles sont plus chères, alors que le R² est proche (0,57 contre 0,60). Leur prix dépend de l'état, du terrain et de l'environnement, peu décrits dans DVF.
- **Périmètre.** Les ventes hors de la fourchette 20 000 € – 1,5 M€ ne sont pas évaluées.
- **Validation croisée non temporelle.** L'encodage de la commune et la recherche d'hyperparamètres utilisent des plis mélangés (non découpés par année), alors que le test final est bien temporel.
- Les données ne contiennent ni l'état du bien, ni le DPE, ni l'étage, ni la vue : ils expliquent probablement une grande partie de l'erreur restante.

## Pistes d'amélioration

- Ajouter des données externes : INSEE (revenus, population par commune), distances par route, gares, DPE.
- Valider l'encodage de la commune et la recherche d'hyperparamètres par année (`TimeSeriesSplit`) plutôt qu'en plis mélangés.
- Intervalles de prédiction (régression quantile) plutôt qu'une valeur unique.
- Comparer avec un petit réseau de neurones (PyTorch).

## Reproduire

```bash
pip install -r requirements.txt
jupyter notebook projet_dvf_doubs.ipynb
```

Le notebook s'exécute de bout en bout en quelques minutes (la recherche d'hyperparamètres de LightGBM est l'étape la plus longue).

## Technologies

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, matplotlib, seaborn.
